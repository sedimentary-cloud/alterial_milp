"""单格点 MILP 构建器：把一条干线的协调问题变成数学规划。

给高中生看的小课堂
------------------
先固定公共周期 C 和速度倍率 κ。
此时所有行程时间、累计时间、窗口秒值都是常数。

我们要决定：
- x：每个路口选哪个方案（0/1 变量）；
- φ：每个路口的周期偏置（秒）；
- δ：每个可调窗口端点微调多少（比例）；
- u, β：每条带什么时候开始、有多宽；
- a：每条带在每个路口使用哪个绿灯窗口；
- n：为了跨周期，带要上下平移几个周期；
- σ：软约束违反了多少。

然后写出很多线性约束，让“带”真的落在绿灯窗口里。
最后让求解器最大化总带宽、最小化违反量。

本文件按顺序完成：
C1 方案选择 -> 有效窗口 -> C2/C3 方案约束 -> C4–C8 绿带
-> C9 红带 -> C10 用户约束 -> C11 双周期一致性 -> 目标函数。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Mapping

from ..errors import GWModelError
from ..preprocess.context import GridPointContext
from ..schema.models import (BandDemandSpec, ConstraintSpec, ConstraintTemplate,
                             Direction, PlanSpec, ProblemInput, Side)
from .constraints import ConstraintMeta, ConstraintStore
from .expressions import LinearExpr
from .meta import (BalancedGroupModel, BigMTable, GreenDemandModel,
                   MarginSlackRef, ModelMeta, RedDemandModel, SoftPenaltyRef)
from .variables import GVarType, VarHandle, VariableRegistry
from .windows import EffectiveWindowBuilder, EffectiveWindowSet, RedWindowBuilder


@dataclass
class ArterialModel:
    vars: VariableRegistry
    cons: ConstraintStore
    objective: LinearExpr | None
    objective_sense: Literal["max", "min"]
    meta: ModelMeta

    # -- accessors -------------------------------------------------------------
    def v_offset(self, i: str) -> VarHandle:
        return self.meta.phi[i]

    def v_win(self, i: str, d: Direction, slot: int, side: Side) -> VarHandle:
        ew = self.meta.effective[(i, d)]
        return ew.s[slot] if side == "start" else ew.e[slot]

    def v_adjust(self, plan_id: str, d: Direction, k: int, side: Side) -> VarHandle | None:
        return self.meta.adj_vars.get((plan_id, d, k, side))

    def v_plan(self, plan_id: str) -> VarHandle:
        return self.meta.plan_vars[plan_id]

    def v_band_u(self, demand: str, q: int) -> VarHandle:
        if demand in self.meta.green:
            return self.meta.green[demand].u[q]
        return self.meta.red_demands[demand].u[q]

    def v_band_beta(self, demand: str, q: int) -> VarHandle:
        if demand in self.meta.green:
            return self.meta.green[demand].beta[q]
        return self.meta.red_demands[demand].beta[q]

    def v_band_e(self, demand: str, q: int) -> VarHandle:
        if demand in self.meta.green:
            return self.meta.green[demand].e[q]
        return self.meta.red_demands[demand].e[q]

    def v_soft(self, constraint_id: str) -> VarHandle:
        for ref in self.meta.soft:
            if ref.constraint_id == constraint_id:
                return self.vars.by_name(ref_label(ref))
        raise GWModelError(f"unknown soft constraint {constraint_id!r}")

    def expr_bandwidth(self, demand: str) -> LinearExpr:
        model = self.meta.green.get(demand) or self.meta.red_demands.get(demand)
        if model is None:
            return LinearExpr()
        expr = LinearExpr()
        for q in range(1, model.qmax + 1):
            expr.add(model.beta[q], 1.0)
        return expr

    def expr_composite(self) -> LinearExpr:
        expr = LinearExpr()
        for b in self.meta.problem.objective.band_demands:
            expr.add(self.expr_bandwidth(b.id), b.weight)
        for m in self.meta.margins:
            for h in list(m.s_vars) + list(m.e_vars):
                expr.add(h, -m.coef)
        for ref in self.meta.soft:
            if ref.owner == "composite":
                expr.add(self.vars.by_name(ref_label(ref)), -ref.coef)
        # 均衡组：把组内最小归一化带宽 z 加入目标；
        # 如果设置了 min_existing 软约束，再减去存在数短缺罚。
        for g in self.meta.balanced.values():
            expr.add(g.z, float(g.group.weight))
            pen = float(getattr(g.group, "min_existing_penalty_s", 0.0))
            if g.deficit is not None and pen > 0:
                expr.add(g.deficit, -pen)
        return expr

    def expr_loss(self) -> LinearExpr:
        expr = LinearExpr()
        for ref in self.meta.soft:
            if ref.owner == "intersection_loss":
                expr.add(self.vars.by_name(ref_label(ref)), ref.coef)
        return expr


def ref_label(ref: SoftPenaltyRef) -> str:
    return ref.label


class ModelBuilder:
    def __init__(self, problem: ProblemInput, ctx: GridPointContext) -> None:
        self.problem = problem
        self.ctx = ctx
        self.C = ctx.cycle_s
        self.vars = VariableRegistry()
        self.cons = ConstraintStore()
        self.m = BigMTable.compute(ctx)
        self.meta = ModelMeta(problem=problem, ctx=ctx, cycle_s=ctx.cycle_s,
                              anchor_intersection=problem.corridor.intersections[0])

    # ------------------------------------------------------------------
    def build(self) -> ArterialModel:
        self._build_plan_selection()
        self._build_offsets()
        self._build_adjustments()
        ew_builder = EffectiveWindowBuilder(self.problem, self.ctx, self.vars, self.cons,
                                            self.meta.plan_vars, self.meta.adj_vars, self.meta.z_vars)
        for iid in self.problem.corridor.intersections:
            for d in ("up", "down"):
                self.meta.effective[(iid, d)] = ew_builder.build(iid, d)
        self._build_plan_constraints()
        self._build_plan_window_constraints()
        self._build_constraint_templates()
        self._build_double_cycle_links()
        self._build_green_bands()
        self._build_red_bands()
        self._build_balanced_groups()
        self._build_user_constraints()
        model = ArterialModel(self.vars, self.cons, None, "max", self.meta)
        model.objective = model.expr_composite()
        model.objective_sense = "max"
        return model

    # ------------------------------------------------------------------
    # C1：每个路口必须且只能选一个方案。
    # 例如“I1 选 P1 还是 P2”就是一个 0/1 变量。
    # ------------------------------------------------------------------
    def _build_plan_selection(self) -> None:
        for iid in self.problem.corridor.intersections:
            plans = self.ctx.feasible_plans.get(iid, ())
            if not plans:
                raise GWModelError(f"no feasible plans for intersection {iid}")
            if len(plans) == 1:
                # 单方案路口：方案已固定，不需要 x 变量，也不需要 C1 约束。
                # 整个模型会直接把 x=1 代入，大幅减少 0/1 变量和 big-M 行。
                self.meta.fixed_plans[iid] = plans[0].id
                continue
            row = LinearExpr()
            for p in plans:
                h = self.vars.add(f"x[{iid},{p.id}]", 0.0, 1.0, GVarType.BINARY,
                                  {"kind": "plan_selection", "intersection": iid, "plan": p.id})
                self.meta.plan_vars[p.id] = h
                row.add(h, 1.0)
            self.cons.add(row, "==", 1.0,
                          ConstraintMeta("C1", f"one plan at {iid}", {"intersection": iid}))

    def _build_offsets(self) -> None:
        # φ_i 表示第 i 个路口的周期从哪里开始。
        # 所有路口平移同一个量不改变协调效果，所以固定第一个路口 φ=0，
        # 防止出现无穷多个等价解。
        for iid in self.problem.corridor.intersections:
            is_anchor = (iid == self.meta.anchor_intersection)
            h = self.vars.add(f"phi[{iid}]", 0.0, 0.0 if is_anchor else self.C,
                              GVarType.CONTINUOUS,
                              {"kind": "offset", "intersection": iid, "anchor": is_anchor})
            self.meta.phi[iid] = h

    def _build_adjustments(self) -> None:
        # 为每个“可调端点”建一个 δ 变量。
        # 端点秒值 = (名义比例 + δ) × C。
        # 固定端点没有 δ，也就没有 z 变量。
        for iid in self.problem.corridor.intersections:
            for p in self.ctx.feasible_plans.get(iid, ()):
                for d in ("up", "down"):
                    for k in range(p.window_count(d)):
                        for side in ("start", "end"):
                            r = p.adjust_range(d, k, side)
                            nominal = p.nominal_endpoint_r(d, k, side)
                            expr = LinearExpr.constant(nominal * self.C)
                            if r is not None:
                                h = self.vars.add(f"adj[{p.id},{d},{k},{side}]", r.lo_r, r.hi_r,
                                                  GVarType.CONTINUOUS,
                                                  {"kind": "adjustment", "plan": p.id,
                                                   "direction": d, "window": k, "side": side})
                                self.meta.adj_vars[(p.id, d, k, side)] = h
                                expr.add(h, self.C)
                            self.meta.plan_endpoint[(p.id, d, k, side)] = expr

    # ------------------------------------------------------------------
    # C2/C3 plan constraints
    # ------------------------------------------------------------------
    def _constraint_expr(self, c: ConstraintSpec, plan: PlanSpec | None = None) -> LinearExpr:
        return self._terms_expr(c.terms, constraint_id=c.id, plan=plan)

    def _terms_expr(self, terms, *, constraint_id: str,
                    plan: PlanSpec | None = None,
                    scope_d: Direction | None = None,
                    scope_k: int | None = None,
                    scope_red=None,
                    scope_rid: int | None = None) -> LinearExpr:
        """把约束项翻译成秒制 LinearExpr。

        ``plan`` 是具体方案上下文；``scope_d``/``scope_k`` 是模板展开时的当前
        方向/窗口占位符。普通 plan 约束走 ``plan``，ConstraintTemplate 走
        ``scope_d``/``scope_k``。
        """
        expr = LinearExpr()
        for t in terms:
            a = t.atom
            if a.kind == "endpoint":
                if plan is None:
                    raise GWModelError("endpoint atom referenced outside a plan constraint")
                key = (a.plan or plan.id, a.direction, a.window, a.side)
                if key not in self.meta.plan_endpoint:
                    raise GWModelError(f"unknown endpoint reference {key}")
                expr.add(self.meta.plan_endpoint[key], t.coef)
            elif a.kind == "window_duration":
                plan_id = a.plan or (plan.id if plan is not None else None)
                if plan_id is None:
                    raise GWModelError("window_duration atom requires a plan context")
                p = plan if (plan is not None and plan.id == plan_id) else self.problem.plan(plan_id)
                d = a.direction if a.direction is not None else scope_d
                k = a.window if a.window is not None else scope_k
                if d is None or k is None:
                    raise GWModelError("window_duration atom requires direction/window scope")
                if k < 0 or k >= p.window_count(d):
                    raise GWModelError(f"window_duration references unknown window {p.id}/{d}/{k}")
                start = self.meta.plan_endpoint[(p.id, d, k, "start")]
                end = self.meta.plan_endpoint[(p.id, d, k, "end")]
                expr.add(start, -t.coef)
                expr.add(end, t.coef)
            elif a.kind == "red_duration":
                if scope_red is None or scope_rid is None:
                    raise GWModelError("red_duration atom requires a red-window scope")
                start = scope_red.start[scope_rid]
                end = scope_red.end[scope_rid]
                expr.add(start, -t.coef)
                expr.add(end, t.coef)
            elif a.kind == "const":
                expr.add_const(t.coef * float(a.const or 0.0))
            elif a.kind == "offset":
                expr.add(self.meta.phi[a.intersection], t.coef)
            elif a.kind == "bandwidth":
                if a.demand not in self.meta.green and a.demand not in self.meta.red_demands:
                    raise GWModelError(f"unknown demand {a.demand!r} in constraint {constraint_id}")
                expr.add(self._bandwidth_expr(a.demand), t.coef)
            else:  # pragma: no cover
                raise GWModelError(f"unsupported atom {a.kind}")
        return expr

    def _bandwidth_expr(self, demand: str) -> LinearExpr:
        model = self.meta.green.get(demand) or self.meta.red_demands.get(demand)
        if model is None:
            return LinearExpr()
        e = LinearExpr()
        for q in range(1, model.qmax + 1):
            e.add(model.beta[q], 1.0)
        return e


    def _max_effective_end_seconds(self, intersection: str, d: Direction) -> float:
        best = 0.0
        for p in self.ctx.feasible_plans.get(intersection, ()):
            for k, w in enumerate(p.effective_windows(d)):
                r = p.adjust_range(d, k, "end")
                hi = r.hi_r if r is not None else 0.0
                best = max(best, (w.end_r + max(0.0, hi)) * self.C)
        return best

    def _green_n_lower(self, intersection: str, d: Direction) -> int:
        max_end = self._max_effective_end_seconds(intersection, d)
        # phi in [0, C]; allow enough negative periods to bring phi + window back
        # close to the non-negative band-time axis.
        return -max(1, int(math.ceil((self.C + max_end) / self.C)) + 1)

    def _constraint_activation_m(self, c: ConstraintSpec, plan: PlanSpec | None) -> float:
        # conservative bound on |lhs-rhs| over declared ranges
        total = 0.0
        for t in c.terms:
            a = t.atom
            if a.kind == "endpoint" and plan is not None:
                nom = plan.nominal_endpoint_r(a.direction, a.window, a.side)
                r = plan.adjust_range(a.direction, a.window, a.side)
                lov = (nom + (r.lo_r if r is not None else 0.0)) * self.C
                hiv = (nom + (r.hi_r if r is not None else 0.0)) * self.C
                atom_ub = max(abs(lov), abs(hiv), 3.0 * self.C)
            elif a.kind == "offset":
                atom_ub = self.C
            elif a.kind == "const":
                atom_ub = abs(float(a.const or 0.0))
            elif a.kind == "bandwidth":
                atom_ub = self.C * (1 + 4)
            else:
                atom_ub = 3.0 * self.C
            total += abs(t.coef) * atom_ub
        return max(total, abs(c.rhs_for_cycle(self.C))) + 1.0

    def _linear_bounds(self, expr: LinearExpr, phi_extra: float = 0.0) -> tuple[float, float]:
        lo = expr.const + phi_extra
        hi = expr.const + phi_extra
        for idx, coef in expr.items():
            lb = self.vars.lbs[idx]
            ub = self.vars.ubs[idx]
            if coef >= 0:
                lo += coef * lb
                hi += coef * ub
            else:
                lo += coef * ub
                hi += coef * lb
        return lo, hi

    def _window_endpoint_bounds(self, plan: PlanSpec, d: Direction, k0: int, side: Side):
        """单个方案某个有效窗口端点的秒值上下界（含可调范围）。"""
        nominal = plan.nominal_endpoint_r(d, k0, side) * self.C
        r = plan.adjust_range(d, k0, side)
        lo = nominal + (r.lo_r if r is not None else 0.0) * self.C
        hi = nominal + (r.hi_r if r is not None else 0.0) * self.C
        return (lo, hi) if lo <= hi else (hi, lo)

    def _window_slot_bounds(self, gw: EffectiveWindowSet):
        """有效窗口槽位的实际端点上下界，按方案取并集。"""
        d = gw.direction
        plans = self.ctx.feasible_plans.get(gw.intersection, ())
        out = {}
        for slot in gw.slots:
            s_mins, s_maxs, e_mins, e_maxs = [], [], [], []
            for p in plans:
                if p.window_count(d) < slot:
                    continue
                slo, shi = self._window_endpoint_bounds(p, d, slot - 1, "start")
                elo, ehi = self._window_endpoint_bounds(p, d, slot - 1, "end")
                s_mins.append(slo); s_maxs.append(shi)
                e_mins.append(elo); e_maxs.append(ehi)
            if s_mins:
                out[slot] = (min(s_mins), max(s_maxs), min(e_mins), max(e_maxs))
        return out

    def _red_slot_bounds(self, red_down: RedWindowSet):
        """下游红窗槽位的实际起止上下界（含可调范围），按方案取并集。"""
        d = red_down.direction
        plans = self.ctx.feasible_plans.get(red_down.intersection, ())
        out = {}
        for rid in red_down.slots:
            after = red_down.red_after_window[rid]
            kind = red_down.kind[rid]
            s_mins, s_maxs, e_mins, e_maxs = [], [], [], []
            for p in plans:
                wins = p.effective_windows(d)
                k_count = len(wins)
                if kind == "gap":
                    if after + 1 > k_count:
                        continue
                    slo, shi = self._window_endpoint_bounds(p, d, after - 1, "end")
                    elo, ehi = self._window_endpoint_bounds(p, d, after, "start")
                else:  # wrap
                    if after > k_count or k_count <= 0:
                        continue
                    slo, shi = self._window_endpoint_bounds(p, d, after - 1, "end")
                    elo, ehi = self._window_endpoint_bounds(p, d, 0, "start")
                    elo += self.C
                    ehi += self.C
                s_mins.append(slo); s_maxs.append(shi)
                e_mins.append(elo); e_maxs.append(ehi)
            if s_mins:
                out[rid] = (min(s_mins), max(s_maxs), min(e_mins), max(e_maxs))
        return out

    def _red_compatible_keys(self, gw_up: EffectiveWindowSet, red_down: RedWindowSet,
                             tau: float):
        """因式分解后的红波兼容性预剪枝。

        返回：(上游绿窗槽 g 的保留列表, 下游红窗实例 (r,k) 的保留列表)。

        判定用变量上下界完成，负责：
        1. 相对偏置 Δφ = φ_j - φ_i ∈ (-C,C) 是否可能重叠；
        2. 绝对时间段是否能被带区间 [u,u+β] 容纳；
        3. 若某 g 与任何 (r,k) 都不兼容，就不建 ag；
           若某 (r,k) 与任何 g 都不兼容，就不建 ar。
        """
        C = self.C
        eps = 1e-9
        up_bounds = self._window_slot_bounds(gw_up)
        red_bounds = self._red_slot_bounds(red_down)
        pairs = set()

        for g in gw_up.slots:
            if g not in up_bounds:
                continue
            s_lo, s_hi, e_lo, e_hi = up_bounds[g]
            for r in red_down.slots:
                if r not in red_bounds:
                    continue
                rs_lo, rs_hi, re_lo, re_hi = red_bounds[r]
                # 相对重叠条件：
                #   s+τ-re-kC < Δφ < e+τ-rs-kC
                # 且 Δφ ∈ (-C,C)。
                k_lo = math.floor((s_lo + tau - re_hi - C) / C + eps) + 1
                k_hi = math.ceil((e_hi + tau - rs_lo + C) / C - eps) - 1
                for k in range(k_lo, k_hi + 1):
                    # 绝对区间：
                    # 上游平移后：φ_i + s + τ，φ_i∈[0,C]（k 在下游）
                    up_lo_min = 0.0 + s_lo + tau
                    up_hi_max = C + e_hi + tau
                    # 下游红窗：φ_j + rs + kC，φ_j∈[0,C]
                    red_lo_min = 0.0 + rs_lo + k * C
                    red_hi_max = C + re_hi + k * C
                    # 和带区间 [u,u+β] 的容纳性检查。
                    ov_lo_min = max(up_lo_min, red_lo_min)
                    ov_hi_max = min(up_hi_max, red_hi_max)
                    if ov_hi_max <= ov_lo_min + eps:
                        continue
                    if ov_lo_min > C + eps:
                        continue
                    up_end_lo = 0.0 + e_lo + tau
                    red_end_lo = 0.0 + re_lo + k * C
                    if min(up_end_lo, red_end_lo) > 2.0 * C + eps:
                        continue
                    # 两个区间本身还要能和 [0,2C] 有交集。
                    if red_hi_max < -eps or red_lo_min > 2.0 * C + eps:
                        continue
                    if up_hi_max < -eps or up_lo_min > 2.0 * C + eps:
                        continue
                    pairs.add((g, r, k))

        ag_slots = tuple(sorted({g for g, _, _ in pairs}))
        ar_keys = tuple(sorted({(r, k) for _, r, k in pairs}))
        return ag_slots, ar_keys

    def _add_activated_hard(self, lhs: LinearExpr, sense: str, rhs: float,
                            x: VarHandle | None, m_big: float,
                            meta_group: str, meta_label: str,
                            refs: dict[str, str]) -> None:
        """加一条“方案选中时才生效”的硬约束。

        单方案路口 x=None，直接加；多方案时用 big-M 松开未选中的方案。
        """
        if x is None:
            self.cons.add(lhs.copy(), sense, rhs, ConstraintMeta(meta_group, meta_label, refs))
            return
        if sense in ("<=", "=="):
            row = lhs.copy()
            row.add(x, m_big)
            self.cons.add(row, "<=", rhs + m_big,
                          ConstraintMeta(meta_group, meta_label, refs))
        if sense in (">=", "=="):
            row = lhs.copy()
            row.add(x, -m_big)
            self.cons.add(row, ">=", rhs - m_big,
                          ConstraintMeta(meta_group, meta_label, refs))

    def _add_activated_soft(self, lhs: LinearExpr, sense: str, rhs: float, *,
                            coef: float, owner: str, plan_id: str | None,
                            x: VarHandle | None, m_big: float,
                            meta_group: str, meta_label_base: str,
                            name_base: str, constraint_id_base: str,
                            refs: dict[str, str]) -> None:
        """加一条“方案选中时才生效”的软约束，并登记到 meta.soft。"""
        if sense in ("<=", "=="):
            var_name = f"viol[{name_base}.pos]"
            sigma = self.vars.add(var_name, 0.0, m_big + abs(rhs) + 1.0,
                                  GVarType.CONTINUOUS, {"kind": "soft_violation"})
            row = lhs.copy()
            row.add(sigma, -1.0)
            rhs_eff = rhs
            if x is not None:
                row.add(x, m_big)
                rhs_eff = rhs + m_big
            self.cons.add(row, "<=", rhs_eff,
                          ConstraintMeta(meta_group, meta_label_base, refs))
            self.meta.soft.append(SoftPenaltyRef(
                constraint_id=constraint_id_base + (".pos" if sense == "==" else ""),
                expr=lhs.copy(), sense="<=", rhs=rhs, coef=coef, owner=owner,
                plan_id=plan_id, label=var_name))
        if sense in (">=", "=="):
            var_name = f"viol[{name_base}.neg]"
            sigma = self.vars.add(var_name, 0.0, m_big + abs(rhs) + 1.0,
                                  GVarType.CONTINUOUS, {"kind": "soft_violation"})
            row = lhs.copy()
            row.add(sigma, 1.0)
            rhs_eff = rhs
            if x is not None:
                row.add(x, -m_big)
                rhs_eff = rhs - m_big
            self.cons.add(row, ">=", rhs_eff,
                          ConstraintMeta(meta_group, meta_label_base, refs))
            self.meta.soft.append(SoftPenaltyRef(
                constraint_id=constraint_id_base + (".neg" if sense == "==" else ""),
                expr=lhs.copy(), sense=">=", rhs=rhs, coef=coef, owner=owner,
                plan_id=plan_id, label=var_name))

    def _add_activated_hard_expr(self, lhs: LinearExpr, sense: str, rhs: float,
                                  act: LinearExpr, m_big: float,
                                  meta_group: str, meta_label: str,
                                  refs: dict[str, str]) -> None:
        """用一般激活表达式 act 控制硬约束是否生效。

        act=1 时约束生效，act=0 时松弛。act 可以是 x 变量，也可以是红窗可用性
        表达式（h_r 或 h_r-h_{r+1}）。
        """
        if act.is_constant():
            if act.const <= 0.5:
                return
            self.cons.add(lhs.copy(), sense, rhs,
                          ConstraintMeta(meta_group, meta_label, refs))
            return
        if sense in ("<=", "=="):
            row = lhs.copy()
            row.add(act, -m_big)
            self.cons.add(row, "<=", rhs - m_big,
                          ConstraintMeta(meta_group, meta_label, refs))
        if sense in (">=", "=="):
            row = lhs.copy()
            row.add(act, m_big)
            self.cons.add(row, ">=", rhs + m_big,
                          ConstraintMeta(meta_group, meta_label, refs))

    def _add_activated_soft_expr(self, lhs: LinearExpr, sense: str, rhs: float,
                                 act: LinearExpr, *,
                                 coef: float, owner: str, plan_id: str | None,
                                 m_big: float, meta_group: str,
                                 meta_label_base: str, name_base: str,
                                 constraint_id_base: str,
                                 refs: dict[str, str]) -> None:
        """用一般激活表达式 act 控制软约束是否生效。"""
        if act.is_constant():
            if act.const <= 0.5:
                return
            self._add_activated_soft(
                lhs, sense, rhs, coef=coef, owner=owner, plan_id=plan_id,
                x=None, m_big=m_big, meta_group=meta_group,
                meta_label_base=meta_label_base, name_base=name_base,
                constraint_id_base=constraint_id_base, refs=refs,
            )
            return
        if sense in ("<=", "=="):
            var_name = f"viol[{name_base}.pos]"
            sigma = self.vars.add(var_name, 0.0, m_big + abs(rhs) + 1.0,
                                  GVarType.CONTINUOUS, {"kind": "soft_violation"})
            row = lhs.copy()
            row.add(sigma, -1.0)
            row.add(act, -m_big)
            self.cons.add(row, "<=", rhs - m_big,
                          ConstraintMeta(meta_group, meta_label_base, refs))
            guard = LinearExpr.of(sigma, 1.0)
            guard.add(act, -m_big)
            self.cons.add(guard, "<=", 0.0,
                          ConstraintMeta(meta_group, f"{meta_label_base}:guard", refs))
            self.meta.soft.append(SoftPenaltyRef(
                constraint_id=constraint_id_base + (".pos" if sense == "==" else ""),
                expr=lhs.copy(), sense="<=", rhs=rhs, coef=coef, owner=owner,
                plan_id=plan_id, label=var_name))
        if sense in (">=", "=="):
            var_name = f"viol[{name_base}.neg]"
            sigma = self.vars.add(var_name, 0.0, m_big + abs(rhs) + 1.0,
                                  GVarType.CONTINUOUS, {"kind": "soft_violation"})
            row = lhs.copy()
            row.add(sigma, 1.0)
            row.add(act, m_big)
            self.cons.add(row, ">=", rhs + m_big,
                          ConstraintMeta(meta_group, meta_label_base, refs))
            guard = LinearExpr.of(sigma, 1.0)
            guard.add(act, -m_big)
            self.cons.add(guard, "<=", 0.0,
                          ConstraintMeta(meta_group, f"{meta_label_base}:guard", refs))
            self.meta.soft.append(SoftPenaltyRef(
                constraint_id=constraint_id_base + (".neg" if sense == "==" else ""),
                expr=lhs.copy(), sense=">=", rhs=rhs, coef=coef, owner=owner,
                plan_id=plan_id, label=var_name))

    def _add_plan_hard(self) -> None:
        # C2：方案内硬约束。
        for iid in self.problem.corridor.intersections:
            fixed = iid in self.meta.fixed_plans
            for p in self.ctx.feasible_plans.get(iid, ()):
                x = None if fixed else self.meta.plan_vars[p.id]
                for c in p.hard_constraints:
                    rhs = c.rhs_for_cycle(self.C)
                    lhs = self._constraint_expr(c, p)
                    M = self._constraint_activation_m(c, p)
                    meta = ConstraintMeta("PLAN_HARD", f"{p.id}:{c.id}",
                                          {"plan": p.id, "constraint": c.id})
                    self._add_activated_hard(lhs, c.sense, rhs, x, M,
                                             meta.group, meta.label, meta.refs)
                    self.meta.plan_constraints.append((c.id, meta))

    def _add_plan_soft(self) -> None:
        # C3：方案内软约束。
        for iid in self.problem.corridor.intersections:
            fixed = iid in self.meta.fixed_plans
            for p in self.ctx.feasible_plans.get(iid, ()):
                x = None if fixed else self.meta.plan_vars[p.id]
                for c in p.soft_constraints:
                    assert c.soft is not None
                    rhs = c.rhs_for_cycle(self.C)
                    lhs = self._constraint_expr(c, p)
                    M = self._constraint_activation_m(c, p)
                    self._add_activated_soft(
                        lhs, c.sense, rhs,
                        coef=c.soft.coef, owner=c.soft.owner, plan_id=p.id,
                        x=x, m_big=M,
                        meta_group="PLAN_SOFT", meta_label_base=f"{p.id}:{c.id}",
                        name_base=f"{p.id}.{c.id}", constraint_id_base=c.id,
                        refs={"plan": p.id, "constraint": c.id},
                    )

    def _red_windows_for_template(self, iid: str, d: Direction):
        """按需构建并缓存某路口方向的红窗补集。"""
        key = (iid, d)
        cached = self.meta.red.get(key)
        if cached is not None:
            return cached
        gw = self.meta.effective[key]
        if not gw.slots:
            raise GWModelError(f"cannot build red windows for {iid}/{d}: no green windows")
        red = RedWindowBuilder(self.C, self.cons).build(gw)
        self.meta.red[key] = red
        return red

    def _nominal_red_duration_s(self, p: PlanSpec, d: Direction, red, rid: int) -> float:
        """用方案的名义窗口估算某个红窗区间的秒数，用于跳过 (0,0) 退化区间。"""
        r = red.red_after_window[rid]
        wins = p.effective_windows(d)
        kind = red.kind[rid]
        if kind == "gap":
            if r >= len(wins):
                return math.nan
            return (wins[r].start_r - wins[r - 1].end_r) * self.C
        if r > len(wins):
            return math.nan
        return (1.0 + wins[0].start_r - wins[r - 1].end_r) * self.C

    def _iter_template_red_scope(self, tmpl: ConstraintTemplate):
        """Yield concrete ``(plan, direction, red_set, red_interval_id)`` tuples."""
        scope = tmpl.scope
        for iid in self.problem.corridor.intersections:
            for p in self.ctx.feasible_plans.get(iid, ()):
                if scope.plan_ids is not None and p.id not in scope.plan_ids:
                    continue
                for d in scope.directions:
                    red = self._red_windows_for_template(iid, d)
                    for rid in red.slots:
                        if scope.window_indices is not None and rid not in scope.window_indices:
                            continue
                        dur = self._nominal_red_duration_s(p, d, red, rid)
                        if not math.isfinite(dur) or dur <= 1e-9:
                            # 跳过 (0,0) / 名义上不存在的红窗区间。
                            continue
                        yield p, d, red, rid

    def _iter_template_scope(self, tmpl: ConstraintTemplate):
        """Yield concrete ``(plan, direction, effective_window_index)`` tuples."""
        scope = tmpl.scope
        for iid in self.problem.corridor.intersections:
            for p in self.ctx.feasible_plans.get(iid, ()):
                if scope.plan_ids is not None and p.id not in scope.plan_ids:
                    continue
                for d in scope.directions:
                    for k in range(p.window_count(d)):
                        if scope.skip_double_cycle_copies and p.effective_is_copy(d, k):
                            continue
                        if scope.window_indices is not None:
                            orig = p.effective_original_index(d, k)
                            if orig not in scope.window_indices:
                                continue
                        yield p, d, k

    def _build_constraint_templates(self) -> None:
        """展开 ConstraintTemplate：统一处理绝对秒/比例 RHS 和作用域。"""
        for tmpl in self.problem.constraint_templates:
            rhs = tmpl.rhs_for_cycle(self.C)
            M = self.m.containment + abs(rhs) + 1.0
            if tmpl.scope.kind == "all_plan_red_windows":
                for p, d, red, rid in self._iter_template_red_scope(tmpl):
                    lhs = self._terms_expr(
                        tmpl.terms, constraint_id=tmpl.id,
                        plan=p, scope_d=d, scope_red=red, scope_rid=rid,
                    )
                    act = red.avail[rid]
                    label = f"{tmpl.id}:{p.id}:{d}:red{rid}"
                    refs = {"template": tmpl.id, "plan": p.id,
                            "direction": d, "red_window": str(rid)}
                    if tmpl.hard:
                        self._add_activated_hard_expr(lhs, tmpl.sense, rhs, act, M,
                                                      "TEMPLATE_HARD", label, refs)
                    else:
                        assert tmpl.soft is not None
                        self._add_activated_soft_expr(
                            lhs, tmpl.sense, rhs, act,
                            coef=tmpl.soft.coef, owner=tmpl.soft.owner,
                            plan_id=p.id, m_big=M,
                            meta_group="TEMPLATE_SOFT", meta_label_base=label,
                            name_base=label, constraint_id_base=label, refs=refs,
                        )
                continue

            for p, d, k in self._iter_template_scope(tmpl):
                lhs = self._terms_expr(
                    tmpl.terms, constraint_id=tmpl.id,
                    plan=p, scope_d=d, scope_k=k,
                )
                fixed = p.intersection in self.meta.fixed_plans
                x = None if fixed else self.meta.plan_vars[p.id]
                label = f"{tmpl.id}:{p.id}:{d}:{k}"
                refs = {"template": tmpl.id, "plan": p.id,
                        "direction": d, "window": str(k)}
                if tmpl.hard:
                    self._add_activated_hard(lhs, tmpl.sense, rhs, x, M,
                                             "TEMPLATE_HARD", label, refs)
                else:
                    assert tmpl.soft is not None
                    self._add_activated_soft(
                        lhs, tmpl.sense, rhs,
                        coef=tmpl.soft.coef, owner=tmpl.soft.owner,
                        plan_id=p.id, x=x, m_big=M,
                        meta_group="TEMPLATE_SOFT", meta_label_base=label,
                        name_base=label, constraint_id_base=label, refs=refs,
                    )

    def _build_plan_constraints(self) -> None:
        self._add_plan_hard()
        self._add_plan_soft()

    # ------------------------------------------------------------------
    # plan-level window ordering (V1/V2 after adjustment)
    # ------------------------------------------------------------------
    def _build_plan_window_constraints(self) -> None:
        # 保证选中方案的窗口在微调后仍然合法：
        # start >= 0，end <= 2C，start <= end，相邻窗口不重叠。
        # 单方案路口可直接添加这些约束，不需要 big-M。
        for iid in self.problem.corridor.intersections:
            fixed = iid in self.meta.fixed_plans
            for p in self.ctx.feasible_plans.get(iid, ()):
                x = None if fixed else self.meta.plan_vars[p.id]
                M = self.m.plan_activation
                for d in ("up", "down"):
                    n = p.window_count(d)
                    for k in range(n):
                        s_expr = self.meta.plan_endpoint[(p.id, d, k, "start")]
                        e_expr = self.meta.plan_endpoint[(p.id, d, k, "end")]
                        for label, row, sense, rhs in (
                            ("start_min", s_expr.copy(), ">=", 0.0),
                            ("end_max", e_expr.copy(), "<=", 2.0 * self.C),
                        ):
                            if x is not None:
                                row.add(x, -M if sense == ">=" else M)
                                rhs_eff = -M if sense == ">=" else 2.0 * self.C + M
                            else:
                                rhs_eff = rhs
                            self.cons.add(row, sense, rhs_eff,
                                          ConstraintMeta("PLAN_WINDOW", f"{p.id}:{d}:{k}:{label}",
                                                         {"plan": p.id}))
                        row = s_expr.copy(); row.add(e_expr, -1.0)
                        if x is not None:
                            row.add(x, M)
                            self.cons.add(row, "<=", M,
                                          ConstraintMeta("PLAN_WINDOW", f"{p.id}:{d}:{k}:order",
                                                         {"plan": p.id}))
                        else:
                            self.cons.add(row, "<=", 0.0,
                                          ConstraintMeta("PLAN_WINDOW", f"{p.id}:{d}:{k}:order",
                                                         {"plan": p.id}))
                    for k in range(n - 1):
                        a = self.meta.plan_endpoint[(p.id, d, k, "end")]
                        b = self.meta.plan_endpoint[(p.id, d, k + 1, "start")]
                        row = a.copy(); row.add(b, -1.0)
                        if x is not None:
                            row.add(x, M)
                            self.cons.add(row, "<=", M,
                                          ConstraintMeta("PLAN_WINDOW", f"{p.id}:{d}:{k}:nonoverlap",
                                                         {"plan": p.id}))
                        else:
                            self.cons.add(row, "<=", 0.0,
                                          ConstraintMeta("PLAN_WINDOW", f"{p.id}:{d}:{k}:nonoverlap",
                                                         {"plan": p.id}))

    # ------------------------------------------------------------------
    # C11 double cycle
    # ------------------------------------------------------------------
    def _build_double_cycle_links(self) -> None:
        # C11：双周期方案的复制窗口必须和原始窗口保持同样的端点微调量。
        # 否则求解器会让两个半周期不对称，等价于偷偷变成全周期方案。
        for iid in self.problem.corridor.intersections:
            for p in self.ctx.feasible_plans.get(iid, ()):
                if not p.double_cycle:
                    continue
                for d in ("up", "down"):
                    orig_to_eff: dict[int, int] = {}
                    for k in range(p.window_count(d)):
                        if not p.effective_is_copy(d, k):
                            orig_to_eff[p.effective_original_index(d, k)] = k
                    for k in range(p.window_count(d)):
                        if not p.effective_is_copy(d, k):
                            continue
                        orig = p.effective_original_index(d, k)
                        ko = orig_to_eff.get(orig)
                        if ko is None:
                            continue
                        for side in ("start", "end"):
                            hc = self.meta.adj_vars.get((p.id, d, k, side))
                            ho = self.meta.adj_vars.get((p.id, d, ko, side))
                            if hc is None or ho is None:
                                continue
                            row = LinearExpr.of(hc, 1.0).add(ho, -1.0)
                            self.cons.add(row, "==", 0.0,
                                          ConstraintMeta("C11", f"{p.id}:{d}:{k}:{side}:double_copy",
                                                         {"plan": p.id, "direction": d,
                                                          "copy_window": str(k), "orig_window": str(ko)}))

    # ------------------------------------------------------------------
    # C4-C8 green bands
    # ------------------------------------------------------------------
    def _build_green_bands(self) -> None:
        # C4–C8：绿波带。
        # 绿波带是“一条连续的时间区间，沿链上每个路口都落在某个绿灯窗口内”。
        # 带宽由链上最紧的窗口决定，不需要额外写 min 函数。
        for b in self.problem.objective.band_demands:
            if b.type != "green":
                continue
            qmax = self.ctx.demand_max_bands(b)
            if qmax <= 0:
                continue
            table = self.ctx.travel[b.id]
            d = b.direction
            u = {q: self.vars.add(f"band[{b.id},{q}].u", 0.0, self.C, GVarType.CONTINUOUS,
                                   {"kind": "band_start", "demand": b.id, "slot": q})
                 for q in range(1, qmax + 1)}
            beta = {q: self.vars.add(f"band[{b.id},{q}].beta", 0.0, self.C, GVarType.CONTINUOUS,
                                      {"kind": "band_width", "demand": b.id, "slot": q})
                    for q in range(1, qmax + 1)}
            e = {q: self.vars.add(f"band[{b.id},{q}].e", 0.0, 1.0, GVarType.BINARY,
                                   {"kind": "band_exists", "demand": b.id, "slot": q})
                 for q in range(1, qmax + 1)}
            a: dict[tuple[int, int, int], VarHandle] = {}
            n: dict[tuple[int, int], VarHandle] = {}
            for q in range(1, qmax + 1):
                for j, node in enumerate(b.nodes):
                    n[(q, j)] = self.vars.add(f"n[{b.id},{q},{node}]",
                                              float(self._green_n_lower(node, d)),
                                              float(table.n_max[j] + 2),
                                              GVarType.INTEGER,
                                              {"kind": "wrap", "demand": b.id, "slot": q, "node": node})
                    for k in range(1, self.ctx.max_window_slots(node, d) + 1):
                        a[(q, j, k)] = self.vars.add(f"asg[{b.id},{q},{node},{k}]", 0.0, 1.0,
                                                     GVarType.BINARY,
                                                     {"kind": "green_assignment", "demand": b.id,
                                                      "slot": q, "node": node, "window": k})
            gmodel = GreenDemandModel(b, qmax, u, beta, e, a, table)
            self.meta.green[b.id] = gmodel

            for q in range(1, qmax + 1):
                # C4：如果这条带存在 e=1，那么链上每个节点都必须恰好
                # 选择一个绿灯窗口槽位；如果 e=0，就一个也不选。
                # 这就是 Σ_k a = e 的含义。
                for j, node in enumerate(b.nodes):
                    row = LinearExpr()
                    for k in range(1, self.ctx.max_window_slots(node, d) + 1):
                        row.add(a[(q, j, k)], 1.0)
                    row.add(e[q], -1.0)
                    self.cons.add(row, "==", 0.0,
                                  ConstraintMeta("C4", f"assign {b.id}/{q}/{node}",
                                                 {"demand": b.id, "slot": str(q), "node": node}))
                    # 只能选择“被选中方案真正提供”的窗口槽位。
                    # h=0 的槽位不能使用。
                    for k in range(1, self.ctx.max_window_slots(node, d) + 1):
                        row = LinearExpr.of(a[(q, j, k)], 1.0)
                        row.add(self.meta.effective[(node, d)].h[k], -1.0)
                        self.cons.add(row, "<=", 0.0,
                                      ConstraintMeta("C4", f"a<=h {b.id}/{q}/{node}/{k}",
                                                     {"demand": b.id, "node": node, "window": str(k)}))
                    # C5：带的包含约束。
                    # 带在第 j 个节点的到达时间是 u + T_j；
                    # 它必须 ≥ 窗口起点，且 ≤ 窗口终点。
                    # n 是整数周期，用来把带平移到正确的周期副本。
                    for k in range(1, self.ctx.max_window_slots(node, d) + 1):
                        s = self.meta.effective[(node, d)].s[k]
                        ew = self.meta.effective[(node, d)].e[k]
                        phi = self.meta.phi[node]
                        nv = n[(q, j)]
                        av = a[(q, j, k)]
                        M = self.m.containment
                        # 下方不等式：带在节点 j 的到达时间 u+T_j
                        # 必须不小于窗口起点 φ+s+nC。
                        # 公式：u + T - φ - s - nC - M(1-a) >= 0。
                        row = LinearExpr.of(u[q], 1.0).add(phi, -1.0).add(s, -1.0)
                        row.add(nv, -self.C).add(av, -M)
                        self.cons.add(row, ">=", -table.cum_T_s[j] - M,
                                      ConstraintMeta("C5", f"band lower {b.id}/{q}/{node}/{k}",
                                                     {"demand": b.id, "slot": str(q), "node": node,
                                                      "window": str(k)}))
                        # 上方不等式：带在节点 j 的离开时间 u+T_j+β
                        # 必须不大于窗口终点 φ+e+nC。
                        # 公式：u + β + T - φ - e - nC + M(1-a) <= 0。
                        row = LinearExpr.of(u[q], 1.0).add(beta[q], 1.0).add(phi, -1.0)
                        row.add(ew, -1.0).add(nv, -self.C).add(av, M)
                        self.cons.add(row, "<=", M - table.cum_T_s[j],
                                      ConstraintMeta("C5", f"band upper {b.id}/{q}/{node}/{k}",
                                                     {"demand": b.id, "slot": str(q), "node": node,
                                                      "window": str(k)}))
                # C6：带不存在时带宽必须为 0。
                # β <= C · e 保证 e=0 时 β=0。
                row = LinearExpr.of(beta[q], 1.0).add(e[q], -self.C)
                self.cons.add(row, "<=", 0.0,
                              ConstraintMeta("C6", f"beta<=C*e {b.id}/{q}",
                                             {"demand": b.id, "slot": str(q)}))
                # C7：多条带按起点排序，避免同一个方案被重复计数。
                # 如果两条带的路径完全一样，它们可以合并，所以不需要同时存在。
                if q < qmax:
                    M = self.m.containment
                    row = LinearExpr.of(u[q], 1.0).add(u[q + 1], -1.0)
                    row.add(e[q], M).add(e[q + 1], M)
                    self.cons.add(row, "<=", 2.0 * M,
                                  ConstraintMeta("C7", f"u order {b.id}/{q}",
                                                 {"demand": b.id, "slot": str(q)}))
                # C7：窗口独占。
                # 同一需求的两条带不能在同一路口使用同一个窗口槽位。
                # 如果打开 allow_window_sharing，就改成“同一槽位时间不重叠”。
                if not self.problem.objective.allow_window_sharing:
                    for j, node in enumerate(b.nodes):
                        for k in range(1, self.ctx.max_window_slots(node, d) + 1):
                            row = LinearExpr()
                            for qq in range(1, qmax + 1):
                                row.add(a[(qq, j, k)], 1.0)
                            self.cons.add(row, "<=", 1.0,
                                          ConstraintMeta("C7", f"window exclusive {b.id}/{node}/{k}",
                                                         {"demand": b.id, "node": node, "window": str(k)}))
                else:
                    # Conditional non-overlap for pairs sharing a window slot.
                    for j, node in enumerate(b.nodes):
                        for k in range(1, self.ctx.max_window_slots(node, d) + 1):
                            for q1 in range(1, qmax + 1):
                                for q2 in range(q1 + 1, qmax + 1):
                                    M = self.m.containment
                                    # u_q1+beta_q1 - u_q2 - T? at node times cancel T
                                    row = LinearExpr.of(u[q1], 1.0).add(beta[q1], 1.0)
                                    row.add(u[q2], -1.0)
                                    row.add(a[(q1, j, k)], M).add(a[(q2, j, k)], M)
                                    self.cons.add(row, "<=", 2.0 * M,
                                                  ConstraintMeta("C7", f"sharing disjoint {b.id}/{node}/{k}",
                                                                 {"demand": b.id, "node": node,
                                                                  "window": str(k)}))
                # C8：margin 短缺。
                # 带离绿灯窗口左右边界太近时，把不足的部分记成 σ，
                # 再乘以系数罚到 composite 目标里。
                margin = self.problem.margin_for(b)
                if margin.delta_min_s > 0 and margin.coef > 0:
                    ref = MarginSlackRef(b.id, margin.coef)
                    for side in ("s", "e"):
                        for j, node in enumerate(b.nodes):
                            for k in range(1, self.ctx.max_window_slots(node, d) + 1):
                                var = self.vars.add(f"marg[{b.id},{q},{node},{k}].{side}", 0.0,
                                                    self.m.containment * 2.0, GVarType.CONTINUOUS,
                                                    {"kind": "margin_shortfall", "demand": b.id,
                                                     "slot": q, "node": node, "window": k, "side": side})
                                if side == "s":
                                    ref.s_vars.append(var)
                                else:
                                    ref.e_vars.append(var)
                                s = self.meta.effective[(node, d)].s[k]
                                ew = self.meta.effective[(node, d)].e[k]
                                phi = self.meta.phi[node]
                                nv = n[(q, j)]
                                av = a[(q, j, k)]
                                M = self.m.soft
                                if side == "s":
                                    # 左 margin 短缺：带宽起点离窗口起点太近。
                                    # σ_left >= δ_min - (u+T-φ-s-nC) - M(1-a)。
                                    # sigma + u - phi - s - nC - M*a >= delta - T - M
                                    row = LinearExpr.of(var, 1.0).add(u[q], 1.0).add(phi, -1.0)
                                    row.add(s, -1.0).add(nv, -self.C).add(av, -M)
                                    self.cons.add(row, ">=", margin.delta_min_s - table.cum_T_s[j] - M,
                                                  ConstraintMeta("C8", f"margin start {b.id}/{q}/{node}/{k}",
                                                                 {"demand": b.id, "slot": str(q), "node": node,
                                                                  "window": str(k)}))
                                else:
                                    # 右 margin 短缺：带宽终点离窗口终点太近。
                                    # σ_right >= δ_min - (φ+e+nC-u-T-β) - M(1-a)。
                                    # sigma + phi + e + nC - u - T - beta - M*a >= delta - M
                                    row = LinearExpr.of(var, 1.0).add(phi, 1.0).add(ew, 1.0)
                                    row.add(nv, self.C).add(u[q], -1.0).add(beta[q], -1.0)
                                    row.add(av, -M)
                                    self.cons.add(row, ">=", margin.delta_min_s + table.cum_T_s[j] - M,
                                                  ConstraintMeta("C8", f"margin end {b.id}/{q}/{node}/{k}",
                                                                 {"demand": b.id, "slot": str(q), "node": node,
                                                                  "window": str(k)}))
                    self.meta.margins.append(ref)

    # ------------------------------------------------------------------
    # C9 red bands
    # ------------------------------------------------------------------
    def _build_red_bands(self) -> None:
        # C9 因式分解版：
        #   ag[q,g]      带槽 q 使用上游绿窗槽 g
        #   ar[q,r,k]    带槽 q 使用下游红窗实例 (r,k)
        # 两组变量通过共享的 u、β 自然耦合。
        for b in self.problem.objective.band_demands:
            if b.type != "red":
                continue
            assert len(b.nodes) == 2
            up_node, down_node = b.nodes
            d = b.direction
            tau = self.ctx.tau_s.get((up_node, d))
            if self.ctx.max_window_slots(up_node, d) <= 0 or self.ctx.max_window_slots(down_node, d) <= 0:
                continue
            if tau is None:
                raise GWModelError(f"no travel time for red demand {b.id}")

            gw_up = self.meta.effective[(up_node, d)]
            gw_down = self.meta.effective[(down_node, d)]
            red_down = RedWindowBuilder(self.C, self.cons).build(gw_down)
            self.meta.red[(down_node, d)] = red_down

            # 预剪枝：保留与某个对方槽位可能配对的 g 和 (r,k)。
            ag_slots, ar_keys = self._red_compatible_keys(gw_up, red_down, tau)

            qmax = self.ctx.demand_max_bands(b)
            u = {q: self.vars.add(f"red[{b.id},{q}].u", 0.0, self.C, GVarType.CONTINUOUS,
                                   {"kind": "red_band_start", "demand": b.id, "slot": q})
                 for q in range(1, qmax + 1)}
            beta = {q: self.vars.add(f"red[{b.id},{q}].beta", 0.0, self.C, GVarType.CONTINUOUS,
                                      {"kind": "red_band_width", "demand": b.id, "slot": q})
                    for q in range(1, qmax + 1)}
            e = {q: self.vars.add(f"red[{b.id},{q}].e", 0.0, 1.0, GVarType.BINARY,
                                   {"kind": "red_band_exists", "demand": b.id, "slot": q})
                 for q in range(1, qmax + 1)}

            ag: dict[tuple[int, int], VarHandle] = {}
            ar: dict[tuple[int, int, int], VarHandle] = {}
            for q in range(1, qmax + 1):
                for g in ag_slots:
                    ag[(q, g)] = self.vars.add(
                        f"ag[{b.id},{q},{g}]", 0.0, 1.0, GVarType.BINARY,
                        {"kind": "red_up_assignment", "demand": b.id, "slot": q, "up_window": g})
                for (r, k) in ar_keys:
                    ar[(q, r, k)] = self.vars.add(
                        f"ar[{b.id},{q},{r},{k}]", 0.0, 1.0, GVarType.BINARY,
                        {"kind": "red_down_assignment", "demand": b.id, "slot": q,
                         "red_rid": r, "shift": k})

            rmodel = RedDemandModel(b, qmax, u, beta, e, ag, ar, gw_up, red_down,
                                    ag_slots, ar_keys)
            self.meta.red_demands[b.id] = rmodel

            M = self.m.containment
            # 上游窗口宽度上界，用于 C13 有效不等式。
            up_bounds = self._window_slot_bounds(gw_up)
            red_bounds = self._red_slot_bounds(red_down)
            Wg = {g: max(0.0, up_bounds[g][3] - up_bounds[g][0]) for g in ag_slots}
            Wr = {(r, k): min(self.C, max(0.0, red_bounds[r][3] - red_bounds[r][0]))
                  for (r, k) in ar_keys}

            for q in range(1, qmax + 1):
                # 指派约束：存在 e 时，上下游各恰好选一个。
                row = LinearExpr()
                for g in ag_slots:
                    row.add(ag[(q, g)], 1.0)
                row.add(e[q], -1.0)
                self.cons.add(row, "==", 0.0,
                              ConstraintMeta("C9", f"red ag assign {b.id}/{q}",
                                             {"demand": b.id, "slot": str(q)}))

                row = LinearExpr()
                for (r, k) in ar_keys:
                    row.add(ar[(q, r, k)], 1.0)
                row.add(e[q], -1.0)
                self.cons.add(row, "==", 0.0,
                              ConstraintMeta("C9", f"red ar assign {b.id}/{q}",
                                             {"demand": b.id, "slot": str(q)}))

                # 可用性：只能选被选中方案提供的窗口/红窗。
                for g in ag_slots:
                    row = LinearExpr.of(ag[(q, g)], 1.0).add(gw_up.h[g], -1.0)
                    self.cons.add(row, "<=", 0.0,
                                  ConstraintMeta("C9", f"red ag<=h {b.id}/{q}/{g}",
                                                 {"demand": b.id, "slot": str(q)}))
                for (r, k) in ar_keys:
                    row = LinearExpr.of(ar[(q, r, k)], 1.0).add(red_down.avail[r], -1.0)
                    self.cons.add(row, "<=", 0.0,
                                  ConstraintMeta("C9", f"red ar<=redavail {b.id}/{q}/{r}/{k}",
                                                 {"demand": b.id, "slot": str(q)}))

                # 上游两条包含约束，只挂 ag。
                for g in ag_slots:
                    av = ag[(q, g)]
                    row = LinearExpr.of(u[q], 1.0).add(self.meta.phi[up_node], -1.0)
                    row.add(gw_up.s[g], -1.0).add(av, -M)
                    self.cons.add(row, ">=", tau - M,
                                  ConstraintMeta("C9", f"red up lower {b.id}/{q}/{g}",
                                                 {"demand": b.id, "slot": str(q)}))
                    row = LinearExpr.of(u[q], 1.0).add(beta[q], 1.0)
                    row.add(self.meta.phi[up_node], -1.0).add(gw_up.e[g], -1.0).add(av, M)
                    self.cons.add(row, "<=", tau + M,
                                  ConstraintMeta("C9", f"red up upper {b.id}/{q}/{g}",
                                                 {"demand": b.id, "slot": str(q)}))

                # 下游两条包含约束，只挂 ar；k 是下游红窗的周期偏移。
                for (r, k) in ar_keys:
                    av = ar[(q, r, k)]
                    row = LinearExpr.of(u[q], 1.0).add(self.meta.phi[down_node], -1.0)
                    row.add(red_down.start[r], -1.0).add(av, -M)
                    self.cons.add(row, ">=", k * self.C - M,
                                  ConstraintMeta("C9", f"red down lower {b.id}/{q}/{r}/{k}",
                                                 {"demand": b.id, "slot": str(q)}))
                    row = LinearExpr.of(u[q], 1.0).add(beta[q], 1.0)
                    row.add(self.meta.phi[down_node], -1.0)
                    row.add(red_down.end[r], -1.0).add(av, M)
                    self.cons.add(row, "<=", k * self.C + M,
                                  ConstraintMeta("C9", f"red down upper {b.id}/{q}/{r}/{k}",
                                                 {"demand": b.id, "slot": str(q)}))

                # 存在性：带宽不能超过 C·e。
                row = LinearExpr.of(beta[q], 1.0).add(e[q], -self.C)
                self.cons.add(row, "<=", 0.0,
                              ConstraintMeta("C9", f"red beta<=C*e {b.id}/{q}",
                                             {"demand": b.id, "slot": str(q)}))

                # C13 有效不等式：带宽不能超过所选上游绿窗/下游红窗的最大宽度。
                row = LinearExpr.of(beta[q], 1.0).add(e[q], self.C)
                for g in ag_slots:
                    row.add(ag[(q, g)], -Wg[g])
                self.cons.add(row, "<=", self.C,
                              ConstraintMeta("C13", f"red width<=up {b.id}/{q}",
                                             {"demand": b.id, "slot": str(q)}))
                row = LinearExpr.of(beta[q], 1.0).add(e[q], self.C)
                for (r, k) in ar_keys:
                    row.add(ar[(q, r, k)], -Wr[(r, k)])
                self.cons.add(row, "<=", self.C,
                              ConstraintMeta("C13", f"red width<=down {b.id}/{q}",
                                             {"demand": b.id, "slot": str(q)}))

                # 时间分离：任意两条都存在的红带不能重叠。
                # 不能只加相邻 q，否则 e_q=0 时 q 与 q+2 仍可能重叠。
                # 按 q 索引强制时间先后，slot 本身对称，因此不丢最优解。
                for q2 in range(q + 1, qmax + 1):
                    row = LinearExpr.of(u[q], 1.0).add(beta[q], 1.0)
                    row.add(u[q2], -1.0).add(e[q], M).add(e[q2], M)
                    self.cons.add(row, "<=", 2.0 * M,
                                  ConstraintMeta("C9", f"red time separation {b.id}/{q}/{q2}",
                                                 {"demand": b.id, "slot": str(q)}))

    # ------------------------------------------------------------------
    # C10 user constraints
    # ------------------------------------------------------------------
    def _build_balanced_groups(self) -> None:
        """构建均衡组目标：最大化组内“归一化总带宽”的最小值 z。

        约束形式：
            z_g <= Β_b / target_b        对所有成员 b
        其中 Β_b = Σ_q β_{b,q} 是需求的聚合带宽。

        on_infeasible="zero" 时，任意成员缺失都会让 z=0；
        on_infeasible="skip" 时，只对存在的成员取最小值，
        并可选地用 min_existing 软约束惩罚“丢成员”行为。
        """
        C = self.C
        for g in self.problem.objective.balanced_groups:
            policy = g.on_infeasible
            band_models = {}
            for did in g.demands:
                bm = self.meta.green.get(did) or self.meta.red_demands.get(did)
                band_models[did] = bm

            # z 的上界：如果 targets 都是 1，就是最大可能带宽；
            # 如果 targets 非 1，按归一化后的值估计。
            def _target_ub(bm, target: float) -> float:
                q = getattr(bm, "qmax", 0) if bm is not None else 0
                return (C * q) / target if target > 0 else C * q

            z_ub = max([_target_ub(band_models[did], g.target_for(did))
                        for did in g.demands] + [1.0])
            z = self.vars.add(f"balance[{g.id}].z", 0.0, z_ub, GVarType.CONTINUOUS,
                              {"kind": "balanced_min_bandwidth", "group": g.id})
            y_vars: dict[str, VarHandle] = {}
            bw_exprs: dict[str, LinearExpr] = {}
            target_vars: dict[str, float] = {}

            for did in g.demands:
                bm = band_models[did]
                target = g.target_for(did)
                target_vars[did] = target
                y = self.vars.add(f"balance[{g.id}].y[{did}]", 0.0, 1.0, GVarType.BINARY,
                                  {"kind": "balanced_demand_present", "group": g.id, "demand": did})
                y_vars[did] = y
                if bm is None:
                    # 当前格点下这个需求结构上不存在，y 固定 0。
                    self.cons.add(LinearExpr.of(y, 1.0), "==", 0.0,
                                  ConstraintMeta("BALANCED", f"{g.id}:{did}:missing",
                                                 {"group": g.id, "demand": did}))
                    bw = LinearExpr()
                else:
                    bw = self._bandwidth_expr(did)
                    bw_exprs[did] = bw
                    # y_b 是“至少存在一条带”的 OR：
                    #   y_b >= e_bq，且 y_b <= Σ_q e_bq。
                    for q in range(1, bm.qmax + 1):
                        row = LinearExpr.of(y, 1.0).add(bm.e[q], -1.0)
                        self.cons.add(row, ">=", 0.0,
                                      ConstraintMeta("BALANCED", f"{g.id}:{did}:y_or_{q}",
                                                     {"group": g.id, "demand": did, "slot": str(q)}))
                    row = LinearExpr.of(y, 1.0)
                    for q in range(1, bm.qmax + 1):
                        row.add(bm.e[q], -1.0)
                    self.cons.add(row, "<=", 0.0,
                                  ConstraintMeta("BALANCED", f"{g.id}:{did}:y_sum",
                                                 {"group": g.id, "demand": did}))

                # 归一化约束：z <= Β_b / target_b。
                # 乘以 target_b 变成线性形式：
                #   target_b·z <= Β_b + target_b·M·(1-y_b)
                # 即
                #   target_b·z - Β_b + target_b·M·y_b <= target_b·M。
                M = z_ub
                row = LinearExpr.of(z, target)
                row.add(bw, -1.0)
                row.add(y, target * M)
                self.cons.add(row, "<=", target * M,
                              ConstraintMeta("BALANCED", f"{g.id}:{did}:z_le_bandwidth",
                                             {"group": g.id, "demand": did}))

                # 严格模式：任一成员缺失则 z=0。
                if policy == "zero":
                    row = LinearExpr.of(z, 1.0).add(y, -M)
                    self.cons.add(row, "<=", 0.0,
                                  ConstraintMeta("BALANCED", f"{g.id}:{did}:missing_zero",
                                                 {"group": g.id, "demand": did}))

            # 跳过模式：至少有一个成员存在，z 才可能为正。
            if policy == "skip":
                row = LinearExpr.of(z, 1.0)
                for y in y_vars.values():
                    row.add(y, -z_ub)
                self.cons.add(row, "<=", 0.0,
                              ConstraintMeta("BALANCED", f"{g.id}:at_least_one",
                                             {"group": g.id}))

            # min_existing 软约束：
            # deficit_g >= min_existing - Σ y_b，deficit_g >= 0，
            # 目标中减去 min_existing_penalty_s · deficit_g。
            deficit = None
            if policy == "skip" and g.min_existing is not None and g.min_existing > 0:
                deficit = self.vars.add(f"balance[{g.id}].deficit", 0.0,
                                        float(g.min_existing), GVarType.CONTINUOUS,
                                        {"kind": "balanced_min_existing_shortfall", "group": g.id})
                row = LinearExpr.of(deficit, 1.0)
                for y in y_vars.values():
                    row.add(y, 1.0)
                self.cons.add(row, ">=", float(g.min_existing),
                              ConstraintMeta("BALANCED", f"{g.id}:min_existing",
                                             {"group": g.id}))

            self.meta.balanced[g.id] = BalancedGroupModel(g, z, y_vars, bw_exprs)
            # 把 deficit 存进 meta，供目标函数使用。
            self.meta.balanced[g.id].deficit = deficit
            self.meta.balanced[g.id].targets_s = target_vars
    def _build_user_constraints(self) -> None:
        # C10：用户自定义全局约束。
        # 可以引用路口偏置 φ、需求聚合带宽、常数；
        # 不能引用方案端点（因为端点归属某个方案）。
        for c in self.problem.global_constraints:
            rhs = c.rhs_for_cycle(self.C)
            lhs = self._constraint_expr(c, None)
            if c.hard:
                self.cons.add(lhs.copy(), c.sense, rhs,
                              ConstraintMeta("USER_HARD", f"user:{c.id}", {"constraint": c.id}))
            else:
                assert c.soft is not None
                coef = c.soft.coef
                owner = c.soft.owner
                M = self.m.soft + abs(rhs) + lhs.max_abs_atom_bound(3.0 * self.C) + 1.0
                if c.sense in ("<=", "=="):
                    sigma = self.vars.add(f"viol[{c.id}.pos]", 0.0, M, GVarType.CONTINUOUS,
                                          {"kind": "soft_violation"})
                    row = lhs.copy().add(sigma, -1.0)
                    self.cons.add(row, "<=", rhs,
                                  ConstraintMeta("USER_SOFT", f"user:{c.id}",
                                                 {"constraint": c.id}))
                    self.meta.soft.append(SoftPenaltyRef(
                        constraint_id=c.id + (".pos" if c.sense == "==" else ""),
                        expr=lhs.copy(), sense="<=", rhs=rhs, coef=coef, owner=owner,
                        label=f"viol[{c.id}.pos]"))
                if c.sense in (">=", "=="):
                    sigma = self.vars.add(f"viol[{c.id}.neg]", 0.0, M, GVarType.CONTINUOUS,
                                          {"kind": "soft_violation"})
                    row = lhs.copy().add(sigma, 1.0)
                    self.cons.add(row, ">=", rhs,
                                  ConstraintMeta("USER_SOFT", f"user:{c.id}",
                                                 {"constraint": c.id}))
                    self.meta.soft.append(SoftPenaltyRef(
                        constraint_id=c.id + (".neg" if c.sense == "==" else ""),
                        expr=lhs.copy(), sense=">=", rhs=rhs, coef=coef, owner=owner,
                        label=f"viol[{c.id}.neg]"))


__all__ = ["ArterialModel", "ModelBuilder", "ref_label"]
