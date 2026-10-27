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
from . import naming
from .expressions import LinearExpr
from .meta import (AggregateModel, BalancedGroupModel, BigMTable,
                   BoundarySlackRef, GreenDemandModel, MarginSlackRef, ModelMeta,
                   RedBoundarySlackRef, RedDemandModel, SoftPenaltyRef)
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

    def v_win(self, i: str, d: Direction, slot: int, side: Side) -> LinearExpr:
        ew = self.meta.effective[(i, d)]
        return ew.s[slot] if side == "start" else ew.e[slot]

    def v_adjust(self, plan_id: str, d: Direction, j: int) -> VarHandle | None:
        return self.meta.boundary_delta.get((plan_id, d, j))

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
            for h, coef in m.slacks:
                expr.add(h, -coef)
        for ref in self.meta.red_boundary:
            for h in ref.start_vars:
                expr.add(h, -ref.start_coef)
            for h in ref.end_vars:
                expr.add(h, -ref.end_coef)
        for ref in self.meta.green_boundary:
            for h in ref.start_vars:
                expr.add(h, -ref.start_coef)
            for h in ref.end_vars:
                expr.add(h, -ref.end_coef)
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
        # 具名聚合的贡献：band objective 只取 target=band_objective 的项。
        for am in self.meta.aggregates.values():
            value = self._aggregate_expr_value(am)
            for c in am.spec.contributions:
                if c.target != "band_objective":
                    continue
                # min 是奖励（+），max 是罚项（−）；两者都把 t 推向正确方向。
                sign = 1.0 if am.spec.op == "min" else -1.0
                expr.add(value, sign * c.weight)
        return expr

    def expr_loss(self) -> LinearExpr:
        expr = LinearExpr()
        for ref in self.meta.soft:
            if ref.owner == "intersection_loss":
                expr.add(self.vars.by_name(ref_label(ref)), ref.coef)
        # 具名聚合的贡献：路口损失（最小化）里，min 是奖励（负项）、max 是罚项。
        for am in self.meta.aggregates.values():
            value = self._aggregate_expr_value(am)
            for c in am.spec.contributions:
                if c.target != "intersection_loss":
                    continue
                sign = -1.0 if am.spec.op == "min" else 1.0
                expr.add(value, sign * c.weight)
        return expr

    @staticmethod
    def _aggregate_expr_value(am: AggregateModel) -> LinearExpr:
        if am.var is not None:
            return LinearExpr.of(am.var, 1.0)
        return LinearExpr.constant(am.constant)


def ref_label(ref: SoftPenaltyRef) -> str:
    return ref.label


class ModelBuilder:
    def __init__(self, problem: ProblemInput, ctx: GridPointContext, *,
                 stage1_symmetry: bool = False,
                 min_band_width_s: float = 0.0,
                 phi_fix: Mapping[str, float] | None = None) -> None:
        self.problem = problem
        self.ctx = ctx
        self.C = ctx.cycle_s
        self.vars = VariableRegistry()
        self.cons = ConstraintStore()
        self.m = BigMTable.compute(ctx)
        self.stage1_symmetry = bool(stage1_symmetry)
        self.min_band_width_s = max(0.0, float(min_band_width_s))
        # 某些路口（例如语义子问题的共享路口）需要在**建模前**就固定 φ，
        # 因为回绕整数 n 的可行区间是用 φ 的上下界推出来的；建完再改 φ 会让
        # n 的区间失效，把最优解卡住。
        self.phi_fix: dict[str, float] = dict(phi_fix or {})
        self.meta = ModelMeta(problem=problem, ctx=ctx, cycle_s=ctx.cycle_s,
                              anchor_intersection=problem.corridor.intersections[0])

    # ------------------------------------------------------------------
    def build(self) -> ArterialModel:
        self._build_plan_selection()
        self._build_offsets()
        self._build_adjustments()
        self._build_aggregates()
        ew_builder = EffectiveWindowBuilder(self.problem, self.ctx, self.vars, self.cons,
                                            self.meta.plan_vars,
                                            self.meta.boundary_delta,
                                            self.meta.boundary_z)
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
                h = self.vars.add(naming.plan_on(iid, p.id), 0.0, 1.0, GVarType.BINARY,
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
            lo, hi = 0.0, 0.0 if is_anchor else self.C
            if iid in self.phi_fix:
                v = min(max(float(self.phi_fix[iid]), 0.0), float(self.C))
                lo = hi = v
            h = self.vars.add(naming.offset(iid), lo, hi, GVarType.CONTINUOUS,
                              {"kind": "offset", "intersection": iid, "anchor": is_anchor})
            self.meta.phi[iid] = h

    def _build_adjustments(self) -> None:
        # v2：为每个“可调相位边界”建 δ 变量。
        # 边界秒值 = (pos_r + δ) × C。
        # 多方案时再建 z = x·δ，用 McCormick 线性化。
        for iid in self.problem.corridor.intersections:
            plans = self.ctx.feasible_plans.get(iid, ())
            fixed = iid in self.meta.fixed_plans
            for p in plans:
                for d in ("up", "down"):
                    for j in range(p.boundary_count(d)):
                        r = p.boundary_adjust_range(d, j)
                        nominal = p.nominal_boundary_r(d, j)
                        expr = LinearExpr.constant(nominal * self.C)
                        if r is not None:
                            delta = self.vars.add(
                                naming.bnd_delta(p.id, d, j), r[0], r[1], GVarType.CONTINUOUS,
                                {"kind": "boundary_adjustment", "plan": p.id,
                                 "direction": d, "boundary": j})
                            self.meta.boundary_delta[(p.id, d, j)] = delta
                            expr.add(delta, self.C)
                            if not fixed:
                                x = self.meta.plan_vars[p.id]
                                z = self.vars.add(
                                    naming.bnd_xdelta(p.id, d, j), min(0.0, r[0]), max(0.0, r[1]),
                                    GVarType.CONTINUOUS,
                                    {"kind": "bilinear_x_delta", "plan": p.id,
                                     "direction": d, "boundary": j})
                                self.meta.boundary_z[(p.id, d, j)] = z
                                self._add_mccormick(x, delta, z, r[0], r[1],
                                                    p.id, d, j)
                        self.meta.plan_boundary[(p.id, d, j)] = expr

    def _build_aggregates(self) -> None:
        """具名聚合：t = max/min(e_i)，t 通过 contributions 进目标/损失。

        只有 boundary/const 原子；所有 boundary 必须属于 ``spec.plan``。
        多方案路口用 x_p 门控：未选中时 t=0（贡献 0）。
        绑定的 plan 在本格点不可行时（on_unavailable="zero"）t 取常数 0。
        """
        for spec in self.problem.aggregates:
            plan = None
            available = True
            if spec.plan is not None:
                try:
                    plan = self.problem.plan(spec.plan)
                except KeyError as exc:
                    raise GWModelError(f"aggregate {spec.id!r} references unknown plan {spec.plan!r}") from exc
                feasible_ids = {p.id for p in self.ctx.feasible_plans.get(plan.intersection, ())}
                available = plan.id in feasible_ids
            if not available:
                self.meta.aggregates[spec.id] = AggregateModel(
                    spec=spec, expressions=(), var=None, constant=0.0, available=False)
                continue
            exprs = tuple(self._aggregate_expr(terms) for terms in spec.expressions)
            x = self.meta.plan_vars.get(plan.id) if plan is not None else None
            # 只有“永远不会被方案选择影响”的聚合才能折叠成常数：
            # plan 为空，或该路口是固定方案（没有 x_p）。
            if all(e.is_constant() for e in exprs) and x is None:
                values = [e.const for e in exprs]
                const = max(values) if spec.op == "max" else min(values)
                self.meta.aggregates[spec.id] = AggregateModel(
                    spec=spec, expressions=exprs, var=None, constant=float(const),
                    available=True)
                continue

            spans = [self._linear_bounds(e) for e in exprs]
            lo_min = min(lo for lo, _ in spans)
            hi_max = max(hi for _, hi in spans)
            L = min(0.0, lo_min)
            U = max(0.0, hi_max)
            var = self.vars.add(naming.agg(spec.id), L, U, GVarType.CONTINUOUS,
                                {"kind": "aggregate", "aggregate": spec.id,
                                 "op": spec.op, "plan": plan.id if plan is not None else None})
            if x is not None:
                # 未选中时把 t 夹到 0（贡献自然为 0）。
                self.cons.add(LinearExpr.of(var, 1.0).add(x, -L), ">=", 0.0,
                              ConstraintMeta("AGG", f"{spec.id}:t>=Lx", {"aggregate": spec.id}))
                self.cons.add(LinearExpr.of(var, 1.0).add(x, -U), "<=", 0.0,
                              ConstraintMeta("AGG", f"{spec.id}:t<=Ux", {"aggregate": spec.id}))
            if spec.op == "max":
                M = hi_max - L + 1.0
                for i, e in enumerate(exprs):
                    row = LinearExpr.of(var, 1.0).add(e, -1.0)
                    if x is not None:
                        row.add(x, -M)
                    self.cons.add(row, ">=", (-M if x is not None else 0.0),
                                  ConstraintMeta("AGG", f"{spec.id}:max{i}", {"aggregate": spec.id}))
            else:
                M = U - lo_min + 1.0
                for i, e in enumerate(exprs):
                    row = LinearExpr.of(var, 1.0).add(e, -1.0)
                    if x is not None:
                        row.add(x, M)
                    self.cons.add(row, "<=", (M if x is not None else 0.0),
                                  ConstraintMeta("AGG", f"{spec.id}:min{i}", {"aggregate": spec.id}))
            self.meta.aggregates[spec.id] = AggregateModel(
                spec=spec, expressions=exprs, var=var, constant=0.0, available=True)

    def _aggregate_expr(self, terms) -> LinearExpr:
        expr = LinearExpr()
        for t in terms:
            a = t.atom
            if a.kind == "boundary":
                key = (a.plan, a.direction, a.index)
                if key not in self.meta.plan_boundary:
                    raise GWModelError(
                        f"aggregate boundary {key} is not available at this grid point")
                expr.add(self.meta.plan_boundary[key], t.coef)
            elif a.kind == "const":
                scale = self.C if a.unit == "ratio" else 1.0
                expr.add_const(t.coef * float(a.const or 0.0) * scale)
            else:
                raise GWModelError(f"unsupported atom {a.kind!r} in aggregate expression")
        return expr

    def _add_mccormick(self, x: VarHandle, delta: VarHandle, z: VarHandle,
                       lo: float, hi: float, plan_id: str, d: Direction, j: int) -> None:
        # z = x·δ，x 是 0/1，δ 在 [lo, hi]。
        row = LinearExpr.of(z, 1.0).add(x, -hi)
        self.cons.add(row, "<=", 0.0,
                      ConstraintMeta("MCC", f"z<=hi*x {plan_id}/{d}/{j}", {"plan": plan_id}))
        row = LinearExpr.of(z, 1.0).add(x, -lo)
        self.cons.add(row, ">=", 0.0,
                      ConstraintMeta("MCC", f"z>=lo*x {plan_id}/{d}/{j}", {"plan": plan_id}))
        row = LinearExpr.of(z, 1.0).add(delta, -1.0).add(x, -lo)
        self.cons.add(row, "<=", -lo,
                      ConstraintMeta("MCC", f"z<=delta-lo(1-x) {plan_id}/{d}/{j}", {"plan": plan_id}))
        row = LinearExpr.of(z, 1.0).add(delta, -1.0).add(x, -hi)
        self.cons.add(row, ">=", -hi,
                      ConstraintMeta("MCC", f"z>=delta-hi(1-x) {plan_id}/{d}/{j}", {"plan": plan_id}))

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
            if a.kind == "boundary":
                if plan is None:
                    raise GWModelError("boundary atom referenced outside a plan constraint")
                key = (a.plan or plan.id, a.direction, a.index)
                if key not in self.meta.plan_boundary:
                    raise GWModelError(f"unknown boundary reference {key}")
                expr.add(self.meta.plan_boundary[key], t.coef)
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
                w = p.window_edge_span(d, k)
                start_e = self._edge_value_expr(p, d, w.start_edge)
                end_e = self._edge_value_expr(p, d, w.end_edge)
                expr.add(start_e, -t.coef)
                expr.add(end_e, t.coef)
                if w.wrap:
                    expr.add_const(t.coef * self.C)
            elif a.kind == "red_duration":
                if plan is not None and (a.plan is None or a.plan == plan.id):
                    d_use = a.direction if a.direction is not None else scope_d
                    rid_use = a.window if a.window is not None else scope_rid
                    if d_use is None or rid_use is None:
                        raise GWModelError("red_duration atom requires direction/rid")
                    reds = plan.derived_red_intervals(d_use)
                    if rid_use < 1 or rid_use > len(reds):
                        raise GWModelError(f"red_duration references unknown rid {rid_use}")
                    red = reds[rid_use - 1]
                    expr.add(self._edge_duration_expr(plan, d_use,
                                                      red.start_edge, red.end_edge,
                                                      red.wrap), t.coef)
                elif scope_red is not None and scope_rid is not None:
                    start = scope_red.start[scope_rid]
                    end = scope_red.end[scope_rid]
                    expr.add(start, -t.coef)
                    expr.add(end, t.coef)
                else:
                    raise GWModelError("red_duration atom requires a plan or red-window scope")
            elif a.kind == "const":
                scale = self.C if a.unit == "ratio" else 1.0
                expr.add_const(t.coef * float(a.const or 0.0) * scale)
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

    def _edge_value_expr(self, plan: PlanSpec, d: Direction, edge: int) -> LinearExpr:
        """Plan-specific expanded edge expression in seconds (x not multiplied)."""
        kind, src, offset = plan.expanded_edge_sources(d)[edge]
        if kind == "const":
            return LinearExpr.constant(float(offset) * self.C)
        expr = self.meta.plan_boundary[(plan.id, d, src)].copy()
        if offset:
            expr.add_const(offset * self.C)
        return expr

    def _edge_duration_expr(self, plan: PlanSpec, d: Direction, start_edge: int,
                            end_edge: int, wrap: int) -> LinearExpr:
        expr = self._edge_value_expr(plan, d, end_edge)
        expr.add(self._edge_value_expr(plan, d, start_edge), -1.0)
        if wrap:
            expr.add_const(self.C)
        return expr

    def _max_effective_end_seconds(self, intersection: str, d: Direction) -> float:
        best = 0.0
        for p in self.ctx.feasible_plans.get(intersection, ()):
            for k in range(p.window_count(d)):
                _, hi = p.window_endpoint_bound_r(d, k, "end")
                best = max(best, hi * self.C)
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
            if a.kind == "boundary" and plan is not None:
                nom = plan.nominal_boundary_r(a.direction, a.index)
                r = plan.boundary_adjust_range(a.direction, a.index)
                lov = (nom + (r[0] if r is not None else 0.0)) * self.C
                hiv = (nom + (r[1] if r is not None else 0.0)) * self.C
                atom_ub = max(abs(lov), abs(hiv), 3.0 * self.C)
            elif a.kind == "window_duration":
                atom_ub = 3.0 * self.C
            elif a.kind == "offset":
                atom_ub = self.C
            elif a.kind == "const":
                scale = self.C if a.unit == "ratio" else 1.0
                atom_ub = abs(float(a.const or 0.0)) * scale
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
        """单个方案某个派生窗口端点的秒值上下界（含边界可调范围）。"""
        lo_r, hi_r = plan.window_endpoint_bound_r(d, k0, side)
        lo = lo_r * self.C
        hi = hi_r * self.C
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

    def _green_wrap_bounds_for_node(self, intersection: str, d: Direction,
                                   cum_T_s: float) -> dict[int, tuple[int, int] | None]:
        """按窗口槽预计算绿带回绕周期 n 的可行整数范围。

        对每个 k：
            n_hi = floor((u_max + T_j - phi_min - s_min) / C)
            n_lo = ceil((u_min + beta_min + T_j - phi_max - e_max) / C)
        若 n_lo > n_hi，则标记该 (node,k) 不可行。
        返回 {k: (n_lo, n_hi) | None}。
        """
        gw = self.meta.effective.get((intersection, d))
        if gw is None:
            return {}
        slot_bounds = self._window_slot_bounds(gw)
        phi = self.meta.phi.get(intersection)
        if phi is None:
            return {k: None for k in slot_bounds}
        phi_lo = float(self.vars.lbs[phi.index])
        phi_hi = float(self.vars.ubs[phi.index])
        u_lo, u_hi = 0.0, float(self.C)
        beta_lo = (float(self.min_band_width_s)
                   if (self.stage1_symmetry and self.min_band_width_s > 0.0) else 0.0)
        C = float(self.C)
        out: dict[int, tuple[int, int] | None] = {}
        for k, (s_lo, _s_hi, _e_lo, e_hi) in slot_bounds.items():
            n_hi = math.floor((u_hi + cum_T_s - phi_lo - s_lo) / C + 1e-9)
            n_lo = math.ceil((u_lo + beta_lo + cum_T_s - phi_hi - e_hi) / C - 1e-9)
            out[k] = (int(n_lo), int(n_hi)) if n_lo <= n_hi else None
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

    def _red_upstream_wrap_bounds(self, gw_up: EffectiveWindowSet, tau: float
                                  ) -> dict[int, tuple[int, int] | None]:
        """红带上游包含式的整数周期下标 m 的可行范围（与绿带的 n 完全对称）。

        红带的上游行（``ag=1`` 时）是：

            s_g ≤ u - τ - φ_up - m·C ≤ e_g

        即

            m_hi = floor((u_max - τ - φ_min - s_min) / C)
            m_lo = ceil ((u_min + β_min - τ - φ_max - e_max) / C)

        没有 m 时，``u - τ - φ_up`` 只能落在 ``(-C-τ, C-τ)``，而上游窗口在
        ``[0, C]``；当上游窗口靠后 + 行程时间较长时，``u - τ - φ_up`` 需要
        ``+C`` 才够得着，模型就表示不出那个物理上完全可行的摆放——表现就是
        "把整组做时间平移后最优值会掉"。m 补上这一维后，红带模型和绿带一样
        对时间平移严格不变。
        """
        C = float(self.C)
        up_bounds = self._window_slot_bounds(gw_up)
        phi = self.meta.phi.get(gw_up.intersection)
        if phi is None:
            return {g: None for g in up_bounds}
        phi_lo = float(self.vars.lbs[phi.index])
        phi_hi = float(self.vars.ubs[phi.index])
        u_lo, u_hi = 0.0, C
        beta_lo = (float(self.min_band_width_s)
                   if (self.stage1_symmetry and self.min_band_width_s > 0.0) else 0.0)
        out: dict[int, tuple[int, int] | None] = {}
        for g, (s_lo, _s_hi, _e_lo, e_hi) in up_bounds.items():
            m_hi = math.floor((u_hi - tau - phi_lo - s_lo) / C + 1e-9)
            m_lo = math.ceil((u_lo + beta_lo - tau - phi_hi - e_hi) / C - 1e-9)
            out[g] = (int(m_lo), int(m_hi)) if m_lo <= m_hi else None
        return out

    def _red_compatible_keys(self, gw_up: EffectiveWindowSet, red_down: RedWindowSet,
                             tau: float,
                             m_ranges: Mapping[int, tuple[int, int] | None] | None = None):
        """因式分解后的红波兼容性预剪枝。

        返回：(上游绿窗槽 g 的保留列表, 下游红窗实例 (r,k) 的保留列表)。

        判定用变量上下界完成，负责：
        1. 相对偏置 Δφ = φ_j - φ_i ∈ (-C,C) 是否可能重叠；
        2. 绝对时间段是否能被带区间 [u,u+β] 容纳（上游侧含周期下标 m 的范围）；
        3. 若某 g 与任何 (r,k) 都不兼容，就不建 ag；
           若某 (r,k) 与任何 g 都不兼容，就不建 ar。
        """
        C = self.C
        eps = 1e-9
        up_bounds = self._window_slot_bounds(gw_up)
        red_bounds = self._red_slot_bounds(red_down)
        if m_ranges is None:
            m_ranges = self._red_upstream_wrap_bounds(gw_up, tau)
        pairs = set()

        for g in gw_up.slots:
            if g not in up_bounds:
                continue
            s_lo, s_hi, e_lo, e_hi = up_bounds[g]
            mr = m_ranges.get(g)
            if mr is None:
                continue                      # 该上游窗口槽根本无可行 m
            m_lo_g, m_hi_g = mr
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
                    # 绝对区间：上游 φ_i∈[0,C] 且 m∈[m_lo,m_hi]。
                    up_lo_min = 0.0 + s_lo + tau + m_lo_g * C
                    up_hi_max = C + e_hi + tau + m_hi_g * C
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
                    up_end_lo = 0.0 + e_lo + tau + m_lo_g * C
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
            var_name = naming.soft_slack(name_base, "pos")
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
            var_name = naming.soft_slack(name_base, "neg")
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
            var_name = naming.soft_slack(name_base, "pos")
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
            var_name = naming.soft_slack(name_base, "neg")
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
                        if scope.skip_double_cycle_copies and p.is_copy_window(d, k):
                            continue
                        if scope.window_indices is not None and k not in scope.window_indices:
                            continue
                        yield p, d, k

    def _build_constraint_templates(self) -> None:
        """展开 ConstraintTemplate：统一处理绝对秒/比例 RHS 和作用域。"""
        for tmpl in self.problem.constraint_templates:
            rhs = tmpl.rhs_for_cycle(self.C)
            M = self.m.containment + abs(rhs) + 1.0
            if tmpl.scope.kind == "all_plan_red_windows":
                scope = tmpl.scope
                for iid in self.problem.corridor.intersections:
                    for p in self.ctx.feasible_plans.get(iid, ()):
                        if scope.plan_ids is not None and p.id not in scope.plan_ids:
                            continue
                        for d in scope.directions:
                            try:
                                reds = p.derived_red_intervals(d)
                            except ValueError:
                                continue
                            for red in reds:
                                if scope.window_indices is not None and red.rid not in scope.window_indices:
                                    continue
                                if red.width_r <= 1e-9:
                                    continue
                                lhs = self._terms_expr(
                                    tmpl.terms, constraint_id=tmpl.id,
                                    plan=p, scope_d=d,
                                    scope_red=None, scope_rid=red.rid,
                                )
                                fixed = p.intersection in self.meta.fixed_plans
                                x = None if fixed else self.meta.plan_vars[p.id]
                                label = f"{tmpl.id}:{p.id}:{d}:red{red.rid}"
                                refs = {"template": tmpl.id, "plan": p.id,
                                        "direction": d, "red_window": str(red.rid)}
                                if tmpl.hard:
                                    self._add_activated_hard(lhs, tmpl.sense, rhs, x, M,
                                                             "TEMPLATE_HARD", label, refs)
                                else:
                                    assert tmpl.soft is not None
                                    self._add_activated_soft(
                                        lhs, tmpl.sense, rhs,
                                        coef=tmpl.soft.coef, owner=tmpl.soft.owner,
                                        plan_id=p.id, x=x, m_big=M,
                                        meta_group="TEMPLATE_SOFT",
                                        meta_label_base=label,
                                        name_base=label,
                                        constraint_id_base=label, refs=refs,
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
        # v2 内置几何约束：边界排序 + min_phase_s。
        # δ 只被本方案的约束引用，未选中时这些约束只约束自由变量，无害。
        g = (min(p.min_phase_s for p in self.problem.plans) / self.C
             if self.problem.plans else 0.0)
        for iid in self.problem.corridor.intersections:
            for p in self.ctx.feasible_plans.get(iid, ()):
                for d in ("up", "down"):
                    m = p.boundary_count(d)
                    if m == 0:
                        continue
                    hi = 0.5 if p.double_cycle else 1.0
                    gg = p.min_phase_s / self.C
                    b0 = self.meta.plan_boundary[(p.id, d, 0)]
                    self.cons.add(b0.copy(), ">=", gg * self.C,
                                  ConstraintMeta("PLAN_WINDOW", f"{p.id}:{d}:b0:min_phase",
                                                 {"plan": p.id, "direction": d}))
                    for j in range(m - 1):
                        a = self.meta.plan_boundary[(p.id, d, j)]
                        b = self.meta.plan_boundary[(p.id, d, j + 1)]
                        row = a.copy()
                        row.add(b, -1.0)
                        self.cons.add(row, "<=", -gg * self.C,
                                      ConstraintMeta("PLAN_WINDOW",
                                                     f"{p.id}:{d}:b{j}:min_phase",
                                                     {"plan": p.id, "direction": d}))
                    bm = self.meta.plan_boundary[(p.id, d, m - 1)]
                    self.cons.add(bm.copy(), "<=", (hi - gg) * self.C,
                                  ConstraintMeta("PLAN_WINDOW", f"{p.id}:{d}:bm:max",
                                                 {"plan": p.id, "direction": d}))

    # ------------------------------------------------------------------
    # C11 double cycle
    # ------------------------------------------------------------------
    def _build_double_cycle_links(self) -> None:
        # v2 的双周期边界在 (0, 0.5) 声明，展开时复制边界直接复用同一个 δ，
        # 因此一致性是结构性的，不需要额外等式。
        return

    # ------------------------------------------------------------------
    # C4-C8 green bands
    # ------------------------------------------------------------------
    def _add_band_slot_symmetry(self, demand_id: str, qmax: int,
                               e: dict[int, VarHandle], beta: dict[int, VarHandle],
                               tag: str) -> None:
        """Stage-1 对称破缺：带槽存在性单调 + 正宽度下界。

        - e_1 >= e_2 >= ...：存在的带槽必须排在前面，消除空槽置换对称；
        - beta_q >= w_min * e_q：禁止 e=1 但宽度为 0 的退化带，减少等价分支。
        """
        if not self.stage1_symmetry:
            return
        for q in range(1, qmax):
            row = LinearExpr.of(e[q], 1.0).add(e[q + 1], -1.0)
            self.cons.add(row, ">=", 0.0,
                          ConstraintMeta("SYM", f"{tag} e slot monotonic {demand_id}/{q}/{q+1}",
                                         {"demand": demand_id, "slot": str(q),
                                          "slot2": str(q + 1)}))
        if self.min_band_width_s > 0.0:
            for q in range(1, qmax + 1):
                row = LinearExpr.of(beta[q], 1.0).add(e[q], -self.min_band_width_s)
                self.cons.add(row, ">=", 0.0,
                              ConstraintMeta("SYM",
                                             f"{tag} positive width {demand_id}/{q}",
                                             {"demand": demand_id, "slot": str(q),
                                              "min_band_width_s": str(self.min_band_width_s)}))

    def _add_green_wrap_range_constraints(self, b: BandDemandSpec, q: int,
                                         e: dict[int, VarHandle],
                                         n: dict[tuple[int, int], VarHandle],
                                         a: dict[tuple[int, int, int], VarHandle],
                                         node_ranges: dict[int, dict[int, tuple[int, int] | None]],
                                         node_global: dict[int, tuple[int, int]]) -> None:
        """把预计算的 n 可行范围接入模型。

        - n 与 e 绑定：e=0 时 n=0；
        - 选中的窗口槽 k 激活对应的 n_lo/n_hi；
        - n_lo > n_hi 的 (j,k) 直接禁止 a=1。
        """
        d = b.direction
        M = self.m.containment
        for j, node in enumerate(b.nodes):
            nv = n[(q, j)]
            lo, hi = node_global[j]
            # e=0 -> n=0；e=1 -> lo <= n <= hi。
            row = LinearExpr.of(nv, 1.0).add(e[q], -hi)
            self.cons.add(row, "<=", 0.0,
                          ConstraintMeta("NRANGE",
                                         f"wrap n<=hi*e {b.id}/{q}/{node}",
                                         {"demand": b.id, "slot": str(q), "node": node}))
            row = LinearExpr.of(nv, 1.0).add(e[q], -lo)
            self.cons.add(row, ">=", 0.0,
                          ConstraintMeta("NRANGE",
                                         f"wrap n>=lo*e {b.id}/{q}/{node}",
                                         {"demand": b.id, "slot": str(q), "node": node}))

            for k in range(1, self.ctx.max_window_slots(node, d) + 1):
                av = a.get((q, j, k))
                if av is None:
                    continue
                rng = node_ranges.get(j, {}).get(k)
                if rng is None:
                    # 该窗口槽下不存在任何整数 n，直接禁止选择它。
                    self.cons.add(LinearExpr.of(av, 1.0), "==", 0.0,
                                  ConstraintMeta("NRANGE",
                                                 f"wrap infeasible {b.id}/{q}/{node}/{k}",
                                                 {"demand": b.id, "slot": str(q),
                                                  "node": node, "window": str(k)}))
                    continue
                n_lo, n_hi = rng
                # n <= n_hi + M(1-a)
                row = LinearExpr.of(nv, 1.0).add(av, M)
                self.cons.add(row, "<=", float(n_hi) + M,
                              ConstraintMeta("NRANGE",
                                             f"wrap n upper {b.id}/{q}/{node}/{k}",
                                             {"demand": b.id, "slot": str(q),
                                              "node": node, "window": str(k)}))
                # n >= n_lo - M(1-a)
                row = LinearExpr.of(nv, 1.0).add(av, -M)
                self.cons.add(row, ">=", float(n_lo) - M,
                              ConstraintMeta("NRANGE",
                                             f"wrap n lower {b.id}/{q}/{node}/{k}",
                                             {"demand": b.id, "slot": str(q),
                                              "node": node, "window": str(k)}))

    def _build_green_bands(self) -> None:
        # C4–C8：绿波带。
        # 绿波带是“一条连续的时间区间，沿链上每个路口都落在某个绿灯窗口内”。
        # 带宽由链上最紧的窗口决定，不需要额外写 min 函数。
        for b in self.problem.objective.band_demands:
            if b.type != "green":
                continue
            qmax = self.ctx.demand_max_bands(
                b, allow_sharing=self.problem.objective.allow_window_sharing)
            if qmax <= 0:
                continue
            table = self.ctx.travel[b.id]
            d = b.direction
            # 按节点和窗口槽预计算 n 的可行整数范围，替代原来的全局保守界。
            node_ranges: dict[int, dict[int, tuple[int, int] | None]] = {}
            node_global: dict[int, tuple[int, int]] = {}
            for j, node in enumerate(b.nodes):
                ranges = self._green_wrap_bounds_for_node(node, d, table.cum_T_s[j])
                node_ranges[j] = ranges
                los = [v[0] for v in ranges.values() if v is not None]
                his = [v[1] for v in ranges.values() if v is not None]
                if los:
                    node_global[j] = (min(0, min(los)), max(0, max(his)))
                else:
                    node_global[j] = (0, 0)
            u = {q: self.vars.add(naming.gw_start(b.id, q), 0.0, self.C, GVarType.CONTINUOUS,
                                   {"kind": "band_start", "demand": b.id, "slot": q})
                 for q in range(1, qmax + 1)}
            beta = {q: self.vars.add(naming.gw_width(b.id, q), 0.0, self.C, GVarType.CONTINUOUS,
                                      {"kind": "band_width", "demand": b.id, "slot": q})
                    for q in range(1, qmax + 1)}
            e = {q: self.vars.add(naming.gw_on(b.id, q), 0.0, 1.0, GVarType.BINARY,
                                   {"kind": "band_exists", "demand": b.id, "slot": q})
                 for q in range(1, qmax + 1)}
            a: dict[tuple[int, int, int], VarHandle] = {}
            n: dict[tuple[int, int], VarHandle] = {}
            for q in range(1, qmax + 1):
                for j, node in enumerate(b.nodes):
                    lo_n, hi_n = node_global[j]
                    n[(q, j)] = self.vars.add(naming.gw_cycle(b.id, q, node),
                                              float(lo_n), float(hi_n),
                                              GVarType.INTEGER,
                                              {"kind": "wrap", "demand": b.id, "slot": q, "node": node})
                    for k in range(1, self.ctx.max_window_slots(node, d) + 1):
                        a[(q, j, k)] = self.vars.add(naming.gw_win(b.id, q, node, k), 0.0, 1.0,
                                                     GVarType.BINARY,
                                                     {"kind": "green_assignment", "demand": b.id,
                                                      "slot": q, "node": node, "window": k})
                self._add_green_wrap_range_constraints(b, q, e, n, a, node_ranges, node_global)
            gmodel = GreenDemandModel(b, qmax, u, beta, e, a, n, table)
            self.meta.green[b.id] = gmodel
            self._add_band_slot_symmetry(b.id, qmax, e, beta, "green")

            # C8_EDGE 的配置：每条需求一份（与红波 C9_RED_EDGE 对称）。
            # 有软罚系数才需要 slack 变量；只有 hard_max_s 时也允许（只加硬约束）。
            gb = self.problem.green_boundary_for(b)
            gb_ref = None
            if gb.enabled and ((gb.start is not None and gb.start.coef > 0)
                               or (gb.end is not None and gb.end.coef > 0)):
                gb_ref = BoundarySlackRef(
                    demand_id=b.id,
                    start_coef=gb.start.coef if gb.start is not None else 0.0,
                    end_coef=gb.end.coef if gb.end is not None else 0.0,
                )
                self.meta.green_boundary.append(gb_ref)

            margin_ref = MarginSlackRef(b.id)

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
                # C7：多带排序 + 圆环时间分离。
                # 两条带在圆环上不能重叠，否则同一条物理带可能被计数两次。
                # u_q <= u_q2 是破对称约束：带槽本身可交换，不丢最优解。
                for q2 in range(q + 1, qmax + 1):
                    M = self.m.containment
                    # u_q <= u_q2
                    row = LinearExpr.of(u[q], 1.0).add(u[q2], -1.0)
                    row.add(e[q], M).add(e[q2], M)
                    self.cons.add(row, "<=", 2.0 * M,
                                  ConstraintMeta("C7", f"u order {b.id}/{q}/{q2}",
                                                 {"demand": b.id, "slot": str(q),
                                                  "slot2": str(q2)}))
                    # q 必须在 q2 起点前结束
                    row = LinearExpr.of(u[q], 1.0).add(beta[q], 1.0).add(u[q2], -1.0)
                    row.add(e[q], M).add(e[q2], M)
                    self.cons.add(row, "<=", 2.0 * M,
                                  ConstraintMeta("C7", f"band disjoint {b.id}/{q}/{q2}",
                                                 {"demand": b.id, "slot": str(q),
                                                  "slot2": str(q2)}))
                    # q2 绕回 0 点时不能在 q 之前结束后再压到 q
                    row = LinearExpr.of(u[q2], 1.0).add(beta[q2], 1.0).add(u[q], -1.0)
                    row.add(e[q], M).add(e[q2], M)
                    self.cons.add(row, "<=", self.C + 2.0 * M,
                                  ConstraintMeta("C7", f"band wrap disjoint {b.id}/{q}/{q2}",
                                                 {"demand": b.id, "slot": str(q),
                                                  "slot2": str(q2)}))
                # C7：窗口独占。allow_window_sharing=true 时允许多条带共用
                # 同一个绿窗，此时由上面的圆环时间分离保证不重叠。
                if not self.problem.objective.allow_window_sharing:
                    for j, node in enumerate(b.nodes):
                        for k in range(1, self.ctx.max_window_slots(node, d) + 1):
                            row = LinearExpr()
                            for qq in range(1, qmax + 1):
                                row.add(a[(qq, j, k)], 1.0)
                            self.cons.add(row, "<=", 1.0,
                                          ConstraintMeta("C7", f"window exclusive {b.id}/{node}/{k}",
                                                         {"demand": b.id, "node": node, "window": str(k)}))
                # C8 / C8_HARD：逐绿窗 margin。
                # margin 现在逐方案、逐绿段配置（plan.green_mask 的 GreenSegmentSpec），
                # 而 MILP 的窗口槽 k 是跨方案混合的（有效窗口 = Σ_p x_p·endpoint_p）：
                #   * 硬下限：gap >= Σ_p A_{p,k}·x_p·hard_p —— RHS 是线性的，不新增变量；
                #   * 软损失：coef 可能逐方案不同，所以每个方案各建一个 slack，
                #     用 a（带选中该窗口）与 x_p（方案被选中）两个 big-M 条件激活。
                # 同色边界 / 红段上的 margin 在 schema 校验时警告并忽略，这里只读绿窗两侧。
                M_hard = self.m.containment
                M_soft = self.m.soft
                for side in ("s", "e"):
                    for j, node in enumerate(b.nodes):
                        plans_node = self.ctx.feasible_plans.get(node, ())
                        for k in range(1, self.ctx.max_window_slots(node, d) + 1):
                            side_specs = []
                            for p in plans_node:
                                if p.window_count(d) < k:
                                    continue
                                ms, me = p.window_margins(d, k - 1)
                                spec = ms if side == "s" else me
                                if spec is None:
                                    continue
                                side_specs.append((p, spec, self.meta.plan_vars.get(p.id)))
                            if not side_specs:
                                continue
                            av = a[(q, j, k)]
                            nv = n[(q, j)]
                            phi = self.meta.phi[node]
                            if side == "s":
                                base = LinearExpr.of(u[q], 1.0).add(phi, -1.0)
                                base.add(self.meta.effective[(node, d)].s[k], -1.0)
                                base.add(nv, -self.C)
                                tag = "start"
                            else:
                                base = LinearExpr.of(phi, 1.0)
                                base.add(self.meta.effective[(node, d)].e[k], 1.0)
                                base.add(nv, self.C).add(u[q], -1.0).add(beta[q], -1.0)
                                tag = "end"

                            # ---- C8_HARD：硬下限 ----
                            hard_specs = [t for t in side_specs if t[1].hard_min_s > 0]
                            if hard_specs:
                                row = base.copy().add(av, -M_hard)
                                x_used = False
                                for p, spec, x in hard_specs:
                                    if x is not None:
                                        row.add(x, -spec.hard_min_s)
                                        x_used = True
                                if x_used:
                                    rhs = -table.cum_T_s[j] - M_hard if side == "s" \
                                          else table.cum_T_s[j] - M_hard
                                else:
                                    # 固定方案：没有 x，RHS 直接是常数。
                                    hard = hard_specs[0][1].hard_min_s
                                    rhs = hard - table.cum_T_s[j] - M_hard if side == "s" \
                                          else hard + table.cum_T_s[j] - M_hard
                                self.cons.add(row, ">=", rhs,
                                              ConstraintMeta("C8_HARD",
                                                             f"margin hard {tag} {b.id}/{q}/{node}/{k}",
                                                             {"demand": b.id, "slot": str(q),
                                                              "node": node, "window": str(k)}))

                            # ---- C8：软损失 ----
                            for p, spec, x in side_specs:
                                if spec.delta_min_s <= 0 or spec.coef <= 0:
                                    continue
                                var = self.vars.add(
                                    naming.margin_name(b.id, q, node, k, side,
                                                       plan=None if x is None else p.id),
                                    0.0, self.m.containment * 2.0, GVarType.CONTINUOUS,
                                    {"kind": "margin_shortfall", "demand": b.id,
                                     "slot": q, "node": node, "window": k,
                                     "side": side, "plan": p.id})
                                margin_ref.slacks.append((var, spec.coef))
                                row = base.copy().add(var, 1.0).add(av, -M_soft)
                                if x is not None:
                                    row.add(x, -M_soft)
                                rhs = (spec.delta_min_s - table.cum_T_s[j] - M_soft
                                       if side == "s"
                                       else spec.delta_min_s + table.cum_T_s[j] - M_soft)
                                if x is not None:
                                    rhs -= M_soft
                                self.cons.add(row, ">=", rhs,
                                              ConstraintMeta("C8",
                                                             f"margin {tag} {b.id}/{q}/{node}/{k}/{p.id}",
                                                             {"demand": b.id, "slot": str(q),
                                                              "node": node, "window": str(k),
                                                              "plan": p.id}))

                # C8_EDGE：绿带边界亲和（贴合损失）。
                # 与 C8 margin 相反：margin 罚“离窗口边界太近”，亲和罚“离窗口边界太远”。
                #   gap_start = (u + T_j) − (φ + s + nC)   希望 ≤ free_s
                #   gap_end   = (φ + e + nC) − (u + T_j + β) 希望 ≤ free_s
                # 默认只开起点侧：让带从绿灯起点出发（“绿波开始贴合”）。
                # 参考窗口就是带自己这个节点的绿窗，所以 penalty 在
                # “带被别的路口卡住、本节点存在富余”时才有值。
                if gb.enabled:
                    M_edge = self.m.soft
                    M_hard = self.m.containment
                    for side in ("s", "e"):
                        spec = gb.start if side == "s" else gb.end
                        if spec is None:
                            continue
                        for j, node in enumerate(b.nodes):
                            for k in range(1, self.ctx.max_window_slots(node, d) + 1):
                                s = self.meta.effective[(node, d)].s[k]
                                ew = self.meta.effective[(node, d)].e[k]
                                phi = self.meta.phi[node]
                                nv = n[(q, j)]
                                av = a[(q, j, k)]
                                if spec.coef > 0:
                                    var = self.vars.add(
                                        naming.edge_name(b.id, q, node, k, side), 0.0,
                                        self.m.containment * 2.0, GVarType.CONTINUOUS,
                                        {"kind": "green_boundary_slack", "demand": b.id,
                                         "slot": q, "node": node, "window": k, "side": side})
                                    if gb_ref is not None:
                                        (gb_ref.start_vars if side == "s"
                                         else gb_ref.end_vars).append(var)
                                    M = M_edge
                                    if side == "s":
                                        # σ >= gap_start - free - M(1-a)
                                        # σ - u + φ + s + nC - M·a >= T - free - M
                                        row = LinearExpr.of(var, 1.0).add(u[q], -1.0)
                                        row.add(phi, 1.0).add(s, 1.0).add(nv, self.C)
                                        row.add(av, -M)
                                        self.cons.add(row, ">=",
                                                      table.cum_T_s[j] - spec.free_s - M,
                                                      ConstraintMeta("C8_EDGE",
                                                                     f"green edge start {b.id}/{q}/{node}/{k}",
                                                                     {"demand": b.id, "slot": str(q),
                                                                      "node": node, "window": str(k)}))
                                    else:
                                        # σ >= gap_end - free - M(1-a)
                                        # σ + u + β + T - φ - e - nC - M·a >= -free - M
                                        row = LinearExpr.of(var, 1.0).add(u[q], 1.0)
                                        row.add(beta[q], 1.0).add(phi, -1.0).add(ew, -1.0)
                                        row.add(nv, -self.C).add(av, -M)
                                        self.cons.add(row, ">=", -spec.free_s - M,
                                                      ConstraintMeta("C8_EDGE",
                                                                     f"green edge end {b.id}/{q}/{node}/{k}",
                                                                     {"demand": b.id, "slot": str(q),
                                                                      "node": node, "window": str(k)}))
                                if spec.hard_max_s is not None:
                                    # gap <= hard_max_s + M(1-a)
                                    if side == "s":
                                        row = LinearExpr.of(u[q], 1.0).add(phi, -1.0)
                                        row.add(s, -1.0).add(nv, -self.C).add(av, M_hard)
                                        self.cons.add(row, "<=",
                                                      spec.hard_max_s - table.cum_T_s[j] + M_hard,
                                                      ConstraintMeta("C8_EDGE_HARD",
                                                                     f"green edge start max {b.id}/{q}/{node}/{k}",
                                                                     {"demand": b.id, "slot": str(q),
                                                                      "node": node, "window": str(k)}))
                                    else:
                                        row = LinearExpr.of(phi, 1.0).add(ew, 1.0)
                                        row.add(nv, self.C).add(u[q], -1.0).add(beta[q], -1.0)
                                        row.add(av, M_hard)
                                        self.cons.add(row, "<=",
                                                      spec.hard_max_s + table.cum_T_s[j] + M_hard,
                                                      ConstraintMeta("C8_EDGE_HARD",
                                                                     f"green edge end max {b.id}/{q}/{node}/{k}",
                                                                     {"demand": b.id, "slot": str(q),
                                                                      "node": node, "window": str(k)}))

            if margin_ref.slacks:
                self.meta.margins.append(margin_ref)

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
            # 上游周期下标 m 的可行范围按"上游绿窗槽"分别推导（与绿带 n 对称）。
            m_ranges = self._red_upstream_wrap_bounds(gw_up, tau)
            ag_slots, ar_keys = self._red_compatible_keys(gw_up, red_down, tau, m_ranges)

            qmax = self.ctx.demand_max_bands(
                b, allow_sharing=self.problem.objective.allow_window_sharing)
            u = {q: self.vars.add(naming.rw_start(b.id, q), 0.0, self.C, GVarType.CONTINUOUS,
                                   {"kind": "red_band_start", "demand": b.id, "slot": q})
                 for q in range(1, qmax + 1)}
            beta = {q: self.vars.add(naming.rw_width(b.id, q), 0.0, self.C, GVarType.CONTINUOUS,
                                      {"kind": "red_band_width", "demand": b.id, "slot": q})
                    for q in range(1, qmax + 1)}
            e = {q: self.vars.add(naming.rw_on(b.id, q), 0.0, 1.0, GVarType.BINARY,
                                   {"kind": "red_band_exists", "demand": b.id, "slot": q})
                 for q in range(1, qmax + 1)}

            # 上游周期下标：与绿带的 n 对称，补上"上游项缺一个周期"那一维。
            feasible_m = [v for v in (m_ranges.get(g) for g in ag_slots) if v is not None]
            if feasible_m:
                m_lb = float(min(0, min(v[0] for v in feasible_m)))
                m_ub = float(max(0, max(v[1] for v in feasible_m)))
            else:
                m_lb = m_ub = 0.0
            m_shift = {q: self.vars.add(naming.rw_up_cycle(b.id, q), m_lb, m_ub, GVarType.INTEGER,
                                        {"kind": "red_up_wrap", "demand": b.id, "slot": q})
                       for q in range(1, qmax + 1)}

            ag: dict[tuple[int, int], VarHandle] = {}
            ar: dict[tuple[int, int, int], VarHandle] = {}
            for q in range(1, qmax + 1):
                for g in ag_slots:
                    ag[(q, g)] = self.vars.add(
                        naming.rw_up_win(b.id, q, g), 0.0, 1.0, GVarType.BINARY,
                        {"kind": "red_up_assignment", "demand": b.id, "slot": q, "up_window": g})
                for (r, k) in ar_keys:
                    ar[(q, r, k)] = self.vars.add(
                        naming.rw_down_win(b.id, q, r, k), 0.0, 1.0, GVarType.BINARY,
                        {"kind": "red_down_assignment", "demand": b.id, "slot": q,
                         "red_rid": r, "shift": k})

            rmodel = RedDemandModel(b, qmax, u, beta, e, ag, ar, gw_up, red_down,
                                    ag_slots, ar_keys, m_shift)
            self.meta.red_demands[b.id] = rmodel
            self._add_band_slot_symmetry(b.id, qmax, e, beta, "red")

            # 红波带边界亲和性：红波带起点尽量贴上游绿窗起点、终点尽量贴上游绿窗终点。
            rb = self.problem.red_boundary_for(b)
            rb_ref = None
            if rb.enabled and ((rb.start is not None and rb.start.coef > 0)
                               or (rb.end is not None and rb.end.coef > 0)):
                rb_ref = RedBoundarySlackRef(
                    demand_id=b.id,
                    start_coef=rb.start.coef if rb.start is not None else 0.0,
                    end_coef=rb.end.coef if rb.end is not None else 0.0,
                )
                self.meta.red_boundary.append(rb_ref)

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

                # 上游两条包含约束，只挂 ag。上游位置是 u - τ - φ_up - m·C，
                # m 是上游周期下标（与绿带的 n 对称），补上"上游项缺一个周期"那一维。
                mv = m_shift[q]
                for g in ag_slots:
                    av = ag[(q, g)]
                    rng = m_ranges.get(g)
                    if rng is None:
                        # 该上游窗口槽不存在任何可行 m，直接禁止使用。
                        self.cons.add(LinearExpr.of(av, 1.0), "==", 0.0,
                                      ConstraintMeta("MRANGE",
                                                     f"red wrap infeasible {b.id}/{q}/{g}",
                                                     {"demand": b.id, "slot": str(q),
                                                      "up_window": str(g)}))
                        continue
                    row = LinearExpr.of(u[q], 1.0).add(self.meta.phi[up_node], -1.0)
                    row.add(mv, -self.C)
                    row.add(gw_up.s[g], -1.0).add(av, -M)
                    self.cons.add(row, ">=", tau - M,
                                  ConstraintMeta("C9", f"red up lower {b.id}/{q}/{g}",
                                                 {"demand": b.id, "slot": str(q)}))
                    row = LinearExpr.of(u[q], 1.0).add(beta[q], 1.0)
                    row.add(self.meta.phi[up_node], -1.0).add(mv, -self.C)
                    row.add(gw_up.e[g], -1.0).add(av, M)
                    self.cons.add(row, "<=", tau + M,
                                  ConstraintMeta("C9", f"red up upper {b.id}/{q}/{g}",
                                                 {"demand": b.id, "slot": str(q)}))
                    # m 的可行范围（按上游窗口槽）：只在选中该槽时生效。
                    # 用**紧的** big-M（m 的跨度只有几），比全局 containment M
                    # 强得多，能明显加强 LP 松弛。
                    m_lo_g, m_hi_g = rng
                    Mm = max(1.0, float(m_ub - m_lb)) + 1.0
                    row = LinearExpr.of(mv, 1.0).add(av, Mm)
                    self.cons.add(row, "<=", float(m_hi_g) + Mm,
                                  ConstraintMeta("MRANGE",
                                                 f"red wrap m upper {b.id}/{q}/{g}",
                                                 {"demand": b.id, "slot": str(q),
                                                  "up_window": str(g)}))
                    row = LinearExpr.of(mv, 1.0).add(av, -Mm)
                    self.cons.add(row, ">=", float(m_lo_g) - Mm,
                                  ConstraintMeta("MRANGE",
                                                 f"red wrap m lower {b.id}/{q}/{g}",
                                                 {"demand": b.id, "slot": str(q),
                                                  "up_window": str(g)}))
                # e=0 时 m=0（与绿带 n 的处理一致）。
                row = LinearExpr.of(mv, 1.0).add(e[q], -m_ub)
                self.cons.add(row, "<=", 0.0,
                              ConstraintMeta("MRANGE", f"red wrap m<=ub*e {b.id}/{q}",
                                             {"demand": b.id, "slot": str(q)}))
                row = LinearExpr.of(mv, 1.0).add(e[q], -m_lb)
                self.cons.add(row, ">=", 0.0,
                              ConstraintMeta("MRANGE", f"red wrap m>=lb*e {b.id}/{q}",
                                             {"demand": b.id, "slot": str(q)}))

                # C9_RED_EDGE：红波带边界与上游绿窗边界的距离损失/约束。
                #
                # gap_start = u - (phi_i + s_g + tau)
                # gap_end   = (phi_i + e_g + tau) - (u + beta)
                #
                # 两者都希望尽量接近 0；超过 free_s 的部分用 sigma 惩罚，
                # 若配置 hard_max_s 则再给距离加上硬上限。
                if rb.enabled:
                    M_edge = self.m.soft
                    M_hard = self.m.containment
                    for g in ag_slots:
                        av = ag[(q, g)]
                        start_spec = rb.start
                        end_spec = rb.end

                        if start_spec is not None:
                            if start_spec.coef > 0:
                                sigma_s = self.vars.add(
                                    naming.rw_edge_start(b.id, q, g), 0.0,
                                    self.m.containment * 2.0, GVarType.CONTINUOUS,
                                    {"kind": "red_boundary_slack", "demand": b.id,
                                     "slot": q, "up_window": g, "side": "start"})
                                if rb_ref is not None:
                                    rb_ref.start_vars.append(sigma_s)
                                row = LinearExpr.of(sigma_s, 1.0).add(u[q], -1.0)
                                row.add(self.meta.phi[up_node], 1.0).add(gw_up.s[g], 1.0)
                                row.add(mv, self.C)
                                row.add(av, -M_edge).add_const(tau)
                                self.cons.add(row, ">=", -start_spec.free_s - M_edge,
                                              ConstraintMeta("C9_RED_EDGE",
                                                             f"red edge start {b.id}/{q}/{g}",
                                                             {"demand": b.id, "slot": str(q)}))
                            if start_spec.hard_max_s is not None:
                                row = LinearExpr.of(u[q], 1.0).add(self.meta.phi[up_node], -1.0)
                                row.add(gw_up.s[g], -1.0).add(mv, -self.C)
                                row.add(av, M_hard).add_const(-tau)
                                self.cons.add(row, "<=", start_spec.hard_max_s + M_hard,
                                              ConstraintMeta("C9_RED_EDGE_HARD",
                                                             f"red edge start max {b.id}/{q}/{g}",
                                                             {"demand": b.id, "slot": str(q)}))

                        if end_spec is not None:
                            if end_spec.coef > 0:
                                sigma_e = self.vars.add(
                                    naming.rw_edge_end(b.id, q, g), 0.0,
                                    self.m.containment * 2.0, GVarType.CONTINUOUS,
                                    {"kind": "red_boundary_slack", "demand": b.id,
                                     "slot": q, "up_window": g, "side": "end"})
                                if rb_ref is not None:
                                    rb_ref.end_vars.append(sigma_e)
                                row = LinearExpr.of(sigma_e, 1.0).add(u[q], 1.0)
                                row.add(beta[q], 1.0).add(self.meta.phi[up_node], -1.0)
                                row.add(gw_up.e[g], -1.0).add(mv, -self.C)
                                row.add(av, -M_edge).add_const(-tau)
                                self.cons.add(row, ">=", -end_spec.free_s - M_edge,
                                              ConstraintMeta("C9_RED_EDGE",
                                                             f"red edge end {b.id}/{q}/{g}",
                                                             {"demand": b.id, "slot": str(q)}))
                            if end_spec.hard_max_s is not None:
                                row = LinearExpr.of(self.meta.phi[up_node], 1.0)
                                row.add(gw_up.e[g], 1.0).add(mv, self.C)
                                row.add(u[q], -1.0).add(beta[q], -1.0)
                                row.add(av, M_hard).add_const(tau)
                                self.cons.add(row, "<=", end_spec.hard_max_s + M_hard,
                                              ConstraintMeta("C9_RED_EDGE_HARD",
                                                             f"red edge end max {b.id}/{q}/{g}",
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

                # 时间分离：任意两条都存在的红带在圆环上不能重叠，
                # 否则同一条物理带可能被计数两次。
                for q2 in range(q + 1, qmax + 1):
                    if self.stage1_symmetry:
                        # 红带 q 槽也按起点排序；不存在时用 big-M 放松。
                        row = LinearExpr.of(u[q], 1.0).add(u[q2], -1.0)
                        row.add(e[q], M).add(e[q2], M)
                        self.cons.add(row, "<=", 2.0 * M,
                                      ConstraintMeta("SYM",
                                                     f"red u order {b.id}/{q}/{q2}",
                                                     {"demand": b.id, "slot": str(q),
                                                      "slot2": str(q2)}))
                    row = LinearExpr.of(u[q], 1.0).add(beta[q], 1.0)
                    row.add(u[q2], -1.0).add(e[q], M).add(e[q2], M)
                    self.cons.add(row, "<=", 2.0 * M,
                                  ConstraintMeta("C9", f"red time separation {b.id}/{q}/{q2}",
                                                 {"demand": b.id, "slot": str(q)}))
                    row = LinearExpr.of(u[q2], 1.0).add(beta[q2], 1.0)
                    row.add(u[q], -1.0).add(e[q], M).add(e[q2], M)
                    self.cons.add(row, "<=", self.C + 2.0 * M,
                                  ConstraintMeta("C9", f"red wrap separation {b.id}/{q}/{q2}",
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
            z = self.vars.add(naming.bal_min(g.id), 0.0, z_ub, GVarType.CONTINUOUS,
                              {"kind": "balanced_min_bandwidth", "group": g.id})
            y_vars: dict[str, VarHandle] = {}
            bw_exprs: dict[str, LinearExpr] = {}
            target_vars: dict[str, float] = {}

            for did in g.demands:
                bm = band_models[did]
                target = g.target_for(did)
                target_vars[did] = target
                y = self.vars.add(naming.bal_present(g.id, did), 0.0, 1.0, GVarType.BINARY,
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
                deficit = self.vars.add(naming.bal_deficit(g.id), 0.0,
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
                    sigma = self.vars.add(naming.soft_slack(c.id, "pos"), 0.0, M, GVarType.CONTINUOUS,
                                          {"kind": "soft_violation"})
                    row = lhs.copy().add(sigma, -1.0)
                    self.cons.add(row, "<=", rhs,
                                  ConstraintMeta("USER_SOFT", f"user:{c.id}",
                                                 {"constraint": c.id}))
                    self.meta.soft.append(SoftPenaltyRef(
                        constraint_id=c.id + (".pos" if c.sense == "==" else ""),
                        expr=lhs.copy(), sense="<=", rhs=rhs, coef=coef, owner=owner,
                        label=naming.soft_slack(c.id, "pos")))
                if c.sense in (">=", "=="):
                    sigma = self.vars.add(naming.soft_slack(c.id, "neg"), 0.0, M, GVarType.CONTINUOUS,
                                          {"kind": "soft_violation"})
                    row = lhs.copy().add(sigma, 1.0)
                    self.cons.add(row, ">=", rhs,
                                  ConstraintMeta("USER_SOFT", f"user:{c.id}",
                                                 {"constraint": c.id}))
                    self.meta.soft.append(SoftPenaltyRef(
                        constraint_id=c.id + (".neg" if c.sense == "==" else ""),
                        expr=lhs.copy(), sense=">=", rhs=rhs, coef=coef, owner=owner,
                        label=naming.soft_slack(c.id, "neg")))


__all__ = ["ArterialModel", "ModelBuilder", "ref_label"]
