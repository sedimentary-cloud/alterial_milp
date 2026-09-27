"""预处理层：判断某个方案在某个周期下到底可不可行。

给高中生看的小课堂
------------------
同一个绿灯窗口，在不同周期下代表的秒数不一样。
例如 0.2 个周期的绿灯：
- C=100s 时是 20s；
- C=60s 时只有 12s。

如果方案里写了“最小绿 15s”，那么 C=60s 时这个方案就不可行。

本文件做两件事：
1. 固定窗口方案：直接代入算术检查；
2. 可调窗口方案：解一个小型线性规划（LP）判断是否存在可行的微调量。
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.optimize import linprog

from ..errors import GWInfeasibleGridPoint
from ..schema.models import (ConstraintSpec, Direction, PlanSpec, ProblemInput,
                             Side)
from .context import (DemandTravelTable, DroppedPlanRecord, GridPointContext,
                      PlanFeasibility)


class FeasibilityFilter:
    """方案可行性过滤器。"""

    def __init__(self, problem: ProblemInput) -> None:
        self.problem = problem
        self._lp_cache: dict[tuple[str, float], PlanFeasibility] = {}

    # ------------------------------------------------------------------
    def check_plan(self, plan: PlanSpec, cycle_s: float) -> PlanFeasibility:
        # 缓存：同一个方案、同一个周期只算一次。
        key = (plan.id, float(cycle_s))
        if key not in self._lp_cache:
            has_adjust = any(
                plan.endpoint_is_adjustable(d, k, side)
                for d in ("up", "down")
                for k in range(plan.window_count(d))
                for side in ("start", "end")
            )
            self._lp_cache[key] = self._check_lp(plan, cycle_s) if has_adjust else self._check_fixed(plan, cycle_s)
        return self._lp_cache[key]

    # -- helpers -------------------------------------------------------
    @staticmethod
    def _endpoint_seconds(plan: PlanSpec, d: Direction, k: int, side: Side,
                          delta: float, cycle_s: float) -> float:
        return (plan.nominal_endpoint_r(d, k, side) + delta) * cycle_s

    def _constraint_value(self, plan: PlanSpec, c: ConstraintSpec, cycle_s: float,
                          deltas: dict[tuple[Direction, int, str], float]) -> float:
        value = 0.0
        for t in c.terms:
            a = t.atom
            if a.kind == "endpoint":
                assert a.direction is not None and a.window is not None and a.side is not None
                delta = deltas.get((a.direction, a.window, a.side), 0.0)
                val = self._endpoint_seconds(plan, a.direction, a.window, a.side, delta, cycle_s)
            elif a.kind == "window_duration":
                assert a.direction is not None and a.window is not None
                ds = deltas.get((a.direction, a.window, "start"), 0.0)
                de = deltas.get((a.direction, a.window, "end"), 0.0)
                start = self._endpoint_seconds(plan, a.direction, a.window, "start", ds, cycle_s)
                end = self._endpoint_seconds(plan, a.direction, a.window, "end", de, cycle_s)
                val = end - start
            elif a.kind == "const":
                val = float(a.const or 0.0)
            else:
                # offset/bandwidth atoms are not meaningful in per-plan feasibility;
                # treat offset as a free variable anchored at 0 here.  Validation
                # warns/errors elsewhere when such references are used.
                val = 0.0
            value += t.coef * val
        return value

    def _check_fixed(self, plan: PlanSpec, cycle_s: float) -> PlanFeasibility:
        """固定窗口方案：把比例乘以 C，逐条代入硬约束做算术检查。"""
        violations: list[str] = []
        # effective window V1/V2 check (can differ from input for double-cycle)
        for d in ("up", "down"):
            eff = plan.effective_windows(d)
            for k, w in enumerate(eff):
                if not (0.0 <= w.start_r < w.end_r <= 2.0):
                    violations.append(f"effective {d} window {k} violates 0<=s<e<=2")
            for k in range(len(eff) - 1):
                if eff[k].end_r > eff[k + 1].start_r + 1e-12:
                    violations.append(f"effective {d} windows {k}/{k+1} overlap/out of order")
        for c in plan.hard_constraints:
            lhs = self._constraint_value(plan, c, cycle_s, {})
            rhs = c.rhs_for_cycle(cycle_s)
            bad = None
            if c.sense == "<=" and lhs > rhs + 1e-7:
                bad = lhs - rhs
            elif c.sense == ">=" and lhs < rhs - 1e-7:
                bad = rhs - lhs
            elif c.sense == "==" and abs(lhs - rhs) > 1e-7:
                bad = abs(lhs - rhs)
            if bad is not None:
                violations.append(f"{c.id}: {c.sense} {rhs:.6g} violated by {bad:.6g}s")
        if violations:
            return PlanFeasibility(False, tuple(violations))
        return PlanFeasibility(True, ())

    def _check_lp(self, plan: PlanSpec, cycle_s: float) -> PlanFeasibility:
        """可调窗口方案：把微调量 δ 当变量，解一个小 LP。

        LP 的约束包括：
        - 方案硬约束；
        - 窗口 start >= 0、end <= 2C；
        - start <= end；
        - 相邻窗口不重叠；
        - 双周期复制端点一致性。
        """
        # 收集所有可调端点，每个端点一个 δ 变量。
        keys: list[tuple[Direction, int, str]] = []
        bounds: list[tuple[float, float]] = []
        for d in ("up", "down"):
            for k in range(plan.window_count(d)):
                for side in ("start", "end"):
                    r = plan.adjust_range(d, k, side)
                    if r is not None:
                        keys.append((d, k, side))
                        bounds.append((r.lo_r, r.hi_r))
        if not keys:
            return self._check_fixed(plan, cycle_s)
        index = {key: i for i, key in enumerate(keys)}
        n = len(keys)

        a_ub: list[list[float]] = []
        b_ub: list[float] = []
        a_eq: list[list[float]] = []
        b_eq: list[float] = []

        def endpoint_coefs(d: Direction, k: int, side: Side) -> tuple[float, dict[int, float]]:
            """constant + delta coefficients for endpoint seconds."""
            const = plan.nominal_endpoint_r(d, k, side) * cycle_s
            j = index.get((d, k, side))
            return const, ({j: cycle_s} if j is not None else {})

        def add_ub(coefs: dict[int, float], constant: float, rhs: float) -> None:
            row = [0.0] * n
            for j, c in coefs.items():
                row[j] += c
            a_ub.append(row)
            b_ub.append(rhs - constant)

        def add_eq(coefs: dict[int, float], constant: float, rhs: float) -> None:
            row = [0.0] * n
            for j, c in coefs.items():
                row[j] += c
            a_eq.append(row)
            b_eq.append(rhs - constant)

        def add_sense(coefs: dict[int, float], constant: float, sense: str,
                      rhs: float, label: str) -> None:
            if sense == "<=":
                add_ub(coefs, constant, rhs)
            elif sense == ">=":
                neg = {j: -c for j, c in coefs.items()}
                add_ub(neg, -constant, -rhs)
            else:
                add_eq(coefs, constant, rhs)

        # 方案声明的硬约束，例如最小绿、窗口间距。
        for c in plan.hard_constraints:
            coefs: dict[int, float] = {}
            constant = 0.0
            for t in c.terms:
                a = t.atom
                if a.kind == "endpoint":
                    assert a.direction is not None and a.window is not None and a.side is not None
                    const, cf = endpoint_coefs(a.direction, a.window, a.side)
                    constant += t.coef * const
                    for j, val in cf.items():
                        coefs[j] = coefs.get(j, 0.0) + t.coef * val
                elif a.kind == "window_duration":
                    assert a.direction is not None and a.window is not None
                    s_const, s_cf = endpoint_coefs(a.direction, a.window, "start")
                    e_const, e_cf = endpoint_coefs(a.direction, a.window, "end")
                    constant += t.coef * (e_const - s_const)
                    for j, val in e_cf.items():
                        coefs[j] = coefs.get(j, 0.0) + t.coef * val
                    for j, val in s_cf.items():
                        coefs[j] = coefs.get(j, 0.0) - t.coef * val
                elif a.kind == "const":
                    constant += t.coef * float(a.const or 0.0)
            add_sense(coefs, constant, c.sense, c.rhs_for_cycle(cycle_s), c.id)

        # 窗口顺序与 V1 边界：
        # 0 <= start <= end <= 2C，且相邻窗口不能重叠。
        for d in ("up", "down"):
            eff = plan.effective_windows(d)
            for k in range(len(eff)):
                s_const, s_cf = endpoint_coefs(d, k, "start")
                e_const, e_cf = endpoint_coefs(d, k, "end")
                # start >= 0
                add_sense(s_cf, s_const, ">=", 0.0, f"{d}.{k}.start_min")
                # end <= 2C (design V1)
                add_sense(e_cf, e_const, "<=", 2.0 * cycle_s, f"{d}.{k}.end_max")
                # start <= end
                diff = dict(s_cf)
                for j, val in e_cf.items():
                    diff[j] = diff.get(j, 0.0) - val
                add_sense(diff, s_const - e_const, "<=", 0.0, f"{d}.{k}.order")
            for k in range(len(eff) - 1):
                a_const, a_cf = endpoint_coefs(d, k, "end")
                b_const, b_cf = endpoint_coefs(d, k + 1, "start")
                coefs = dict(a_cf)
                for j, val in b_cf.items():
                    coefs[j] = coefs.get(j, 0.0) - val
                add_sense(coefs, a_const - b_const, "<=", 0.0, f"{d}.{k}.nonoverlap")

        # 双周期一致性：复制窗口的微调量必须等于原始窗口。
        if plan.double_cycle:
            # map original index -> effective index for originals
            orig_to_eff: dict[int, int] = {}
            for d in ("up", "down"):
                for k in range(plan.window_count(d)):
                    if not plan.effective_is_copy(d, k):
                        orig_to_eff[(d, plan.effective_original_index(d, k))] = k
                for k in range(plan.window_count(d)):
                    if not plan.effective_is_copy(d, k):
                        continue
                    orig_idx = plan.effective_original_index(d, k)
                    orig_k = orig_to_eff.get((d, orig_idx))
                    if orig_k is None:
                        continue
                    for side in ("start", "end"):
                        jc = index.get((d, k, side))
                        jo = index.get((d, orig_k, side))
                        if jc is None or jo is None:
                            continue
                        add_eq({jc: 1.0, jo: -1.0}, 0.0, 0.0)

        if not a_ub and not a_eq:
            feasible = True
            x = None
        else:
            res = linprog(
                c=np.zeros(n),
                A_ub=np.asarray(a_ub, dtype=float) if a_ub else None,
                b_ub=np.asarray(b_ub, dtype=float) if b_ub else None,
                A_eq=np.asarray(a_eq, dtype=float) if a_eq else None,
                b_eq=np.asarray(b_eq, dtype=float) if b_eq else None,
                bounds=bounds,
                method="highs",
            )
            feasible = bool(res.status == 0)
            x = res.x if feasible else None

        if feasible:
            witness = tuple((d, k, side, float(x[index[(d, k, side)]]))
                            for (d, k, side) in keys)
            return PlanFeasibility(True, (), witness)
        reasons = [f"{c.id}: hard constraint infeasible together with window ordering/bounds"
                   for c in plan.hard_constraints]
        if not reasons:
            reasons = ["adjustable windows cannot satisfy V1/V2 window ordering/bounds"]
        return PlanFeasibility(False, tuple(reasons))

    # ------------------------------------------------------------------
    def filter(self, cycle_s: float, kappa: float) -> GridPointContext:
        """逐路口过滤方案，并计算该格点下的所有常数。

        如果某个路口一个可行方案都没有，就抛出 GWInfeasibleGridPoint，
        由上层跳过该格点并记录诊断。
        """
        feasible: dict[str, tuple[PlanSpec, ...]] = {}
        dropped: list[DroppedPlanRecord] = []
        bad: dict[str, tuple[str, ...]] = {}
        for iid in self.problem.corridor.intersections:
            good: list[PlanSpec] = []
            for p in self.problem.plans_of(iid):
                status = self.check_plan(p, cycle_s)
                if status.feasible:
                    good.append(p)
                else:
                    dropped.append(DroppedPlanRecord(iid, p.id, float(cycle_s), status.violations))
            if not good:
                bad[iid] = tuple(v for r in dropped if r.intersection == iid for v in r.violations)
            else:
                feasible[iid] = tuple(good)
        if bad:
            raise GWInfeasibleGridPoint(cycle_s, kappa, bad, dropped=tuple(dropped))

        # plan witnesses
        witnesses: dict[str, tuple[tuple[Direction, int, str, float], ...]] = {}
        for iid, plans in feasible.items():
            for p in plans:
                witnesses[p.id] = self.check_plan(p, cycle_s).witness_r

        # 计算每个路段在每个方向的行程时间 τ = 距离 / (κ × 速度)。
        # 这些值在当前格点内是常数。
        tau: dict[tuple[str, Direction], float] = {}
        corr = self.problem.corridor
        for idx, seg in enumerate(corr.segments):
            tau[(corr.intersections[idx], "up")] = seg.distance_for("up") / (kappa * seg.speed_up_mps)
        for idx in range(1, len(corr.intersections)):
            seg = corr.segments[idx - 1]
            tau[(corr.intersections[idx], "down")] = seg.distance_for("down") / (kappa * seg.speed_down_mps)

        # 对每条带需求，计算沿链的累计行程时间 T_j。
        # 若带在链首时间为 u，则到第 j 个节点的时间就是 u + T_j。
        travel: dict[str, DemandTravelTable] = {}
        for b in self.problem.objective.band_demands:
            cum = [0.0]
            for a, bnode in zip(b.nodes, b.nodes[1:]):
                t = tau.get((a, b.direction))
                if t is None:
                    raise GWInfeasibleGridPoint(cycle_s, kappa,
                                                {a: (f"no segment along {b.direction} for demand {b.id}",)})
                cum.append(cum[-1] + t)
            n_max = tuple(int(math.ceil(v / cycle_s - 1e-12)) if v > 0 else 0 for v in cum)
            travel[b.id] = DemandTravelTable(
                demand_id=b.id,
                nodes=b.nodes,
                cum_T_s=tuple(cum),
                span_s=cum[-1],
                n_max=n_max,
            )
        return GridPointContext(
            cycle_s=float(cycle_s),
            kappa=float(kappa),
            tau_s=tau,
            feasible_plans=feasible,
            dropped=tuple(dropped),
            travel=travel,
            plan_witness_r=witnesses,
        )


__all__ = ["FeasibilityFilter"]
