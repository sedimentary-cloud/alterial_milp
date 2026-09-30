"""预处理层：判断某个相位边界方案在某个周期下是否可行。

v2 的可行性变量是每个方向的可调相位边界 δ（比例域）。
硬约束包括：
- 边界排序 + min_phase_s（内置几何约束）；
- 方案声明的 hard_constraints（boundary / window_duration 原子）；
- 双周期声明边界在半周期内，复制边界复用同一个 δ，因此不需要额外一致性等式。
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.optimize import linprog

from ..errors import GWInfeasibleGridPoint
from ..schema.models import ConstraintSpec, Direction, PlanSpec, ProblemInput, Side
from .context import (DemandTravelTable, DroppedPlanRecord, GridPointContext,
                      PlanFeasibility)


class FeasibilityFilter:
    """方案可行性过滤器。"""

    def __init__(self, problem: ProblemInput) -> None:
        self.problem = problem
        self._lp_cache: dict[tuple[str, float], PlanFeasibility] = {}

    # ------------------------------------------------------------------
    def check_plan(self, plan: PlanSpec, cycle_s: float) -> PlanFeasibility:
        key = (plan.id, float(cycle_s))
        if key not in self._lp_cache:
            has_adjust = any(
                plan.boundary_is_adjustable(d, j)
                for d in ("up", "down")
                for j in range(plan.boundary_count(d))
            )
            self._lp_cache[key] = (self._check_lp(plan, cycle_s)
                                   if has_adjust else self._check_fixed(plan, cycle_s))
        return self._lp_cache[key]

    # -- helpers -------------------------------------------------------
    @staticmethod
    def _boundary_seconds(plan: PlanSpec, d: Direction, j: int, delta: float,
                          cycle_s: float) -> float:
        return (plan.nominal_boundary_r(d, j) + delta) * cycle_s

    def _edge_seconds(self, plan: PlanSpec, d: Direction, edge: int, cycle_s: float,
                      deltas: dict[tuple[Direction, int], float]) -> float:
        """Expanded edge value in seconds, including +0.5 copy offsets."""
        kind, src, offset = plan.expanded_edge_sources(d)[edge]
        if kind == "const":
            val = float(offset)
        else:
            val = plan.nominal_boundary_r(d, src) + offset + deltas.get((d, src), 0.0)
        return val * cycle_s

    def _constraint_value(self, plan: PlanSpec, c: ConstraintSpec, cycle_s: float,
                          deltas: dict[tuple[Direction, int], float]) -> float:
        value = 0.0
        for t in c.terms:
            a = t.atom
            if a.kind == "boundary":
                assert a.direction is not None and a.index is not None
                val = self._boundary_seconds(plan, a.direction, a.index,
                                             deltas.get((a.direction, a.index), 0.0), cycle_s)
            elif a.kind == "window_duration":
                assert a.direction is not None and a.window is not None
                w = plan.window_edge_span(a.direction, a.window)
                start = self._edge_seconds(plan, a.direction, w.start_edge, cycle_s, deltas)
                end = self._edge_seconds(plan, a.direction, w.end_edge, cycle_s, deltas)
                val = end + w.wrap * cycle_s - start
            elif a.kind == "red_duration":
                assert a.direction is not None and a.window is not None
                reds = plan.derived_red_intervals(a.direction)
                if a.window < 1 or a.window > len(reds):
                    raise ValueError(f"red_duration references unknown rid {a.window}")
                r = reds[a.window - 1]
                start = self._edge_seconds(plan, a.direction, r.start_edge, cycle_s, deltas)
                end = self._edge_seconds(plan, a.direction, r.end_edge, cycle_s, deltas)
                val = end + r.wrap * cycle_s - start
            elif a.kind == "const":
                val = float(a.const or 0.0)
            else:
                val = 0.0
            value += t.coef * val
        return value

    @staticmethod
    def _geometry_violations(plan: PlanSpec, cycle_s: float,
                             deltas: dict[tuple[Direction, int], float]) -> list[str]:
        """Check boundary ordering / min_phase / range for both directions."""
        g = plan.min_phase_s / cycle_s if cycle_s > 1e-12 else float("inf")
        out: list[str] = []
        for d in ("up", "down"):
            m = plan.boundary_count(d)
            vals = [plan.nominal_boundary_r(d, j) + deltas.get((d, j), 0.0)
                    for j in range(m)]
            hi = 0.5 if plan.double_cycle else 1.0
            if g > 0 and m == 0:
                if hi < g - 1e-12:
                    out.append(f"{d}: cycle length below min_phase_s")
                continue
            if m:
                if vals[0] < g - 1e-9:
                    out.append(f"{d}: b0 below min_phase_s")
                for j in range(m - 1):
                    if vals[j + 1] - vals[j] < g - 1e-9:
                        out.append(f"{d}: boundary {j}/{j+1} gap below min_phase_s")
                if hi - vals[-1] < g - 1e-9:
                    out.append(f"{d}: last boundary to cycle end below min_phase_s")
        return out

    def _check_fixed(self, plan: PlanSpec, cycle_s: float) -> PlanFeasibility:
        deltas: dict[tuple[Direction, int], float] = {}
        violations = list(self._geometry_violations(plan, cycle_s, deltas))
        for c in plan.hard_constraints:
            lhs = self._constraint_value(plan, c, cycle_s, deltas)
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
        keys: list[tuple[Direction, int]] = []
        bounds: list[tuple[float, float]] = []
        for d in ("up", "down"):
            for j in range(plan.boundary_count(d)):
                r = plan.boundary_adjust_range(d, j)
                if r is not None:
                    keys.append((d, j))
                    bounds.append((r[0], r[1]))
        if not keys:
            return self._check_fixed(plan, cycle_s)
        index = {key: i for i, key in enumerate(keys)}
        n = len(keys)

        a_ub: list[list[float]] = []
        b_ub: list[float] = []
        a_eq: list[list[float]] = []
        b_eq: list[float] = []

        def boundary_coefs(d: Direction, j: int) -> tuple[float, dict[int, float]]:
            const = plan.nominal_boundary_r(d, j) * cycle_s
            i = index.get((d, j))
            return const, ({i: cycle_s} if i is not None else {})

        def add_ub(coefs: dict[int, float], constant: float, rhs: float) -> None:
            row = [0.0] * n
            for i, c in coefs.items():
                row[i] += c
            a_ub.append(row)
            b_ub.append(rhs - constant)

        def add_eq(coefs: dict[int, float], constant: float, rhs: float) -> None:
            row = [0.0] * n
            for i, c in coefs.items():
                row[i] += c
            a_eq.append(row)
            b_eq.append(rhs - constant)

        def add_sense(coefs: dict[int, float], constant: float, sense: str,
                      rhs: float, label: str) -> None:
            if sense == "<=":
                add_ub(coefs, constant, rhs)
            elif sense == ">=":
                add_ub({i: -c for i, c in coefs.items()}, -constant, -rhs)
            else:
                add_eq(coefs, constant, rhs)

        def edge_seconds_coefs(d: Direction, edge: int) -> tuple[float, dict[int, float]]:
            kind, src, offset = plan.expanded_edge_sources(d)[edge]
            if kind == "const":
                return float(offset) * cycle_s, {}
            const, coefs = boundary_coefs(d, src)
            const += offset * cycle_s
            return const, coefs

        # 方案声明的硬约束。
        for c in plan.hard_constraints:
            coefs: dict[int, float] = {}
            constant = 0.0
            for t in c.terms:
                a = t.atom
                if a.kind == "boundary":
                    assert a.direction is not None and a.index is not None
                    const, cf = boundary_coefs(a.direction, a.index)
                    constant += t.coef * const
                    for i, val in cf.items():
                        coefs[i] = coefs.get(i, 0.0) + t.coef * val
                elif a.kind == "window_duration":
                    assert a.direction is not None and a.window is not None
                    w = plan.window_edge_span(a.direction, a.window)
                    s_const, s_cf = edge_seconds_coefs(a.direction, w.start_edge)
                    e_const, e_cf = edge_seconds_coefs(a.direction, w.end_edge)
                    e_const += w.wrap * cycle_s
                    constant += t.coef * (e_const - s_const)
                    for i, val in e_cf.items():
                        coefs[i] = coefs.get(i, 0.0) + t.coef * val
                    for i, val in s_cf.items():
                        coefs[i] = coefs.get(i, 0.0) - t.coef * val
                elif a.kind == "red_duration":
                    assert a.direction is not None and a.window is not None
                    reds = plan.derived_red_intervals(a.direction)
                    if a.window < 1 or a.window > len(reds):
                        raise ValueError(f"red_duration references unknown rid {a.window}")
                    r = reds[a.window - 1]
                    s_const, s_cf = edge_seconds_coefs(a.direction, r.start_edge)
                    e_const, e_cf = edge_seconds_coefs(a.direction, r.end_edge)
                    e_const += r.wrap * cycle_s
                    constant += t.coef * (e_const - s_const)
                    for i, val in e_cf.items():
                        coefs[i] = coefs.get(i, 0.0) + t.coef * val
                    for i, val in s_cf.items():
                        coefs[i] = coefs.get(i, 0.0) - t.coef * val
                elif a.kind == "const":
                    constant += t.coef * float(a.const or 0.0)
            add_sense(coefs, constant, c.sense, c.rhs_for_cycle(cycle_s), c.id)

        # 内置几何约束：排序 + min_phase_s。
        g = plan.min_phase_s / cycle_s if cycle_s > 1e-12 else 0.0
        for d in ("up", "down"):
            m = plan.boundary_count(d)
            hi = 0.5 if plan.double_cycle else 1.0
            if m == 0:
                continue
            # b0 >= g
            const, cf = boundary_coefs(d, 0)
            add_sense(cf, const, ">=", g * cycle_s, f"{d}.b0.min_phase")
            # gaps
            for j in range(m - 1):
                c1, cf1 = boundary_coefs(d, j)
                c2, cf2 = boundary_coefs(d, j + 1)
                coefs = dict(cf2)
                for i, val in cf1.items():
                    coefs[i] = coefs.get(i, 0.0) - val
                add_sense(coefs, c2 - c1, ">=", g * cycle_s, f"{d}.b{j}.min_phase")
            # hi - bm >= g  ->  bm <= hi - g
            const, cf = boundary_coefs(d, m - 1)
            add_sense(cf, const, "<=", (hi - g) * cycle_s, f"{d}.bm.max")

        if not a_ub and not a_eq:
            feasible = True
            x = None
        else:
            res = linprog(
                c=np.zeros(n),
                A_ub=np.asarray(a_ub, dtype=float) if a_ub else None,
                b_ub=np.asarray(b_ub, dtype=float) if b_ub else None,
                A_eq=np.asarray(a_eq, dtype=float) if a_eq else None,
                b_eq=np.asarray(b_eq, dtype=float) if a_eq else None,
                bounds=bounds,
                method="highs",
            )
            feasible = bool(res.status == 0)
            x = res.x if feasible else None

        if feasible:
            witness = tuple((d, j, float(x[index[(d, j)]])) for (d, j) in keys)
            return PlanFeasibility(True, (), witness)
        reasons = list(self._geometry_violations(plan, cycle_s, {}))
        reasons += [f"{c.id}: hard constraint infeasible together with boundary ordering/min_phase"
                    for c in plan.hard_constraints]
        if not reasons:
            reasons = ["adjustable boundaries cannot satisfy ordering/min_phase/hard constraints"]
        return PlanFeasibility(False, tuple(reasons))

    # ------------------------------------------------------------------
    def filter(self, cycle_s: float, kappa: float) -> GridPointContext:
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

        witnesses: dict[str, tuple[tuple[Direction, int, float], ...]] = {}
        for iid, plans in feasible.items():
            for p in plans:
                witnesses[p.id] = self.check_plan(p, cycle_s).witness_r

        tau: dict[tuple[str, Direction], float] = {}
        corr = self.problem.corridor
        for idx, seg in enumerate(corr.segments):
            tau[(corr.intersections[idx], "up")] = seg.distance_for("up") / (kappa * seg.speed_up_mps)
        for idx in range(1, len(corr.intersections)):
            seg = corr.segments[idx - 1]
            tau[(corr.intersections[idx], "down")] = seg.distance_for("down") / (kappa * seg.speed_down_mps)

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
