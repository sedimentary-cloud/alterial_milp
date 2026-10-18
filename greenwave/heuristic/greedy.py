"""贪心初始解：先快速找一个“还不错”的可行方案。

给高中生看的小课堂
------------------
MILP 求解器虽然能找到最优解，但有时很慢。
我们可以先自己快速拼一个方案：
- 每个路口先选一个看起来最不亏的方案；
- 再用枚举事件点的办法，快速给每个路口试几个偏置 φ；
- 然后构造绿波带/红波带，算一算总带宽；
- 最后做几轮局部微调。

这个贪心解有三个作用：
1. 给 MILP 一个下界，帮助判断求解结果是否正常；
2. 给格点排序：贪心分高的格点先算；
3. 如果 MILP 超时，可以拿它当备选方案。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Iterable, Mapping

from ..preprocess.context import GridPointContext
from ..schema.models import (BandDemandSpec, ConstraintSpec, Direction, PlanSpec,
                             ProblemInput, Side)


# 一条实际构造出来的带：方向、宽度、起点、用了哪些窗口。
@dataclass(frozen=True)
class ConstructedBand:
    demand: str
    slot: int
    exists: bool
    width_s: float
    start_s: float
    windows: Mapping[str, int] = field(default_factory=dict)
    red_instance: tuple[int, ...] | None = None
    # 红带上游周期下标 m（与绿带的 n 对称）：s_g ≤ u - τ - φ_up - m·C ≤ e_g。
    red_up_wrap: int = 0
    # (demand, slot, node) -> 回绕周期 n。SolutionDecoder/GreedyConstructor 都写入。
    wraps: Mapping[str, int] = field(default_factory=dict)


# 贪心解：每个路口选什么方案、偏置多少、带有多宽。
@dataclass(frozen=True)
class GreedySolution:
    plan_selection: Mapping[str, str]
    offsets_s: Mapping[str, float]
    adjustments_r: Mapping[tuple[str, Direction, int], float]
    bands: tuple[ConstructedBand, ...]
    composite_est: float
    loss_est: float = 0.0


class GreedyConstructor:
    """贪心构造器。"""

    def __init__(self, problem: ProblemInput, ctx: GridPointContext) -> None:
        self.problem = problem
        self.ctx = ctx
        self.C = ctx.cycle_s
        self._witness = {plan_id: ctx.witness_for(plan_id) for plan_id in ctx.plan_witness_r}

    # -- v2 boundary helpers --------------------------------------------------
    def _edge_value_r(self, plan: PlanSpec, d: Direction, edge: int,
                      witness: Mapping[tuple[Direction, int], float]) -> float:
        kind, src, offset = plan.expanded_edge_sources(d)[edge]
        if kind == "const":
            return float(offset)
        return plan.nominal_boundary_r(d, src) + offset + witness.get((d, src), 0.0)

    def _window_endpoint_r(self, plan: PlanSpec, d: Direction, k: int, side: str,
                           witness: Mapping[tuple[Direction, int], float]) -> float:
        w = plan.window_edge_span(d, k)
        if side == "start":
            return self._edge_value_r(plan, d, w.start_edge, witness)
        return self._edge_value_r(plan, d, w.end_edge, witness) + w.wrap

    def _edge_duration_s(self, plan: PlanSpec, d: Direction, start_edge: int,
                         end_edge: int, wrap: int,
                         witness: Mapping[tuple[Direction, int], float]) -> float:
        return (self._edge_value_r(plan, d, end_edge, witness) + wrap
                - self._edge_value_r(plan, d, start_edge, witness)) * self.C

    # ------------------------------------------------------------------
    # public entry
    # ------------------------------------------------------------------
    def run(self) -> GreedySolution:
        # 完整流程：选方案 -> 定偏置 -> 构造带 -> 局部 refinement。
        selection = self._select_plans()
        offsets = self._compute_offsets(selection)
        bands = self._construct_bands(selection, offsets)
        sol = self._make_solution(selection, offsets, bands)
        if self.problem.solver.heuristic.refine_rounds > 0:
            sol = self._refine(sol, self.problem.solver.heuristic.refine_rounds)
        return sol

    # ------------------------------------------------------------------
    # step 1: plan selection
    # ------------------------------------------------------------------
    def _plan_loss(self, plan: PlanSpec) -> float:
        witness = self._witness.get(plan.id, {})
        total = 0.0
        for c in plan.soft_constraints:
            total += self._soft_violation_plan(plan, c, witness)
        return total

    def _plan_green_score(self, plan: PlanSpec) -> float:
        total = 0.0
        for d in ("up", "down"):
            total += sum(w.width_r for w in plan.effective_windows(d))
        return total

    def _select_plans(self) -> dict[str, PlanSpec]:
        """每个路口先选一个路口损失最小、绿灯总窗口尽量大的方案。"""
        out: dict[str, PlanSpec] = {}
        for iid in self.problem.corridor.intersections:
            plans = self.ctx.feasible_plans.get(iid, ())
            if not plans:
                raise RuntimeError(f"no feasible plans for {iid}")
            best = min(plans, key=lambda p: (self._plan_loss(p), -self._plan_green_score(p), p.id))
            out[iid] = best
        return out

    # ------------------------------------------------------------------
    # step 2: offsets
    # ------------------------------------------------------------------
    def _window_intervals_abs(self, plan: PlanSpec, d: Direction, phi_s: float,
                              cycle_s: float):
        """Return list of (start_s, end_s, k) for one nominal public cycle."""
        witness = self._witness.get(plan.id, {})
        out = []
        for k in range(plan.window_count(d)):
            s_r = self._window_endpoint_r(plan, d, k, "start", witness)
            e_r = self._window_endpoint_r(plan, d, k, "end", witness)
            out.append((s_r * cycle_s + phi_s, e_r * cycle_s + phi_s, k))
        return out

    def _demand_targets(self, u_ref: float) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for b in self.problem.objective.band_demands:
            table = self.ctx.travel[b.id]
            out[b.id] = {node: (u_ref + t) % self.C for node, t in zip(table.nodes, table.cum_T_s)}
        return out

    def _score_offset(self, i: str, phi_s: float, selection: Mapping[str, PlanSpec],
                      targets: Mapping[str, Mapping[str, float]]) -> float:
        score = 0.0
        for b in self.problem.objective.band_demands:
            if i not in b.nodes:
                continue
            plan = selection[i]
            target = targets[b.id][i]
            intervals = self._window_intervals_abs(plan, b.direction, phi_s, self.C)
            if b.type == "green":
                # Estimate the widest band that can be hosted locally.
                width_est = min(min((w.width_r for w in selection[node].effective_windows(b.direction)),
                                    default=0.0) for node in b.nodes) * self.C
            else:
                width_est = 0.25 * self.C
            best = 0.0
            for s, e, k in intervals:
                if e <= s:
                    continue
                base = math.floor((target - s) / self.C)
                for n in range(int(base) - 2, int(base) + 4):
                    ss = s + n * self.C
                    ee = e + n * self.C
                    overlap = max(0.0, min(target + width_est, ee) - max(target, ss))
                    if overlap > best:
                        best = overlap
            score += best if b.type == "green" else 0.5 * best
        return score

    def _candidate_offsets(self, i: str, selection: Mapping[str, PlanSpec],
                           targets: Mapping[str, Mapping[str, float]]) -> list[float]:
        cands = {0.0}
        for b in self.problem.objective.band_demands:
            if i not in b.nodes:
                continue
            plan = selection[i]
            target = targets[b.id][i]
            witness = self._witness.get(plan.id, {})
            for k in range(plan.window_count(b.direction)):
                for nominal in (self._window_endpoint_r(plan, b.direction, k, "start", witness),
                                self._window_endpoint_r(plan, b.direction, k, "end", witness)):
                    event = nominal * self.C
                    # place the event close to target, with optional margin offsets
                    for target_shift in (0.0, 2.0, -2.0):
                        n = math.floor((target + target_shift - event) / self.C)
                        for nn in (n, n + 1):
                            phi = (target + target_shift - event - nn * self.C) % self.C
                            cands.add(round(phi, 9))
        return sorted(cands)

    def _compute_offsets(self, selection: Mapping[str, PlanSpec]) -> dict[str, float]:
        """给每个路口试一些候选 φ，选择让带更宽的偏置。

        偏置 φ 可以理解为“这个路口的周期从第几秒开始”。
        """
        anchor = self.problem.corridor.intersections[0]
        # reference band start: first window start of first green demand
        green_demands = [b for b in self.problem.objective.band_demands if b.type == "green"]
        ref_u = 0.0
        if green_demands:
            b = green_demands[0]
            p = selection[b.nodes[0]]
            w = p.effective_windows(b.direction)[0]
            ref_u = (w.start_r * self.C) % self.C
        targets = self._demand_targets(ref_u)
        offsets: dict[str, float] = {iid: 0.0 for iid in self.problem.corridor.intersections}
        # anchor fixed at zero; optimise remaining intersections in physical order
        for iid in self.problem.corridor.intersections[1:]:
            best_phi = 0.0
            best_score = -1e18
            for phi in self._candidate_offsets(iid, selection, targets):
                sc = self._score_offset(iid, phi, selection, targets)
                if sc > best_score + 1e-12:
                    best_score = sc
                    best_phi = phi
            offsets[iid] = best_phi
        # global shift so anchor is exactly 0 (translation symmetry)
        shift = offsets[anchor]
        for iid in offsets:
            offsets[iid] = (offsets[iid] - shift) % self.C
        offsets[anchor] = 0.0
        return offsets

    # ------------------------------------------------------------------
    # step 3: construct bands
    # ------------------------------------------------------------------
    def _plan_window_abs(self, plan: PlanSpec, d: Direction, phi_s: float,
                         cycle_s: float) -> list[tuple[float, float, int]]:
        return self._window_intervals_abs(plan, d, phi_s, cycle_s)

    def _construct_green_bands(self, b: BandDemandSpec, selection, offsets) -> list[ConstructedBand]:
        table = self.ctx.travel[b.id]
        nodes = b.nodes
        first = nodes[0]
        first_plan = selection[first]
        first_intervals = self._plan_window_abs(first_plan, b.direction, offsets[first], self.C)
        qmax = self.ctx.demand_max_bands(
            b, allow_sharing=self.problem.objective.allow_window_sharing)
        candidates: list[ConstructedBand] = []
        for k0, (s0, e0, _) in enumerate(first_intervals):
            # 对外统一使用 1-based 窗口槽位；内部 helper 仍以 0-based 索引方案窗口。
            k0_out = k0 + 1
            for u0 in {s0, (s0 + e0) / 2.0, e0, s0 + 2.0, e0 - 2.0}:
                lower = u0
                upper = e0
                assignment: dict[str, int] = {first: k0_out}
                wraps: dict[str, int] = {first: 0}
                ok = True
                for j, node in enumerate(nodes[1:], start=1):
                    target = u0 + table.cum_T_s[j]
                    plan = selection[node]
                    intervals = self._plan_window_abs(plan, b.direction, offsets[node], self.C)
                    best = None
                    # choose the periodic copy that overlaps the current band best
                    for s, e, k0_inner in intervals:
                        n0 = math.floor((target - s) / self.C)
                        for n in range(n0 - 2, n0 + 4):
                            ss = s + n * self.C
                            ee = e + n * self.C
                            lo = max(lower, ss - table.cum_T_s[j])
                            hi = min(upper, ee - table.cum_T_s[j])
                            if hi > lo + 1e-9:
                                # (width, lo, hi, k, n)
                                cand = (hi - lo, lo, hi, k0_inner + 1, n)
                                if best is None or cand > best:
                                    best = cand
                    if best is None:
                        ok = False
                        break
                    _, lower, upper, k, n = best
                    assignment[node] = k
                    wraps[node] = n
                if not ok:
                    continue
                width = upper - lower
                if width > 1e-7:
                    candidates.append(ConstructedBand(b.id, 0, True, width, lower,
                                                      dict(assignment), wraps=dict(wraps)))
        # keep widest non-conflicting bands
        candidates.sort(key=lambda c: (-c.width_s, c.start_s))
        chosen: list[ConstructedBand] = []
        occupied: set[tuple[str, int]] = set()
        for cand in candidates:
            conflict = any((node, slot) in occupied for node, slot in cand.windows.items())
            if conflict:
                continue
            occupied.update(cand.windows.items())
            chosen.append(replace(cand, slot=len(chosen) + 1))
            if len(chosen) >= qmax:
                break
        return chosen

    def _downstream_red_intervals(self, plan: PlanSpec, d: Direction, phi_s: float):
        """Return red intervals as (start, end, kind, r_index) over one/two cycles."""
        intervals = self._plan_window_abs(plan, d, phi_s, self.C)
        intervals.sort(key=lambda x: (x[0], x[1]))
        reds: list[tuple[float, float, int]] = []
        for i in range(len(intervals) - 1):
            reds.append((intervals[i][1], intervals[i + 1][0], i + 1))
        if intervals:
            reds.append((intervals[-1][1], intervals[0][0] + self.C, len(intervals)))
        # tile ± one cycle so shifted upstream windows can match
        out = []
        for s, e, ridx in reds:
            for n in (-1, 0, 1):
                out.append((s + n * self.C, e + n * self.C, ridx))
        return out

    def _construct_red_bands(self, b: BandDemandSpec, selection, offsets) -> list[ConstructedBand]:
        assert len(b.nodes) == 2
        up_node, down_node = b.nodes
        d = b.direction
        up_plan = selection[up_node]
        down_plan = selection[down_node]
        tau = self.ctx.tau_s.get((up_node, d))
        if tau is None:
            return []
        up_windows = self._plan_window_abs(up_plan, d, offsets[up_node], self.C)
        red_intervals = self._downstream_red_intervals(down_plan, d, offsets[down_node])
        qmax = self.ctx.demand_max_bands(
            b, allow_sharing=self.problem.objective.allow_window_sharing)
        candidates = []
        k_upper = int(math.ceil((tau + 2 * self.C) / self.C)) + 1
        for s_up, e_up, g in up_windows:
            for s_red, e_red, r in red_intervals:
                for k in range(0, k_upper + 1):
                    lo = max(s_red, s_up + tau - k * self.C)
                    hi = min(e_red, e_up + tau - k * self.C)
                    if hi > lo + 1e-9:
                        # Normalize to the representative copy with u in [0, C);
                        # otherwise periodic copies of one physical band would be
                        # counted multiple times by the heuristic bound.
                        start = lo % self.C
                        candidates.append((hi - lo, start, g, r, k))
        candidates.sort(key=lambda x: (-x[0], x[1]))
        chosen: list[ConstructedBand] = []
        used: set[tuple[float, float, int, int]] = set()
        for width, start, g, r, k in candidates:
            key = (round(start, 6), round(width, 6), g, r)
            if key in used:
                continue
            used.add(key)
            g_out = g + 1
            chosen.append(ConstructedBand(b.id, len(chosen) + 1, True, width, start,
                                          {up_node: g_out, down_node: r},
                                          red_instance=(g_out, r, k)))
            if len(chosen) >= qmax:
                break
        return chosen

    def _construct_bands(self, selection, offsets) -> tuple[ConstructedBand, ...]:
        """在偏置固定后，直接算出每条绿带/红带的真实宽度。

        对固定偏置来说，窗口位置是确定的，所以带的宽度可以精确计算，
        不需要用求解器。
        """
        out: list[ConstructedBand] = []
        for b in self.problem.objective.band_demands:
            if b.type == "green":
                out.extend(self._construct_green_bands(b, selection, offsets))
            else:
                out.extend(self._construct_red_bands(b, selection, offsets))
        return tuple(out)

    # ------------------------------------------------------------------
    # evaluation / refinement
    # ------------------------------------------------------------------
    def _soft_violation_plan(self, plan: PlanSpec, c: ConstraintSpec,
                             witness: Mapping[tuple[Direction, int], float]) -> float:
        lhs = 0.0
        for t in c.terms:
            a = t.atom
            if a.kind == "boundary":
                assert a.direction is not None and a.index is not None
                delta = witness.get((a.direction, a.index), 0.0)
                val = (plan.nominal_boundary_r(a.direction, a.index) + delta) * self.C
            elif a.kind == "window_duration":
                assert a.direction is not None and a.window is not None
                w = plan.window_edge_span(a.direction, a.window)
                val = self._edge_duration_s(plan, a.direction, w.start_edge, w.end_edge,
                                            w.wrap, witness)
            elif a.kind == "red_duration":
                assert a.direction is not None and a.window is not None
                reds = plan.derived_red_intervals(a.direction)
                r = reds[a.window - 1]
                val = self._edge_duration_s(plan, a.direction, r.start_edge, r.end_edge,
                                            r.wrap, witness)
            elif a.kind == "const":
                val = float(a.const or 0.0)
            else:
                val = 0.0
            lhs += t.coef * val
        rhs = c.rhs_for_cycle(self.C)
        if c.sense == "<=":
            viol = max(0.0, lhs - rhs)
        elif c.sense == ">=":
            viol = max(0.0, rhs - lhs)
        else:
            viol = abs(lhs - rhs)
        coef = c.soft.coef if c.soft is not None else 0.0
        return coef * viol

    def _band_margin_penalty(self, band: ConstructedBand, b: BandDemandSpec,
                             selection, offsets) -> float:
        if b.type != "green":
            return 0.0
        table = self.ctx.travel[b.id]
        margin = self.problem.margin_for(b)
        if margin.delta_min_s <= 0 or margin.coef <= 0:
            return 0.0
        pen = 0.0
        for j, node in enumerate(b.nodes if hasattr(b, "nodes") else table.nodes):
            slot = band.windows.get(node)
            if slot is None:
                continue
            plan = selection[node]
            witness = self._witness.get(plan.id, {})
            # 对外 windows 统一是 1-based，内部 helper 需要 0-based。
            slot_inner = slot - 1
            s = self._window_endpoint_r(plan, b.direction, slot_inner, "start", witness) * self.C + offsets[node]
            e = self._window_endpoint_r(plan, b.direction, slot_inner, "end", witness) * self.C + offsets[node]
            band_lo = band.start_s + table.cum_T_s[j]
            band_hi = band_lo + band.width_s
            # choose periodic copy containing the band
            n = math.floor((band_lo - s) / self.C)
            for nn in (n, n + 1):
                ss = s + nn * self.C
                ee = e + nn * self.C
                if band_lo >= ss - 1e-7 and band_hi <= ee + 1e-7:
                    pen += margin.coef * max(0.0, margin.delta_min_s - (band_lo - ss))
                    pen += margin.coef * max(0.0, margin.delta_min_s - (ee - band_hi))
                    break
        return pen

    def _band_edge_penalty(self, band: ConstructedBand, b: BandDemandSpec,
                           selection, offsets) -> float:
        """绿波边界亲和（贴合损失）：带端离本节点绿窗对应端点超出 free_s 的部分。"""
        if b.type != "green":
            return 0.0
        gb = self.problem.green_boundary_for(b)
        if not gb.enabled:
            return 0.0
        start = gb.start if (gb.start is not None and gb.start.coef > 0) else None
        end = gb.end if (gb.end is not None and gb.end.coef > 0) else None
        if start is None and end is None:
            return 0.0
        table = self.ctx.travel[b.id]
        pen = 0.0
        for j, node in enumerate(b.nodes if hasattr(b, "nodes") else table.nodes):
            slot = band.windows.get(node)
            if slot is None:
                continue
            plan = selection[node]
            witness = self._witness.get(plan.id, {})
            slot_inner = slot - 1
            s = self._window_endpoint_r(plan, b.direction, slot_inner, "start", witness) * self.C + offsets[node]
            e = self._window_endpoint_r(plan, b.direction, slot_inner, "end", witness) * self.C + offsets[node]
            band_lo = band.start_s + table.cum_T_s[j]
            band_hi = band_lo + band.width_s
            n = math.floor((band_lo - s) / self.C)
            for nn in (n, n + 1):
                ss = s + nn * self.C
                ee = e + nn * self.C
                if band_lo >= ss - 1e-7 and band_hi <= ee + 1e-7:
                    if start is not None:
                        pen += start.coef * max(0.0, (band_lo - ss) - start.free_s)
                    if end is not None:
                        pen += end.coef * max(0.0, (ee - band_hi) - end.free_s)
                    break
        return pen

    def _global_value(self, c, selection, offsets, bands) -> float:
        total = 0.0
        for t in c.terms:
            a = t.atom
            if a.kind == "offset":
                total += t.coef * offsets.get(a.intersection, 0.0)
            elif a.kind == "const":
                total += t.coef * float(a.const or 0.0)
            elif a.kind == "bandwidth":
                total += t.coef * sum(bd.width_s for bd in bands if bd.demand == a.demand)
            else:
                total += 0.0
        return total

    def _global_violation(self, c, selection, offsets, bands) -> float:
        lhs = self._global_value(c, selection, offsets, bands)
        rhs = c.rhs_for_cycle(self.C)
        if c.sense == "<=":
            return max(0.0, lhs - rhs)
        if c.sense == ">=":
            return max(0.0, rhs - lhs)
        return abs(lhs - rhs)

    def _evaluate(self, selection, offsets, bands) -> tuple[float, float, float]:
        """计算一个贪心方案的 composite、路口损失和总带宽。"""
        reward = 0.0
        for b in self.problem.objective.band_demands:
            for band in bands:
                if band.demand == b.id:
                    reward += b.weight * band.width_s
        margin_pen = 0.0
        edge_pen = 0.0
        loss = 0.0
        composite_soft = 0.0
        for b in self.problem.objective.band_demands:
            for band in bands:
                if band.demand == b.id:
                    margin_pen += self._band_margin_penalty(band, b, selection, offsets)
                    edge_pen += self._band_edge_penalty(band, b, selection, offsets)
        for iid, plan in selection.items():
            witness = self._witness.get(plan.id, {})
            for c in plan.soft_constraints:
                cost = self._soft_violation_plan(plan, c, witness)
                coef = c.soft.coef if c.soft is not None else 0.0
                owner = c.soft.owner if c.soft is not None else "intersection_loss"
                if owner == "intersection_loss":
                    loss += cost
                else:
                    composite_soft += cost
        # global hard/soft constraints
        for c in self.problem.global_constraints:
            viol = self._global_violation(c, selection, offsets, bands)
            if c.hard:
                if viol > 1e-7:
                    return -1e18, loss, reward
            else:
                assert c.soft is not None
                cost = c.soft.coef * viol
                if c.soft.owner == "intersection_loss":
                    loss += cost
                else:
                    composite_soft += cost

        # 均衡组：按实际构造出的带，计算归一化后的组内最小带宽。
        agg = {b.id: sum(band.width_s for band in bands if band.demand == b.id)
               for b in self.problem.objective.band_demands}
        balanced_reward = 0.0
        for g in self.problem.objective.balanced_groups:
            present = []
            missing = 0
            for did in g.demands:
                bw = agg.get(did, 0.0)
                if bw > 0:
                    present.append(bw / g.target_for(did))
                else:
                    missing += 1
            if g.on_infeasible == "zero" and missing > 0:
                z_val = 0.0
            else:
                z_val = min(present) if present else 0.0
            balanced_reward += g.weight * z_val
            if g.on_infeasible == "skip" and g.min_existing is not None and g.min_existing_penalty_s > 0:
                actual = len(g.demands) - missing
                deficit = max(0, g.min_existing - actual)
                balanced_reward -= g.min_existing_penalty_s * deficit

        composite = reward - margin_pen - edge_pen - composite_soft + balanced_reward
        return composite, loss, reward

    def _make_solution(self, selection, offsets, bands) -> GreedySolution:
        comp, loss, _ = self._evaluate(selection, offsets, bands)
        adjustments: dict[tuple[str, Direction, int], float] = {}
        for iid, plan in selection.items():
            witness = self._witness.get(plan.id, {})
            for d in ("up", "down"):
                for j in range(plan.boundary_count(d)):
                    if plan.boundary_is_adjustable(d, j):
                        v = witness.get((d, j), 0.0)
                        if abs(v) > 1e-12:
                            adjustments[(plan.id, d, j)] = float(v)
        return GreedySolution(
            plan_selection={k: v.id for k, v in selection.items()},
            offsets_s={k: float(v) for k, v in offsets.items()},
            adjustments_r=adjustments,
            bands=bands,
            composite_est=float(comp),
            loss_est=float(loss),
        )

    def _refine(self, sol: GreedySolution, rounds: int) -> GreedySolution:
        current = sol
        for _ in range(max(0, rounds)):
            selection = {iid: self.problem.plan(pid) for iid, pid in current.plan_selection.items()}
            offsets = dict(current.offsets_s)
            improved = False
            for iid in self.problem.corridor.intersections:
                best_plan = selection[iid]
                best_comp, _, _ = self._evaluate(selection, offsets,
                                                 self._construct_bands(selection, offsets))
                for plan in self.ctx.feasible_plans.get(iid, ()):
                    trial = dict(selection)
                    trial[iid] = plan
                    bands = self._construct_bands(trial, offsets)
                    comp, _, _ = self._evaluate(trial, offsets, bands)
                    if comp > best_comp + 1e-9:
                        best_comp = comp
                        best_plan = plan
                if best_plan.id != selection[iid].id:
                    selection[iid] = best_plan
                    improved = True
            offsets = self._compute_offsets(selection)
            bands = self._construct_bands(selection, offsets)
            current = self._make_solution(selection, offsets, bands)
            if not improved:
                break
        return current


__all__ = ["ConstructedBand", "GreedySolution", "GreedyConstructor"]
