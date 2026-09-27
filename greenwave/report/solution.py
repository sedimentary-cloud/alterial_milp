"""结果解码与报告：把求解器返回的数字变成人能看懂的方案。

给高中生看的小课堂
------------------
求解器只会给我们一串数字：
    [v0, v1, v2, ...]

我们需要知道：
- v0 = 1 表示选了哪个方案；
- v1 = 45 表示某个路口偏置是 45 秒；
- 哪几个数字组成一条绿波/红波带；
- 带宽是多少；
- 有没有软约束被违反。

本文件负责这种“翻译”工作，并组装成 JSON 报告。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from ..heuristic.greedy import ConstructedBand
from ..model.builder import ArterialModel
from ..model.meta import SoftPenaltyRef


# 一条软约束的违反记录：违反了多少秒、罚了多少、归属谁。
@dataclass(frozen=True)
class ViolationRecord:
    constraint_id: str
    violation_s: float
    cost: float
    owner: str


# 一个完整解：方案、偏置、微调、带、违反、composite、路口损失。
@dataclass(frozen=True)
class SolutionRecord:
    plan_selection: Mapping[str, str]
    offsets_s: Mapping[str, float]
    adjustments_r: Mapping[tuple[str, str, int, str], float]
    bands: tuple[ConstructedBand, ...]
    violations: tuple[ViolationRecord, ...]
    composite_s: float
    loss_s: float
    balanced_groups: Mapping[str, float] = field(default_factory=dict)
    # 下面三个字段用于格点热力图：
    # band_reward_s：Σ weight_b · Β_b，只含正带宽收益；
    # margin_loss_s：绿波 margin 短缺罚的正值；
    # loss_s 已经是路口损失的正值。
    band_reward_s: float = 0.0
    margin_loss_s: float = 0.0


def _violation(ref: SoftPenaltyRef, values: np.ndarray) -> float:
    lhs = ref.expr.evaluate(values)
    if ref.sense == "<=":
        return max(0.0, lhs - ref.rhs)
    if ref.sense == ">=":
        return max(0.0, ref.rhs - lhs)
    return abs(lhs - ref.rhs)


# 解解码器：只依赖模型访问器 + 元数据，不直接碰求解器对象。
class SolutionDecoder:
    def __init__(self, model: ArterialModel) -> None:
        self.model = model

    def decode(self, values: np.ndarray) -> SolutionRecord:
        # 第一步：把 x 变量翻译成每个路口选中的方案。
        values = np.asarray(values, dtype=float)
        meta = self.model.meta
        # plan selection
        selection: dict[str, str] = {}
        for iid in meta.problem.corridor.intersections:
            if iid in meta.fixed_plans:
                # 单方案路口没有 x 变量，直接采用固定方案。
                selection[iid] = meta.fixed_plans[iid]
                continue
            best = None
            for p in meta.ctx.feasible_plans[iid]:
                v = values[meta.plan_vars[p.id].index]
                if best is None or v > best[0]:
                    best = (v, p.id)
            selection[iid] = best[1] if best else meta.ctx.feasible_plans[iid][0].id
        offsets = {iid: float(values[meta.phi[iid].index]) for iid in meta.problem.corridor.intersections}

        # adjustments on selected plans
        adjustments: dict[tuple[str, str, int, str], float] = {}
        for iid, pid in selection.items():
            plan = meta.problem.plan(pid)
            for d in ("up", "down"):
                for k in range(plan.window_count(d)):
                    for side in ("start", "end"):
                        h = meta.adj_vars.get((pid, d, k, side))
                        if h is not None:
                            adjustments[(pid, d, k, side)] = float(values[h.index])

        bands: list[ConstructedBand] = []
        # 第二步：读出每条绿带和红带。
        # green demands
        for b in meta.problem.objective.band_demands:
            if b.type == "green":
                gm = meta.green.get(b.id)
                if gm is None:
                    continue
                table = gm.table
                for q in range(1, gm.qmax + 1):
                    if values[gm.e[q].index] < 0.5:
                        continue
                    width = float(values[gm.beta[q].index])
                    start = float(values[gm.u[q].index])
                    windows: dict[str, int] = {}
                    for j, node in enumerate(b.nodes):
                        best_k, best_v = None, -1.0
                        for k in range(1, meta.ctx.max_window_slots(node, b.direction) + 1):
                            v = values[gm.a[(q, j, k)].index]
                            if v > best_v:
                                best_v, best_k = v, k
                        if best_k is not None:
                            windows[node] = best_k
                    bands.append(ConstructedBand(b.id, q, True, width, start, windows))
            else:
                rm = meta.red_demands.get(b.id)
                if rm is None:
                    continue
                for q in range(1, rm.qmax + 1):
                    if values[rm.e[q].index] < 0.5:
                        continue
                    width = float(values[rm.beta[q].index])
                    start = float(values[rm.u[q].index])
                    # 上游绿窗指派 argmax ag，下游红窗实例 argmax ar。
                    best_g, best_gv = None, -1.0
                    for g in rm.ag_slots:
                        v = values[rm.ag[(q, g)].index]
                        if v > best_gv:
                            best_gv, best_g = v, g
                    best_rk, best_rv = None, -1.0
                    for (r, k) in rm.ar_keys:
                        v = values[rm.ar[(q, r, k)].index]
                        if v > best_rv:
                            best_rv, best_rk = v, (r, k)
                    windows = {}
                    red_instance = None
                    if best_g is not None:
                        windows[b.nodes[0]] = best_g
                    if best_rk is not None:
                        r, k = best_rk
                        windows[b.nodes[1]] = r
                        if best_g is not None:
                            red_instance = (best_g, r, k)
                    bands.append(ConstructedBand(b.id, q, True, width, start, windows,
                                                 red_instance=red_instance))

        # 第三步：重新计算真实的违反量和目标值。
        # 注意：不能直接使用求解器内部的松弛变量值，
        # 因为它可能大于真实违反量。
        violations: list[ViolationRecord] = []
        composite_soft = 0.0
        loss = 0.0
        for ref in meta.soft:
            # Soft constraints belonging to a plan are active only when that
            # plan is selected.  Decoding them unconditionally can report a
            # phantom loss for an unselected alternative plan.
            if ref.plan_id is not None:
                plan_obj = meta.problem.plan(ref.plan_id)
                if selection.get(plan_obj.intersection) != ref.plan_id:
                    continue
            viol = _violation(ref, values)
            cost = ref.coef * viol
            violations.append(ViolationRecord(ref.constraint_id, float(viol), float(cost), ref.owner))
            if ref.owner == "composite":
                composite_soft += cost
            else:
                loss += cost
        reward = 0.0
        for band in bands:
            for b in meta.problem.objective.band_demands:
                if b.id == band.demand:
                    reward += b.weight * band.width_s
        margin_pen = self._actual_margin_penalty(values, bands)

        # 红波边界亲和损失：直接从 slack 变量值读取，目标里已最小化它们。
        red_boundary_pen = 0.0
        for ref in meta.red_boundary:
            for side, vars_, coef in (("start", ref.start_vars, ref.start_coef),
                                      ("end", ref.end_vars, ref.end_coef)):
                if coef <= 0:
                    continue
                for h in vars_:
                    val = float(values[h.index])
                    if val <= 1e-9:
                        continue
                    cost = coef * val
                    red_boundary_pen += cost
                    violations.append(ViolationRecord(
                        f"red_boundary_{side}:{ref.demand_id}:{self.model.vars.names[h.index]}",
                        float(val), float(cost), "composite"))

        # 均衡组：实际 z 值 + min_existing 软约束的短缺罚。
        balanced_values: dict[str, float] = {}
        balanced_reward = 0.0
        for gid, gm in meta.balanced.items():
            z_val = float(values[gm.z.index])
            balanced_values[gid] = z_val
            balanced_reward += float(gm.group.weight) * z_val
            pen = float(getattr(gm.group, "min_existing_penalty_s", 0.0))
            if (getattr(gm.group, "on_infeasible", "zero") == "skip"
                    and gm.deficit is not None and pen > 0):
                balanced_reward -= pen * float(values[gm.deficit.index])

        composite = reward - margin_pen - composite_soft + balanced_reward - red_boundary_pen
        return SolutionRecord(dict(selection), offsets, adjustments, tuple(bands),
                              tuple(violations), float(composite), float(loss),
                              balanced_values, float(reward), float(margin_pen))

    # ------------------------------------------------------------------
    def _actual_margin_penalty(self, values: np.ndarray, bands) -> float:
        meta = self.model.meta
        total = 0.0
        for band in bands:
            b = meta.problem.demand(band.demand)
            if b.type != "green":
                continue
            margin = meta.problem.margin_for(b)
            if margin.delta_min_s <= 0 or margin.coef <= 0:
                continue
            gm = meta.green[b.id]
            table = gm.table
            for j, node in enumerate(b.nodes):
                k = band.windows.get(node)
                if k is None:
                    continue
                ew = meta.effective[(node, b.direction)]
                phi = float(values[meta.phi[node].index])
                s = phi + float(values[ew.s[k].index])
                e = phi + float(values[ew.e[k].index])
                band_lo = band.start_s + table.cum_T_s[j]
                band_hi = band_lo + band.width_s
                n = round((band_lo - s) / self.model.meta.cycle_s)
                for nn in (n - 1, n, n + 1):
                    ss = s + nn * self.model.meta.cycle_s
                    ee = e + nn * self.model.meta.cycle_s
                    if band_lo >= ss - 1e-7 and band_hi <= ee + 1e-7:
                        total += margin.coef * max(0.0, margin.delta_min_s - (band_lo - ss))
                        total += margin.coef * max(0.0, margin.delta_min_s - (ee - band_hi))
                        break
        return total


# 报告组装器：把内部结果拼成对外 JSON。
class ReportBuilder:
    @staticmethod
    def build(grid_report, pareto: list[list]) -> dict:
        grid_results = []
        for r in grid_report.results:
            grid_results.append({
                "cycle_s": r.cycle_s,
                "speed_ratio": r.kappa,
                "status": r.status,
                "composite_best": r.composite_best,
                "heuristic_bound": None if r.heuristic is None else r.heuristic.composite_est,
                "solve_time_s": r.time_s,
                "dropped_plans": [{"intersection": d.intersection, "plan": d.plan,
                                   "reason": "; ".join(d.violations)} for d in r.dropped],
                "message": r.message,
                "bandwidth_objective_s": r.band_reward_s,
                "negative_margin_loss_s": None if r.margin_loss_s is None else -r.margin_loss_s,
                "negative_intersection_loss_s": None if r.intersection_loss_s is None else -r.intersection_loss_s,
                # 归一化版本：除以周期 C，单位变成“周期比例”，可跨周期比较。
                "bandwidth_objective_ratio": (
                    None if r.band_reward_s is None or r.cycle_s <= 0 else r.band_reward_s / r.cycle_s),
                "negative_margin_loss_ratio": (
                    None if r.margin_loss_s is None or r.cycle_s <= 0 else -r.margin_loss_s / r.cycle_s),
                "negative_intersection_loss_ratio": (
                    None if r.intersection_loss_s is None or r.cycle_s <= 0 else -r.intersection_loss_s / r.cycle_s),
            })
        pareto_out: Any
        if len(pareto) == 1:
            base = getattr(pareto[0], "base_grid", None)
            pareto_out = {
                "grid": None if base is None else {"cycle_s": base.cycle_s,
                                                     "speed_ratio": base.kappa},
                "points": [ReportBuilder._point(p) for p in pareto[0]],
            }
        else:
            pareto_out = []
            for group in pareto:
                base = getattr(group, "base_grid", None)
                pareto_out.append({
                    "grid": None if base is None else {"cycle_s": base.cycle_s,
                                                       "speed_ratio": base.kappa},
                    "points": [ReportBuilder._point(p) for p in group],
                })
        return {"grid_results": grid_results, "pareto": pareto_out}

    @staticmethod
    def _point(p) -> dict:
        sol = p.solution
        return {
            "epsilon": p.epsilon_s,
            "composite_actual": p.composite_actual_s,
            "intersection_loss": p.intersection_loss_s,
            "solution": ReportBuilder._solution(sol),
        }

    @staticmethod
    def _solution(sol: SolutionRecord) -> dict:
        return {
            "plan_selection": dict(sol.plan_selection),
            "offsets_s": {k: float(v) for k, v in sol.offsets_s.items()},
            "adjustments": {f"{p}.{d}.{k}.{s}": float(v) for (p, d, k, s), v in sol.adjustments_r.items()},
            "bands": [{
                "demand": b.demand,
                "slot": b.slot,
                "exists": b.exists,
                "width_s": b.width_s,
                "start_at_anchor_s": b.start_s,
                "windows": dict(b.windows),
                "red_instance": None if b.red_instance is None else list(b.red_instance),
            } for b in sol.bands],
            "constraint_violations": [{
                "id": v.constraint_id,
                "violation_s": v.violation_s,
                "cost": v.cost,
                "owner": v.owner,
            } for v in sol.violations if v.violation_s > 1e-7],
            "balanced_groups": {k: float(v) for k, v in sol.balanced_groups.items()},
            "band_reward_s": sol.band_reward_s,
            "margin_loss_s": sol.margin_loss_s,
            "intersection_loss_s": sol.loss_s,
        }


__all__ = ["ViolationRecord", "SolutionRecord", "SolutionDecoder", "ReportBuilder"]
