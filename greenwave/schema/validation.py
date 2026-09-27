"""输入校验：在建模之前先把明显不合法的数据拦下来。

给高中生看的小课堂
------------------
写程序时最怕“垃圾进、垃圾出”。
所以先检查用户输入是否符合常识：
- 绿灯窗口的起点必须小于终点；
- 同一个方向的绿灯窗口不能重叠；
- 红波带必须刚好两个相邻路口；
- 上下行节点顺序必须按行驶方向排列；
- 距离、速度、周期必须是正数。

这些检查与“能不能求解”无关，只检查数据本身是否合法。
真正跟周期相关的可行性，在后面 preprocessing 层用线性规划判断。
"""
from __future__ import annotations

from typing import Iterable

from ..errors import ValidationIssue
from .models import (AtomRef, BandDemandSpec, ConstraintSpec, ConstraintTemplate,
                     Direction, PlanSpec, ProblemInput)


def _ordered_monotonic(nodes: tuple[str, ...], corridor, d: Direction) -> bool:
    """检查节点是否按行驶方向单调排列。

    上行：物理序号越来越大；
    下行：物理序号越来越小。
    """
    idx = [corridor.index(n) for n in nodes]
    if d == "up":
        return all(idx[i] < idx[i + 1] for i in range(len(idx) - 1))
    return all(idx[i] > idx[i + 1] for i in range(len(idx) - 1))


def _consecutive(nodes: tuple[str, ...], corridor, d: Direction) -> bool:
    """检查节点在物理上是否连续，中间不能跳过路口。"""
    idx = [corridor.index(n) for n in nodes]
    if d == "up":
        return all(idx[i + 1] == idx[i] + 1 for i in range(len(idx) - 1))
    return all(idx[i + 1] == idx[i] - 1 for i in range(len(idx) - 1))


def _validate_constraint_static(c: ConstraintSpec, *, is_global: bool,
                                owner_label: str, issues: list[ValidationIssue]) -> None:
    if c.hard and c.soft is not None:
        issues.append(ValidationIssue("error", "V6",
                                      f"{owner_label} hard constraint {c.id!r} must not declare soft",
                                      (c.id,)))
    if not c.hard and c.soft is None:
        issues.append(ValidationIssue("error", "V6",
                                      f"{owner_label} soft constraint {c.id!r} requires soft penalty",
                                      (c.id,)))
    if c.soft is not None and c.soft.owner not in ("composite", "intersection_loss"):
        issues.append(ValidationIssue("error", "V6",
                                      f"{owner_label} constraint {c.id!r} has invalid owner",
                                      (c.id,)))
    for t in c.terms:
        a = t.atom
        if is_global and a.kind in ("endpoint", "window_duration", "red_duration"):
            issues.append(ValidationIssue("error", "V7",
                                          f"global constraint {c.id!r} must not reference scoped atoms "
                                          f"(endpoint/window_duration/red_duration)",
                                          (c.id,)))
        if a.kind == "endpoint" and a.intersection is None:
            issues.append(ValidationIssue("error", "V7", f"endpoint atom in {c.id!r} missing intersection", (c.id,)))


def validate_problem(problem: ProblemInput) -> list[ValidationIssue]:
    """执行 V1–V10 以及实现中额外补充的静态校验。

    返回值是一个 issue 列表：
    - level="error" 表示必须停止；
    - level="warning" 表示可以继续但建议检查。
    """
    issues: list[ValidationIssue] = []
    corr = problem.corridor
    N = len(corr.intersections)
    # 干线至少要两个路口，否则没有路段，也形成不了带。
    if N < 2:
        issues.append(ValidationIssue("error", "V4", "corridor must contain at least two intersections"))
    if len(corr.segments) != max(0, N - 1):
        issues.append(ValidationIssue("error", "V8", "number of segments must equal N-1"))
    # V8：路段距离和速度必须是正数。
    # 距离支持“公共距离 + 分方向覆盖”，所以三个字段都要检查。
    for i, s in enumerate(corr.segments):
        if s.distance_m is not None and s.distance_m <= 0:
            issues.append(ValidationIssue("error", "V8",
                                          f"segment {i} has non-positive distance_m", (s.from_, s.to)))
        if s.distance_up_m is not None and s.distance_up_m <= 0:
            issues.append(ValidationIssue("error", "V8",
                                          f"segment {i} has non-positive distance_up_m", (s.from_, s.to)))
        if s.distance_down_m is not None and s.distance_down_m <= 0:
            issues.append(ValidationIssue("error", "V8",
                                          f"segment {i} has non-positive distance_down_m", (s.from_, s.to)))
        if s.speed_up_mps <= 0 or s.speed_down_mps <= 0:
            issues.append(ValidationIssue("error", "V8", f"segment {i} has non-positive speed", (s.from_, s.to)))
        for d in ("up", "down"):
            try:
                if s.distance_for(d) <= 0:
                    issues.append(ValidationIssue("error", "V8",
                                                  f"segment {i} effective {d} distance must be positive",
                                                  (s.from_, s.to)))
            except ValueError:
                issues.append(ValidationIssue("error", "V8",
                                              f"segment {i} has no effective distance for {d} direction",
                                              (s.from_, s.to)))
    if not problem.grid.cycles_s or any(c <= 0 for c in problem.grid.cycles_s):
        issues.append(ValidationIssue("error", "V8", "cycles_s must be non-empty and positive"))
    if not problem.grid.speed_ratios or any(k <= 0 for k in problem.grid.speed_ratios):
        issues.append(ValidationIssue("error", "V8", "speed_ratios must be non-empty and positive"))

    # V9: intersections/plans consistency
    plan_ids = [p.id for p in problem.plans]
    if len(set(plan_ids)) != len(plan_ids):
        issues.append(ValidationIssue("error", "V9", "plan ids must be unique"))
    for iid in corr.intersections:
        if not problem.plans_of(iid):
            issues.append(ValidationIssue("error", "V9", f"intersection {iid!r} has no plans", (iid,)))
    valid_intersections = set(corr.intersections)
    for p in problem.plans:
        if p.intersection not in valid_intersections:
            issues.append(ValidationIssue("error", "V9",
                                          f"plan {p.id!r} references unknown intersection {p.intersection!r}",
                                          (p.id, p.intersection)))

    # V1/V2/V3：窗口比例、窗口顺序、可调范围。
    for p in problem.plans:
        for d in ("up", "down"):
            originals = p.windows.get(d, ())
            for wi, w in enumerate(originals):
                if not (0.0 <= w.start_r < w.end_r <= 2.0):
                    issues.append(ValidationIssue("error", "V1",
                                                  f"plan {p.id!r} {d} window {wi} violates 0<=s<e<=2",
                                                  (p.id, d, str(wi))))
            # effective windows after double-cycle copying must be clean
            eff = p.effective_windows(d)
            for wi, w in enumerate(eff):
                if not (0.0 <= w.start_r < w.end_r <= 2.0):
                    issues.append(ValidationIssue("error", "V1",
                                                  f"plan {p.id!r} effective {d} window {wi} violates 0<=s<e<=2",
                                                  (p.id, d, str(wi))))
            for a, b in zip(eff, eff[1:]):
                if a.end_r > b.start_r + 1e-12:
                    issues.append(ValidationIssue("error", "V2",
                                                  f"plan {p.id!r} {d} effective windows overlap/out of order",
                                                  (p.id, d)))
            spec = p.adjustable.get(d, {})
            for k, adj in spec.items():
                if k < 0 or k >= len(originals):
                    issues.append(ValidationIssue("error", "V3",
                                                  f"plan {p.id!r} {d} adjusts unknown original window {k}",
                                                  (p.id, d, str(k))))
                    continue
                for side_name, r in (("start", adj.start), ("end", adj.end)):
                    if r is not None and not (r.lo_r <= 0.0 <= r.hi_r):
                        issues.append(ValidationIssue("error", "V3",
                                                      f"plan {p.id!r} {d} window {k} {side_name} range must satisfy lo<=0<=hi",
                                                      (p.id, d, str(k), side_name)))
        for c in list(p.hard_constraints) + list(p.soft_constraints):
            _validate_constraint_static(c, is_global=False, owner_label=f"plan {p.id!r}", issues=issues)
            for t in c.terms:
                a = t.atom
                if a.kind == "endpoint":
                    if a.plan != p.id:
                        issues.append(ValidationIssue("error", "V6",
                                                      f"plan constraint {c.id!r} references another plan",
                                                      (c.id,)))
                    elif a.direction is None or a.window is None or a.side is None:
                        pass
                    elif a.window < 0 or a.window >= p.window_count(a.direction):
                        issues.append(ValidationIssue("error", "V6",
                                                      f"plan constraint {c.id!r} references unknown effective window {a.window}",
                                                      (c.id,)))
                if a.kind == "window_duration":
                    if a.plan is not None and a.plan != p.id:
                        issues.append(ValidationIssue("error", "V6",
                                                      f"plan constraint {c.id!r} references another plan",
                                                      (c.id,)))
                    if a.direction is None or a.window is None:
                        issues.append(ValidationIssue("error", "V6",
                                                      f"window_duration atom in {c.id!r} must bind direction/window",
                                                      (c.id,)))
                    elif a.window < 0 or a.window >= p.window_count(a.direction):
                        issues.append(ValidationIssue("error", "V6",
                                                      f"plan constraint {c.id!r} references unknown effective window {a.window}",
                                                      (c.id,)))
                if a.kind == "red_duration":
                    issues.append(ValidationIssue("error", "V6",
                                                  f"plan constraint {c.id!r} must not reference red_duration; "
                                                  f"use constraint_templates instead",
                                                  (c.id,)))
                if a.kind == "bandwidth":
                    issues.append(ValidationIssue("error", "V6",
                                                  f"plan constraint {c.id!r} must not reference bandwidth",
                                                  (c.id,)))
                if a.kind == "offset" and a.intersection not in valid_intersections:
                    issues.append(ValidationIssue("error", "V6",
                                                  f"plan constraint {c.id!r} references unknown intersection",
                                                  (c.id,)))

    # V4/V5/V10：带需求的节点、类型、顺序、margin。
    demand_ids = [b.id for b in problem.objective.band_demands]
    if len(set(demand_ids)) != len(demand_ids):
        issues.append(ValidationIssue("error", "V4", "band demand ids must be unique"))
    for b in problem.objective.band_demands:
        # V4
        if b.type == "red" and len(b.nodes) != 2:
            issues.append(ValidationIssue("error", "V4", f"red demand {b.id!r} must have exactly 2 nodes", (b.id,)))
        if b.type == "green" and len(b.nodes) < 2:
            issues.append(ValidationIssue("error", "V4", f"green demand {b.id!r} must have at least 2 nodes", (b.id,)))
        unknown = [n for n in b.nodes if n not in valid_intersections]
        if unknown:
            issues.append(ValidationIssue("error", "V4", f"demand {b.id!r} references unknown nodes {unknown}", (b.id,)))
            continue
        if not _ordered_monotonic(b.nodes, corr, b.direction):
            issues.append(ValidationIssue("error", "V4",
                                          f"demand {b.id!r} nodes are not ordered along direction {b.direction}",
                                          (b.id,)))
        if not _consecutive(b.nodes, corr, b.direction):
            issues.append(ValidationIssue("error", "V5", f"demand {b.id!r} nodes are not physically consecutive", (b.id,)))
        if b.max_bands is not None and b.max_bands < 1:
            issues.append(ValidationIssue("error", "V4", f"demand {b.id!r} max_bands must be >=1", (b.id,)))
        if b.weight < 0:
            issues.append(ValidationIssue("error", "V4", f"demand {b.id!r} weight must be >=0", (b.id,)))
        # V10 warning
        if b.type == "green":
            margin = problem.margin_for(b)
            if margin.delta_min_s < 0:
                issues.append(ValidationIssue("error", "V10", f"demand {b.id!r} delta_min_s must be >=0", (b.id,)))
            if margin.hard_min_s < 0:
                issues.append(ValidationIssue("error", "V10", f"demand {b.id!r} hard_min_s must be >=0", (b.id,)))
            if margin.hard_min_s > margin.delta_min_s:
                issues.append(ValidationIssue("warning", "V10",
                                              f"demand {b.id!r} hard_min_s > delta_min_s; "
                                              f"soft margin target is weaker than hard limit",
                                              (b.id,)))
            else:
                widths = []
                for node in b.nodes:
                    for p in problem.plans_of(node):
                        for w in p.effective_windows(b.direction):
                            widths.append(w.width_r)
                if widths and margin.delta_min_s >= min(widths):
                    issues.append(ValidationIssue("warning", "V10",
                                                  f"demand {b.id!r} delta_min_s >= min window width; margin target may be unreachable",
                                                  (b.id,)))

    # 均衡组校验：组内需求必须存在、不能重复、权重和缺失罚系数非负。
    group_ids = [g.id for g in problem.objective.balanced_groups]
    if len(set(group_ids)) != len(group_ids):
        issues.append(ValidationIssue("error", "V4", "balanced group ids must be unique"))
    for g in problem.objective.balanced_groups:
        if not g.demands:
            issues.append(ValidationIssue("error", "V4",
                                          f"balanced group {g.id!r} must contain at least one demand",
                                          (g.id,)))
        if len(set(g.demands)) != len(g.demands):
            issues.append(ValidationIssue("error", "V4",
                                          f"balanced group {g.id!r} contains duplicate demands",
                                          (g.id,)))
        for did in g.demands:
            if problem.demand_or_none(did) is None:
                issues.append(ValidationIssue("error", "V4",
                                              f"balanced group {g.id!r} references unknown demand {did!r}",
                                              (g.id, did)))
        if g.weight < 0:
            issues.append(ValidationIssue("error", "V8",
                                          f"balanced group {g.id!r} weight must be >= 0",
                                          (g.id,)))
        if g.on_infeasible not in ("zero", "skip"):
            issues.append(ValidationIssue("error", "V4",
                                          f"balanced group {g.id!r} has invalid on_infeasible",
                                          (g.id,)))
        for did, target in g.targets_s.items():
            if did not in g.demands:
                issues.append(ValidationIssue("error", "V4",
                                              f"balanced group {g.id!r} targets_s has unknown demand {did!r}",
                                              (g.id, did)))
            if target <= 0:
                issues.append(ValidationIssue("error", "V8",
                                              f"balanced group {g.id!r} target_s for {did!r} must be positive",
                                              (g.id, did)))
        if g.min_existing is not None and not (0 <= g.min_existing <= len(g.demands)):
            issues.append(ValidationIssue("error", "V4",
                                          f"balanced group {g.id!r} min_existing must be in [0, len(demands)]",
                                          (g.id,)))
        if g.min_existing_penalty_s < 0:
            issues.append(ValidationIssue("error", "V8",
                                          f"balanced group {g.id!r} min_existing_penalty_s must be >= 0",
                                          (g.id,)))

    # V7：全局约束不能引用方案终点（endpoint）。
    for c in problem.global_constraints:
        _validate_constraint_static(c, is_global=True, owner_label="global", issues=issues)
        for t in c.terms:
            if t.atom.kind == "bandwidth" and problem.demand_or_none(t.atom.demand or "") is None:
                issues.append(ValidationIssue("error", "V7",
                                              f"global constraint {c.id!r} references unknown demand {t.atom.demand!r}",
                                              (c.id,)))
            if t.atom.kind == "offset" and t.atom.intersection not in valid_intersections:
                issues.append(ValidationIssue("error", "V7",
                                              f"global constraint {c.id!r} references unknown intersection {t.atom.intersection!r}",
                                              (c.id,)))

    # V11：作用域约束模板。
    template_ids = [t.id for t in problem.constraint_templates]
    if len(set(template_ids)) != len(template_ids):
        issues.append(ValidationIssue("error", "V11", "constraint template ids must be unique"))
    valid_plan_ids = {p.id for p in problem.plans}
    for tmpl in problem.constraint_templates:
        if tmpl.hard and tmpl.soft is not None:
            issues.append(ValidationIssue("error", "V6",
                                          f"constraint template {tmpl.id!r} hard must not declare soft",
                                          (tmpl.id,)))
        if not tmpl.hard and tmpl.soft is None:
            issues.append(ValidationIssue("error", "V6",
                                          f"constraint template {tmpl.id!r} requires soft penalty",
                                          (tmpl.id,)))
        if tmpl.soft is not None and tmpl.soft.owner not in ("composite", "intersection_loss"):
            issues.append(ValidationIssue("error", "V6",
                                          f"constraint template {tmpl.id!r} has invalid owner",
                                          (tmpl.id,)))
        scope = tmpl.scope
        if scope.plan_ids is not None:
            unknown = [pid for pid in scope.plan_ids if pid not in valid_plan_ids]
            if unknown:
                issues.append(ValidationIssue("error", "V11",
                                              f"constraint template {tmpl.id!r} references unknown plans {unknown}",
                                              (tmpl.id,)))
        for t in tmpl.terms:
            a = t.atom
            if a.kind not in ("window_duration", "red_duration", "const"):
                issues.append(ValidationIssue("error", "V11",
                                              f"constraint template {tmpl.id!r} only supports "
                                              f"window_duration/red_duration/const atoms at the moment",
                                              (tmpl.id,)))
            if a.kind in ("window_duration", "red_duration") and (
                    a.plan is not None or a.direction is not None or a.window is not None):
                issues.append(ValidationIssue("error", "V11",
                                              f"constraint template {tmpl.id!r} {a.kind} "
                                              f"must be unbound; use scope to select plan/direction/window",
                                              (tmpl.id,)))
    return issues


__all__ = ["validate_problem"]
