"""语义寻址：把用户 WindowRef/Target 解析成具体 plan / 有效窗口 / 红窗 rid。"""
from __future__ import annotations

from dataclasses import dataclass

from ..errors import GWSemanticError
from ..model.red_math import RedInterval
from ..schema.models import Direction, PlanSpec, ProblemInput
from .ledger import describe_red_ledger, red_ledger
from .models import Target, WindowRef


@dataclass(frozen=True)
class GreenSlotRef:
    plan: PlanSpec
    direction: Direction
    original_index: int
    effective_index: int


@dataclass(frozen=True)
class RedIntervalRef:
    plan: PlanSpec
    direction: Direction
    rid: int
    interval: RedInterval


def _fail(rule_id: str, target_desc: str, reason: str, candidates=()) -> GWSemanticError:
    return GWSemanticError(rule_id, target_desc, reason, tuple(candidates))


def _target_desc(target: Target) -> str:
    w = target.window
    if w.kind == "green":
        where = f"index={w.index}" if w.index is not None else f"position={w.position}"
    else:
        where = f"index={w.index}" if w.index is not None else f"after_green={w.after_green}"
    return f"{target.intersection}/{target.plan}/{target.direction}/{w.kind}({where})"


def _plan_for_target(problem: ProblemInput, target: Target, rule_id: str) -> PlanSpec:
    if target.intersection not in problem.corridor.intersections:
        raise _fail(rule_id, _target_desc(target),
                    f"unknown intersection {target.intersection!r}")
    try:
        plan = problem.plan(target.plan)
    except KeyError:
        raise _fail(rule_id, _target_desc(target),
                    f"unknown plan {target.plan!r}",
                    [p.id for p in problem.plans_of(target.intersection)])
    if plan.intersection != target.intersection:
        raise _fail(rule_id, _target_desc(target),
                    f"plan {plan.id!r} belongs to {plan.intersection!r}, "
                    f"not {target.intersection!r}")
    if not plan.effective_windows(target.direction):
        raise _fail(rule_id, _target_desc(target),
                    f"plan {plan.id!r} has no {target.direction} green windows")
    return plan


def resolve_green_slot(problem: ProblemInput, target: Target, rule_id: str) -> GreenSlotRef:
    plan = _plan_for_target(problem, target, rule_id)
    ref = target.window
    if ref.kind != "green":
        raise _fail(rule_id, _target_desc(target), "expected a green window reference")
    entries = plan._effective_entries(target.direction)
    originals = [(i, e) for i, e in enumerate(entries) if not e[2]]
    if ref.index is not None:
        match = [(eff_i, e) for eff_i, e in originals if e[1] == ref.index]
        if not match:
            raise _fail(rule_id, _target_desc(target),
                        f"declared green window index {ref.index} does not exist",
                        [f"declared index {e[1]} -> effective index {eff_i}"
                         for eff_i, e in originals])
        eff_i, entry = match[0]
        return GreenSlotRef(plan, target.direction, entry[1], eff_i)
    # position first/last by start_r among original windows
    sorted_originals = sorted(originals, key=lambda pair: pair[1][0].start_r)
    if not sorted_originals:
        raise _fail(rule_id, _target_desc(target), "plan has no non-copy green windows")
    eff_i, entry = sorted_originals[0 if ref.position == "first" else -1]
    return GreenSlotRef(plan, target.direction, entry[1], eff_i)


def resolve_red_interval(problem: ProblemInput, target: Target, rule_id: str) -> RedIntervalRef:
    plan = _plan_for_target(problem, target, rule_id)
    ref = target.window
    if ref.kind != "red":
        raise _fail(rule_id, _target_desc(target), "expected a red window reference")
    # v1: 红窗寻址只允许单方案路口，避免运行时 rid 漂移。
    same_intersection = problem.plans_of(target.intersection)
    if len(same_intersection) != 1:
        raise _fail(rule_id, _target_desc(target),
                    "red-window addressing is only supported for a single-plan "
                    "intersection in v1",
                    [p.id for p in same_intersection])
    ledger = red_ledger(plan, target.direction)
    if not ledger:
        raise _fail(rule_id, _target_desc(target),
                    f"plan {plan.id!r} has no red intervals")
    by_rid = {item.rid: item for item in ledger}
    if ref.index is not None:
        rid = ref.index
    else:
        k = int(ref.after_green or 0)
        k_green = len(plan.effective_windows(target.direction))
        if k < 1 or k > k_green:
            raise _fail(rule_id, _target_desc(target),
                        f"after_green={k} out of range 1..{k_green}",
                        describe_red_ledger(plan, target.direction))
        # k < K: gap after green k = rid k; k == K: wrap after green K = rid 2K-1
        rid = k if k < k_green else (2 * k_green - 1)
    item = by_rid.get(rid)
    if item is None:
        raise _fail(rule_id, _target_desc(target),
                    f"red interval rid={rid} does not exist",
                    describe_red_ledger(plan, target.direction))
    if item.degenerate:
        raise _fail(rule_id, _target_desc(target),
                    f"red interval rid={rid} is degenerate/zero-length",
                    describe_red_ledger(plan, target.direction))
    return RedIntervalRef(plan, target.direction, rid, item)


__all__ = ["GreenSlotRef", "RedIntervalRef", "resolve_green_slot", "resolve_red_interval"]
