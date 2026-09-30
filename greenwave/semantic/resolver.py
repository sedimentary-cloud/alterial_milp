"""语义寻址：把用户 WindowRef/Target 解析成具体 plan / 派生窗口 / 红窗 rid。

v2 是 plan-native 的：
- 绿窗由 plan 自己的 boundaries + green_mask 派生，不依赖路口里有多少个 plan；
- 红窗也由同一个 plan 派生，rid 1..K 固定；
- 双周期时语义寻址使用“前半周期代表窗口”，复制窗口结构性共享同一组边界。
"""
from __future__ import annotations

from dataclasses import dataclass

from ..errors import GWSemanticError
from ..schema.models import Direction, PlanSpec, ProblemInput
from .ledger import describe_red_ledger, red_ledger
from .models import Target


@dataclass(frozen=True)
class GreenSlotRef:
    plan: PlanSpec
    direction: Direction
    original_index: int       # declared window index (0-based, half-cycle for double)
    effective_index: int      # index in plan.derived_windows(d)


@dataclass(frozen=True)
class RedIntervalRef:
    plan: PlanSpec
    direction: Direction
    rid: int
    interval: object


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
    if not plan.derived_windows(target.direction):
        raise _fail(rule_id, _target_desc(target),
                    f"plan {plan.id!r} has no {target.direction} green windows")
    return plan


def resolve_green_slot(problem: ProblemInput, target: Target, rule_id: str) -> GreenSlotRef:
    plan = _plan_for_target(problem, target, rule_id)
    ref = target.window
    if ref.kind != "green":
        raise _fail(rule_id, _target_desc(target), "expected a green window reference")
    d = target.direction
    all_windows = plan.derived_windows(d)
    declared = [k for k in range(len(all_windows)) if not plan.is_copy_window(d, k)]
    if not declared:
        raise _fail(rule_id, _target_desc(target), "plan has no declared green windows")
    if ref.index is not None:
        if ref.index < 0 or ref.index >= len(declared):
            raise _fail(rule_id, _target_desc(target),
                        f"declared green window index {ref.index} does not exist",
                        [f"declared index {i} -> effective index {declared[i]}"
                         for i in range(len(declared))])
        declared_index = ref.index
    else:
        declared_index = 0 if ref.position == "first" else len(declared) - 1
    return GreenSlotRef(plan, d, declared_index, declared[declared_index])


def resolve_red_interval(problem: ProblemInput, target: Target, rule_id: str) -> RedIntervalRef:
    plan = _plan_for_target(problem, target, rule_id)
    ref = target.window
    if ref.kind != "red":
        raise _fail(rule_id, _target_desc(target), "expected a red window reference")
    d = target.direction
    ledger = red_ledger(plan, d)
    if not ledger:
        raise _fail(rule_id, _target_desc(target),
                    f"plan {plan.id!r} has no red intervals")
    if ref.index is not None:
        rid = ref.index
        if rid < 1 or rid > len(ledger):
            raise _fail(rule_id, _target_desc(target),
                        f"red interval rid={rid} does not exist",
                        describe_red_ledger(plan, d))
    else:
        k = int(ref.after_green or 0)
        if k < 1 or k > len(ledger):
            raise _fail(rule_id, _target_desc(target),
                        f"after_green={k} out of range 1..{len(ledger)}",
                        describe_red_ledger(plan, d))
        rid = k
    item = ledger[rid - 1]
    if item.width_r <= 1e-9:
        raise _fail(rule_id, _target_desc(target),
                    f"red interval rid={rid} is degenerate/zero-length",
                    describe_red_ledger(plan, d))
    return RedIntervalRef(plan, d, rid, item)


__all__ = ["GreenSlotRef", "RedIntervalRef", "resolve_green_slot", "resolve_red_interval"]
