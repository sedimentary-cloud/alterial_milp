"""红窗台账：把声明方案编译成用户可理解的 red rid 列表。"""
from __future__ import annotations

from ..model.red_math import RedInterval, enumerate_red_intervals
from ..schema.models import Direction, PlanSpec


def red_ledger(plan: PlanSpec, direction: Direction) -> tuple[RedInterval, ...]:
    """Return the canonical red interval ledger for one plan/direction."""
    return enumerate_red_intervals(plan.effective_windows(direction))


def describe_red_ledger(plan: PlanSpec, direction: Direction) -> tuple[str, ...]:
    lines: list[str] = []
    for item in red_ledger(plan, direction):
        deg = " [degenerate]" if item.degenerate else ""
        lines.append(
            f"rid={item.rid} {item.kind} after_green={item.after_slot} "
            f"nominal=[{item.nominal_start_r:.6g}, {item.nominal_end_r:.6g}] "
            f"width={item.nominal_width_r:.6g}{deg}"
        )
    return tuple(lines)


__all__ = ["red_ledger", "describe_red_ledger"]
