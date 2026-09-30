"""红窗台账：把声明方案编译成 plan-native 的 red rid 列表。

v2 中每个 plan 的红窗完全由自己的 boundaries + green_mask 决定：
- 绿窗编号 1..K；
- rid i (i<K) 是绿窗 i 与 i+1 之间的 gap；
- rid K 是绿窗 K 之后、绿窗 1 之前的循环红区间，可能跨 0/C。
"""
from __future__ import annotations

from ..schema.models import Direction, PlanSpec


def red_ledger(plan: PlanSpec, direction: Direction):
    return plan.derived_red_intervals(direction)


def describe_red_ledger(plan: PlanSpec, direction: Direction) -> tuple[str, ...]:
    lines: list[str] = []
    for item in red_ledger(plan, direction):
        deg = " [degenerate]" if item.width_r <= 1e-9 else ""
        lines.append(
            f"rid={item.rid} {item.kind} after_green={item.after_green} "
            f"nominal=[{item.start_r:.6g}, {item.end_r:.6g}] "
            f"width={item.width_r:.6g}{deg}"
        )
    return tuple(lines)


__all__ = ["red_ledger", "describe_red_ledger"]
