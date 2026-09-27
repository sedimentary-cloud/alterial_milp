"""纯函数红窗枚举：语义层和建模层共用同一份红窗编号契约。

给定一个已按 start_r 排序、包含双周期副本的有效窗口序列，
红窗补集的编号规则与 RedWindowBuilder 保持一致：

- 先 gap：槽位 r 结束 -> 槽位 r+1 开始，rid = 1..k-1；
- 再 wrap：槽位 r 结束 -> 槽位 1 开始 + 1.0 周期，rid = k..2k-1。

这样用户说的“第几段红灯”在编译期和建模期是同一个对象。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..schema.models import WindowSpec


@dataclass(frozen=True)
class RedInterval:
    rid: int
    kind: str                 # "gap" | "wrap"
    after_slot: int           # 1-based green slot this red interval follows
    nominal_start_r: float
    nominal_end_r: float
    nominal_width_r: float

    @property
    def degenerate(self) -> bool:
        return self.nominal_width_r <= 1e-9


def enumerate_red_intervals(windows: Iterable[WindowSpec]) -> tuple[RedInterval, ...]:
    """Return red intervals in the canonical gap-then-wrap order."""
    wins = tuple(windows)
    k = len(wins)
    if k <= 0:
        return ()
    out: list[RedInterval] = []
    rid = 0
    # gap after each green slot r = 1..k-1
    for r in range(1, k):
        rid += 1
        start = wins[r - 1].end_r
        end = wins[r].start_r
        out.append(RedInterval(rid, "gap", r, start, end, end - start))
    # wrap after each green slot r = 1..k
    for r in range(1, k + 1):
        rid += 1
        start = wins[r - 1].end_r
        end = wins[0].start_r + 1.0
        out.append(RedInterval(rid, "wrap", r, start, end, end - start))
    return tuple(out)


def nominal_red_width_r(windows: Iterable[WindowSpec], rid: int) -> float:
    for item in enumerate_red_intervals(windows):
        if item.rid == rid:
            return item.nominal_width_r
    return 0.0


__all__ = ["RedInterval", "enumerate_red_intervals", "nominal_red_width_r"]
