"""纯函数红窗枚举：语义层和建模层共用的 plan-native 红窗编号契约。

v2 约定：
- 绿窗不跨周期边界，按 start_r 排序编号 1..K；
- 红窗 = 极大连续非绿区间，也编号 1..K：
  - rid = i (1..K-1)：绿窗 i 与绿窗 i+1 之间的 gap；
  - rid = K       ：绿窗 K 之后、绿窗 1 之前的循环红区间（可能跨 0/C）；
- 跨 0/C 的红区间用 wrap=1 表示，不拆分 ID；
- after_green = k 直接对应 rid = k。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..schema.models import DerivedInterval


@dataclass(frozen=True)
class RedInterval:
    rid: int
    kind: str                 # "gap" | "wrap"
    after_slot: int           # 1-based green window this red interval follows
    start_r: float
    end_r: float
    wrap: int                 # 0/1; 1 means end_r is already expanded (may be > 1)
    nominal_width_r: float

    @property
    def degenerate(self) -> bool:
        return self.nominal_width_r <= 1e-9


def enumerate_red_intervals(green_windows: Iterable[DerivedInterval]) -> tuple[RedInterval, ...]:
    """Return plan-native red intervals for ordered, non-wrapping green windows."""
    green = tuple(green_windows)
    if not green:
        return ()
    out: list[RedInterval] = []
    K = len(green)
    for i in range(K):
        if i < K - 1:
            start_r = green[i].end_r
            end_r = green[i + 1].start_r
            wrap = 0
            kind = "gap"
            after = i + 1
        else:
            start_r = green[-1].end_r
            end_r = green[0].start_r + 1.0
            wrap = 1
            kind = "wrap"
            after = K
        out.append(RedInterval(
            rid=i + 1, kind=kind, after_slot=after,
            start_r=start_r, end_r=end_r, wrap=wrap,
            nominal_width_r=end_r - start_r,
        ))
    return tuple(out)


def nominal_red_width_r(green_windows: Iterable[DerivedInterval], rid: int) -> float:
    for item in enumerate_red_intervals(green_windows):
        if item.rid == rid:
            return item.nominal_width_r
    return 0.0


__all__ = ["RedInterval", "enumerate_red_intervals", "nominal_red_width_r"]
