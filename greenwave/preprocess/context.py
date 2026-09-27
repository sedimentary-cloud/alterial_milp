"""格点上下文：把某个 (C, κ) 下的常数和过滤结果打包起来。

给高中生看的小课堂
------------------
公共周期 C 和速度倍率 κ 一旦固定，
所有路段的行程时间 τ、累计时间 T、绿灯窗口秒值都变成了常数。

本文件定义：
- 每个需求的累计行程时间表；
- 每个方案在当前周期下是否可行；
- 被剔除的方案及原因；
- 每个路口在当前格点下还剩哪些可行方案。

建模层拿到这个上下文后，就可以直接拼 MILP 了。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from ..schema.models import BandDemandSpec, Direction, PlanSpec, ProblemInput


# 一个方案在某个周期下是否可行，以及不可行原因。
# witness_r 里保存一组具体可行的端点微调量，便于贪心/绘图使用。
@dataclass(frozen=True)
class PlanFeasibility:
    feasible: bool
    violations: tuple[str, ...] = ()
    # witness adjustment values in ratio domain, keyed (direction, effective_window, side)
    witness_r: tuple[tuple[Direction, int, str, float], ...] = ()


# 某条带需求的累计行程时间表。
# nodes 按行驶方向排列；cum_T_s[j] 表示从链首到第 j 个节点的秒数。
@dataclass(frozen=True)
class DemandTravelTable:
    demand_id: str
    nodes: tuple[str, ...]
    cum_T_s: tuple[float, ...]
    span_s: float
    n_max: tuple[int, ...]

    def cumulative(self, intersection: str) -> float:
        return self.cum_T_s[self.nodes.index(intersection)]

    def n_upper(self, intersection: str) -> int:
        return self.n_max[self.nodes.index(intersection)]


# 在某个周期下被预处理剔除的方案记录。
@dataclass(frozen=True)
class DroppedPlanRecord:
    intersection: str
    plan: str
    cycle_s: float
    violations: tuple[str, ...]


# 一个格点 (C, κ) 的全部预处理结果。
# 这是 heuristic、model、report 共用的“唯一事实来源”。
@dataclass(frozen=True)
class GridPointContext:
    cycle_s: float
    kappa: float
    tau_s: Mapping[tuple[str, Direction], float]
    feasible_plans: Mapping[str, tuple[PlanSpec, ...]]
    dropped: tuple[DroppedPlanRecord, ...]
    travel: Mapping[str, DemandTravelTable]
    plan_witness_r: Mapping[str, tuple[tuple[Direction, int, str, float], ...]] = field(default_factory=dict)

    def max_window_slots(self, intersection: str, d: Direction) -> int:
        plans = self.feasible_plans.get(intersection, ())
        return max((p.window_count(d) for p in plans), default=0)

    def demand_max_bands(self, demand: BandDemandSpec) -> int:
        if demand.type == "red":
            if len(demand.nodes) < 2:
                return 0
            if self.max_window_slots(demand.nodes[1], demand.direction) <= 0:
                return 0
            return max(1, demand.max_bands if demand.max_bands is not None else 3)
        chain_max = min((self.max_window_slots(node, demand.direction) for node in demand.nodes), default=0)
        if chain_max <= 0:
            return 0
        if demand.max_bands is None:
            q = chain_max
        else:
            q = min(int(demand.max_bands), chain_max)
        return max(1, q)

    def max_cum_T_s(self) -> float:
        return max((t.span_s for t in self.travel.values()), default=0.0)

    def witness_for(self, plan_id: str) -> dict[tuple[Direction, int, str], float]:
        return {(d, k, side): v for d, k, side, v in self.plan_witness_r.get(plan_id, ())}


__all__ = ["PlanFeasibility", "DemandTravelTable", "DroppedPlanRecord", "GridPointContext"]
