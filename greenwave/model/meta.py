"""模型元数据：记录变量表之外，解码和目标装配需要的信息。

给高中生看的小课堂
------------------
MILP 解出来只是一串数字。
要知道每个数字代表什么，就需要“字典”：
- 哪个变量是哪个路口的选择 x；
- 哪个变量是哪个窗口端点；
- 哪条带用了哪个窗口；
- 哪些软约束需要计入哪个目标。

本文件定义这些字典和容器，供目标函数、结果解码和报告共用。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from ..preprocess.context import GridPointContext
from ..schema.models import BandDemandSpec, Direction, ProblemInput, Side
from .constraints import ConstraintMeta
from .expressions import LinearExpr
from .variables import VarHandle
from .windows import EffectiveWindowSet, RedWindowSet


# big-M 表：把约束“关掉”时需要一个足够大的数 M。
# M 不能乱取 1e6，否则求解器数值不稳定；这里按数据规模估算。
@dataclass(frozen=True)
class BigMTable:
    containment: float
    plan_activation: float
    soft: float

    @classmethod
    def compute(cls, ctx: GridPointContext) -> "BigMTable":
        tmax = ctx.max_cum_T_s()
        c = ctx.cycle_s
        return cls(containment=tmax + 6.0 * c + 10.0,
                   plan_activation=tmax + 6.0 * c + 10.0,
                   soft=tmax + 6.0 * c + 10.0)


# 一条软约束的引用：表达式、方向、右边值、系数、归属。
@dataclass
class SoftPenaltyRef:
    constraint_id: str
    expr: LinearExpr
    sense: str
    rhs: float
    coef: float
    owner: str
    plan_id: str | None = None
    label: str = ""


# 一条绿波需求的 margin 短缺变量集合。
@dataclass
class MarginSlackRef:
    demand_id: str
    coef: float
    s_vars: list[VarHandle] = field(default_factory=list)
    e_vars: list[VarHandle] = field(default_factory=list)


# 一个绿波带需求的全部 MILP 变量。
@dataclass
class GreenDemandModel:
    demand: BandDemandSpec
    qmax: int
    u: dict[int, VarHandle]
    beta: dict[int, VarHandle]
    e: dict[int, VarHandle]
    a: dict[tuple[int, int, int], VarHandle]   # (q, chain_node_index, green_slot_1based)
    table: object


# 一个红波带需求的全部 MILP 变量。
# 红带因式分解为 ag（上游绿窗指派）和 ar（下游红窗实例指派）
@dataclass
class RedDemandModel:
    demand: BandDemandSpec
    qmax: int
    u: dict[int, VarHandle]
    beta: dict[int, VarHandle]
    e: dict[int, VarHandle]
    # ag：带槽 q 使用上游绿窗槽 g
    ag: dict[tuple[int, int], VarHandle]
    # ar：带槽 q 使用下游红窗实例 (r,k)
    ar: dict[tuple[int, int, int], VarHandle]  # (q, r, k)
    green_up: EffectiveWindowSet
    red_down: RedWindowSet
    ag_slots: tuple[int, ...]
    ar_keys: tuple[tuple[int, int], ...]  # (r,k)


# 整个模型的元数据总表。

# 一个均衡组：z 是组内最小带宽，y_b 表示需求 b 是否存在。
@dataclass
class BalancedGroupModel:
    group: object                 # BalancedGroupSpec
    z: VarHandle
    y: dict[str, VarHandle]
    bandwidth_exprs: dict[str, LinearExpr]
    deficit: VarHandle | None = None
    targets_s: dict[str, float] = field(default_factory=dict)

@dataclass
class ModelMeta:
    problem: ProblemInput
    ctx: GridPointContext
    cycle_s: float
    anchor_intersection: str
    plan_vars: dict[str, VarHandle] = field(default_factory=dict)
    # 单方案路口：不需要 0/1 方案选择变量，直接固定方案 id。
    fixed_plans: dict[str, str] = field(default_factory=dict)
    phi: dict[str, VarHandle] = field(default_factory=dict)
    adj_vars: dict[tuple[str, Direction, int, Side], VarHandle] = field(default_factory=dict)
    z_vars: dict[tuple[str, Direction, int, Side], VarHandle] = field(default_factory=dict)
    plan_endpoint: dict[tuple[str, Direction, int, Side], LinearExpr] = field(default_factory=dict)
    effective: dict[tuple[str, Direction], EffectiveWindowSet] = field(default_factory=dict)
    red: dict[tuple[str, Direction], RedWindowSet] = field(default_factory=dict)
    green: dict[str, GreenDemandModel] = field(default_factory=dict)
    red_demands: dict[str, RedDemandModel] = field(default_factory=dict)
    soft: list[SoftPenaltyRef] = field(default_factory=list)
    margins: list[MarginSlackRef] = field(default_factory=list)
    plan_constraints: list[tuple[str, ConstraintMeta]] = field(default_factory=list)
    balanced: dict[str, BalancedGroupModel] = field(default_factory=dict)

    def effective_slot_vars(self, intersection: str, d: Direction):
        return self.effective[(intersection, d)]


__all__ = ["BigMTable", "SoftPenaltyRef", "MarginSlackRef", "GreenDemandModel",
           "RedDemandModel", "BalancedGroupModel", "ModelMeta"]
