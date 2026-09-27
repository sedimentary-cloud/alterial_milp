"""求解器后端接口：规定“求解器必须提供哪些能力”。

给高中生看的小课堂
------------------
模型建好以后，真正去“解方程”的是 scipy 或 highspy。
为了不把建模代码和某一个求解器绑死，我们定义一个抽象接口。

以后如果换用别的求解器，只要实现：
- create：把模型交给求解器；
- set_objective：设置最大化/最小化目标；
- add_constraint：追加一条约束；
- solve：开始求解并返回结果。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Protocol

import numpy as np

from ..model.builder import ArterialModel
from ..model.constraints import ConstraintMeta
from ..model.expressions import LinearExpr


# 求解状态：最优、找到可行解、不可行、超时、错误。
class SolveStatus(Enum):
    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    TIME_LIMIT = "time_limit"
    ERROR = "error"


# 求解选项：时间上限、相对 gap、是否预处理。
@dataclass(frozen=True)
class SolveOptions:
    time_limit_s: float
    mip_rel_gap: float
    presolve: bool = True


# 求解结果：状态、目标值、变量值、界、耗时、消息。
@dataclass(frozen=True)
class SolveResult:
    status: SolveStatus
    objective: float | None
    values: np.ndarray | None
    bound: float | None
    time_s: float
    message: str = ""


class ModelSession(Protocol):
    backend_name: str


# 所有求解器后端的“父类”，定义统一接口。
class SolverBackend(ABC):
    name: str

    @abstractmethod
    def create(self, model: ArterialModel) -> ModelSession: ...

    @abstractmethod
    def solve(self, session: ModelSession, options: SolveOptions) -> SolveResult: ...

    @abstractmethod
    def set_objective(self, session: ModelSession, expr: LinearExpr,
                      sense: Literal["max", "min"]) -> None: ...

    @abstractmethod
    def add_constraint(self, session: ModelSession, expr: LinearExpr,
                       sense: str, rhs: float, meta: ConstraintMeta) -> int: ...


__all__ = ["SolveStatus", "SolveOptions", "SolveResult", "ModelSession", "SolverBackend"]
