"""变量注册表：记录 MILP 里所有“未知数”。

给高中生看的小课堂
------------------
优化模型就像解方程组，只不过方程组里有一些“可以自己选的数”，
这些数就叫“变量”。

变量分三类：
- 连续变量：可以取任意小数，比如 3.7、10.25；
- 二进制变量：只能取 0 或 1，常用来表示“选/不选”；
- 整数变量：只能取整数，比如 -2、0、1、5。

本文件不关心用哪个求解器，只负责把变量编号、起名字、记录上下界，
最后交给 scipy 或 highspy 去解。这样建模层和求解器层就不会绑死。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Iterable

from ..errors import GWModelError


# 变量的三种类型：和 scipy 的约定保持一致，方便后端转换。
class GVarType(IntEnum):
    CONTINUOUS = 0   # 连续变量，可以取小数
    BINARY = 1      # 0/1 变量，表示“选或不选”
    INTEGER = 2     # 整数变量，例如“绕了几个周期”


# 变量的“身份证”：index 是它在数组里的位置，name 是给人看的名字。
@dataclass(frozen=True)
class VarHandle:
    index: int
    name: str

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.name


class VariableRegistry:
    """变量登记册：按加入顺序保存所有变量。

    为什么需要它？
    - 建模代码只需要拿到一个 VarHandle；
    - 求解器后端只需要读取 names/lbs/ubs/integrality；
    - 中间不依赖任何 scipy 特有对象，换求解器很容易。
    """

    def __init__(self) -> None:
        self._names: list[str] = []
        self._lbs: list[float] = []
        self._ubs: list[float] = []
        self._types: list[GVarType] = []
        self._metas: list[dict[str, Any]] = []
        self._by_name: dict[str, VarHandle] = {}

    def add(self, name: str, lb: float, ub: float, vtype: GVarType,
            meta: dict[str, Any] | None = None) -> VarHandle:
        # lb / ub 是变量的下界和上界。
        # 例如二进制变量通常是 0 到 1，整数开绕变量可能是 0 到 3。
        if name in self._by_name:
            raise GWModelError(f"duplicate variable name: {name}")
        idx = len(self._names)
        h = VarHandle(idx, name)
        self._names.append(name)
        self._lbs.append(float(lb))
        self._ubs.append(float(ub))
        self._types.append(vtype)
        self._metas.append(dict(meta or {}))
        self._by_name[name] = h
        return h

    def by_name(self, name: str) -> VarHandle:
        # 根据名字找变量，调试和结果解码时常用。
        try:
            return self._by_name[name]
        except KeyError as exc:
            raise GWModelError(f"unknown variable {name!r}") from exc

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._names)

    @property
    def lbs(self) -> tuple[float, ...]:
        return tuple(self._lbs)

    @property
    def ubs(self) -> tuple[float, ...]:
        return tuple(self._ubs)

    @property
    def integrality(self) -> tuple[int, ...]:
        # scipy 的约定：0 表示连续，1 表示整数。
        # 二进制变量本质上是“上下界为 0/1 的整数变量”，所以也标成 1。
        return tuple(1 if t in (GVarType.BINARY, GVarType.INTEGER) else 0 for t in self._types)

    @property
    def types(self) -> tuple[GVarType, ...]:
        return tuple(self._types)

    @property
    def metas(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._metas)

    def __len__(self) -> int:
        return len(self._names)

    def __iter__(self):
        return iter((VarHandle(i, n) for i, n in enumerate(self._names)))


__all__ = ["GVarType", "VarHandle", "VariableRegistry"]
