"""线性表达式：像“3.5 × 变量A − 2 × 变量B + 10”这样的式子。

给高中生看的小课堂
------------------
线性表达式只允许变量乘以一个常数再相加，不允许出现 x²、x·y 这种乘法。
这就是线性规划 / 整数线性规划的基础。

例如：
    3.5 * φ_I1 - 2.0 * β + 10

本类用字典保存“变量编号 -> 系数”，常数单独保存，
这样拼约束、代值计算都很方便。
"""
from __future__ import annotations

from typing import Iterable, Mapping

from ..errors import GWModelError
from .variables import VarHandle


class LinearExpr:
    """稀疏线性表达式：Σ(系数 × 变量) + 常数。

    “稀疏”意思是只保存系数不为 0 的变量，节省内存。
    """

    __slots__ = ("_coefs", "_const")

    def __init__(self, coefs: Mapping[int, float] | None = None, const: float = 0.0):
        self._coefs: dict[int, float] = {}
        if coefs:
            for idx, c in coefs.items():
                if abs(c) > 1e-15:
                    self._coefs[int(idx)] = float(c)
        self._const = float(const)

    # -- factories -------------------------------------------------------------
    @classmethod
    def of(cls, h: VarHandle, coef: float = 1.0) -> "LinearExpr":
        e = cls()
        e.add(h, coef)
        return e

    @classmethod
    def constant(cls, c: float) -> "LinearExpr":
        return cls(const=c)

    # -- manipulation ----------------------------------------------------------
    def add(self, h: VarHandle | "LinearExpr", coef: float = 1.0) -> "LinearExpr":
        # 支持两种加法：
        # 1) 加一个变量：expr.add(x, 3.0)  -> 系数增加 3
        # 2) 加另一个表达式：expr.add(other, 2.0)
        if isinstance(h, LinearExpr):
            for idx, c in h._coefs.items():
                self._coefs[idx] = self._coefs.get(idx, 0.0) + coef * c
                if abs(self._coefs[idx]) <= 1e-15:
                    del self._coefs[idx]
            self._const += coef * h._const
            return self
        idx = h.index
        self._coefs[idx] = self._coefs.get(idx, 0.0) + float(coef)
        if abs(self._coefs[idx]) <= 1e-15:
            del self._coefs[idx]
        return self

    def add_const(self, c: float) -> "LinearExpr":
        self._const += float(c)
        return self

    def sub(self, other: "LinearExpr", coef: float = 1.0) -> "LinearExpr":
        return self.add(other, -coef)

    def copy(self) -> "LinearExpr":
        return LinearExpr(self._coefs, self._const)

    def scaled(self, factor: float) -> "LinearExpr":
        return LinearExpr({k: v * factor for k, v in self._coefs.items()}, self._const * factor)

    def plus(self, other: "LinearExpr") -> "LinearExpr":
        e = self.copy()
        e.add(other, 1.0)
        return e

    # -- inspection ------------------------------------------------------------
    @property
    def const(self) -> float:
        return self._const

    @property
    def coefs(self) -> dict[int, float]:
        return dict(self._coefs)

    def items(self) -> Iterable[tuple[int, float]]:
        return self._coefs.items()

    def is_constant(self) -> bool:
        return not self._coefs

    def evaluate(self, values) -> float:
        # 把求解器返回的一组变量值代进表达式，算出实际数值。
        # 用于检查约束违反量、计算实际目标值。
        total = self._const
        for idx, c in self._coefs.items():
            total += c * float(values[idx])
        return total

    def max_abs_atom_bound(self, atom_ub: float) -> float:
        return sum(abs(c) for c in self._coefs.values()) * atom_ub + abs(self._const)

    def __bool__(self) -> bool:
        return bool(self._coefs) or abs(self._const) > 1e-15

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        parts = [f"{c:+.6g}*v[{i}]" for i, c in sorted(self._coefs.items())]
        if abs(self._const) > 1e-15:
            parts.append(f"{self._const:+.6g}")
        return " ".join(parts) or "0"


__all__ = ["LinearExpr"]
