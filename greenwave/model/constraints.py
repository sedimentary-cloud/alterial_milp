"""约束仓库：保存所有“必须满足的条件”。

给高中生看的小课堂
------------------
一个线性约束长这样：

    expr <= rhs
    expr >= rhs
    expr == rhs

其中 expr 是线性表达式，rhs 是右边的常数（秒或比例换算后的秒）。

本仓库还给每条约束标注：
- 组号，例如 C1、C5、C9；
- 语义标签，例如 “band lower”；
- 相关对象 id，便于出错时定位。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..errors import GWModelError
from .expressions import LinearExpr


# 约束的“标签”：出问题时能知道是哪条约束、属于哪一组。
@dataclass(frozen=True)
class ConstraintMeta:
    group: str
    label: str
    refs: dict[str, str] = field(default_factory=dict)


@dataclass
class ConstraintRow:
    expr: LinearExpr
    sense: str
    rhs: float
    meta: ConstraintMeta


class ConstraintStore:
    """所有约束行的集合，可以转换成求解器要的稀疏矩阵。"""

    def __init__(self) -> None:
        self.rows: list[ConstraintRow] = []

    def add(self, expr: LinearExpr, sense: str, rhs: float,
            meta: ConstraintMeta) -> int:
        if sense not in ("<=", ">=", "=="):
            raise GWModelError(f"unsupported sense {sense!r}")
        self.rows.append(ConstraintRow(expr.copy(), sense, float(rhs), meta))
        return len(self.rows) - 1

    def __len__(self) -> int:
        return len(self.rows)

    def to_sparse(self, nvars: int):
        """转换成稀疏矩阵，供 scipy / highspy 使用。

        返回：
        - A：约束矩阵，每一行是一条约束；
        - lower / upper：每行允许的下界和上界。

        例如 expr <= rhs 会变成 lower=-inf, upper=rhs；
        expr >= rhs 会变成 lower=rhs, upper=inf；
        expr == rhs 会变成上下界都等于 rhs。
        """
        from scipy.sparse import csr_matrix
        data: list[float] = []
        indices: list[int] = []
        indptr: list[int] = [0]
        lower = np.empty(len(self.rows), dtype=float)
        upper = np.empty(len(self.rows), dtype=float)
        for r, row in enumerate(self.rows):
            for idx, coef in row.expr.items():
                if idx < 0 or idx >= nvars:
                    raise GWModelError(f"constraint row {r} references variable index {idx} out of range")
                data.append(coef)
                indices.append(idx)
            indptr.append(len(data))
            const = row.expr.const
            if row.sense == "<=":
                lower[r] = -np.inf
                upper[r] = row.rhs - const
            elif row.sense == ">=":
                lower[r] = row.rhs - const
                upper[r] = np.inf
            else:
                lower[r] = row.rhs - const
                upper[r] = row.rhs - const
        mat = csr_matrix((np.asarray(data, dtype=float), np.asarray(indices, dtype=int), np.asarray(indptr, dtype=int)),
                         shape=(len(self.rows), nvars))
        return mat, lower, upper

    def groups(self) -> dict[str, list[int]]:
        out: dict[str, list[int]] = {}
        for i, row in enumerate(self.rows):
            out.setdefault(row.meta.group, []).append(i)
        return out


__all__ = ["ConstraintMeta", "ConstraintRow", "ConstraintStore"]
