"""错误类型：把不同的错误分门别类，方便定位问题。

给高中生看的小课堂
------------------
程序出错时，我们希望知道错在哪一类：
- 输入数据写错了？
- 某个周期下方案不可行？
- 建模代码有 bug？
- 求解器本身报错？

不同错误用不同的异常类表示，捕获处理时更清楚。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


# 一条校验结果：级别、编号、说明、相关对象。
@dataclass(frozen=True)
class ValidationIssue:
    """A single static input-validation finding.

    ``level`` is one of ``"error"``, ``"warning"`` or ``"info"``.  ``code``
    carries the V-number from the design document (V1..V10).
    """

    level: str
    code: str
    message: str
    refs: tuple[str, ...] = ()


class GWError(Exception):
    """Base class for all greenwave errors."""


class GWValidationError(GWError):
    """Raised when static input validation finds at least one error."""

    def __init__(self, issues: Iterable[ValidationIssue]):
        self.issues = tuple(issues)
        super().__init__("; ".join(f"{i.code}: {i.message}" for i in self.issues))


class GWInfeasibleGridPoint(GWError):
    """Raised by the preprocessing layer when some intersection has no feasible plan."""

    def __init__(self, cycle_s: float, kappa: float,
                 records: dict[str, tuple[str, ...]] | None = None,
                 dropped: tuple = ()):
        self.cycle_s = cycle_s
        self.kappa = kappa
        self.records = dict(records or {})
        self.dropped = tuple(dropped)
        detail = "; ".join(f"{k}: {', '.join(v)}" for k, v in self.records.items())
        super().__init__(f"grid point C={cycle_s}, kappa={kappa} infeasible: {detail}")


class GWModelError(GWError):
    """Raised for internal modelling errors (a bug or unsupported reference)."""


class GWSemanticError(GWError):
    """Raised when the high-level semantic compiler cannot resolve a rule."""

    def __init__(self, rule_id: str, target_desc: str, reason: str,
                 candidates: tuple[str, ...] = ()):  # pragma: no cover - message helper
        self.rule_id = rule_id
        self.target_desc = target_desc
        self.reason = reason
        self.candidates = tuple(candidates)
        lines = [f"semantic rule {rule_id!r} failed: {reason}"]
        if target_desc:
            lines.append(f"target: {target_desc}")
        if self.candidates:
            lines.append("candidates:")
            lines.extend(f"  {c}" for c in self.candidates)
        super().__init__("\n".join(lines))


class GWSolveError(GWError):
    """Raised when a solver backend reports an unrecoverable error."""


__all__ = [
    "ValidationIssue",
    "GWError",
    "GWValidationError",
    "GWInfeasibleGridPoint",
    "GWModelError",
    "GWSemanticError",
    "GWSolveError",
]
