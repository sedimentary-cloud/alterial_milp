"""错误类型：把不同的错误分门别类，方便定位问题。

给高中生看的小课堂
------------------
程序出错时，我们希望知道错在哪一类：
- 输入数据写错了？
- 某个周期下方案不可行？
- 建模代码有 bug？
- 求解器本身报错？

不同错误用不同的异常类表示，捕获处理时更清楚。每个异常都带：
- ``code``：稳定的错误编号（便于用户搜索/反馈）；
- ``cause``：最可能的原因；
- ``hint``：建议的排查/解决方法。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


# 一条校验结果：级别、编号、说明、相关对象。
@dataclass(frozen=True)
class ValidationIssue:
    """A single static input-validation finding.

    ``level`` is one of ``"error"``, ``"warning"`` or ``"info"``.  ``code``
    carries the V-number from the design document (V1..V14).
    """

    level: str
    code: str
    message: str
    refs: tuple[str, ...] = ()


class GWError(Exception):
    """所有算法包错误的基类。

    用户看到的格式：

        [GW-INPUT] <原始错误信息>
        可能原因：...
        建议：...
    """

    code: str = "GW-ERROR"
    default_cause: str = ""
    default_hint: str = ""

    def __init__(self, message: str, *, code: str | None = None,
                 cause: str | None = None, hint: str | None = None,
                 context: Mapping[str, Any] | None = None) -> None:
        self.message = str(message)
        self.code = code or self.code
        self.cause = cause if cause is not None else self.default_cause
        self.hint = hint if hint is not None else self.default_hint
        self.context = dict(context or {})
        super().__init__(self.message)

    def __str__(self) -> str:
        lines = [f"[{self.code}] {self.message}"]
        if self.cause:
            lines.append(f"可能原因：{self.cause}")
        if self.hint:
            lines.append(f"建议：{self.hint}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """给 CLI/JSON 报告用的结构化错误。"""
        return {
            "code": self.code,
            "message": self.message,
            "cause": self.cause,
            "hint": self.hint,
            "context": dict(self.context),
        }


class GWInputError(GWError, ValueError):
    """输入 JSON / 配置不合法。"""

    code = "GW-INPUT"
    default_cause = "输入 JSON 不符合 schema：字段缺失、类型/取值范围不对，或引用了不存在的对象。"
    default_hint = ("对照报错中的字段/对象名检查配置；字段规则见 "
                    "docs/配置手册与数学原理.md；也可先运行 ProblemInput.validate_all()。")


class GWInvariantError(GWError, AssertionError):
    """内部不变式被破坏（通常意味着代码 bug 或输入组合超出覆盖范围）。"""

    code = "GW-INTERNAL"
    default_cause = "触发了本应恒成立的不变式，通常是代码 bug，或输入组合超出了当前实现的覆盖范围。"
    default_hint = ("请保留 traceback、输入 JSON 和格点 (C, κ) 反馈给维护者；"
                    "可先用 validate_all() 排除输入错误。")


class GWValidationError(GWError):
    """静态输入校验发现至少一个 error。"""

    code = "GW-VALIDATION"
    default_cause = "输入通过了 JSON 解析，但违反了业务校验规则（V1..V14）。"
    default_hint = "逐条修正 issues 中的错误；规则说明见 docs/配置手册与数学原理.md 的校验章节。"

    def __init__(self, issues: Iterable[ValidationIssue]):
        self.issues = tuple(issues)
        super().__init__("; ".join(f"{i.code}: {i.message}" for i in self.issues),
                         context={"issues": [i.code for i in self.issues]})


class GWInfeasibleGridPoint(GWError):
    """预处理层：某个路口在这个 (C, κ) 下没有任何可行方案。"""

    code = "GW-GRID-INFEASIBLE"
    default_cause = "该格点的周期/速度组合下，至少一个路口的所有候选方案都不满足硬约束或可调边界不可行。"
    default_hint = ("查看 dropped 方案的原因；可放宽 hard_constraints/min_phase_s，"
                    "增加候选方案，或从 grid.cycles_s / speed_ratios 中移除该格点。")

    def __init__(self, cycle_s: float, kappa: float,
                 records: dict[str, tuple[str, ...]] | None = None,
                 dropped: tuple = ()):
        self.cycle_s = cycle_s
        self.kappa = kappa
        self.records = dict(records or {})
        self.dropped = tuple(dropped)
        detail = "; ".join(f"{k}: {', '.join(v)}" for k, v in self.records.items())
        super().__init__(
            f"grid point C={cycle_s}, kappa={kappa} infeasible: {detail}",
            context={"cycle_s": cycle_s, "kappa": kappa,
                     "intersections": sorted(self.records)},
        )


class GWModelError(GWError):
    """建模层错误（不支持的引用、模型内部不一致等）。"""

    code = "GW-MODEL"
    default_cause = "模型构建遇到了不支持的原子/引用组合，或内部数据不一致。"
    default_hint = ("检查约束/聚合引用的 plan、direction、index、window 是否存在；"
                    "若输入已通过 validate_all()，请保留 traceback 反馈给维护者。")


class GWSemanticError(GWError):
    """高层语义规则无法编译。"""

    code = "GW-SEMANTIC"
    default_cause = "语义规则的 target/window/plan 无法解析，或规则组合互相矛盾。"
    default_hint = "检查语义规则的 target、plan_id、direction、window 引用；可先单独编译该规则。"

    def __init__(self, rule_id: str, target_desc: str, reason: str,
                 candidates: tuple[str, ...] = ()) -> None:
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
        super().__init__("\n".join(lines), context={"rule_id": rule_id})


class GWSolveError(GWError):
    """求解器后端报错。"""

    code = "GW-SOLVE"
    default_cause = "求解器不可用、返回错误，或模型在数值上无法求解。"
    default_hint = ("检查 backend 安装（scipy/highspy）、time_limit_s、mip_rel_gap；"
                    "可换一个 backend 或先运行不可行诊断。")


__all__ = [
    "ValidationIssue",
    "GWError",
    "GWInputError",
    "GWInvariantError",
    "GWValidationError",
    "GWInfeasibleGridPoint",
    "GWModelError",
    "GWSemanticError",
    "GWSolveError",
]
