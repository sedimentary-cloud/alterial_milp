"""语义编译报告：台账、生成清单、warnings。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CompileReport:
    red_ledgers: tuple[dict[str, Any], ...] = ()
    locked_endpoints: tuple[dict[str, Any], ...] = ()
    generated_constraints: tuple[dict[str, Any], ...] = ()
    generated_templates: tuple[dict[str, Any], ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "red_ledgers": list(self.red_ledgers),
            "locked_endpoints": list(self.locked_endpoints),
            "generated_constraints": list(self.generated_constraints),
            "generated_templates": list(self.generated_templates),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class CompileResult:
    problem_json: dict[str, Any]
    report: CompileReport


__all__ = ["CompileReport", "CompileResult"]
