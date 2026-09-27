"""公开 API：high-level semantic rules -> compiled problem JSON."""
from __future__ import annotations

from typing import Any

from .compiler import compile_semantics as _compile_semantics
from .report import CompileReport, CompileResult

__all__ = ["compile_semantics", "CompileResult", "CompileReport"]


def compile_semantics(problem_json: dict[str, Any],
                      semantics_json: dict[str, Any] | list[dict[str, Any]]) -> CompileResult:
    _, report, compiled_json = _compile_semantics(problem_json, semantics_json)
    return CompileResult(problem_json=compiled_json, report=report)
