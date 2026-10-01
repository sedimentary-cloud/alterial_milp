"""Green/red-wave arterial coordination MILP package.

Entry point::

    from greenwave import run
    result = run(problem_dict)
"""
from .errors import (GWError, GWInfeasibleGridPoint, GWModelError,
                     GWSolveError, GWValidationError, ValidationIssue)
from .pipeline import run, run_multistage, run_optimization
from .schema.models import ProblemInput

__all__ = [
    "run", "run_optimization", "run_multistage", "ProblemInput",
    "GWError", "GWValidationError", "GWInfeasibleGridPoint", "GWModelError",
    "GWSolveError", "ValidationIssue",
]
