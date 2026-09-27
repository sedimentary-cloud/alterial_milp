"""总流程：校验 -> 格点阶段 1 -> Pareto 阶段 2 -> 报告。

给高中生看的小课堂
------------------
整个算法分几步：
1. 检查输入 JSON 是否合法；
2. 对每个周期 C、速度倍率 κ 组合，先过滤方案，再求最大总带宽；
3. 从所有格点里挑出最优的几个；
4. 在这些格点上做 ε-constraint，得到 Pareto 前沿；
5. 把结果整理成 JSON。
"""
from __future__ import annotations

from typing import Any

from .errors import GWValidationError
from .grid.runner import GridRunner, GridRunReport
from .pareto.epsilon import EpsilonConstraintScanner
from .preprocess.feasibility import FeasibilityFilter
from .report.solution import ReportBuilder
from .schema.models import ProblemInput
from .solve import make_backend


def run(problem_json: dict[str, Any], semantics: dict[str, Any] | list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """用户最常调用的入口函数。

    ``semantics`` 不为 None 时，先经过语义编译层，再走原优化管线。
    """
    if semantics is not None:
        from .semantic import compile_semantics
        problem_json = compile_semantics(problem_json, semantics).problem_json
    problem = ProblemInput.from_obj(problem_json)
    issues = problem.validate_all()
    errors = [i for i in issues if i.level == "error"]
    if errors:
        raise GWValidationError(errors)
    cfg = problem.solver
    grid_report: GridRunReport = GridRunner(problem, lambda: make_backend(cfg.backend), cfg).run()
    bases = grid_report.best(k=problem.objective.pareto.topk_grids)
    pareto_groups = []
    for base in bases:
        ctx = base.ctx
        if ctx is None:
            ctx = FeasibilityFilter(problem).filter(base.cycle_s, base.kappa)
        scanner = EpsilonConstraintScanner(problem, ctx, make_backend(cfg.backend),
                                           problem.objective.pareto)
        pareto_groups.append(scanner.scan(base))
    # Attach base grid to each ParetoFront (scanner already does via base_grid).
    return ReportBuilder.build(grid_report, pareto_groups)


run_optimization = run

__all__ = ["run", "run_optimization"]
