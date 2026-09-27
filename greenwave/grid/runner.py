"""格点遍历与阶段 1 并行求解。

给高中生看的小课堂
------------------
公共周期 C 和速度倍率 κ 有很多候选组合，
每个组合叫一个“格点”。

每个格点都独立地：
1. 过滤不可行方案；
2. 跑贪心初始解；
3. 构建 MILP 并最大化 composite。

这些格点互不影响，所以可以并行计算。
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..diagnostics.infeasible import InfeasibilityDiagnoser
from ..errors import GWInfeasibleGridPoint, GWValidationError
from ..heuristic.greedy import GreedyConstructor, GreedySolution
from ..model.builder import ModelBuilder
from ..preprocess.context import GridPointContext
from ..preprocess.feasibility import FeasibilityFilter
from ..report.solution import SolutionDecoder, SolutionRecord
from ..schema.models import ProblemInput, SolverConfig
from ..solve import make_backend
from ..solve.backend_base import SolveOptions, SolveStatus


# 单个格点的求解结果。
@dataclass(frozen=True)
class GridPointResult:
    cycle_s: float
    kappa: float
    status: str
    composite_best: float | None
    heuristic: GreedySolution | None
    solution: SolutionRecord | None
    dropped: tuple = ()
    time_s: float = 0.0
    message: str = ""
    ctx: GridPointContext | None = None
    band_reward_s: float | None = None
    margin_loss_s: float | None = None
    intersection_loss_s: float | None = None


# 所有格点的汇总报告。
@dataclass(frozen=True)
class GridRunReport:
    results: tuple[GridPointResult, ...]

    def best(self, k: int = 1) -> list[GridPointResult]:
        valid = [r for r in self.results if r.composite_best is not None]
        valid.sort(key=lambda r: r.composite_best, reverse=True)
        return valid[:max(0, k)]


def _stage1_worker(payload):
    # 进程池的入口函数：在子进程里重新过滤、建模型、求解。
    # 这样每个格点互不共享内存，避免并行冲突。
    problem, cycle_s, kappa, solver_cfg = payload
    try:
        ctx = FeasibilityFilter(problem).filter(cycle_s, kappa)
        result = _solve_stage1(problem, ctx, solver_cfg)
        return result
    except GWInfeasibleGridPoint as exc:
        return GridPointResult(cycle_s, kappa, "skipped_infeasible", None, None, None,
                               exc.dropped, 0.0, str(exc), None)
    except Exception as exc:  # pragma: no cover - worker isolation
        return GridPointResult(cycle_s, kappa, "error", None, None, None, (), 0.0,
                               f"{type(exc).__name__}: {exc}", None)


def _solve_stage1(problem: ProblemInput, ctx: GridPointContext,
                  solver_cfg: SolverConfig) -> GridPointResult:
    model = ModelBuilder(problem, ctx).build()
    backend = make_backend(solver_cfg.backend)
    session = backend.create(model)
    backend.set_objective(session, model.expr_composite(), "max")
    res = backend.solve(session, SolveOptions(time_limit_s=solver_cfg.time_limit_s,
                                              mip_rel_gap=solver_cfg.mip_rel_gap))
    status = res.status.value
    solution = None
    composite = res.objective
    message = res.message
    if res.status == SolveStatus.INFEASIBLE:
        try:
            diag = InfeasibilityDiagnoser(model, backend).diagnose()
            message = f"{message} | diagnosis groups={diag.culprit_groups}: {diag.detail}"
        except Exception as exc:  # diagnosis must never hide the original status
            message = f"{message} | diagnosis failed: {exc}"
    if res.values is not None and res.status not in (SolveStatus.INFEASIBLE, SolveStatus.ERROR):
        solution = SolutionDecoder(model).decode(res.values)
        # ``composite`` remains the solver objective value / certified bound.  The
        # decoder's ``composite_s`` is the actual attained value and may differ by
        # solver tolerances; stage 2 must use the model-certified B*.
        if composite is None:
            composite = solution.composite_s
    return GridPointResult(
        cycle_s=ctx.cycle_s,
        kappa=ctx.kappa,
        status=status,
        composite_best=None if composite is None else float(composite),
        heuristic=None,
        solution=solution,
        dropped=ctx.dropped,
        time_s=res.time_s,
        message=message,
        ctx=ctx,
        band_reward_s=None if solution is None else solution.band_reward_s,
        margin_loss_s=None if solution is None else solution.margin_loss_s,
        intersection_loss_s=None if solution is None else solution.loss_s,
    )


class GridRunner:
    """格点运行器：负责串行/并行调度所有 (C, κ)。"""

    def __init__(self, problem: ProblemInput,
                 backend_factory: Callable[[], object] | None,
                 cfg: SolverConfig) -> None:
        self.problem = problem
        self.backend_factory = backend_factory
        self.cfg = cfg
        self._filter = FeasibilityFilter(problem)

    def run(self) -> GridRunReport:
        # 阶段 A：所有格点先做过滤和贪心（很便宜）。
        candidates: list[tuple[float, float, GridPointContext, GreedySolution]] = []
        skipped: list[GridPointResult] = []
        # Phase A: filtering + greedy scoring (cheap, sequential)
        for cycle_s in self.problem.grid.cycles_s:
            for kappa in self.problem.grid.speed_ratios:
                try:
                    ctx = self._filter.filter(cycle_s, kappa)
                except GWInfeasibleGridPoint as exc:
                    skipped.append(GridPointResult(cycle_s, kappa, "skipped_infeasible",
                                                   None, None, None, exc.dropped, 0.0,
                                                   str(exc), None))
                    continue
                greedy = None
                if self.cfg.heuristic.enabled:
                    try:
                        greedy = GreedyConstructor(self.problem, ctx).run()
                    except Exception as exc:  # heuristic must never block the pipeline
                        greedy = None
                candidates.append((cycle_s, kappa, ctx, greedy))
        # Sort stage-1 submissions by greedy score descending (7.2)
        candidates.sort(key=lambda t: (t[3].composite_est if t[3] is not None else -1e18),
                        reverse=True)

        results: list[GridPointResult] = []
        payloads = [(self.problem, c, k, self.cfg) for c, k, _, _ in candidates]
        if self.cfg.num_workers and self.cfg.num_workers > 1 and len(payloads) > 1:
            try:
                with ProcessPoolExecutor(max_workers=int(self.cfg.num_workers)) as ex:
                    futs = {ex.submit(_stage1_worker, p): i for i, p in enumerate(payloads)}
                    ordered: list[GridPointResult | None] = [None] * len(payloads)
                    for fut in as_completed(futs):
                        ordered[futs[fut]] = fut.result()
                    results = [r for r in ordered if r is not None]
            except Exception:
                # fall back to sequential on platforms where process pools fail
                results = [_stage1_worker(p) for p in payloads]
        else:
            results = [_stage1_worker(p) for p in payloads]

        # attach heuristic solutions to their matching grid result
        heur_map = {(c, k): g for c, k, _, g in candidates}
        results2 = []
        for r in results:
            g = heur_map.get((r.cycle_s, r.kappa))
            if g is not None:
                r = GridPointResult(r.cycle_s, r.kappa, r.status, r.composite_best,
                                    g, r.solution, r.dropped, r.time_s, r.message, r.ctx,
                                    r.band_reward_s, r.margin_loss_s, r.intersection_loss_s)
            results2.append(r)
        return GridRunReport(tuple(results2 + skipped))


__all__ = ["GridPointResult", "GridRunReport", "GridRunner"]
