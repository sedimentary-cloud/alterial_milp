"""单格点运行耗时拆解：看看时间到底花在哪一步。

运行：
    python examples/profile_pipeline.py
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from greenwave.heuristic.greedy import GreedyConstructor  # noqa: E402
from greenwave.model.builder import ModelBuilder  # noqa: E402
from greenwave.preprocess.feasibility import FeasibilityFilter  # noqa: E402
from greenwave.report.solution import SolutionDecoder  # noqa: E402
from greenwave.schema.models import ProblemInput  # noqa: E402
from greenwave.solve import make_backend  # noqa: E402
from greenwave.solve.backend_base import SolveOptions  # noqa: E402


def _tic():
    return time.perf_counter()


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "standard_problem.json"), "r", encoding="utf-8") as f:
        raw = json.load(f)
    raw["grid"] = {"cycles_s": [140.0], "speed_ratios": [1.0]}

    t0 = _tic()
    problem = ProblemInput.from_obj(raw)
    t_parse = _tic() - t0

    t0 = _tic()
    issues = problem.validate_all()
    t_validate = _tic() - t0

    C = 140.0
    kappa = 1.0

    t0 = _tic()
    ctx = FeasibilityFilter(problem).filter(C, kappa)
    t_filter = _tic() - t0

    t0 = _tic()
    greedy = GreedyConstructor(problem, ctx).run()
    t_greedy = _tic() - t0

    t0 = _tic()
    model = ModelBuilder(problem, ctx).build()
    t_build = _tic() - t0

    backend = make_backend(problem.solver.backend)
    t0 = _tic()
    session = backend.create(model)
    backend.set_objective(session, model.expr_composite(), "max")
    t_session = _tic() - t0

    t0 = _tic()
    result = backend.solve(session, SolveOptions(
        time_limit_s=problem.solver.time_limit_s,
        mip_rel_gap=problem.solver.mip_rel_gap,
    ))
    t_solve = _tic() - t0

    t0 = _tic()
    if result.values is not None:
        _ = SolutionDecoder(model).decode(result.values)
    t_decode = _tic() - t0

    total = t_parse + t_validate + t_filter + t_greedy + t_build + t_session + t_solve + t_decode
    rows = [
        ("parse JSON -> dataclass", t_parse),
        ("validate", t_validate),
        ("preprocess filter", t_filter),
        ("greedy initial solution", t_greedy),
        ("build MILP model", t_build),
        ("create solver session", t_session),
        ("solve MILP", t_solve),
        ("decode solution", t_decode),
    ]
    print(f"{'step':32s} {'seconds':>10s} {'percent':>9s}")
    print("-" * 55)
    for name, dt in rows:
        print(f"{name:32s} {dt:10.4f} {100.0 * dt / total:8.2f}%")
    print("-" * 55)
    print(f"{'total':32s} {total:10.4f} {100.0:8.2f}%")
    print(f"\nsolver status={result.status.value} objective={result.objective}")
    print(f"greedy bound={greedy.composite_est}")


if __name__ == "__main__":
    main()
