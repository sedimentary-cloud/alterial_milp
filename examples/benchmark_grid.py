"""单格点与并行格点搜索计时脚本。

用法：
    python examples/benchmark_grid.py

默认使用 examples/standard_problem.json，并替换为 3×2 格点：
    cycles_s    = [130, 140, 150]
    speed_ratios = [0.95, 1.0]

分别用 num_workers = 1 和 3 运行 GridRunner，输出：
- 总墙钟时间；
- 各格点求解器耗时；
- 并行加速比。
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from greenwave.grid.runner import GridRunner  # noqa: E402
from greenwave.schema.models import ProblemInput, SolverConfig  # noqa: E402
from greenwave.solve import make_backend  # noqa: E402


def _load_problem() -> ProblemInput:
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "standard_problem.json"), "r", encoding="utf-8") as f:
        raw = json.load(f)
    raw["grid"] = {
        "cycles_s": [130.0, 140.0, 150.0, 160.0],
        "speed_ratios": [0.95, 1.0, 1.05],
    }
    raw["solver"]["num_workers"] = 1
    raw["solver"]["heuristic"]["enabled"] = True
    raw["objective"]["pareto"] = {"num_points": 0, "relax_max": 0.0, "topk_grids": 1}
    return ProblemInput.from_obj(raw)


def _run(problem: ProblemInput, workers: int):
    cfg = replace(problem.solver, num_workers=workers)
    t0 = time.perf_counter()
    report = GridRunner(problem, lambda: make_backend(cfg.backend), cfg).run()
    wall = time.perf_counter() - t0
    return wall, report


def main() -> None:
    problem = _load_problem()

    # 先单独跑一个格点，测“一个方案多久出来”。
    single_cfg = replace(problem.solver, num_workers=1)
    single_problem = replace(problem, grid=replace(problem.grid, cycles_s=(140.0,),
                                                   speed_ratios=(1.0,)))
    t0 = time.perf_counter()
    single_report = GridRunner(single_problem, lambda: make_backend(single_cfg.backend),
                               single_cfg).run()
    single_wall = time.perf_counter() - t0
    print("=== single grid point ===")
    for r in single_report.results:
        print(f"C={r.cycle_s:g}s kappa={r.kappa:g} status={r.status} "
              f"model+solve={r.time_s:.3f}s wall_total={single_wall:.3f}s "
              f"composite={r.composite_best}")
    print(f"single wall total: {single_wall:.3f}s")

    print("\n=== parallel benchmark (4×3 grids) ===")
    results = {}
    for workers in (1, 2, 3):
        wall, report = _run(problem, workers)
        solver_sum = sum(r.time_s for r in report.results)
        results[workers] = (wall, solver_sum, report)
        print(f"workers={workers}: wall={wall:.3f}s "
              f"solver_sum={solver_sum:.3f}s overhead={wall - solver_sum:.3f}s")
        for r in sorted(report.results, key=lambda x: (x.cycle_s, x.kappa)):
            print(f"  C={r.cycle_s:g}s kappa={r.kappa:g} "
                  f"status={r.status} solve={r.time_s:.3f}s "
                  f"band={r.band_reward_s} margin={r.margin_loss_s} "
                  f"loss={r.intersection_loss_s}")

    w1 = results[1][0]
    for workers in (2, 3):
        if workers in results and results[workers][0] > 0:
            print(f"speedup (workers=1 -> {workers}): {w1 / results[workers][0]:.2f}×")


if __name__ == "__main__":
    main()
