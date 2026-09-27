"""标准综合案例：6 路口、140s 周期、上行绿波 + 下行红波。

问题定义已经单独写到：

    examples/standard_problem.json

本脚本只负责：
1. 读取这份 JSON；
2. 调用 run() 做两阶段求解；
3. 生成结果 JSON、时距图和 Pareto 图。

运行：
    python run_standard.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from greenwave import run  # noqa: E402
from greenwave.model.builder import ModelBuilder  # noqa: E402
from greenwave.preprocess.feasibility import FeasibilityFilter  # noqa: E402
from greenwave.report import (ParetoFrontierDiagram, SolutionDecoder,  # noqa: E402
                              TimeSpaceDiagram)
from greenwave.schema.models import ProblemInput  # noqa: E402
from greenwave.solve import make_backend  # noqa: E402
from greenwave.solve.backend_base import SolveOptions  # noqa: E402


PROBLEM_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "examples", "standard_problem.json")


def build_standard_problem() -> dict:
    """读取 examples/standard_problem.json。"""
    with open(PROBLEM_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _solve_stage1(problem_dict: dict, cycle_s: float, kappa: float):
    """单独跑一次阶段 1，用来画时距图。"""
    problem = ProblemInput.from_obj(problem_dict)
    ctx = FeasibilityFilter(problem).filter(cycle_s, kappa)
    model = ModelBuilder(problem, ctx).build()
    backend = make_backend(problem.solver.backend)
    session = backend.create(model)
    backend.set_objective(session, model.expr_composite(), "max")
    result = backend.solve(session, SolveOptions(
        time_limit_s=problem.solver.time_limit_s,
        mip_rel_gap=problem.solver.mip_rel_gap,
    ))
    if result.values is None:
        raise RuntimeError(f"standard case solve failed: {result.status} {result.message}")
    return problem, ctx, SolutionDecoder(model).decode(result.values)


def main() -> None:
    problem = build_standard_problem()
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "examples", "standard_output")
    os.makedirs(out_dir, exist_ok=True)

    # 完整两阶段求解 + JSON 报告
    result = run(problem)
    result_path = os.path.join(out_dir, "standard_result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"wrote {result_path}")

    # Pareto 前沿图
    pareto_path = os.path.join(out_dir, "standard_pareto.png")
    ParetoFrontierDiagram.render(result, pareto_path,
                                 title="Standard Case: up green vs intersection loss")
    print(f"wrote {pareto_path}")

    # 阶段 1 时距图
    cycle_s = problem["grid"]["cycles_s"][0]
    kappa = problem["grid"]["speed_ratios"][0]
    problem_obj, ctx, solution = _solve_stage1(problem, cycle_s, kappa)
    ts_path = os.path.join(out_dir, "standard_time_space.png")
    TimeSpaceDiagram.render(solution, problem_obj, ctx, ts_path,
                            title="Standard Case: up green + down red")
    print(f"wrote {ts_path}")

    # 文字摘要
    print("grid statuses:")
    for grid in result["grid_results"]:
        print(f"  C={grid['cycle_s']:.0f}s kappa={grid['speed_ratio']:.2f} "
              f"status={grid['status']} composite={grid['composite_best']}")
    print(f"pareto points: {len(result['pareto']['points'])}")
    for band in solution.bands:
        print(f"  band {band.demand}: width={band.width_s:.2f}s start={band.start_s:.2f}s")


if __name__ == "__main__":
    main()
