"""Generate a Pareto-frontier plot for examples/standard_problem.json.

The standard case contains one grid point: C=140s, kappa=1.0.
This script scans 20 epsilon-constraint points and draws the
non-dominated composite-vs-intersection-loss frontier.

Run from the project root:

    python examples/generate_pareto.py
"""
from __future__ import annotations

import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from greenwave import run  # noqa: E402
from greenwave.report import ParetoFrontierDiagram  # noqa: E402


def build_problem() -> dict:
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "standard_problem.json"), "r", encoding="utf-8") as f:
        problem = json.load(f)

    # ParetoFrontierDiagram 会画 num_points+1 个 epsilon 采样点；
    # 这里要 20 个点，所以 num_points=19。
    problem = copy.deepcopy(problem)
    problem["objective"]["pareto"] = {
        "num_points": 19,
        "relax_max": problem["objective"].get("pareto", {}).get("relax_max", 0.3),
        "topk_grids": 1,
    }
    return problem


def main() -> None:
    result = run(build_problem())
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diagrams")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "08_pareto_frontier.png")
    ParetoFrontierDiagram.render(
        result, out_path,
        title="Standard Case Pareto Frontier: composite vs intersection loss",
    )
    print(f"wrote {out_path}")
    points = result["pareto"]["points"]
    print(f"pareto points: {len(points)}")
    for p in points:
        print(f"epsilon={p['epsilon']:.3f}s  composite={p['composite_actual']:.3f}s  "
              f"loss={p['intersection_loss']:.3f}s")


if __name__ == "__main__":
    main()
