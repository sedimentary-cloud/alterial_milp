"""Run the bundled minimal problem and print a compact summary."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from greenwave import run  # noqa: E402


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "minimal_problem.json"), "r", encoding="utf-8") as f:
        problem = json.load(f)
    result = run(problem)
    for grid in result["grid_results"]:
        print(grid["cycle_s"], grid["speed_ratio"], grid["status"], grid["composite_best"])
    print("pareto points:", len(result["pareto"]["points"]))


if __name__ == "__main__":
    main()
