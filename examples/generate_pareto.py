"""Generate a Pareto-frontier plot for one selected case.

The selected case is the T6-style scenario from the design document:
an adjustable green-window end with an intersection-loss soft constraint.
The epsilon scan is fine enough to expose several non-dominated points.

Run from the project root:

    python examples/generate_pareto.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from greenwave import run  # noqa: E402
from greenwave.report import ParetoFrontierDiagram  # noqa: E402


def _seg(from_, to, dist=500.0, v=11.1111111111):
    return {"from": from_, "to": to, "distance_m": dist,
            "speed_up_mps": v, "speed_down_mps": v}


def _plan(iid, up, down, **extra):
    obj = {"id": f"{iid}_P1", "intersection": iid, "double_cycle": False,
           "windows": {"up": up, "down": down},
           "adjustable": {"up": {}, "down": {}}}
    obj.update(extra)
    return obj


def build_problem() -> dict:
    soft = [{
        "id": "endsoft",
        "terms": [{
            "atom": {"kind": "endpoint", "intersection": "I1", "plan": "I1_P1",
                     "direction": "up", "window": 0, "side": "end"},
            "coef": 1.0,
        }],
        "sense": "<=",
        "rhs_ratio": 0.40,
        "hard": False,
        "soft": {"coef": 10.0, "owner": "intersection_loss"},
    }]
    plans = [
        _plan("I1", [[0.0, 0.45]], [[0.5, 0.95]],
              adjustable={"up": {"0": {"end": [-0.05, 0.05]}}, "down": {}},
              soft_constraints=soft),
        _plan("I2", [[0.0, 0.45]], [[0.5, 0.95]]),
        _plan("I3", [[0.0, 0.45]], [[0.5, 0.95]]),
    ]
    return {
        "corridor": {
            "corridor_id": "pareto_adjustable_soft",
            "intersections": ["I1", "I2", "I3"],
            "segments": [_seg("I1", "I2"), _seg("I2", "I3")],
        },
        "plans": plans,
        "objective": {
            "band_demands": [{
                "id": "GW_up", "type": "green", "direction": "up",
                "nodes": ["I1", "I2", "I3"], "max_bands": 1, "weight": 1.0,
                "margin": {"delta_min_s": 0.0, "coef": 1.0},
            }],
            "margin_default": {"delta_min_s": 0.0, "coef": 1.0},
            "pareto": {"num_points": 16, "relax_max": 0.5, "topk_grids": 1},
        },
        "grid": {"cycles_s": [90.0], "speed_ratios": [1.0]},
        "solver": {"backend": "scipy", "time_limit_s": 30, "mip_rel_gap": 0.0,
                   "num_workers": 1, "heuristic": {"enabled": False, "refine_rounds": 0}},
    }


def main() -> None:
    result = run(build_problem())
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diagrams")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "08_pareto_frontier.png")
    ParetoFrontierDiagram.render(
        result, out_path,
        title="Pareto Frontier: adjustable green end vs intersection loss",
    )
    print(f"wrote {out_path}")
    for p in result["pareto"]["points"]:
        print(f"epsilon={p['epsilon']:.3f}s  composite={p['composite_actual']:.3f}s  "
              f"loss={p['intersection_loss']:.3f}s")


if __name__ == "__main__":
    main()
