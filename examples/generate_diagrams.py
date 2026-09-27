"""Generate a set of time-space diagrams for the synthetic design-document cases.

Run from the project root:

    python examples/generate_diagrams.py
"""
from __future__ import annotations

import os
import sys

# Make the project importable when this file is executed directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from greenwave.model.builder import ModelBuilder  # noqa: E402
from greenwave.preprocess.feasibility import FeasibilityFilter  # noqa: E402
from greenwave.report import TimeSpaceDiagram, SolutionDecoder  # noqa: E402
from greenwave.schema.models import ProblemInput  # noqa: E402
from greenwave.solve import make_backend  # noqa: E402
from greenwave.solve.backend_base import SolveOptions  # noqa: E402


def _seg(from_: str, to: str, dist: float = 500.0, v: float = 11.1111111111,
         dist_up: float | None = None, dist_down: float | None = None) -> dict:
    obj = {"from": from_, "to": to, "speed_up_mps": v, "speed_down_mps": v}
    if dist is not None:
        obj["distance_m"] = dist
    if dist_up is not None:
        obj["distance_up_m"] = dist_up
    if dist_down is not None:
        obj["distance_down_m"] = dist_down
    return obj


def _plan(iid: str, up, down, double: bool = False, adjustable=None,
          hard=(), soft=()) -> dict:
    return {"id": f"{iid}_P1", "intersection": iid, "double_cycle": double,
            "windows": {"up": up, "down": down},
            "adjustable": adjustable or {"up": {}, "down": {}},
            "hard_constraints": list(hard), "soft_constraints": list(soft)}


def _problem(corridor, plans, demands, cycle=90.0, margin=0.0) -> dict:
    return {
        "corridor": corridor,
        "plans": plans,
        "objective": {
            "band_demands": demands,
            "margin_default": {"delta_min_s": margin, "coef": 1.0},
            "pareto": {"num_points": 1, "relax_max": 0.0, "topk_grids": 1},
        },
        "grid": {"cycles_s": [cycle], "speed_ratios": [1.0]},
        "solver": {"backend": "scipy", "time_limit_s": 30, "mip_rel_gap": 0.0,
                   "num_workers": 1, "heuristic": {"enabled": False, "refine_rounds": 0}},
    }


def solve_solution(problem_dict: dict, cycle: float, kappa: float = 1.0):
    problem = ProblemInput.from_obj(problem_dict)
    ctx = FeasibilityFilter(problem).filter(cycle, kappa)
    model = ModelBuilder(problem, ctx).build()
    backend = make_backend(problem.solver.backend)
    session = backend.create(model)
    backend.set_objective(session, model.expr_composite(), "max")
    result = backend.solve(session, SolveOptions(problem.solver.time_limit_s,
                                                 problem.solver.mip_rel_gap))
    if result.values is None:
        raise RuntimeError(f"no solution for case {problem.corridor.corridor_id}: "
                           f"{result.status} {result.message}")
    return problem, ctx, SolutionDecoder(model).decode(result.values)


def render(name: str, problem_dict: dict, cycle: float = 90.0, kappa: float = 1.0) -> None:
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diagrams")
    os.makedirs(out_dir, exist_ok=True)
    problem, ctx, solution = solve_solution(problem_dict, cycle, kappa)
    out_path = os.path.join(out_dir, f"{name}.png")
    TimeSpaceDiagram.render(solution, problem, ctx, out_path)
    print(f"wrote {out_path}")


def main() -> None:
    # 01: perfect one-way green wave (T1)
    corridor3 = {
        "corridor_id": "perfect_green",
        "intersections": ["I1", "I2", "I3"],
        "segments": [_seg("I1", "I2"), _seg("I2", "I3")],
    }
    plans3 = [_plan(i, [[0.0, 0.45]], [[0.5, 0.95]]) for i in ("I1", "I2", "I3")]
    render("01_perfect_green_wave", _problem(corridor3, plans3, [
        {"id": "GW_up", "type": "green", "direction": "up", "nodes": ["I1", "I2", "I3"],
         "weight": 1.0, "margin": {"delta_min_s": 0.0}},
    ]))

    # 02: bidirectional green waves (T3)
    plans_bi = [_plan(i, [[0.0, 0.45]], [[0.0, 0.45]]) for i in ("I1", "I2", "I3")]
    render("02_bidirectional_green", _problem(corridor3, plans_bi, [
        {"id": "GW_up", "type": "green", "direction": "up", "nodes": ["I1", "I2", "I3"],
         "weight": 1.0, "margin": {"delta_min_s": 0.0}},
        {"id": "GW_down", "type": "green", "direction": "down", "nodes": ["I3", "I2", "I1"],
         "weight": 1.0, "margin": {"delta_min_s": 0.0}},
    ]))

    # 03: multi-window / multi-band (T4)
    plans_mw = [_plan(i, [[0.0, 0.2], [0.5, 0.7]], [[0.0, 0.2], [0.5, 0.7]])
                for i in ("I1", "I2", "I3")]
    render("03_multi_window_multi_band", _problem(corridor3, plans_mw, [
        {"id": "GW_up", "type": "green", "direction": "up", "nodes": ["I1", "I2", "I3"],
         "max_bands": 2, "weight": 1.0, "margin": {"delta_min_s": 0.0}},
    ]))

    # 04: red wave across one adjacent pair (T5)
    corridor2 = {
        "corridor_id": "red_pair",
        "intersections": ["I1", "I2"],
        "segments": [_seg("I1", "I2", dist=200.0, v=10.0)],
    }
    plans_red = [_plan("I1", [[0.0, 0.45]], [[0.0, 0.45]]),
                 _plan("I2", [[0.0, 0.45]], [[0.0, 0.45]])]
    render("04_red_wave_pair", _problem(corridor2, plans_red, [
        {"id": "RW_up", "type": "red", "direction": "up", "nodes": ["I1", "I2"],
         "max_bands": 3, "weight": 1.0},
    ]))

    # 05: double-cycle plan with consistency constraint (T8)
    corridor_dc = {
        "corridor_id": "double_cycle",
        "intersections": ["I1", "I2"],
        "segments": [_seg("I1", "I2")],
    }
    plans_dc = [
        _plan("I1", [[0.0, 0.2]], [[0.0, 0.2]], double=True,
              adjustable={"up": {"0": {"start": [-0.02, 0.02], "end": [-0.02, 0.02]}},
                          "down": {}}),
        _plan("I2", [[0.0, 0.45]], [[0.0, 0.45]]),
    ]
    render("05_double_cycle", _problem(corridor_dc, plans_dc, [
        {"id": "GW_up", "type": "green", "direction": "up", "nodes": ["I1", "I2"],
         "weight": 1.0, "margin": {"delta_min_s": 0.0}},
    ]))

    # 06: directional segment lengths (up 500 m, down 650 m)
    corridor_dir = {
        "corridor_id": "directional_lengths",
        "intersections": ["I1", "I2", "I3"],
        "segments": [_seg("I1", "I2", dist=None, dist_up=500.0, dist_down=650.0),
                     _seg("I2", "I3", dist=None, dist_up=500.0, dist_down=650.0)],
    }
    render("06_directional_segment_lengths", _problem(corridor_dir, plans_bi, [
        {"id": "GW_up", "type": "green", "direction": "up", "nodes": ["I1", "I2", "I3"],
         "weight": 1.0, "margin": {"delta_min_s": 0.0}},
        {"id": "GW_down", "type": "green", "direction": "down", "nodes": ["I3", "I2", "I1"],
         "weight": 1.0, "margin": {"delta_min_s": 0.0}},
    ]))

    # 07: up green wave + down red wave
    # Up direction gets a full green band; down direction has tailored red
    # windows so that two adjacent-pair red bands also survive.
    plans_ugdr = [
        _plan("I1", [[0.0, 0.45]], [[0.7, 0.9]]),
        _plan("I2", [[0.0, 0.45]], [[0.3, 0.5]]),
        _plan("I3", [[0.0, 0.45]], [[0.4, 0.6]]),
    ]
    render("07_up_green_down_red", _problem(corridor3, plans_ugdr, [
        {"id": "GW_up", "type": "green", "direction": "up", "nodes": ["I1", "I2", "I3"],
         "weight": 2.0, "margin": {"delta_min_s": 0.0}},
        {"id": "RW_down_32", "type": "red", "direction": "down",
         "nodes": ["I3", "I2"], "max_bands": 1, "weight": 1.0},
        {"id": "RW_down_21", "type": "red", "direction": "down",
         "nodes": ["I2", "I1"], "max_bands": 1, "weight": 1.0},
    ]))


if __name__ == "__main__":
    main()
