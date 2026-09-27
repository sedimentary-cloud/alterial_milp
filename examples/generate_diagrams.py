"""Generate a set of time-space diagrams for the synthetic design-document cases.

Run from the project root:

    python examples/generate_diagrams.py
"""
from __future__ import annotations

import copy
import json
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


def render(name: str, problem_dict: dict, cycle: float = 90.0, kappa: float = 1.0,
           title: str | None = None) -> None:
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diagrams")
    os.makedirs(out_dir, exist_ok=True)
    problem, ctx, solution = solve_solution(problem_dict, cycle, kappa)
    out_path = os.path.join(out_dir, f"{name}.png")
    TimeSpaceDiagram.render(solution, problem, ctx, out_path, title=title)
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

    # standard_problem.json 的四类 objective 对照场景
    _run_standard_objective_scenarios()




# ---------------------------------------------------------------------------
# standard_problem.json 的四类 objective 场景
# ---------------------------------------------------------------------------

def _standard_demand(did: str, kind: str, direction: str, nodes, weight: float = 1.0):
    obj = {
        "id": did,
        "type": kind,
        "direction": direction,
        "nodes": list(nodes),
        "max_bands": 1,
        "weight": weight,
    }
    if kind == "green":
        obj["margin"] = {"delta_min_s": 0.0, "coef": 0.0}
    return obj


def _run_standard_objective_scenarios() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "standard_problem.json"), "r", encoding="utf-8") as f:
        base = json.load(f)

    nodes = list(base["corridor"]["intersections"])
    pairs = [(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]

    scenarios = []

    # 1) 双向全局绿波：最大化两个方向的最小带宽。
    scenarios.append((
        "standard_case_01_bidirectional_green_maxmin",
        "Standard Case 1: bidirectional global green, max-min bandwidth",
        [
            _standard_demand("GW_up_full", "green", "up", nodes, weight=0.0),
            _standard_demand("GW_down_full", "green", "down", list(reversed(nodes)), weight=0.0),
        ],
        [{
            "id": "balance_bidir_green",
            "demands": ["GW_up_full", "GW_down_full"],
            "weight": 1.0,
            "on_infeasible": "zero",
            "targets_s": {},
        }],
    ))

    # 2) 单绿多红：一个全局上行绿波 + 下行每个相邻对一条红波，最大化带宽和。
    down_red_demands = []
    for i in range(len(nodes) - 1, 0, -1):
        down_red_demands.append(_standard_demand(
            f"RW_down_{i + 1}_{i}", "red", "down",
            [nodes[i], nodes[i - 1]], weight=1.0))
    scenarios.append((
        "standard_case_02_single_green_single_red_sum",
        "Standard Case 2: single green + down red waves, max bandwidth sum",
        [
            _standard_demand("GW_up_full", "green", "up", nodes, weight=1.0),
            *down_red_demands,
        ],
        [],
    ))

    # 3) 双向红波：每个相邻对、上下行各一条红波，最大化带宽和。
    red_demands = []
    for i, (a, b) in enumerate(pairs):
        red_demands.append(_standard_demand(
            f"RW_up_{i + 1}_{i + 2}", "red", "up", [a, b], weight=1.0))
        red_demands.append(_standard_demand(
            f"RW_down_{i + 2}_{i + 1}", "red", "down", [b, a], weight=1.0))
    scenarios.append((
        "standard_case_03_bidirectional_red_sum",
        "Standard Case 3: bidirectional red waves, max bandwidth sum",
        red_demands,
        [],
    ))

    # 4) 所有相邻路口的绿波：每个相邻对、上下行各一条绿波，最大化带宽和。
    green_demands = []
    for i, (a, b) in enumerate(pairs):
        green_demands.append(_standard_demand(
            f"GW_up_{i + 1}_{i + 2}", "green", "up", [a, b], weight=1.0))
        green_demands.append(_standard_demand(
            f"GW_down_{i + 2}_{i + 1}", "green", "down", [b, a], weight=1.0))
    scenarios.append((
        "standard_case_04_all_adjacent_green_sum",
        "Standard Case 4: all adjacent-pair green waves, max bandwidth sum",
        green_demands,
        [],
    ))

    for name, title, demands, balanced_groups in scenarios:
        problem = copy.deepcopy(base)
        problem["objective"]["band_demands"] = demands
        problem["objective"]["balanced_groups"] = balanced_groups
        problem["objective"]["margin_default"] = {"delta_min_s": 0.0, "coef": 0.0}
        problem["objective"]["pareto"] = {"num_points": 0, "relax_max": 0.0, "topk_grids": 0}
        # 这些是 objective 对照场景，去掉只针对原上行带宽的 global constraints。
        problem["global_constraints"] = []
        problem["grid"] = {"cycles_s": [140.0], "speed_ratios": [1.0]}
        problem["solver"]["heuristic"]["enabled"] = False
        problem["solver"]["num_workers"] = 1
        render(name, problem, cycle=140.0, kappa=1.0, title=title)

if __name__ == "__main__":
    main()
