"""Prototype: pairwise decomposition for fixed-boundary bidirectional window green.

Assumption:
  - phase boundaries are fixed (no adjust)
  - objective is only adjacent-pair green waves (both directions)
  - no global chain green wave
  - no cross-demand window capacity constraints

Then each edge (i, i+1) is independent and depends only on the relative
offset Delta_e = phi_{i+1} - phi_i.  We solve one tiny MILP per edge and
compose the global offsets by accumulating the optimal Deltas.
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from greenwave.model.builder import ModelBuilder  # noqa: E402
from greenwave.model.variables import GVarType  # noqa: E402
from greenwave.preprocess.feasibility import FeasibilityFilter  # noqa: E402
from greenwave.report import SolutionDecoder  # noqa: E402
from greenwave.schema.models import ProblemInput  # noqa: E402
from greenwave.solve import make_backend  # noqa: E402
from greenwave.solve.backend_base import SolveOptions  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.join(HERE, "standard_problem.json")
CYCLE = 140.0
KAPPA = 1.0
TIME_LIMIT = 60.0


def _load() -> dict:
    with open(BASE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _fix_boundaries(raw: dict) -> None:
    for plan in raw["plans"]:
        for d in ("up", "down"):
            for boundary in plan["directions"][d]["boundaries"]:
                boundary.pop("adjust", None)


def _demand(did: str, direction: str, nodes: list[str], max_bands: int) -> dict:
    return {
        "id": did, "type": "green", "direction": direction, "nodes": list(nodes),
        "max_bands": max_bands, "weight": 1.0,
        "margin": {"delta_min_s": 0.0, "coef": 1.0},
    }


def _pair_demands(full: dict, i: int, max_bands: int) -> list[dict]:
    nodes = list(full["corridor"]["intersections"])
    a, b = nodes[i], nodes[i + 1]
    return [
        _demand(f"GW_up_{a}_{b}", "up", [a, b], max_bands),
        _demand(f"GW_down_{b}_{a}", "down", [b, a], max_bands),
    ]


def _plan_map(raw: dict) -> dict:
    return {p["id"]: p for p in raw["plans"]}


def _build_joint(full: dict, fixed: bool = True) -> ProblemInput:
    raw = copy.deepcopy(full)
    if fixed:
        _fix_boundaries(raw)
    nodes = list(raw["corridor"]["intersections"])
    demands = []
    for i in range(len(nodes) - 1):
        pair = [nodes[i], nodes[i + 1]]
        mb = 2 if "I3" in pair else 1
        demands.extend(_pair_demands(raw, i, mb))
    obj = raw["objective"]
    obj["band_demands"] = demands
    obj["balanced_groups"] = []
    obj["margin_default"] = {"delta_min_s": 0.0, "coef": 1.0}
    obj["allow_window_sharing"] = True
    obj["pareto"] = {"num_points": 0, "relax_max": 0.0, "topk_grids": 1}
    raw["global_constraints"] = []
    raw["constraint_templates"] = []
    raw["grid"] = {"cycles_s": [CYCLE], "speed_ratios": [KAPPA]}
    raw["solver"]["heuristic"]["enabled"] = False
    raw["solver"]["num_workers"] = 1
    raw["solver"]["time_limit_s"] = TIME_LIMIT
    raw["solver"]["mip_rel_gap"] = 0.0
    return ProblemInput.from_obj(raw)


def _build_pair(full: dict, i: int, fixed: bool = True) -> ProblemInput:
    raw = copy.deepcopy(full)
    if fixed:
        _fix_boundaries(raw)
    nodes = list(raw["corridor"]["intersections"])
    a, b = nodes[i], nodes[i + 1]
    plans = [p for p in raw["plans"] if p["intersection"] in (a, b)]
    pair = [a, b]
    mb = 2 if "I3" in pair else 1
    raw["corridor"] = {
        "corridor_id": f"pair_{a}_{b}",
        "intersections": [a, b],
        "segments": [copy.deepcopy(raw["corridor"]["segments"][i])],
    }
    raw["plans"] = plans
    obj = raw["objective"]
    obj["band_demands"] = _pair_demands(raw, 0, mb)
    obj["balanced_groups"] = []
    obj["margin_default"] = {"delta_min_s": 0.0, "coef": 1.0}
    obj["allow_window_sharing"] = True
    obj["pareto"] = {"num_points": 0, "relax_max": 0.0, "topk_grids": 1}
    raw["global_constraints"] = []
    raw["constraint_templates"] = []
    raw["grid"] = {"cycles_s": [CYCLE], "speed_ratios": [KAPPA]}
    raw["solver"]["heuristic"]["enabled"] = False
    raw["solver"]["num_workers"] = 1
    raw["solver"]["time_limit_s"] = TIME_LIMIT
    raw["solver"]["mip_rel_gap"] = 0.0
    return ProblemInput.from_obj(raw)


def _solve(problem: ProblemInput, decode: bool = False):
    ctx = FeasibilityFilter(problem).filter(CYCLE, KAPPA)
    model = ModelBuilder(problem, ctx).build()
    n_bin = sum(1 for t in model.vars.types if t == GVarType.BINARY)
    n_int = sum(1 for t in model.vars.types if t == GVarType.INTEGER)
    n_cons = len(model.cons.rows)
    n_nz = sum(len(row.expr.coefs) for row in model.cons.rows)
    backend = make_backend(problem.solver.backend)
    session = backend.create(model)
    backend.set_objective(session, model.expr_composite(), "max")
    t0 = time.perf_counter()
    res = backend.solve(session, SolveOptions(problem.solver.time_limit_s,
                                              problem.solver.mip_rel_gap))
    solve_s = time.perf_counter() - t0
    sol = None
    if decode and res.values is not None:
        sol = SolutionDecoder(model).decode(res.values)
    return {
        "status": res.status.value,
        "solve_s": solve_s,
        "objective": None if res.objective is None else float(res.objective),
        "vars": len(model.vars),
        "binary": n_bin,
        "integer": n_int,
        "cons": n_cons,
        "nnz": n_nz,
        "solution": sol,
    }


def main() -> None:
    full = _load()
    nodes = list(full["corridor"]["intersections"])

    print("=== joint MILP (fixed boundaries) ===")
    joint = _solve(_build_joint(full, fixed=True))
    print(f"status={joint['status']} objective={joint['objective']} "
          f"solve={joint['solve_s']:.3f}s vars={joint['vars']} "
          f"binary={joint['binary']} cons={joint['cons']} nnz={joint['nnz']}")

    print("\n=== pairwise decomposition (fixed boundaries) ===")
    pairwise_total = 0.0
    edge_rows = []
    deltas = []
    for i in range(len(nodes) - 1):
        t0 = time.perf_counter()
        out = _solve(_build_pair(full, i, fixed=True), decode=True)
        wall = time.perf_counter() - t0
        pairwise_total += out["solve_s"]
        sol = out["solution"]
        width_sum = 0.0
        if sol is not None:
            width_sum = sum(float(b.width_s) for b in sol.bands)
        delta = None
        if sol is not None:
            # Pair subproblem anchor is the first intersection phi=0.
            delta = float(sol.offsets_s.get(nodes[i + 1], 0.0)) % CYCLE
        deltas.append(delta)
        edge_rows.append((nodes[i], nodes[i + 1], out, width_sum, delta, wall))
        print(f"edge {nodes[i]}->{nodes[i+1]}: status={out['status']} "
              f"band_sum={width_sum:.3f}s solve={out['solve_s']:.4f}s "
              f"build+filter+solve={wall:.4f}s delta={delta} "
              f"vars={out['vars']} binary={out['binary']} cons={out['cons']}")

    # Compose global offsets.
    offsets = {nodes[0]: 0.0}
    for i, delta in enumerate(deltas):
        offsets[nodes[i + 1]] = (offsets[nodes[i]] + (delta or 0.0)) % CYCLE

    print("\n=== composed global offsets ===")
    print({k: round(v, 4) for k, v in offsets.items()})
    print(f"pairwise objective = {sum(r[3] for r in edge_rows):.6f}s")
    print(f"joint objective    = {joint['objective']}")
    print(f"pairwise solve sum = {pairwise_total:.4f}s")
    print(f"joint solve        = {joint['solve_s']:.4f}s")
    if pairwise_total > 1e-12:
        print(f"speedup            = {joint['solve_s'] / pairwise_total:.2f}x")


if __name__ == "__main__":
    main()
