"""标准案例配置对比实验：双周期情况 × 目标类型。

3 种双周期配置：
  none        : I3 不是双周期
  double_mb1  : I3 是双周期，但所有 demand max_bands=1
  double_mb2  : I3 是双周期，涉及 I3 / 全局的 demand max_bands=2，并打开共享

3 种目标：
  bidir_green : 双向全局绿波
  up_green_down_red : 上行全局绿波 + 下行相邻对红波
  bidir_window: 双向相邻对窗口绿波

固定 C=140s, κ=1.0，关闭 heuristic，单线程。
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
from greenwave.schema.models import ProblemInput  # noqa: E402
from greenwave.solve import make_backend  # noqa: E402
from greenwave.solve.backend_base import SolveOptions  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.join(HERE, "standard_problem.json")
CYCLE = 140.0
KAPPA = 1.0

OBJECTIVES = ("bidir_green", "up_green_down_red", "bidir_window")
FACTORS = ("none", "double_mb1", "double_mb2")

OBJECTIVE_LABEL = {
    "bidir_green": "双向全局绿波",
    "up_green_down_red": "上绿下红",
    "bidir_window": "双向窗口绿波",
}
FACTOR_LABEL = {
    "none": "无双周期",
    "double_mb1": "双周期 mb=1",
    "double_mb2": "双周期 mb=2",
}


def _demand(did: str, kind: str, direction: str, nodes: list[str],
            weight: float = 1.0) -> dict:
    return {
        "id": did, "type": kind, "direction": direction, "nodes": list(nodes),
        "max_bands": 1, "weight": weight,
        "margin": {"delta_min_s": 0.0, "coef": 1.0},
    }


def _make_no_double_i3(plan: dict) -> dict:
    plan = copy.deepcopy(plan)
    plan["double_cycle"] = False
    # 只保留单周期下有意义的原约束；红窗 rid2 相关约束删除。
    keep_hard = {"I3_up_end_hard_070", "I3_down_end_hard_070", "I3_sync_up_down_end"}
    keep_soft = {"I3_up_end_soft_055", "I3_down_end_soft_055"}
    plan["hard_constraints"] = [c for c in plan.get("hard_constraints", []) if c["id"] in keep_hard]
    plan["soft_constraints"] = [c for c in plan.get("soft_constraints", []) if c["id"] in keep_soft]
    return plan


def _fix_all_adjustments(raw: dict) -> None:
    """Remove every adjustable range so only phi offsets remain free."""
    for plan in raw["plans"]:
        for d in ("up", "down"):
            for boundary in plan["directions"][d]["boundaries"]:
                boundary.pop("adjust", None)


def _build_objective_demands(objective: str, nodes: list[str]) -> list[dict]:
    pairs = [(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]
    if objective == "bidir_green":
        return [
            _demand("GW_up_full", "green", "up", nodes),
            _demand("GW_down_full", "green", "down", list(reversed(nodes))),
        ]
    if objective == "up_green_down_red":
        out = [_demand("GW_up_full", "green", "up", nodes)]
        for a, b in pairs:
            out.append(_demand(f"RW_down_{b}_{a}", "red", "down", [b, a]))
        return out
    # bidir_window: 每个相邻对、上下行各一条窗口绿波
    out = []
    for a, b in pairs:
        out.append(_demand(f"GW_up_{a}_{b}", "green", "up", [a, b]))
        out.append(_demand(f"GW_down_{b}_{a}", "green", "down", [b, a]))
    return out


def _needs_two(demand: dict, nodes: list[str]) -> bool:
    seq = tuple(demand["nodes"])
    if seq == tuple(nodes) or seq == tuple(reversed(nodes)):
        return True
    return "I3" in demand["nodes"]


def _build_problem(objective: str, factor: str,
                   fixed_boundaries: bool = False) -> ProblemInput:
    with open(BASE_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    raw = copy.deepcopy(raw)
    nodes = list(raw["corridor"]["intersections"])

    if factor == "none":
        raw["plans"] = [
            _make_no_double_i3(p) if p["id"] == "I3_P1" else p for p in raw["plans"]
        ]

    if fixed_boundaries:
        _fix_all_adjustments(raw)

    demands = _build_objective_demands(objective, nodes)
    sharing = factor == "double_mb2"
    for d in demands:
        if factor == "double_mb2" and _needs_two(d, nodes):
            d["max_bands"] = 2
        else:
            d["max_bands"] = 1
    obj = raw["objective"]
    obj["band_demands"] = demands
    obj["balanced_groups"] = []
    obj["margin_default"] = {"delta_min_s": 0.0, "coef": 1.0}
    obj["allow_window_sharing"] = sharing
    obj["pareto"] = {"num_points": 0, "relax_max": 0.0, "topk_grids": 1}
    raw["global_constraints"] = []
    raw["constraint_templates"] = []
    raw["grid"] = {"cycles_s": [CYCLE], "speed_ratios": [KAPPA]}
    raw["solver"]["heuristic"]["enabled"] = False
    raw["solver"]["num_workers"] = 1
    raw["solver"]["time_limit_s"] = 60.0
    raw["solver"]["mip_rel_gap"] = 0.0
    return ProblemInput.from_obj(raw)


def _run_one(objective: str, factor: str, fixed_boundaries: bool = False) -> dict[str, Any]:
    problem = _build_problem(objective, factor, fixed_boundaries=fixed_boundaries)
    issues = [i for i in problem.validate_all() if i.level == "error"]
    if issues:
        return {"status": "validation_error", "message": "; ".join(i.message for i in issues[:2])}
    ctx = FeasibilityFilter(problem).filter(CYCLE, KAPPA)
    model = ModelBuilder(problem, ctx).build()
    n_bin = sum(1 for t in model.vars.types if t == GVarType.BINARY)
    n_int = sum(1 for t in model.vars.types if t == GVarType.INTEGER)
    n_cons = len(model.cons.rows)
    n_nz = sum(len(row.expr.coefs) for row in model.cons.rows)
    t0 = time.perf_counter()
    backend = make_backend(problem.solver.backend)
    session = backend.create(model)
    backend.set_objective(session, model.expr_composite(), "max")
    res = backend.solve(session, SolveOptions(problem.solver.time_limit_s,
                                              problem.solver.mip_rel_gap))
    solve_s = time.perf_counter() - t0
    return {
        "status": res.status.value,
        "solve_s": solve_s,
        "objective": None if res.objective is None else float(res.objective),
        "vars": len(model.vars),
        "binary": n_bin,
        "integer": n_int,
        "cons": n_cons,
        "nnz": n_nz,
        "message": res.message,
    }


def main() -> None:
    rows = []
    for factor in FACTORS:
        for objective in OBJECTIVES:
            t0 = time.perf_counter()
            out = _run_one(objective, factor)
            out["wall_s"] = time.perf_counter() - t0
            rows.append((factor, objective, out))
            print(f"done {factor:11s} {objective:16s} "
                  f"status={out.get('status')} solve={out.get('solve_s', 0):.3f}s")

    print("\n")
    hdr = (f"{'双周期配置':<12}{'目标':<16}{'状态':<10}"
           f"{'变量':>6}{'二进制':>8}{'整数':>6}{'约束':>7}{'非零元':>8}"
           f"{'求解(s)':>10}{'墙钟(s)':>10}{'目标值':>12}")
    print(hdr)
    print("-" * len(hdr.encode("gbk", errors="ignore")))
    for factor, objective, out in rows:
        print(f"{FACTOR_LABEL[factor]:<12}{OBJECTIVE_LABEL[objective]:<16}"
              f"{str(out.get('status')):<10}"
              f"{out.get('vars', 0):>6}{out.get('binary', 0):>8}{out.get('integer', 0):>6}"
              f"{out.get('cons', 0):>7}{out.get('nnz', 0):>8}"
              f"{out.get('solve_s', 0):>10.3f}{out.get('wall_s', 0):>10.3f}"
              f"{(out.get('objective') if out.get('objective') is not None else float('nan')):>12.2f}")

    slow = sorted(rows, key=lambda r: r[2].get("solve_s", -1), reverse=True)[:3]
    print("\n最耗时 Top 3：")
    for factor, objective, out in slow:
        print(f"  {FACTOR_LABEL[factor]} + {OBJECTIVE_LABEL[objective]} "
              f": solve={out.get('solve_s', 0):.3f}s "
              f"binary={out.get('binary', 0)} cons={out.get('cons', 0)} "
              f"nnz={out.get('nnz', 0)}")


if __name__ == "__main__":
    main()
