"""教学小案例：用来复现 `docs/MILP建模教学-绿波篇.md` §8 的所有数字。

四个案例都是 3 路口（I1→I2→I3）、C=100s、单方案、边界固定：

1. 单方向带 + φ 全自由：带宽 = 链上最窄绿窗（φ 可以自由把窗口挪到带的位置上）；
2. 窗口选择：I1 有两个绿段（5s / 50s），模型会挑能用的那一段（各路口**独立**挑）；
3. 双向冲突：两个方向共享每个路口的 φ，工程上真正难的地方；
4. margin：**每个绿段**的左右边界各留 δ_min 秒 → 可用宽度 = 绿窗宽 − 2δ_min。

用法：
    python examples/tutorial_small_cases.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from greenwave.model.builder import ModelBuilder  # noqa: E402
from greenwave.preprocess.feasibility import FeasibilityFilter  # noqa: E402
from greenwave.report.solution import SolutionDecoder  # noqa: E402
from greenwave.schema.models import ProblemInput  # noqa: E402
from greenwave.solve import make_backend  # noqa: E402
from greenwave.solve.backend_base import SolveOptions  # noqa: E402

C = 100.0
SPEED = 15.0


def segment(a: str, b: str, distance_m: float = 300.0) -> dict:
    return {"from": a, "to": b, "distance_m": distance_m,
            "speed_up_mps": SPEED, "speed_down_mps": SPEED}


def _segments(mask: list[bool], *, margin_s: float = 0.0, coef: float = 1.0,
              hard_min_s: float = 0.0) -> list[dict]:
    """绿掩码 -> 对象列表；margin 只挂在每个绿段（颜色变化边界）的两端。"""
    segs = [{"green": bool(x)} for x in mask]
    if margin_s <= 0 and hard_min_s <= 0:
        return segs
    spec = {"delta_min_s": margin_s, "coef": coef, "hard_min_s": hard_min_s}
    i = 0
    while i < len(mask):
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(mask) and mask[j + 1]:
            j += 1
        segs[i]["margin"] = {"start": dict(spec)}
        segs[j].setdefault("margin", {})["end"] = dict(spec)
        i = j + 1
    return segs


def plan(pid: str, iid: str, boundaries: list[float], mask: list[bool], *,
         margin_s: float = 0.0, coef: float = 1.0, hard_min_s: float = 0.0) -> dict:
    dirs = {d: {"boundaries": [{"pos_r": x} for x in boundaries],
                "green_mask": _segments(mask, margin_s=margin_s, coef=coef,
                                        hard_min_s=hard_min_s)}
            for d in ("up", "down")}
    return {"id": pid, "intersection": iid, "double_cycle": False,
            "min_phase_s": 0.001, "directions": dirs,
            "hard_constraints": [], "soft_constraints": []}


def demand(did: str, direction: str, nodes: list[str], *, weight: float = 1.0) -> dict:
    return {"id": did, "type": "green", "direction": direction, "nodes": list(nodes),
            "max_bands": 1, "weight": weight}


def build(plans: list[dict], demands: list[dict], *, distance_m: float = 300.0
          ) -> ProblemInput:
    return ProblemInput.from_obj({
        "corridor": {"corridor_id": "tutorial", "intersections": ["I1", "I2", "I3"],
                     "segments": [segment("I1", "I2", distance_m),
                                  segment("I2", "I3", distance_m)]},
        "plans": plans,
        "objective": {"band_demands": demands, "balanced_groups": [],
                      "pareto": {"num_points": 0, "relax_max": 0.0, "topk_grids": 1},
                      "allow_window_sharing": False},
        "grid": {"cycles_s": [C], "speed_ratios": [1.0]},
        "solver": {"backend": "scipy", "time_limit_s": 60.0, "mip_rel_gap": 0.0,
                   "num_workers": 1, "heuristic": {"enabled": False}},
        "global_constraints": [], "constraint_templates": []})


def solve(problem: ProblemInput, title: str) -> None:
    ctx = FeasibilityFilter(problem).filter(C, 1.0)
    widths = {iid: [round((w.end_r - w.start_r) * C, 2) for w in p.effective_windows("up")]
              for iid, plans in sorted(ctx.feasible_plans.items()) for p in plans}
    model = ModelBuilder(problem, ctx).build()
    backend = make_backend(problem.solver.backend)
    session = backend.create(model)
    backend.set_objective(session, model.expr_composite(), "max")
    res = backend.solve(session, SolveOptions(problem.solver.time_limit_s, 0.0))
    sol = SolutionDecoder(model).decode(res.values)
    print(f"--- {title}")
    print(f"    各路口绿窗宽度(s): {widths}")
    print(f"    status={res.status.value}  composite={sol.composite_s:.3f}  "
          f"phi=" + ", ".join(f"{k}={v:.1f}" for k, v in sorted(sol.offsets_s.items())))
    if not sol.bands:
        print("    （没有宽度 > 0 的带）")
    for band in sol.bands:
        b = problem.demand(band.demand)
        print(f"    {band.demand}: β={band.width_s:6.2f}s u={band.start_s:6.2f} "
              f"用窗口槽={dict(band.windows)} n={dict(band.wraps)}")
        for j, node in enumerate(b.nodes):
            T = model.meta.green[band.demand].table.cum_T_s[j]
            w = band.windows[node]
            n = band.wraps.get(node, 0)
            gw = model.meta.effective[(node, b.direction)]
            ws = sol.offsets_s[node] + gw.s[w].evaluate(res.values) + n * C
            we = sol.offsets_s[node] + gw.e[w].evaluate(res.values) + n * C
            start = band.start_s + T
            print(f"      {node}: 到达 {start:6.2f} 离开 {start + band.width_s:6.2f}"
                  f"  窗口 [{ws:6.2f}, {we:6.2f}] 槽{w} n={n}"
                  f"  左余 {start - ws:5.2f} 右余 {we - start - band.width_s:5.2f}")
    print()


def case_1_single_band() -> None:
    plans = [plan("I1_P1", "I1", [0.6], [True, False]),
             plan("I2_P1", "I2", [0.5], [False, True]),
             plan("I3_P1", "I3", [0.7], [True, False])]
    solve(build(plans, [demand("GW_up", "up", ["I1", "I2", "I3"])]),
          "案例 1：单方向带，φ 全自由 → β = 最窄绿窗 = 50s")


def case_2_window_choice() -> None:
    plans = [plan("I1_P1", "I1", [0.05, 0.4, 0.9], [True, False, True, False]),
             plan("I2_P1", "I2", [0.3], [False, True]),
             plan("I3_P1", "I3", [0.7], [True, False])]
    solve(build(plans, [demand("GW_up", "up", ["I1", "I2", "I3"])]),
          "案例 2：I1 有 5s / 50s 两段绿 → 模型挑槽 2（各路口独立挑）")


def case_3_bidirectional() -> None:
    print("=== 案例 3：双向冲突（每个方向各一条跨全线带，权重都 1）===")
    print(f"{'绿窗宽':>7}{'单段行程':>9}{'Σβ 最优':>9}{'β_up':>8}{'β_down':>8}")
    for boundary in (0.5, 0.7):
        for distance in (300.0, 100.0):
            plans = [plan(f"I{i}_P1", f"I{i}", [boundary], [True, False]) for i in (1, 2, 3)]
            problem = build(plans, [demand("GW_up", "up", ["I1", "I2", "I3"]),
                                    demand("GW_down", "down", ["I3", "I2", "I1"])],
                            distance_m=distance)
            ctx = FeasibilityFilter(problem).filter(C, 1.0)
            model = ModelBuilder(problem, ctx).build()
            backend = make_backend(problem.solver.backend)
            session = backend.create(model)
            backend.set_objective(session, model.expr_composite(), "max")
            res = backend.solve(session, SolveOptions(problem.solver.time_limit_s, 0.0))
            sol = SolutionDecoder(model).decode(res.values)
            got = {b.demand: b.width_s for b in sol.bands}
            print(f"{boundary * C:>7.0f}{distance / SPEED:>9.2f}{sol.composite_s:>9.2f}"
                  f"{got.get('GW_up', 0.0):>8.2f}{got.get('GW_down', 0.0):>8.2f}")
    print("  → 绿窗越宽、行程越短，两个方向越能同时拿到带宽；")
    print("    绿窗 50s / 行程 20s 时几乎完全互斥（Σβ = 一个绿窗的宽度）。\n")


def case_4_margin() -> None:
    plans = [plan(f"I{i}_P1", f"I{i}", [0.5], [True, False], margin_s=5.0)
             for i in (1, 2, 3)]
    solve(build(plans, [demand("GW_up", "up", ["I1", "I2", "I3"]),
                        demand("GW_down", "down", ["I3", "I2", "I1"])]),
          "案例 4：双向 + 每个绿段 margin(δmin=5s, coef=1) → 可用宽度 50−2×5 = 40s")


if __name__ == "__main__":
    case_1_single_band()
    case_2_window_choice()
    case_3_bidirectional()
    case_4_margin()
