"""格点搜索热力图 + Pareto knee 时距图示例。

运行：
    python examples/plot_grid_heatmaps.py

会读取 examples/standard_problem.json，替换其中的 grid 配置，
跑一轮格点搜索，然后：

1. 生成 2×2 布局的四张热力图；
2. 选取 composite band objective / C 最大的格点；
3. 用 epsilon-constraint 扫描该格点的 Pareto 前沿；
4. 找 Pareto 前沿的 knee point；
5. 输出 knee point 对应的时距图。
"""
from __future__ import annotations

import json
import math
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from greenwave import run  # noqa: E402
from greenwave.pareto.epsilon import EpsilonConstraintScanner  # noqa: E402
from greenwave.preprocess.feasibility import FeasibilityFilter  # noqa: E402
from greenwave.report import GridHeatmapDiagram, TimeSpaceDiagram  # noqa: E402
from greenwave.schema.models import ParetoConfig, ProblemInput  # noqa: E402
from greenwave.solve import make_backend  # noqa: E402


def _non_dominated(points):
    """去掉被其他 Pareto 点支配的点：composite 更小且 loss 更大。"""
    out = []
    eps = 1e-9
    for p in points:
        dominated = False
        for q in points:
            if p is q:
                continue
            if (q.composite_actual_s >= p.composite_actual_s - eps and
                    q.intersection_loss_s <= p.intersection_loss_s + eps and
                    (q.composite_actual_s > p.composite_actual_s + eps or
                     q.intersection_loss_s < p.intersection_loss_s - eps)):
                dominated = True
                break
        if not dominated:
            out.append(p)
    return out


def _find_knee(front):
    """在 Pareto 前沿上找 knee point。

    目标空间：
    - composite 越大越好；
    - intersection_loss 越小越好。
    先归一化到 [0,1]，再找离“两个端点连线”最远的内部点。
    """
    points = [p for p in front if math.isfinite(p.composite_actual_s)
              and math.isfinite(p.intersection_loss_s)]
    if not points:
        return None
    points = _non_dominated(points)
    # 去重，避免重复点干扰距离计算。
    seen = set()
    unique = []
    for p in points:
        key = (round(p.composite_actual_s, 6), round(p.intersection_loss_s, 6))
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)
    if len(unique) <= 2:
        return max(unique, key=lambda p: p.composite_actual_s)

    comps = [p.composite_actual_s for p in unique]
    losses = [p.intersection_loss_s for p in unique]
    c_min, c_max = min(comps), max(comps)
    l_min, l_max = min(losses), max(losses)

    def norm(p):
        c = 0.0 if c_max - c_min < 1e-12 else (p.composite_actual_s - c_min) / (c_max - c_min)
        l = 0.0 if l_max - l_min < 1e-12 else (p.intersection_loss_s - l_min) / (l_max - l_min)
        return l, c  # x=loss, y=composite

    # 两个端点：最高 composite 点 和 最低 loss 点。
    a = max(unique, key=lambda p: p.composite_actual_s)
    b = min(unique, key=lambda p: p.intersection_loss_s)
    ax, ay = norm(a)
    bx, by = norm(b)
    denom = math.hypot(bx - ax, by - ay)
    if denom < 1e-12:
        return a

    best = None
    best_dist = -1.0
    for p in unique:
        px, py = norm(p)
        dist = abs((by - ay) * px - (bx - ax) * py + bx * ay - by * ax) / denom
        if dist > best_dist:
            best_dist = dist
            best = p
    return best if best is not None else a


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    problem_path = os.path.join(here, "standard_problem.json")
    with open(problem_path, "r", encoding="utf-8") as f:
        problem = json.load(f)

    # 高分辨率局部格点：周期和速度步数接近。
    n_cycles = 25
    n_speeds = 25
    cycles = [round(60.0 + 100.0 * i / (n_cycles - 1), 3)
              for i in range(n_cycles)]
    speeds = [round(0.30 + 1.20 * i / (n_speeds - 1), 4)
              for i in range(n_speeds)]
    problem["grid"] = {"cycles_s": cycles, "speed_ratios": speeds}
    problem["solver"]["num_workers"] = 1
    # 阶段 1 先只求每个格点最优；Pareto knee 稍后对选中的格点单独扫描。
    problem["objective"]["pareto"] = {"num_points": 0, "relax_max": 0.0, "topk_grids": 1}

    result = run(problem)
    out_dir = os.path.join(here, "diagrams")
    out_path = os.path.join(out_dir, "09_grid_heatmaps.png")
    GridHeatmapDiagram.render(
        result, out_path,
        title=(f"Grid Search Heatmaps ({n_cycles}×{n_speeds}): "
               f"C ∈ [{cycles[0]:g},{cycles[-1]:g}], "
               f"κ ∈ [{speeds[0]:g},{speeds[-1]:g}]"),
    )
    print(f"wrote {out_path}")

    # 打印格点结果，方便核对。
    for row in result["grid_results"]:
        print(f"C={row['cycle_s']:.3f}s κ={row['speed_ratio']:.4f} "
              f"status={row['status']} "
              f"band={row.get('bandwidth_objective_s')} "
              f"margin={row.get('negative_margin_loss_s')} "
              f"loss={row.get('negative_intersection_loss_s')}")

    # ------------------------------------------------------------------
    # 找到 composite band objective / C 最大的格点。
    # ------------------------------------------------------------------
    valid = [r for r in result["grid_results"]
             if r.get("status") == "optimal"
             and r.get("composite_best") is not None
             and float(r.get("cycle_s") or 0.0) > 0.0]
    if not valid:
        print("no valid grid point; skip Pareto knee scan")
        return
    best_row = max(valid, key=lambda r: float(r["composite_best"]) / float(r["cycle_s"]))
    best_C = float(best_row["cycle_s"])
    best_kappa = float(best_row["speed_ratio"])
    best_ratio = float(best_row["composite_best"]) / best_C
    print("\nbest composite/C grid point:")
    print(f"  C={best_C:g}s κ={best_kappa:g} "
          f"composite={best_row['composite_best']:.6f}s "
          f"composite/C={best_ratio:.6f}")

    # 对选中的格点重新构造 ctx，并做 epsilon-constraint Pareto 扫描。
    problem_obj = ProblemInput.from_obj(problem)
    ctx = FeasibilityFilter(problem_obj).filter(best_C, best_kappa)
    scanner_cfg = ParetoConfig(num_points=12, relax_max=0.3, topk_grids=1)
    base = SimpleNamespace(composite_best=float(best_row["composite_best"]),
                           cycle_s=best_C, kappa=best_kappa)
    front = EpsilonConstraintScanner(
        problem_obj, ctx, make_backend(problem_obj.solver.backend), scanner_cfg,
    ).scan(base)

    print("\nPareto front:")
    for p in front:
        print(f"  ε={p.epsilon_s:.4f}s  composite={p.composite_actual_s:.6f}s  "
              f"loss={p.intersection_loss_s:.6f}s")

    knee = _find_knee(front)
    if knee is None:
        print("no Pareto point; skip knee time-space diagram")
        return
    print("\nknee point:")
    print(f"  ε={knee.epsilon_s:.6f}s  composite={knee.composite_actual_s:.6f}s  "
          f"loss={knee.intersection_loss_s:.6f}s")

    knee_path = os.path.join(out_dir, "10_pareto_knee_time_space.png")
    TimeSpaceDiagram.render(
        knee.solution, problem_obj, ctx, knee_path,
        title=(f"Pareto Knee: C={best_C:g}s κ={best_kappa:g} "
               f"ε={knee.epsilon_s:.3f}s"),
    )
    print(f"wrote {knee_path}")


if __name__ == "__main__":
    main()
