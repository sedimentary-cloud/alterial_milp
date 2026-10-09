"""标准案例的**精细格点搜索**（整模）+ 可交互 HTML 热力图。

搜索对象（按需求固定）
----------------------
* 案例：`examples/standard_problem.json`，**无双周期 · 单方案 · 可调相位边界**；
* 目标：**上下全局带宽均衡组 × 1.0 + 上下全局绿波带宽和 × 0.5**
  （`composite = z + 0.5·(β_up + β_down)`，`z = min(β_up, β_down)`）；
* 方法：**整模**（一次性完整 MILP，δ 自由）；
* 网格：公共周期 `C ∈ [60, 150]s` × 速度倍率 `κ ∈ [0.3, 1.5]`。

口径与 `benchmark_matrix.py` 一致：`mip_rel_gap=0`、heuristic 关、单线程、
stage-1 对称破缺 + 最小带宽 1s、每个格点独立时限。单格点约 0.03s，
33×33 = 1089 个格点连建模开销约 1 分钟。

用法：
    python examples/benchmark_grid_search.py                  # 33×33
    python examples/benchmark_grid_search.py --cells 41       # 更细
    python examples/benchmark_grid_search.py --workers 4      # 并行（会干扰时间指标）
    python examples/benchmark_grid_search.py --html-only      # 只重画 HTML
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Mapping, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(HERE)
for _p in (HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bench_configs as bc  # noqa: E402
import grid_report as gr  # noqa: E402
import ts_canvas  # noqa: E402

from greenwave.errors import GWInfeasibleGridPoint  # noqa: E402
from greenwave.model.builder import ModelBuilder  # noqa: E402
from greenwave.preprocess.feasibility import FeasibilityFilter  # noqa: E402
from greenwave.report.solution import SolutionDecoder  # noqa: E402
from greenwave.solve import make_backend  # noqa: E402
from greenwave.solve.backend_base import SolveOptions  # noqa: E402

DEFAULT_OUT = os.path.join(HERE, "bench_output", "grid_search.json")
DEFAULT_HTML = os.path.join(HERE, "bench_output", "grid_search.html")

#: 固定案例：无双周期 · 单方案 · 可调边界 × 上下全局均衡组+带宽和
CASE_GROUP = "G1"
CASE_OBJECTIVE = "GW_both_global_balanced"
CASE_BOUNDARIES = "adjustable"
CASE_PLANS = 1

STAGE1_SYMMETRY = True
MIN_BAND_WIDTH_S = 1.0

#: 判定"最优格点"用的指标：**归一化目标**（composite / C，周期比例）。
#: 理由：绝对秒数会天然偏向大周期，跨 C 比较没有意义（详见 README 的实测对比）。
BEST_METRIC = "composite_objective_ratio"
_ABS_FALLBACK = "composite_s"

#: 每个格点记录的指标（键 → 说明），报告的指标选择器由它生成。
#:
#: **归一化口径**（与 matplotlib 版 `GridHeatmapDiagram` 一致）：
#: 凡是"秒为单位的带宽目标 / 损失"都要除以对应格点的周期 C，变成**周期比例**
#: （无量纲），这样横跨不同 C 才能直接比较；损失类取负号，使"越大越好"统一。
#: 命名沿用库里那套：`*_s` = 绝对秒，`*_ratio` = 除以 C 后的比例。
METRIC_NOTES = {
    # —— 归一化（周期比例）——
    "composite_objective_ratio": "目标 composite / C　（周期比例，越大越好）",
    "bandwidth_objective_ratio": "带宽奖励 Σw·β / C　（周期比例）",
    "balanced_z_ratio": "均衡组 z = min(β↑,β↓) / C　（周期比例）",
    "beta_up_ratio": "上行总带宽 β↑ / C　（周期比例）",
    "beta_down_ratio": "下行总带宽 β↓ / C　（周期比例）",
    "beta_both_ratio": "(β↑+β↓) / C　（周期比例）",
    "negative_margin_loss_ratio": "−margin 短缺 / C　（取负号，越大越好；本目标已关闭 = 0）",
    # —— 绝对秒 ——
    "composite_s": "目标 composite（秒，绝对值）",
    "reward_s": "带宽奖励 Σw·β（秒）",
    "balanced_z_s": "均衡组 z（秒）",
    "beta_up_s": "上行总带宽 β↑（秒）",
    "beta_down_s": "下行总带宽 β↓（秒）",
    "margin_loss_s": "margin 短缺（秒，已关闭 = 0）",
    # —— 规模与开销 ——
    "n_bands": "存在的带槽条数",
    "wall_s": "建模 + 求解墙钟（秒，时间开销不归一化）",
}


#: 旧默认的步长（C 取 60..150 共 33 点、κ 取 0.3..1.5 共 33 点）
BASE_CYCLE_STEP = (150.0 - 60.0) / 32.0      # 2.8125 s
BASE_KAPPA_STEP = (1.5 - 0.3) / 32.0         # 0.0375
DEFAULT_STEP_SCALE = 0.75


def _norm(rec: dict[str, Any]) -> None:
    """把"秒为单位的带宽目标 / 损失"除以周期 C，补上 `*_ratio`（就地修改）。

    口径与 `greenwave/report/heatmap.py` 的 `GridHeatmapDiagram` 一致：
    损失取负号（`negative_*`），使"越大越好"统一。
    """
    C = float(rec.get("cycle_s") or 0.0)
    if C <= 0:
        return
    pairs = (("composite_s", "composite_objective_ratio"),
             ("reward_s", "bandwidth_objective_ratio"),
             ("balanced_z_s", "balanced_z_ratio"),
             ("beta_up_s", "beta_up_ratio"),
             ("beta_down_s", "beta_down_ratio"),
             ("beta_both_s", "beta_both_ratio"))
    for src, dst in pairs:
        v = rec.get(src)
        if v is not None:
            rec[dst] = float(v) / C
    ml = rec.get("margin_loss_s")
    if ml is not None:
        rec["negative_margin_loss_ratio"] = -float(ml) / C


def round_floats(obj: Any, digits: int = 6) -> Any:
    """递归把 payload 里的浮点收成 ``digits`` 位小数（JSON/HTML 体积能小一半）。"""
    if isinstance(obj, float):
        return round(obj, digits)
    if isinstance(obj, dict):
        return {k: round_floats(v, digits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [round_floats(v, digits) for v in obj]
    return obj


def grid_values_step(lo: float, hi: float, step: float) -> list[float]:
    """按**目标步长**取点：点数 = ceil((hi-lo)/step) + 1（步长会取整到尽量接近）。"""
    if step <= 0:
        raise ValueError("--step 必须为正")
    cells = max(2, int(math.ceil((float(hi) - float(lo)) / float(step))) + 1)
    return grid_values(lo, hi, cells)


def grid_values(lo: float, hi: float, cells: int) -> list[float]:
    if cells < 2:
        raise ValueError("--cells 至少 2")
    step = (float(hi) - float(lo)) / (cells - 1)
    return [round(float(lo) + i * step, 6) for i in range(cells)]


#: 每次运行只算一次：时距图静态上下文（含 δ 的扁平顺序）
_TS_CTX: list[Any] = [None]


def solve_point(problem, cycle_s: float, kappa: float, *,
                time_limit_s: float, mip_rel_gap: float) -> dict[str, Any]:
    """一个格点：过滤 → 建模 → 求解 → 解码 → 记录指标。"""
    t0 = time.perf_counter()
    rec: dict[str, Any] = {"cycle_s": float(cycle_s), "kappa": float(kappa)}
    try:
        ctx = FeasibilityFilter(problem).filter(cycle_s, kappa)
    except GWInfeasibleGridPoint as exc:
        rec.update(status="infeasible", message=str(exc)[:200],
                   wall_s=time.perf_counter() - t0)
        return rec
    except Exception as exc:                          # noqa: BLE001
        rec.update(status="error", message=f"{type(exc).__name__}: {exc}"[:200],
                   wall_s=time.perf_counter() - t0)
        return rec

    try:
        model = ModelBuilder(problem, ctx, stage1_symmetry=STAGE1_SYMMETRY,
                             min_band_width_s=MIN_BAND_WIDTH_S).build()
        backend = make_backend(problem.solver.backend)
        session = backend.create(model)
        backend.set_objective(session, model.expr_composite(), "max")
        res = backend.solve(session, SolveOptions(time_limit_s, mip_rel_gap))
        wall = time.perf_counter() - t0
        rec.update(status=res.status.value, solver_s=float(res.time_s),
                   message=(res.message or "")[:200])
        if res.values is None:
            rec["wall_s"] = wall
            return rec
        sol = SolutionDecoder(model).decode(res.values)
        beta_up = sum(b.width_s for b in sol.bands if b.demand == "GW_up_full")
        beta_down = sum(b.width_s for b in sol.bands if b.demand == "GW_down_full")
        # 时距图所需的"每点解"（共享契约 ts_canvas.draw_payload）
        rec["ts"] = ts_canvas.draw_payload(sol, _TS_CTX[0])
        rec.update(
            composite_s=float(sol.composite_s),
            reward_s=float(sol.band_reward_s),
            balanced_z_s=float(sol.balanced_groups.get("BG_up_down_global", 0.0)),
            beta_up_s=float(beta_up),
            beta_down_s=float(beta_down),
            beta_both_s=float(beta_up + beta_down),
            margin_loss_s=float(sol.margin_loss_s),
            n_bands=int(len(sol.bands)),
            model_vars=int(len(model.vars)),
            model_cons=int(len(model.cons.rows)),
            dropped_plans=int(len(ctx.dropped)),
        )
        _norm(rec)
        rec["wall_s"] = wall
        return rec
    except Exception as exc:                          # noqa: BLE001
        rec.update(status="error", message=f"{type(exc).__name__}: {exc}"[:200],
                   wall_s=time.perf_counter() - t0)
        return rec


def _worker(args):
    problem, cycle_s, kappa, time_limit_s, mip_rel_gap = args
    return solve_point(problem, cycle_s, kappa,
                       time_limit_s=time_limit_s, mip_rel_gap=mip_rel_gap)


def run_grid(problem, cycles: Sequence[float], kappas: Sequence[float], *,
             time_limit_s: float, mip_rel_gap: float, workers: int,
             quiet: bool = False) -> list[dict[str, Any]]:
    jobs = [(problem, c, k, float(time_limit_s), float(mip_rel_gap))
            for k in kappas for c in cycles]
    total = len(jobs)
    points: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    if workers and workers > 1:
        with ProcessPoolExecutor(max_workers=int(workers)) as pool:
            for i, rec in enumerate(pool.map(_worker, jobs, chunksize=8), 1):
                points.append(rec)
                if not quiet and (i % 50 == 0 or i == total):
                    _progress(i, total, t0, rec)
    else:
        for i, job in enumerate(jobs, 1):
            rec = _worker(job)
            points.append(rec)
            if not quiet and (i % 50 == 0 or i == total):
                _progress(i, total, t0, rec)
    return points


def _progress(i: int, total: int, t0: float, rec: Mapping[str, Any]) -> None:
    dt = time.perf_counter() - t0
    eta = dt / i * (total - i)
    obj = rec.get("composite_s")
    print(f"[{i:5d}/{total}] C={rec['cycle_s']:6.1f}s κ={rec['kappa']:.3f} "
          f"{rec['status']:<10} obj={('-' if obj is None else format(obj, '.3f')):>8} "
          f"已用 {dt:5.1f}s ETA {eta:5.1f}s", flush=True)


def _rank(p: Mapping[str, Any]) -> float:
    """排序/择优的键：优先归一化目标，缺了才退回绝对秒。"""
    v = p.get(BEST_METRIC)
    if v is None:
        v = p.get(_ABS_FALLBACK)
    return -math.inf if v is None else float(v)


def summarise(points: Sequence[Mapping[str, Any]], *, wall_s: float,
              cells: int, cycles: Sequence[float], kappas: Sequence[float],
              time_limit_s: float, mip_rel_gap: float, workers: int) -> dict[str, Any]:
    ok = [p for p in points if p.get("composite_s") is not None]
    ok.sort(key=_rank, reverse=True)          # 按归一化目标择优
    status_count: dict[str, int] = {}
    for p in points:
        status_count[p["status"]] = status_count.get(p["status"], 0) + 1
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "wall_s": wall_s,
        "case": {
            "group": CASE_GROUP, "objective": CASE_OBJECTIVE,
            "boundaries": CASE_BOUNDARIES, "n_plans": CASE_PLANS,
            "spec_id": f"{CASE_GROUP}|{CASE_OBJECTIVE}|S|p{CASE_PLANS}|adj",
            "objective_label": bc.objective_spec(CASE_OBJECTIVE).label,
            "objective_desc": bc.objective_spec(CASE_OBJECTIVE).desc,
        },
        "config": {
            "cells": int(cells), "cycle_min": cycles[0], "cycle_max": cycles[-1],
            "kappa_min": kappas[0], "kappa_max": kappas[-1],
            "n_cycles": len(cycles), "n_kappas": len(kappas),
            "time_limit_s": time_limit_s, "mip_rel_gap": mip_rel_gap,
            "backend": "scipy(HiGHS)", "heuristic": False, "num_workers": workers,
            "stage1_symmetry": STAGE1_SYMMETRY, "min_band_width_s": MIN_BAND_WIDTH_S,
            "method": "整模",
            "cycle_step": cycles[1] - cycles[0] if len(cycles) > 1 else 0.0,
            "kappa_step": kappas[1] - kappas[0] if len(kappas) > 1 else 0.0,
            "step_scale": DEFAULT_STEP_SCALE,
            "normalized_metrics": [k for k in METRIC_NOTES if k.endswith("_ratio")],
        },
        "status_count": status_count,
        "n_feasible": len(ok),
        "best_metric": BEST_METRIC,
        "top": [
            {k: p.get(k) for k in ("cycle_s", "kappa", BEST_METRIC, "composite_s",
                                   "reward_s", "balanced_z_s", "beta_up_s",
                                   "beta_down_s", "beta_both_s", "wall_s", "status")}
            for p in ok[:15]
        ],
        "metrics": METRIC_NOTES,
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="标准案例精细格点搜索（整模）")
    ap.add_argument("--cells", type=int, default=0,
                    help="每个轴固定取多少点；给 0（默认）则按 --step-scale 决定点数")
    ap.add_argument("--step-scale", type=float, default=DEFAULT_STEP_SCALE,
                    help="步长相对旧默认（C 2.8125s / κ 0.0375）的比例，默认 0.75 即加密到 3/4")
    ap.add_argument("--cycle-step", type=float, default=None, help="直接指定 C 步长（覆盖 --step-scale）")
    ap.add_argument("--kappa-step", type=float, default=None, help="直接指定 κ 步长")
    ap.add_argument("--cycle-min", type=float, default=60.0)
    ap.add_argument("--cycle-max", type=float, default=180.0)
    ap.add_argument("--kappa-min", type=float, default=0.3)
    ap.add_argument("--kappa-max", type=float, default=1.5)
    ap.add_argument("--time-limit", type=float, default=10.0, help="单格点求解时限（秒）")
    ap.add_argument("--mip-rel-gap", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=1, help="并行进程数（默认 1，串行）")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--html", default=DEFAULT_HTML)
    ap.add_argument("--html-only", action="store_true", help="不求解，只重画 HTML")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    if args.html_only:
        old = gr.load_payload(args.out)
        gr.write_report(args.out, args.html, old)
        print(f"[html-only] 报告 -> {args.html}")
        return 0

    if args.cells and args.cells >= 2:
        cycles = grid_values(args.cycle_min, args.cycle_max, args.cells)
        kappas = grid_values(args.kappa_min, args.kappa_max, args.cells)
    else:
        c_step = (args.cycle_step if args.cycle_step
                  else BASE_CYCLE_STEP * args.step_scale)
        k_step = (args.kappa_step if args.kappa_step
                  else BASE_KAPPA_STEP * args.step_scale)
        cycles = grid_values_step(args.cycle_min, args.cycle_max, c_step)
        kappas = grid_values_step(args.kappa_min, args.kappa_max, k_step)
    spec = bc.CaseSpec(CASE_GROUP, CASE_OBJECTIVE, False, CASE_PLANS, CASE_BOUNDARIES)
    build = bc.build_case(spec, time_limit_s=args.time_limit,
                          grid=(cycles, kappas))
    problem = build.problem

    ts_ctx = ts_canvas.draw_context(problem)
    _TS_CTX[0] = ts_ctx
    print(f"案例 {spec.id}（{bc.objective_spec(CASE_OBJECTIVE).label}）· 整模")
    c_step = cycles[1] - cycles[0] if len(cycles) > 1 else 0.0
    k_step = kappas[1] - kappas[0] if len(kappas) > 1 else 0.0
    print(f"网格 {len(cycles)}×{len(kappas)} = {len(cycles) * len(kappas)} 个格点："
          f"C ∈ [{cycles[0]:g}, {cycles[-1]:g}]s（步长 {c_step:.4g}s = 基准的 "
          f"{c_step / BASE_CYCLE_STEP:.0%}），κ ∈ [{kappas[0]:g}, {kappas[-1]:g}]"
          f"（步长 {k_step:.4g} = 基准的 {k_step / BASE_KAPPA_STEP:.0%}）")
    t0 = time.perf_counter()
    points = run_grid(problem, cycles, kappas,
                      time_limit_s=args.time_limit, mip_rel_gap=args.mip_rel_gap,
                      workers=args.workers, quiet=args.quiet)
    wall = time.perf_counter() - t0

    payload = {
        "grid": {"cycles": cycles, "kappas": kappas},
        "ts_ctx": ts_ctx,
        "points": points,                       # 行优先：κ 主序、C 次序
        "summary": summarise(points, wall_s=wall, cells=args.cells, cycles=cycles,
                             kappas=kappas, time_limit_s=args.time_limit,
                             mip_rel_gap=args.mip_rel_gap, workers=args.workers),
    }
    payload = round_floats(payload, 6)
    gr.write_json(args.out, payload)
    gr.write_report(args.out, args.html, payload)
    s = payload["summary"]
    best = s["top"][0] if s["top"] else None
    print(f"\n{len(points)} 个格点，用时 {wall:.1f}s，状态分布 {s['status_count']}")
    if best:
        print(f"最优格点（按 {BEST_METRIC}）：C={best['cycle_s']:.2f}s κ={best['kappa']:.3f} "
              f"composite/C={best[BEST_METRIC]:.4f} "
              f"(composite={best['composite_s']:.2f}s，β↑={best['beta_up_s']:.2f} "
              f"β↓={best['beta_down_s']:.2f} z={best['balanced_z_s']:.2f})")
    print(f"数据 -> {args.out}\n热力图 -> {args.html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
