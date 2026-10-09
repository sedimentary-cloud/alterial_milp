"""标准案例全矩阵性能测试：整模 vs 分阶段。

矩阵（每个案例都跑 ≥2 次，单次求解上限默认 20s）
-------------------------------------------------
分组                          边界模式            对比
G1 无双周期·单方案(1.1–1.4)   fixed              整模 vs stage-1（分解/DP）
G2 无双周期·多方案 k=2..5     fixed              同上
G3 双周期·单方案              adjustable         一次性完整 MILP（δ 自由）
G4 双周期·多方案 k=2..5                           vs multistage 全链路
                              ↑ fixed 与 adjustable 各跑一遍，即 1.1–1.4 / 1.5–1.8

目标（6 种，见 ``bench_configs.OBJECTIVES``）：
    上下全局绿波带宽和 / 全局均衡组×1+带宽和×0.5 / 双向窗口绿波 /
    上绿逐对+下红逐对 / 双向红波 / 上全局绿+下红逐对

用法：
    python examples/benchmark_matrix.py                       # 跑全矩阵
    python examples/benchmark_matrix.py --limit 4             # 先试 4 个案例
    python examples/benchmark_matrix.py --resume              # 断点续跑
    python examples/benchmark_matrix.py --html-only           # 只重生成 HTML
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import sys
import time
from dataclasses import replace
from typing import Any, Iterable, Mapping, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(HERE)
for _p in (HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bench_configs as bc  # noqa: E402
import bench_report as br  # noqa: E402
import ts_canvas as tsc  # noqa: E402

from greenwave.model.builder import ModelBuilder  # noqa: E402
from greenwave.multistage import (solve_fixed_boundaries,  # noqa: E402
                                  solve_multistage, stage1_path_label)
from greenwave.preprocess.feasibility import FeasibilityFilter  # noqa: E402
from greenwave.report.solution import SolutionDecoder  # noqa: E402
from greenwave.solve import make_backend  # noqa: E402
from greenwave.solve.backend_base import SolveOptions  # noqa: E402

DEFAULT_OUT = os.path.join(HERE, "bench_output", "bench_matrix.json")
DEFAULT_HTML = os.path.join(HERE, "bench_output", "bench_matrix.html")
#: 时空图默认放在 JSON 同级目录下的这个子目录里，每个案例一个子目录
DEFAULT_DIAGRAM_DIR = "diagrams"

METHODS = ("monolith", "staged")
METHOD_LABEL = {"monolith": "整模", "staged": "分阶段"}
STAGE1_SYMMETRY = True
MIN_BAND_WIDTH_S = 1.0
#: 标题要写中文，显式找一个 CJK 字体文件（matplotlib 在本机不做逐字回退，
#: 挂 font.sans-serif 没用，必须把 FontProperties 传给 set_title）
_CJK_FONT_PATHS = (
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "/System/Library/Fonts/PingFang.ttc",
)
_CJK_PROP = "unset"

#: 标题里没有 CJK 字体时，把路径标签换成 ASCII（否则出方框）
_PATH_ASCII = {"整模": "monolith", "语义分解": "decomposed", "方案枚举DP": "plan-DP"}


def ascii_path_label(path: str) -> str:
    out = str(path or "-")
    for zh, en in _PATH_ASCII.items():
        out = out.replace(zh, en)
    return out
#: 标题里真正会用到的字符（中文方法名 + 拉丁/数字/符号）；字体覆盖不全就不用中文
_TITLE_PROBE = "整模分阶段·0123456789AaPp_.-=|"


def cjk_title_font():
    """返回**能画全标题**的 CJK 字体；找不到就返回 None（标题退化成 ASCII）。

    注意：matplotlib 不做逐字回退——把一个只有中文的 fallback 字体塞给标题，
    拉丁字母和数字会变成方框（本机 `DroidSansFallbackFull.ttf` 就是这种纯 CJK 字体）。
    所以这里逐个候选字体查 cmap，只有"中文 + 拉丁 + 数字"全覆盖才采用。
    """
    global _CJK_PROP
    if _CJK_PROP == "unset":
        _CJK_PROP = None
        try:
            from matplotlib import font_manager
            from matplotlib.ft2font import FT2Font
            for path in _CJK_FONT_PATHS:
                if not os.path.exists(path):
                    continue
                try:
                    cmap = FT2Font(path).get_charmap()
                except Exception:                     # noqa: BLE001
                    continue
                if all(ord(ch) in cmap for ch in _TITLE_PROBE):
                    _CJK_PROP = font_manager.FontProperties(fname=path)
                    break
        except Exception:                             # noqa: BLE001
            _CJK_PROP = None
    return _CJK_PROP


# ---------------------------------------------------------------------------
# 单次求解
# ---------------------------------------------------------------------------

def _rel_gap(objective: float | None, bound: float | None) -> float | None:
    if objective is None or bound is None:
        return None
    return (float(bound) - float(objective)) / max(abs(float(objective)), 1e-9)


def _record(wall_s: float, objective: float | None, status: str, path: str,
            *,
            solver_s: float | None = None, bound: float | None = None,
            message: str = "", **extra: Any) -> dict[str, Any]:
    rec = {
        "wall_s": float(wall_s),
        "solver_s": None if solver_s is None else float(solver_s),
        "status": str(status),
        "objective": None if objective is None else float(objective),
        "bound": None if bound is None else float(bound),
        "rel_gap": _rel_gap(objective, bound),
        "path": path,
        "message": (message or "").strip()[:240],
    }
    rec.update(extra)
    return rec


def _solve_full_milp(problem, ctx, time_limit_s: float, mip_rel_gap: float):
    """一次性完整 MILP（δ 自由，所有整数+连续变量一起解）。"""
    model = ModelBuilder(problem, ctx, stage1_symmetry=STAGE1_SYMMETRY,
                         min_band_width_s=MIN_BAND_WIDTH_S).build()
    backend = make_backend(problem.solver.backend)
    session = backend.create(model)
    backend.set_objective(session, model.expr_composite(), "max")
    res = backend.solve(session, SolveOptions(time_limit_s, mip_rel_gap))
    sol = SolutionDecoder(model).decode(res.values) if res.values is not None else None
    return model, sol, res


def run_monolith(spec: bc.CaseSpec, build: bc.CaseBuild, ctx, *,
                 time_limit_s: float, mip_rel_gap: float) -> dict[str, Any]:
    problem = build.problem
    t0 = time.perf_counter()
    if spec.boundaries == "fixed":
        out = solve_fixed_boundaries(problem, ctx, {}, time_limit_s=time_limit_s,
                                     mip_rel_gap=mip_rel_gap, decompose=False)
        wall = time.perf_counter() - t0
        if out is None:
            return _record(wall, None, "no_solution", "整模")
        sol, _prior, _model, res = out
        return _record(wall, sol.composite_s, res.status.value, "整模",
                       solver_s=res.time_s, bound=res.bound, message=res.message,
                       _solution=sol)
    model, sol, res = _solve_full_milp(problem, ctx, time_limit_s, mip_rel_gap)
    wall = time.perf_counter() - t0
    return _record(wall, None if sol is None else sol.composite_s, res.status.value,
                   "整模", solver_s=res.time_s, bound=res.bound,
                   message=res.message, _solution=sol)


def run_staged(spec: bc.CaseSpec, build: bc.CaseBuild, ctx, *,
               time_limit_s: float, mip_rel_gap: float,
               max_rounds: int) -> dict[str, Any]:
    problem = build.problem
    t0 = time.perf_counter()
    if spec.boundaries == "fixed":
        # δ 被冻结：没有 stage-2 可言，"分阶段"就是 stage-1（内部分解 / DP）。
        out = solve_fixed_boundaries(problem, ctx, {}, time_limit_s=time_limit_s,
                                     mip_rel_gap=mip_rel_gap, decompose=True)
        wall = time.perf_counter() - t0
        if out is None:
            return _record(wall, None, "no_solution", "整模")
        sol, _prior, _model, res = out
        return _record(wall, sol.composite_s, res.status.value,
                       stage1_path_label(res.message), solver_s=res.time_s,
                       bound=res.bound, message=res.message, _solution=sol)
    cfg = replace(problem.solver.multistage, enabled=True, max_rounds=int(max_rounds),
                  restarts=1, decompose_stage1=True,
                  stage1_time_limit_s=time_limit_s, stage2_time_limit_s=time_limit_s)
    res = solve_multistage(problem, ctx, cfg)
    wall = time.perf_counter() - t0
    paths = tuple(dict.fromkeys(res.stage1_paths)) or ("整模",)
    return _record(wall, res.objective, res.status, " + ".join(paths),
                   solver_s=res.time_s, message=res.message,
                   stage1_s=res.stage1_time_s, stage2_s=res.stage2_time_s,
                   rounds=res.rounds, history=list(res.history),
                   stage1_paths=list(res.stage1_paths), _solution=res.solution)


def _diagram_run(runs: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """挑一张最有代表性的解：优先"有解且目标值最好"的那一次重复。"""
    with_sol = [r for r in runs if r.get("_solution") is not None]
    if not with_sol:
        return None
    with_obj = [r for r in with_sol if r.get("objective") is not None]
    pool = with_obj or with_sol
    return max(pool, key=lambda r: (r.get("objective") or -math.inf,
                                    -float(r.get("wall_s") or 0.0)))


def build_ts_payload(case_id: str, ts_ctx: Mapping[str, Any],
                     rec: dict[str, Any]) -> None:
    """给两种方法各压一份时距图载荷（挂在 ``rec["ts_data"]`` 上）。

    这份载荷让报告能在浏览器里**就地渲染**时距图，从而不必再输出 240 张 PNG。
    """
    data: dict[str, Any] = {}
    for method in METHODS:
        run = _diagram_run(rec.get("runs", {}).get(method, ()))
        if run is None:
            continue
        sol = run["_solution"]
        # 结构遵循共享契约：ts.draw(rec, title) 需要 rec.ts + rec.cycle_s/kappa
        data[method] = {
            "ts": tsc.draw_payload(sol, ts_ctx),
            "cycle_s": bc.CYCLE,
            "kappa": bc.KAPPA,
            "method": method,
            "method_label": METHOD_LABEL[method],
            "objective": run.get("objective"),
            "status": run.get("status"),
            "path": run.get("path", "-"),
            "wall_s": run.get("wall_s"),
        }
    if data:
        rec["ts_data"] = data


def _optional_png(case_id: str, problem, ctx, rec: dict[str, Any], *,
                  out_json: str, diagram_dir: str) -> dict[str, str]:
    """可选：仍然把每案例两张 PNG 写到磁盘（``--png-diagrams`` 时才走这条路）。"""
    try:
        from greenwave.report.diagram import TimeSpaceDiagram
    except Exception as exc:                          # noqa: BLE001
        rec.setdefault("notes", []).append(f"跳过时空图：无法导入绘图模块（{exc}）")
        return {}
    base_abs = os.path.join(os.path.dirname(os.path.abspath(out_json)), diagram_dir)
    sub = safe_case_dir(case_id)
    out: dict[str, str] = {}
    for method in METHODS:
        run = _diagram_run(rec.get("runs", {}).get(method, ()))
        if run is None:
            continue
        rel = os.path.join(diagram_dir, sub, f"{method}.png")
        abs_path = os.path.join(base_abs, sub, f"{method}.png")
        obj = run.get("objective")
        cjk = cjk_title_font()
        label = METHOD_LABEL[method] if cjk is not None else method
        path_txt = run.get("path", "-")
        if cjk is None:
            path_txt = ascii_path_label(path_txt)
        title = (f"{case_id}\n{label} · "
                 f"obj={('-' if obj is None else format(obj, '.2f'))} · "
                 f"{float(run.get('wall_s') or 0.0):.2f}s · {run.get('status')} · "
                 f"{path_txt}")
        try:
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            TimeSpaceDiagram.render(run["_solution"], problem, ctx, abs_path,
                                    title=title, title_font=cjk)
            out[method] = rel.replace(os.sep, "/")
        except Exception as exc:                      # noqa: BLE001
            rec.setdefault("notes", []).append(
                f"{method} 时空图失败：{type(exc).__name__}: {exc}")
    return out


# ---------------------------------------------------------------------------
# 案例
# ---------------------------------------------------------------------------

def run_case(spec: bc.CaseSpec, *, reps: int, time_limit_s: float,
             mip_rel_gap: float, max_rounds: int, out_json: str = DEFAULT_OUT,
             diagrams: bool = True, png_diagrams: bool = False,
             diagram_dir: str = DEFAULT_DIAGRAM_DIR) -> dict[str, Any]:
    t_case = time.perf_counter()
    rec: dict[str, Any] = {
        "id": spec.id,
        "group": spec.group,
        "group_label": bc.group_spec(spec.group).label,
        "objective": spec.objective,
        "objective_label": bc.objective_spec(spec.objective).label,
        "objective_desc": bc.objective_spec(spec.objective).desc,
        "double_cycle": bool(spec.double_cycle),
        "n_plans": int(spec.n_plans),
        "boundaries": spec.boundaries,
        "runs": {m: [] for m in METHODS},
    }
    try:
        build = bc.build_case(spec, time_limit_s=time_limit_s)
        ctx = FeasibilityFilter(build.problem).filter(bc.CYCLE, bc.KAPPA)
        rec["plan_span"] = build.plan_span
        rec["notes"] = list(build.notes)
        rec["structure"] = bc.case_structure(build.problem)
        ts_ctx = tsc.draw_context(build.problem)
        rec["ts_ctx"] = ts_ctx
    except Exception as exc:                          # noqa: BLE001
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["wall_s"] = time.perf_counter() - t_case
        return rec

    for method in METHODS:
        for rep in range(int(reps)):
            try:
                if method == "monolith":
                    out = run_monolith(spec, build, ctx, time_limit_s=time_limit_s,
                                       mip_rel_gap=mip_rel_gap)
                else:
                    out = run_staged(spec, build, ctx, time_limit_s=time_limit_s,
                                     mip_rel_gap=mip_rel_gap, max_rounds=max_rounds)
            except Exception as exc:                  # noqa: BLE001
                out = _record(0.0, None, "error", "-",
                              message=f"{type(exc).__name__}: {exc}")
            out["rep"] = rep
            rec["runs"][method].append(out)

    if diagrams:
        # 默认只内嵌数据（报告里点击就地渲染），不再产出 PNG 文件
        build_ts_payload(spec.id, ts_ctx, rec)
    if png_diagrams:
        rec["diagrams"] = _optional_png(spec.id, build.problem, ctx, rec,
                                        out_json=out_json, diagram_dir=diagram_dir)

    # 解对象不写进 JSON（只用于画图）。
    for method in METHODS:
        for run in rec["runs"][method]:
            run.pop("_solution", None)
    rec["wall_s"] = time.perf_counter() - t_case
    return rec


# ---------------------------------------------------------------------------
# 持久化
# ---------------------------------------------------------------------------

def _payload_meta(args: argparse.Namespace, started: float) -> dict[str, Any]:
    import hashlib
    key = json.dumps({
        "cycle": bc.CYCLE, "kappa": bc.KAPPA, "reps": args.reps,
        "time_limit_s": args.time_limit, "mip_rel_gap": args.mip_rel_gap,
        "max_rounds": args.max_rounds, "plan_span": bc.DEFAULT_PLAN_SPAN,
        "stage1_symmetry": STAGE1_SYMMETRY, "min_band_width_s": MIN_BAND_WIDTH_S,
        "objectives": list(bc.OBJECTIVE_IDS),
        "diagrams": bool(args.diagrams), "png_diagrams": bool(args.png_diagrams),
    }, sort_keys=True)
    return {
        "schema": 1,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started)),
        "wall_s": None,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "config": {
            "cycle_s": bc.CYCLE, "kappa": bc.KAPPA, "reps": args.reps,
            "time_limit_s": args.time_limit, "mip_rel_gap": args.mip_rel_gap,
            "max_rounds": args.max_rounds, "plan_span": bc.DEFAULT_PLAN_SPAN,
            "backend": "scipy(HiGHS)", "heuristic": False, "num_workers": 1,
            "stage1_symmetry": STAGE1_SYMMETRY,
            "min_band_width_s": MIN_BAND_WIDTH_S,
            "diagrams": bool(args.diagrams),
            "png_diagrams": bool(args.png_diagrams),
            "diagram_dir": args.diagram_dir,
        },
        "config_hash": hashlib.sha1(key.encode("utf-8")).hexdigest()[:12],
    }


def write_json(path: str, payload: Mapping[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def load_json(path: str) -> dict[str, Any] | None:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="标准案例全矩阵性能测试")
    ap.add_argument("--reps", type=int, default=2, help="每个案例每种方法重复次数")
    ap.add_argument("--time-limit", type=float, default=20.0, help="单次求解上限（秒）")
    ap.add_argument("--mip-rel-gap", type=float, default=0.0)
    ap.add_argument("--max-rounds", type=int, default=3, help="可调边界时 multistage 轮数")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--html", default=DEFAULT_HTML)
    ap.add_argument("--diagrams", dest="diagrams", action="store_true", default=True,
                    help="内嵌时距图数据，报告里点击就地渲染（默认开，不产出 PNG）")
    ap.add_argument("--no-diagrams", dest="diagrams", action="store_false",
                    help="不内嵌时距图数据")
    ap.add_argument("--png-diagrams", action="store_true",
                    help="额外把每案例两张 PNG 写到 --diagram-dir（默认不写，省空间）")
    ap.add_argument("--diagram-dir", default=DEFAULT_DIAGRAM_DIR,
                    help="时空图子目录名（相对 --out 所在目录）")
    ap.add_argument("--groups", default="", help="只看这些组，如 G1,G3")
    ap.add_argument("--objectives", default="", help="只看这些目标 id")
    ap.add_argument("--boundaries", default="", help="fixed / adjustable")
    ap.add_argument("--plan-counts", default="", help="只看这些方案数，如 1,2,3")
    ap.add_argument("--limit", type=int, default=0, help="最多跑多少个案例（调试用）")
    ap.add_argument("--ids", default="", help="只跑这些案例 id（逗号分隔）")
    ap.add_argument("--merge", action="store_true",
                    help="把本次结果合并进已有 JSON（同 id 覆盖），配合 --ids 做局部重跑")
    ap.add_argument("--resume", action="store_true", help="复用已有 JSON 里的结果")
    ap.add_argument("--html-only", action="store_true", help="不求解，只重生成 HTML")
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args(argv)


def _select_cases(args: argparse.Namespace) -> list[bc.CaseSpec]:
    cases = bc.all_cases()
    if args.groups:
        keep = {s.strip().upper() for s in args.groups.split(",") if s.strip()}
        cases = [c for c in cases if c.group in keep]
    if args.objectives:
        keep = {s.strip() for s in args.objectives.split(",") if s.strip()}
        cases = [c for c in cases if c.objective in keep]
    if args.boundaries:
        keep = {s.strip() for s in args.boundaries.split(",") if s.strip()}
        cases = [c for c in cases if c.boundaries in keep]
    if args.plan_counts:
        keep = {int(s) for s in args.plan_counts.split(",") if s.strip()}
        cases = [c for c in cases if c.n_plans in keep]
    if args.ids:
        keep = {s.strip() for s in args.ids.split(",") if s.strip()}
        cases = [c for c in cases if c.id in keep]
        if len(cases) != len(keep):
            have = {c.id for c in cases}
            print(f"[warn] 有 {len(keep - have)} 个 id 不在矩阵里：{sorted(keep - have)[:3]}",
                  file=sys.stderr)
    if args.limit:
        cases = cases[: args.limit]
    return cases


def _fmt_secs(x: float | None) -> str:
    return "-" if x is None else f"{x:.3f}s"


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    started = time.time()
    meta = _payload_meta(args, started)

    if args.html_only:
        old = load_json(args.out)
        if old is None:
            print(f"[html-only] 找不到 {args.out}", file=sys.stderr)
            return 2
        old.setdefault("meta", meta)
        br.write_report(args.out, args.html, old)
        print(f"[html-only] 报告 -> {args.html}")
        return 0

    cases = _select_cases(args)
    payload: dict[str, Any] = {"meta": meta, "cases": []}
    if args.merge:
        old = load_json(args.out)
        if old is None:
            print(f"[merge] 找不到 {args.out}，按全新结果处理", file=sys.stderr)
        else:
            payload = old
            payload["meta"].update({k: v for k, v in meta.items() if k != "wall_s"})
            payload["meta"]["config_hash"] = meta["config_hash"]
            merged = {c["id"] for c in payload.get("cases", ())}
            print(f"[merge] 已载入 {len(merged)} 个案例，将覆盖 {len(cases)} 个")
    elif args.resume:
        old = load_json(args.out)
        if old and old.get("meta", {}).get("config_hash") == meta["config_hash"]:
            payload["cases"] = list(old.get("cases", ()))
            print(f"[resume] 复用 {len(payload['cases'])} 个已有案例")
        elif old:
            print("[resume] 口径不一致（config_hash 变了），重新跑")

    done = set() if args.merge else {c["id"] for c in payload["cases"] if "error" not in c}
    todo = [c for c in cases if c.id not in done]
    total = len(todo)
    print(f"案例总数 {len(cases)}，待跑 {total}，每案例 {args.reps} 次 × 2 方法，"
          f"单次上限 {args.time_limit:.0f}s")

    t_all = time.perf_counter()
    for idx, spec in enumerate(todo, 1):
        t0 = time.perf_counter()
        rec = run_case(spec, reps=args.reps, time_limit_s=args.time_limit,
                       mip_rel_gap=args.mip_rel_gap, max_rounds=args.max_rounds,
                       out_json=args.out, diagrams=args.diagrams,
                       png_diagrams=args.png_diagrams,
                       diagram_dir=args.diagram_dir)
        if args.merge:
            for i, old_rec in enumerate(payload["cases"]):
                if old_rec["id"] == rec["id"]:
                    payload["cases"][i] = rec
                    break
            else:
                payload["cases"].append(rec)
        else:
            payload["cases"].append(rec)
        payload["meta"]["wall_s"] = time.perf_counter() - t_all
        write_json(args.out, payload)
        dt = time.perf_counter() - t0
        if not args.quiet:
            s = br.summarise(rec)
            print(f"[{idx:3d}/{total}] {spec.id:52s} "
                  f"整模 {_fmt_secs(s['monolith']['wall_median']):>9} "
                  f"分阶段 {_fmt_secs(s['staged']['wall_median']):>9} "
                  f"{('%.2fx' % s['speedup']) if s['speedup'] else '   -  ':>7} "
                  f"路径={s['staged']['path']:<10} "
                  f"时距图={len(rec.get('ts_data', {})) or '-'} ({dt:.1f}s)", flush=True)
            for note in rec.get("notes", ()):
                print(f"          ! {note}", flush=True)

    payload["meta"]["wall_s"] = time.perf_counter() - t_all
    write_json(args.out, payload)
    br.write_report(args.out, args.html, payload)
    print(f"\n结果 -> {args.out}\n报告 -> {args.html}")
    if args.diagrams:
        print("时距图已内嵌进 JSON/HTML：在报告明细表点「整模 / 分阶段」即可就地渲染"
              "（不再输出 PNG 文件）")
    if args.png_diagrams:
        print(f"PNG 时空图 -> "
              f"{os.path.join(os.path.dirname(os.path.abspath(args.out)), args.diagram_dir)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
