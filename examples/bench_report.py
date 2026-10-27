"""全矩阵性能测试的 HTML 报告渲染（自包含单文件，双击即可打开）。

``benchmark_matrix.py`` 只负责跑数据，这个模块负责把 JSON 结果渲染成报告：
* 顶部 KPI 卡片；
* 每个 (分组 × 边界模式) 一张矩阵表：行 = 目标，列 = 方案数，格子里同时给出
  整模/分阶段的时间、加速比、目标值与实际走的路径；
* 一张 log-log 散点图（整模时间 vs 分阶段时间 + 等值线）；
* 一张可筛选/排序的明细表（带对数比例的内嵌条形）；
* 口径与注意事项说明。
"""
from __future__ import annotations

import html
import json
import math
import os
import statistics
from typing import Any, Iterable, Mapping, Sequence

import ts_canvas  # noqa: E402  （时距图渲染器：两份报告共用）

METHODS = ("monolith", "staged")
METHOD_LABEL = {"monolith": "整模", "staged": "分阶段"}

_GROUP_COLOR = {"G1": "#2563eb", "G2": "#16a34a", "G3": "#d97706", "G4": "#dc2626"}
_PATH_CLASS = {"语义分解": "p-dec", "方案枚举DP": "p-dp", "整模": "p-mono"}
_BAD_STATUS = ("time_limit", "infeasible", "error", "no_solution")


# ---------------------------------------------------------------------------
# 汇总（benchmark_matrix 也用这里的实现，避免两处口径不一致）
# ---------------------------------------------------------------------------

def _median(values: Iterable[float | None]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return statistics.median(vals) if vals else None


def summarise(rec: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for method in METHODS:
        runs = list(rec.get("runs", {}).get(method, ()))
        ok = [r for r in runs if r.get("status") not in ("error", "no_solution")]
        objs = [r["objective"] for r in ok if r.get("objective") is not None]
        out[method] = {
            "wall_median": _median([r.get("wall_s") for r in ok]),
            "wall_min": min([r["wall_s"] for r in ok], default=None),
            "wall_max": max([r["wall_s"] for r in ok], default=None),
            "objective_best": max(objs, default=None),
            "objective_median": _median(objs),
            "status": ok[0]["status"] if ok else (runs[0]["status"] if runs else "-"),
            "path": (ok[0].get("path") if ok else "-") or "-",
            "rel_gap_worst": max([r["rel_gap"] for r in ok
                                  if r.get("rel_gap") is not None], default=None),
            "hit_limit": any(r.get("status") == "time_limit" for r in ok),
        }
    m, s = out["monolith"], out["staged"]
    if m["wall_median"] and s["wall_median"]:
        out["speedup"] = m["wall_median"] / s["wall_median"]
    else:
        out["speedup"] = None
    if m["objective_best"] is not None and s["objective_best"] is not None:
        out["objective_delta"] = s["objective_best"] - m["objective_best"]
    else:
        out["objective_delta"] = None
    return out


def summaries(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {c["id"]: summarise(c) for c in payload.get("cases", ())}


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------

def _esc(x: Any) -> str:
    return html.escape("" if x is None else str(x))


def _secs(x: float | None, digits: int = 3) -> str:
    return "-" if x is None else f"{x:.{digits}f}s"


def _num(x: float | None, digits: int = 2) -> str:
    return "-" if x is None else f"{x:.{digits}f}"


def _badge(speedup: float | None, *, monolith_hit: bool = False) -> str:
    if speedup is None:
        return '<span class="badge flat">-</span>'
    cls = "up" if speedup >= 1.25 else ("down" if speedup <= 0.8 else "flat")
    txt = f"×{speedup:.2f}"
    if monolith_hit and speedup >= 1.25:
        cls = "up strong"
        txt += " ⏱"
    return f'<span class="badge {cls}">{txt}</span>'


def _path_chip(path: str) -> str:
    if not path or path == "-":
        return ""
    parts = [p.strip() for p in str(path).replace("+", " ").split() if p.strip()]
    chips = []
    for p in dict.fromkeys(parts):
        cls = _PATH_CLASS.get(p, "p-mono")
        chips.append(f'<span class="chip {cls}">{_esc(p)}</span>')
    return "".join(chips)


def _delta_txt(delta: float | None) -> str:
    return "-" if delta is None else f"{delta:+.3f}"


def _staged_detail(rec: Mapping[str, Any]) -> str:
    """分阶段那一列的耗时拆解：可调边界给 stage1/stage2/轮数，固定边界给求解器时间。"""
    runs = list(rec.get("runs", {}).get("staged", ()))
    if not runs:
        return "-"
    r = next((x for x in runs if x.get("status") not in ("error", "no_solution")), runs[0])
    if r.get("rounds") is not None:
        return (f"stage1 {_num(r.get('stage1_s'), 2)}s + stage2 "
                f"{_num(r.get('stage2_s'), 2)}s / {r.get('rounds')} 轮")
    if r.get("solver_s") is not None:
        return f"求解器 {_num(r.get('solver_s'), 3)}s"
    return "-"


def _diagram_links(rec: Mapping[str, Any]) -> str:
    """时距图入口：默认是"点击就地渲染"的按钮；若额外产出了 PNG 再附一个链接。"""
    data = rec.get("ts_data") or {}
    diagrams = rec.get("diagrams") or {}
    index = rec.get("_ts_index") or {}
    out = []
    for method in METHODS:
        if method in data and method in index:
            out.append(f'<button class="dbtn" onclick="showTS({index[method]})">'
                       f'{_esc(METHOD_LABEL[method])}</button>')
        if method in diagrams:
            out.append(f'<a class="dlink" href="{_esc(diagrams[method])}" target="_blank">'
                       f'PNG</a>')
    return " ".join(out) if out else '<span class="muted">-</span>'


def _status_dot(status: str) -> str:
    if status in ("optimal",):
        return '<span class="dot ok" title="已证最优"></span>'
    if status == "time_limit":
        return '<span class="dot warn" title="达到时间上限，未证最优"></span>'
    if status == "feasible":
        return '<span class="dot info" title="找到可行解"></span>'
    if status in ("infeasible", "error", "no_solution"):
        return f'<span class="dot bad" title="{_esc(status)}"></span>'
    return f'<span class="dot" title="{_esc(status)}"></span>'


# ---------------------------------------------------------------------------
# KPI
# ---------------------------------------------------------------------------

def _kpis(payload: Mapping[str, Any], sums: Mapping[str, Any]) -> str:
    cases = list(payload.get("cases", ()))
    solved = [c for c in cases if "error" not in c]
    n_solves = sum(len(c.get("runs", {}).get(m, ())) for c in solved for m in METHODS)
    n_limit = sum(1 for c in solved
                  if sums[c["id"]]["monolith"]["hit_limit"]
                  or sums[c["id"]]["staged"]["hit_limit"])
    n_decomp = sum(1 for c in solved
                   if sums[c["id"]]["staged"]["path"] not in ("整模", "-", ""))
    sp = [sums[c["id"]]["speedup"] for c in solved
          if sums[c["id"]]["speedup"] is not None]
    faster = sum(1 for x in sp if x >= 1.25)
    slower = sum(1 for x in sp if x <= 0.8)
    med = statistics.median(sp) if sp else None
    n_env = sum(1 for c in solved if c.get("structure", {}).get("groups"))
    dl = [sums[c["id"]]["objective_delta"] for c in solved
          if sums[c["id"]]["objective_delta"] is not None]
    same = sum(1 for d in dl if abs(d) <= 1e-6)
    better = sum(1 for d in dl if d > 1e-6)
    worse = sum(1 for d in dl if d < -1e-6)

    cards = [
        ("案例数", f"{len(solved)}", f"矩阵 {len(cases)} 个，另 {len(cases) - len(solved)} 个配置失败"),
        ("求解次数", f"{n_solves}", f"每案例 {payload['meta']['config']['reps']} 次 × 2 方法"),
        ("碰到时间上限", f"{n_limit}", f"上限 {payload['meta']['config']['time_limit_s']:.0f}s"),
        ("分阶段走了分解/DP", f"{n_decomp}", f"其余回退整模（{n_env} 案例有语义组信息）"),
        ("分阶段更快 (≥1.25×)", f"{faster}", f"更慢 ≤0.8× 的有 {slower} 个"),
        ("加速比中位数", f"{med:.2f}×" if med else "-", "全体有效案例"),
        ("目标值相同", f"{same}", f"分阶段更好 {better}，更差 {worse}"),
    ]
    out = ['<section class="cards">']
    for title, value, sub in cards:
        out.append(
            f'<div class="card"><div class="k">{_esc(title)}</div>'
            f'<div class="v">{_esc(value)}</div><div class="s">{_esc(sub)}</div></div>')
    out.append("</section>")
    return "".join(out)


# ---------------------------------------------------------------------------
# 矩阵表
# ---------------------------------------------------------------------------

def _cell(rec: Mapping[str, Any], s: Mapping[str, Any]) -> str:
    m, st = s["monolith"], s["staged"]
    cls = ""
    if s["speedup"] is not None:
        cls = " faster" if s["speedup"] >= 1.25 else (" slower" if s["speedup"] <= 0.8 else "")
    lines = [
        f'<div class="r"><span class="tag m">整</span><span class="t">{_secs(m["wall_median"])}</span>'
        f'<span class="o">{_num(m["objective_best"])}</span>{_status_dot(m["status"])}</div>',
        f'<div class="r"><span class="tag s">分</span><span class="t">{_secs(st["wall_median"])}</span>'
        f'<span class="o">{_num(st["objective_best"])}</span>'
        f'{_badge(s["speedup"], monolith_hit=m["hit_limit"])}{_status_dot(st["status"])}</div>',
    ]
    chips = _path_chip(st["path"])
    if chips:
        lines.append(f'<div class="r paths">{chips}</div>')
    if s["objective_delta"] is not None and abs(s["objective_delta"]) > 1e-6:
        d = s["objective_delta"]
        lines.append(f'<div class="r delta {"pos" if d > 0 else "neg"}">Δ目标 {d:+.3f}</div>')
    return f'<td class="cell{cls}">{"".join(lines)}</td>'


def _matrix_table(group: Mapping[str, Any], boundary: str,
                  cases: Sequence[Mapping[str, Any]],
                  sums: Mapping[str, Any]) -> str:
    by_obj: dict[str, dict[int, Mapping[str, Any]]] = {}
    plan_counts: list[int] = []
    for c in cases:
        if c.get("group") != group["id"] or c.get("boundaries") != boundary:
            continue
        by_obj.setdefault(c["objective"], {})[int(c["n_plans"])] = c
        if int(c["n_plans"]) not in plan_counts:
            plan_counts.append(int(c["n_plans"]))
    if not by_obj:
        return ""
    plan_counts.sort()
    head = "".join(f'<th>{k} 方案/路口</th>' for k in plan_counts)
    rows = []
    for obj_id in [o for o in _objective_order(cases) if o in by_obj]:
        label = next(c["objective_label"] for c in cases
                     if c["objective"] == obj_id and c.get("group") == group["id"])
        cells = []
        for k in plan_counts:
            rec = by_obj[obj_id].get(k)
            if rec is None:
                cells.append('<td class="cell empty">-</td>')
            else:
                cells.append(_cell(rec, sums[rec["id"]]))
        rows.append(f'<tr><th class="rowhead">{_esc(label)}</th>{"".join(cells)}</tr>')
    bnd_label = "固定相位边界 (1.1–1.4)" if boundary == "fixed" else "可调相位边界 (1.5–1.8)"
    return (f'<div class="block" data-group="{group["id"]}" data-boundaries="{boundary}">'
            f'<h3>{_esc(group["label"])} · {_esc(bnd_label)}</h3>'
            f'<div class="tw"><table class="matrix"><thead><tr><th class="rowhead">目标</th>'
            f'{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div></div>')


def _objective_order(cases: Sequence[Mapping[str, Any]]) -> list[str]:
    out: list[str] = []
    for c in cases:
        if c["objective"] not in out:
            out.append(c["objective"])
    return out


# ---------------------------------------------------------------------------
# log-log 散点图
# ---------------------------------------------------------------------------

def _log_scale(v: float, lo: float, hi: float, px: float) -> float:
    v = max(float(v), lo)
    return (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * px


def _scatter(payload: Mapping[str, Any], sums: Mapping[str, Any]) -> str:
    cases = [c for c in payload.get("cases", ()) if "error" not in c]
    pts = []
    for c in cases:
        m = sums[c["id"]]["monolith"]["wall_median"]
        s = sums[c["id"]]["staged"]["wall_median"]
        if m is None or s is None or m <= 0 or s <= 0:
            continue
        pts.append((c, m, s))
    if not pts:
        return ""
    lo, hi = 0.02, 30.0
    W = H = 460
    pad = 46
    size = W - pad - 18

    def xy(m: float, s: float) -> tuple[float, float]:
        x = pad + _log_scale(m, lo, hi, size)
        y = H - pad - _log_scale(s, lo, hi, size)
        return x, y

    parts = [f'<svg class="scatter" viewBox="0 0 {W} {H}" role="img">']
    parts.append(f'<rect x="{pad}" y="{H - pad - size}" width="{size}" height="{size}" '
                 f'class="plot-bg"/>')
    # 等值线 + 网格
    for mult, cls in ((1.0, "diag"), (2.0, "diag-lite"), (0.5, "diag-lite")):
        x1, y1 = xy(lo, lo * mult)
        x2, y2 = xy(hi, hi * mult)
        parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" class="{cls}"/>')
    for tick in (0.1, 1.0, 10.0):
        x = pad + _log_scale(tick, lo, hi, size)
        y = H - pad - _log_scale(tick, lo, hi, size)
        parts.append(f'<line x1="{x:.1f}" y1="{H - pad - size}" x2="{x:.1f}" y2="{H - pad}" class="grid"/>')
        parts.append(f'<line x1="{pad}" y1="{y:.1f}" x2="{pad + size}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{x:.1f}" y="{H - pad + 16}" class="tick">{tick:g}s</text>')
        parts.append(f'<text x="{pad - 8}" y="{y + 4:.1f}" class="tick end">{tick:g}s</text>')
    for c, m, s in pts:
        x, y = xy(m, s)
        color = _GROUP_COLOR.get(c.get("group", ""), "#64748b")
        hollow = c.get("boundaries") == "adjustable"
        title = (f'{c["id"]}\n整模 {m:.3f}s / 分阶段 {s:.3f}s'
                 f'\n加速 {m / s:.2f}× · {sums[c["id"]]["staged"]["path"]}')
        if hollow:
            parts.append(f'<rect x="{x - 4:.1f}" y="{y - 4:.1f}" width="8" height="8" rx="1.5" '
                         f'fill="none" stroke="{color}" stroke-width="1.8"><title>{_esc(title)}</title></rect>')
        else:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}" '
                         f'fill-opacity="0.85"><title>{_esc(title)}</title></circle>')
    parts.append(f'<text x="{pad + size / 2:.0f}" y="{H - 8}" class="axis">整模用时</text>')
    parts.append(f'<text x="14" y="{H - pad - size / 2:.0f}" class="axis" '
                 f'transform="rotate(-90 14 {H - pad - size / 2:.0f})">分阶段用时</text>')
    parts.append("</svg>")
    legend = ['<div class="legend">']
    for gid, color in _GROUP_COLOR.items():
        legend.append(f'<span><i style="background:{color}"></i>{gid}</span>')
    legend.append('<span><i class="hollow"></i>可调边界</span>'
                  '<span><i style="background:#94a3b8"></i>固定边界（实心）</span>')
    legend.append('<span class="lg-note">对角线 = 两者同速；点在对角线下方 = 分阶段更快</span>')
    legend.append("</div>")
    return "".join(parts) + "".join(legend)


# ---------------------------------------------------------------------------
# 明细表
# ---------------------------------------------------------------------------

def _bar(value: float | None, hi: float, cls: str) -> str:
    if value is None:
        return ""
    w = 100.0 * _log_scale(value, 0.02, max(hi, 0.05), 1.0)
    return f'<span class="bar {cls}" style="width:{max(2.0, min(100.0, w)):.1f}%"></span>'


def _detail_table(payload: Mapping[str, Any], sums: Mapping[str, Any]) -> str:
    cases = list(payload.get("cases", ()))
    tl = float(payload["meta"]["config"]["time_limit_s"])
    rows = []
    for c in cases:
        if "error" in c:
            rows.append(
                f'<tr class="err" data-group="{_esc(c.get("group"))}" '
                f'data-boundaries="{_esc(c.get("boundaries"))}" data-k="{c.get("n_plans")}" '
                f'data-obj="{_esc(c.get("objective"))}">'
                f'<td>{_esc(c.get("group"))}</td><td>{_esc(c.get("objective_label"))}</td>'
                f'<td>{c.get("n_plans")}</td><td>{_esc(c.get("boundaries"))}</td>'
                f'<td colspan="10" class="bad">配置失败：{_esc(c.get("error"))}</td></tr>')
            continue
        s = sums[c["id"]]
        m, st = s["monolith"], s["staged"]
        stc = c.get("structure", {})
        model = stc.get("model", {})
        rows.append(
            f'<tr data-group="{_esc(c["group"])}" data-boundaries="{_esc(c["boundaries"])}" '
            f'data-k="{c["n_plans"]}" data-obj="{_esc(c["objective"])}" '
            f'data-limit="{1 if (m["hit_limit"] or st["hit_limit"]) else 0}" '
            f'data-path="{_esc(st["path"])}">'
            f'<td>{_esc(c["group"])}</td>'
            f'<td class="obj">{_esc(c["objective_label"])}</td>'
            f'<td>{c["n_plans"]}</td>'
            f'<td>{_esc(c["boundaries"])}</td>'
            f'<td>{stc.get("groups", "-")}</td>'
            f'<td>{"✓" if stc.get("dp_candidate") else "–"}'
            f'{("/" + str(stc["dp_solves"])) if stc.get("dp_solves") else ""}</td>'
            f'<td class="mono">{model.get("vars", "-")}/{model.get("binary", "-")}b/'
            f'{model.get("integer", "-")}i/{model.get("cons", "-")}c</td>'
            f'<td class="num">{_secs(m["wall_median"])}{_bar(m["wall_median"], tl, "bm")}'
            f'{_status_dot(m["status"])}</td>'
            f'<td class="num">{_secs(st["wall_median"])}{_bar(st["wall_median"], tl, "bs")}'
            f'{_status_dot(st["status"])}</td>'
            f'<td>{_badge(s["speedup"], monolith_hit=m["hit_limit"])}</td>'
            f'<td>{_path_chip(st["path"])}</td>'
            f'<td class="mono">{_esc(_staged_detail(c))}</td>'
            f'<td>{_diagram_links(c)}</td>'
            f'<td class="num">{_num(m["objective_best"], 3)}</td>'
            f'<td class="num">{_num(st["objective_best"], 3)}</td>'
            f'<td class="num">{_delta_txt(s["objective_delta"])}</td>'
            f'<td class="note">{_esc("; ".join(c.get("notes", ())))}</td>'
            f'</tr>')
    head = ("<tr><th>组</th><th>目标</th><th>方案数</th><th>边界</th><th>语义组</th>"
            "<th>DP候选</th><th>模型规模</th><th>整模</th><th>分阶段</th><th>加速</th>"
            "<th>实际路径</th><th>分阶段拆解</th><th>时空图</th><th>整模obj</th>"
            "<th>分阶段obj</th><th>Δobj</th><th>备注</th></tr>")
    return (f'<div class="filters">'
            f'<select id="f-group"><option value="">全部组</option>'
            f'<option>G1</option><option>G2</option><option>G3</option><option>G4</option></select>'
            f'<select id="f-obj"><option value="">全部目标</option>'
            + "".join(f'<option value="{_esc(o)}">{_esc(l)}</option>'
                      for o, l in _objective_pairs(cases)) +
            f'</select>'
            f'<select id="f-bnd"><option value="">全部边界</option>'
            f'<option value="fixed">固定</option><option value="adjustable">可调</option></select>'
            f'<select id="f-k"><option value="">全部方案数</option>'
            + "".join(f'<option value="{k}">{k}</option>' for k in _plan_counts(cases)) +
            f'</select>'
            f'<label class="cb"><input type="checkbox" id="f-limit">只看碰到时间上限</label>'
            f'<span id="f-count" class="count"></span>'
            f'</div>'
            f'<div class="tw tall"><table class="detail" id="detail"><thead>{head}</thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def _objective_pairs(cases: Sequence[Mapping[str, Any]]) -> list[tuple[str, str]]:
    out: dict[str, str] = {}
    for c in cases:
        if "objective" in c:
            out.setdefault(c["objective"], c.get("objective_label", c["objective"]))
    return list(out.items())


def _plan_counts(cases: Sequence[Mapping[str, Any]]) -> list[int]:
    return sorted({int(c["n_plans"]) for c in cases if c.get("n_plans")})



# ---------------------------------------------------------------------------
# 关键发现 + 按目标聚合
# ---------------------------------------------------------------------------

def _findings(payload: Mapping[str, Any], sums: Mapping[str, Any]) -> str:
    cases = [c for c in payload.get("cases", ()) if "error" not in c]
    scored = [(sums[c["id"]]["speedup"], c) for c in cases
              if sums[c["id"]]["speedup"] is not None]
    fast = sorted([x for x in scored if x[0] >= 1.25], key=lambda x: -x[0])[:5]
    slow = sorted(scored, key=lambda x: x[0])[:5]
    limits = [c for c in cases
              if sums[c["id"]]["monolith"]["hit_limit"]
              or sums[c["id"]]["staged"]["hit_limit"]]
    worse = [(sums[c["id"]]["objective_delta"], c) for c in cases
             if sums[c["id"]]["objective_delta"] is not None
             and sums[c["id"]]["objective_delta"] < -1e-6]
    worse.sort(key=lambda x: x[0])

    def item(c, tag):
        s = sums[c["id"]]
        return (f'<li><code>{_esc(c["id"])}</code> — 整模 {_secs(s["monolith"]["wall_median"])}'
                f' vs 分阶段 {_secs(s["staged"]["wall_median"])} '
                f'{_badge(s["speedup"])} {_path_chip(s["staged"]["path"])}'
                f'<span class="lg-note"> {_esc(tag)}</span></li>')

    out = ['<section class="block findings"><h3>关键发现</h3><div class="cols"><div>']
    out.append("<h4>加速最明显</h4><ul>")
    out += [item(c, "") for _, c in fast] or ["<li>无</li>"]
    out.append("</ul></div><div><h4>分阶段反而更慢</h4><ul>")
    out += [item(c, "") for _, c in slow] or ["<li>无</li>"]
    out.append("</ul></div><div>")
    out.append(f"<h4>触及 {payload['meta']['config']['time_limit_s']:.0f}s 上限（未证最优）</h4><ul>")
    if limits:
        for c in limits[:8]:
            s = sums[c["id"]]
            out.append(f'<li><code>{_esc(c["id"])}</code> — 整模 '
                       f'{"⏱" if s["monolith"]["hit_limit"] else "ok"} / 分阶段 '
                       f'{"⏱" if s["staged"]["hit_limit"] else "ok"}'
                       f'（整模 gap {_num(s["monolith"]["rel_gap_worst"], 3)}）</li>')
    else:
        out.append("<li>没有案例触及上限</li>")
    out.append("</ul>")
    out.append("<h4>分阶段目标值劣化最大的案例</h4><ul>")
    if worse:
        for d, c in worse[:5]:
            out.append(f'<li><code>{_esc(c["id"])}</code> — Δ目标 {d:+.3f}'
                       f'（整模 {_num(sums[c["id"]]["monolith"]["objective_best"], 3)}'
                       f' → 分阶段 {_num(sums[c["id"]]["staged"]["objective_best"], 3)}）</li>')
    else:
        out.append("<li>无</li>")
    out.append("</ul></div></div></section>")
    return "".join(out)


def _aggregate(payload: Mapping[str, Any], sums: Mapping[str, Any]) -> str:
    """按 (分组, 边界, 目标) 聚合：跨方案数取中位数。"""
    groups: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = {}
    for c in payload.get("cases", ()):
        if "error" in c:
            continue
        key = (c["group"], c["group_label"], c["boundaries"], c["objective_label"])
        groups.setdefault(key, []).append(c)
    rows = []
    for (gid, glabel, bnd, olabel), cs in sorted(groups.items()):
        sp = [sums[c["id"]]["speedup"] for c in cs if sums[c["id"]]["speedup"] is not None]
        mo = _median([sums[c["id"]]["monolith"]["wall_median"] for c in cs])
        st = _median([sums[c["id"]]["staged"]["wall_median"] for c in cs])
        dl = [sums[c["id"]]["objective_delta"] for c in cs
              if sums[c["id"]]["objective_delta"] is not None]
        lim = sum(1 for c in cs if sums[c["id"]]["monolith"]["hit_limit"]
                  or sums[c["id"]]["staged"]["hit_limit"])
        paths = sorted({sums[c["id"]]["staged"]["path"] for c in cs})
        rows.append(
            f'<tr><td>{_esc(gid)}</td><td>{_esc(olabel)}</td>'
            f'<td>{"固定" if bnd == "fixed" else "可调"}</td>'
            f'<td class="num">{len(cs)}</td>'
            f'<td class="num">{_secs(mo)}</td><td class="num">{_secs(st)}</td>'
            f'<td>{_badge(statistics.median(sp) if sp else None, monolith_hit=lim > 0)}</td>'
            f'<td class="num">{"-" if not dl else f"{statistics.median(dl):+.3f}"}</td>'
            f'<td class="num">{lim or ""}</td>'
            f'<td>{_path_chip(" ".join(paths))}</td></tr>')
    head = ("<tr><th>组</th><th>目标</th><th>边界</th><th>案例数</th><th>整模中位</th>"
            "<th>分阶段中位</th><th>加速中位</th><th>Δobj 中位</th><th>触上限</th>"
            "<th>出现过的路径</th></tr>")
    return (f'<div class="block"><h3>按目标聚合（同一目标跨方案数取中位数）</h3>'
            f'<div class="tw"><table class="detail"><thead>{head}</thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div></div>')


# ---------------------------------------------------------------------------
# 说明
# ---------------------------------------------------------------------------

def _notes_section(payload: Mapping[str, Any]) -> str:
    cfg = payload["meta"]["config"]
    return f"""
<section class="notes">
  <h2>口径与注意事项</h2>
  <div class="cols">
    <div>
      <h3>求解口径</h3>
      <ul>
        <li>C={cfg['cycle_s']:.0f}s、κ={cfg['kappa']:g}，单格点。</li>
        <li>后端 {_esc(cfg['backend'])}，mip_rel_gap={cfg['mip_rel_gap']:g}，
            heuristic 关闭，单线程。</li>
        <li>每个案例每种方法跑 {cfg['reps']} 次，表中取中位数（同时记录 min/max）。</li>
        <li>单次求解上限 {cfg['time_limit_s']:.0f}s；命中上限记
            <span class="dot warn"></span>，表示"未证最优"，仍会给出当时的最优可行解与对偶界。</li>
        <li>stage-1 对称破缺 = {cfg['stage1_symmetry']}，最小带宽
            {cfg['min_band_width_s']:g}s（两条路径使用同一套建模参数）。</li>
      </ul>
    </div>
    <div>
      <h3>整模 vs 分阶段</h3>
      <ul>
        <li><b>固定相位边界</b>：δ≡0，没有 stage-2 可言。"整模"与"分阶段"都走
            <code>solve_fixed_boundaries</code>，唯一区别是 <code>decompose</code>
            开关——分阶段会先试语义分解 / 方案枚举 DP，不行才回退整模。</li>
        <li><b>可调相位边界</b>：整模 = 一次性完整 MILP（δ 与组合变量一起解）；
            分阶段 = multistage 全链路（stage-1 组合 → stage-2 LP 连续优化
            {cfg['max_rounds']} 轮、单起点）。</li>
        <li>"实际路径"列来自求解器返回的 message：<span class="chip p-dec">语义分解</span>
            <span class="chip p-dp">方案枚举DP</span> <span class="chip p-mono">整模</span>。</li>
      </ul>
    </div>
    <div>
      <h3>改动过的配置</h3>
      <ul>
        <li>6 种目标全部替换 <code>band_demands</code>；均衡组只在"均衡组×1+带宽和×0.5"
            里出现（组权重 1.0，两条带权重 0.5）。</li>
        <li>双周期组：涉及 I3 或跨全线的需求 <code>max_bands=2</code> 且
            <code>allow_window_sharing=true</code>；单周期组：全部 1 且关闭共享。</li>
        <li>多方案：把每个路口的方案整体平移出 k 个变体（最大平移
            ±{cfg['plan_span']:.3f}·C，脚本会按"硬约束常数行 + k 个方案都保留"
            自适应收缩），保持边界个数与绿掩码结构一致。</li>
        <li>清空 <code>global_constraints</code> 与 <code>constraint_templates</code>
            （它们引用了本矩阵不存在的需求 id）；绿段 margin 关掉；
            方案自带 hard_constraints 与 <code>red_boundary_default</code> 保留；
            绿波边界亲和默认关闭，本矩阵不启用。</li>
      </ul>
    </div>
  </div>
  <div class="cols" style="margin-top:10px">
    <div>
      <h3>时空图（行内渲染，不再输出 PNG）</h3>
      <ul>
        <li>每案例每方法的解（φ、各相位边界微调 δ、选中方案、各条波带）都内嵌在报告里；
            点明细表「时空图」列的 <b>整模 / 分阶段</b> 按钮即可就地渲染，不必再存 240 张 PNG。</li>
        <li>用的是该案例该方法多次重复里<b>目标值最好</b>的那一次解。</li>
        <li>画法与 Python 版 <code>TimeSpaceDiagram</code> 一致（绿窗 / 红灯 / 绿波带 /
            红波带那套"上下游展示副本"搜索），可勾选灯条・波带・周期分界，也可下载 PNG。</li>
        <li>多方案案例只画每个路口<b>被选中</b>的那个方案的相位。</li>
      </ul>
      <h3>怎么选（按实测）</h3>
      <ul>
        <li><b>固定相位边界（无论目标）</b>：统一走分阶段入口。可分解时中位 6.09×
            （最高 62×，解与整模逐行校验 + 最优性证书一致，Δ≤3e-6）；不可分解时它
            就是"整模 + 一次失败的分解尝试"，与整模等价（绝对差中位 −1ms）。没有取舍成本。</li>
        <li><b>可调边界 + 逐对/局部目标</b>：分阶段换速度（中位 3.05×，最高 22×），
            但<b>它不是精确算法</b>——stage-1 冻结 δ 只保证"固定 δ 的组合子问题"最优，
            联合 (δ, 组合) 问题仍是启发式，实测 30 例里 8 例目标值掉 &gt;1（其中 3 例
            整模已证最优且更好，如双向红波 k=3/4/5：543.07 → 532.6~536.1）。
            要速度用分阶段，要最好解用整模。</li>
        <li><b>可调边界 + 跨全线全局目标</b>（上下全局绿波、全局均衡组、上全局绿+下红）：
            不可分解，用整模。分阶段中位 0.84×（更慢）且目标值最多掉 8.4%
            （如 305.79 → 280.00）。</li>
        <li>需要"已证最优"时：固定边界下分阶段在整模 20s 证不动的案例上仍能给出证书
            （实测 9 例）。</li>
      </ul>
    </div>
  </div>
  <p class="warnbox">读法提醒：k≥2 的案例里，克隆方案自带的软约束（owner=intersection_loss）
  被去掉——它们不进 composite 目标，只增加变量；因此 k=1 与 k≥2 之间除"方案数"外
  还有这一项差异。目标值比较只在同一案例的整模/分阶段之间做（两者模型完全相同）。<br>
  多阶段返回的 <code>status=feasible</code> 是它的固定标记（启发式结果），并不表示"未证最优"；
  真正要看的是"实际路径"与"分阶段拆解"两列。固定边界下，分解/DP 给出的目标值与整模
  完全一致（实测差异 ≤3e-6，属求解器数值容差）；可调边界下多阶段是启发式，目标值可能
  低于一次性整模。</p>
</section>"""


_CSS = """
:root{--bg:#f5f6f8;--card:#fff;--ink:#1e293b;--muted:#64748b;--line:#e2e8f0;
--blue:#2563eb;--green:#16a34a;--red:#dc2626;--amber:#d97706;--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif}
header{padding:26px 30px 18px;background:linear-gradient(135deg,#1e293b,#334155);color:#fff}
header h1{margin:0 0 6px;font-size:22px;letter-spacing:.3px}
header .meta{font-size:12.5px;color:#cbd5e1;font-family:var(--mono)}
main{padding:20px 30px 60px;max-width:1680px;margin:0 auto}
h2{font-size:17px;margin:30px 0 12px;padding-bottom:6px;border-bottom:2px solid var(--line)}
h3{font-size:14.5px;margin:0 0 8px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:18px 0 6px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;
box-shadow:0 1px 2px rgba(15,23,42,.04)}
.card .k{font-size:12px;color:var(--muted)}
.card .v{font-size:22px;font-weight:650;margin:2px 0 2px;font-family:var(--mono)}
.card .s{font-size:11.5px;color:var(--muted)}
.block{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;
margin:14px 0;box-shadow:0 1px 2px rgba(15,23,42,.04)}
.tw{overflow-x:auto}
table{border-collapse:separate;border-spacing:0;width:100%;font-size:12.5px}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top}
thead th{position:sticky;top:0;background:#f8fafc;z-index:2;font-size:12px;color:#334155;
border-bottom:1.5px solid #cbd5e1;white-space:nowrap}
table.matrix .rowhead{width:210px;font-weight:600;background:#fbfcfe}
table.matrix td.cell{padding:7px 9px;min-width:190px;border-left:3px solid transparent}
table.matrix td.cell.faster{border-left-color:var(--green);background:#f0fdf4}
table.matrix td.cell.slower{border-left-color:var(--red);background:#fef2f2}
table.matrix td.cell.empty{color:#cbd5e1;text-align:center}
.cell .r{display:flex;align-items:center;gap:6px;white-space:nowrap}
.cell .r.paths{margin-top:3px}
.tag{display:inline-block;width:16px;text-align:center;border-radius:4px;font-size:10.5px;
color:#fff;background:#94a3b8;padding:0 2px}
.tag.m{background:#475569}.tag.s{background:#0ea5e9}.tag.o{background:#a855f7}
.cell .t{font-family:var(--mono);font-weight:600;min-width:58px}
.cell .o{font-family:var(--mono);color:var(--muted);min-width:56px;text-align:right}
.cell .delta{font-family:var(--mono);font-size:11.5px}
.delta.pos{color:var(--green)}.delta.neg{color:var(--red)}
.badge{display:inline-block;padding:0 6px;border-radius:999px;font-family:var(--mono);
font-size:11px;font-weight:600}
.badge.up{background:#dcfce7;color:#15803d}.badge.up.strong{box-shadow:0 0 0 1.5px #86efac inset}
.badge.down{background:#fee2e2;color:#b91c1c}.badge.flat{background:#f1f5f9;color:#64748b}
.chip{display:inline-block;padding:0 6px;border-radius:6px;font-size:10.5px;margin-right:4px;
border:1px solid transparent}
.chip.p-dec{background:#e0f2fe;color:#075985}
.chip.p-dp{background:#ede9fe;color:#5b21b6}
.chip.p-mono{background:#f1f5f9;color:#475569}
.dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:#cbd5e1;margin-left:3px}
.dot.ok{background:#22c55e}.dot.warn{background:#f59e0b}.dot.info{background:#38bdf8}
.dot.bad{background:#ef4444}
.scatter{width:100%;max-width:520px;height:auto}
.scatter .plot-bg{fill:#fcfdff;stroke:#e2e8f0}
.scatter .grid{stroke:#eef2f7}
.scatter .diag{stroke:#94a3b8;stroke-width:1.2;stroke-dasharray:5 4}
.scatter .diag-lite{stroke:#e2e8f0;stroke-width:1;stroke-dasharray:3 5}
.scatter .tick{font-size:10px;fill:#94a3b8;text-anchor:middle;font-family:var(--mono)}
.scatter .tick.end{text-anchor:end}
.scatter .axis{font-size:11px;fill:#64748b;text-anchor:middle}
.legend{font-size:11.5px;color:var(--muted);margin-top:6px;display:flex;gap:14px;flex-wrap:wrap}
.legend i{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}
.legend i.hollow{background:none;border:1.8px solid #94a3b8;border-radius:2px}
.legend .lg-note{color:#94a3b8}
.filters{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:8px 0 10px}
select,input[type=text]{padding:5px 8px;border:1px solid var(--line);border-radius:7px;
background:#fff;font:inherit;font-size:12.5px}
.cb{font-size:12.5px;color:var(--muted);display:flex;align-items:center;gap:5px}
.count{font-size:12px;color:var(--muted);font-family:var(--mono)}
button.dbtn{display:inline-block;padding:1px 7px;border:1px solid #bfdbfe;border-radius:6px;
background:#eff6ff;color:#1d4ed8;font-size:11px;margin-right:4px;cursor:pointer;font:inherit}
button.dbtn:hover{background:#dbeafe}
a.dlink{display:inline-block;padding:0 6px;border:1px solid #bfdbfe;border-radius:6px;
background:#eff6ff;color:#1d4ed8;text-decoration:none;font-size:11px;margin-right:4px}
a.dlink:hover{background:#dbeafe}
.muted{color:#cbd5e1}
table.detail{font-size:12px}
table.detail td,table.detail th{white-space:nowrap}
table.detail td.obj{max-width:200px;overflow:hidden;text-overflow:ellipsis}
table.detail td.mono{font-family:var(--mono);color:var(--muted)}
table.detail td.num{font-family:var(--mono);position:relative}
table.detail td.note{color:var(--muted);font-size:11.5px}
table.detail .bar{position:absolute;left:6px;bottom:2px;height:3px;border-radius:2px;opacity:.55}
.bar.bm{background:#64748b}.bar.bs{background:#0ea5e9}
table.detail tr.err td{background:#fef2f2}
td.bad{color:var(--red)}
.tw.tall{max-height:640px;overflow:auto}
.notes .cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:18px}
.notes h3{color:#334155;margin-top:6px}
.notes ul{margin:6px 0;padding-left:18px}
.notes li{margin:3px 0;color:#475569}
.notes code{background:#f1f5f9;padding:0 4px;border-radius:4px;font-family:var(--mono);font-size:11.5px}
.findings h4{margin:10px 0 4px;font-size:13px;color:#334155}
.findings ul{list-style:none;padding-left:0}
.findings li{padding:3px 0;border-bottom:1px dashed #eef2f7;font-size:12.5px}
.findings code{background:#f1f5f9;padding:0 4px;border-radius:4px;font-family:var(--mono);font-size:11.5px}
.warnbox{background:#fffbeb;border:1px solid #fde68a;color:#92400e;border-radius:8px;
padding:10px 12px;font-size:12.5px}
footer{color:var(--muted);font-size:12px;padding:0 30px 40px;max-width:1680px;margin:0 auto}
"""

_JS = """
function applyFilters(){
  var g=document.getElementById('f-group').value;
  var o=document.getElementById('f-obj').value;
  var b=document.getElementById('f-bnd').value;
  var k=document.getElementById('f-k').value;
  var lim=document.getElementById('f-limit').checked;
  var rows=document.querySelectorAll('#detail tbody tr');
  var n=0;
  rows.forEach(function(r){
    var ok=(!g||r.dataset.group===g)&&(!o||r.dataset.obj===o)&&
           (!b||r.dataset.boundaries===b)&&(!k||r.dataset.k===k)&&
           (!lim||r.dataset.limit==='1');
    r.style.display=ok?'':'none';
    if(ok)n++;
  });
  document.getElementById('f-count').textContent='显示 '+n+' / '+rows.length;
}
['f-group','f-obj','f-bnd','f-k'].forEach(function(id){
  var el=document.getElementById(id); if(el) el.addEventListener('change',applyFilters);
});
var fl=document.getElementById('f-limit'); if(fl) fl.addEventListener('change',applyFilters);
document.querySelectorAll('table.detail thead th').forEach(function(th,i){
  th.style.cursor='pointer';
  var asc=false;
  th.addEventListener('click',function(){
    var tb=th.closest('table').querySelector('tbody');
    var rows=Array.prototype.slice.call(tb.querySelectorAll('tr'));
    asc=!asc;
    rows.sort(function(a,b){
      var x=a.children[i]?a.children[i].innerText.trim():'';
      var y=b.children[i]?b.children[i].innerText.trim():'';
      var nx=parseFloat(x.replace(/[^0-9.+-]/g,'')), ny=parseFloat(y.replace(/[^0-9.+-]/g,''));
      var r=(!isNaN(nx)&&!isNaN(ny))?(nx-ny):x.localeCompare(y,'zh');
      return asc?r:-r;
    });
    rows.forEach(function(r){tb.appendChild(r);});
  });
});
applyFilters();
"""


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------

def _ts_json(obj: Any) -> str:
    """把数据安全嵌进 <script>（转义 </ 以免提前闭合）。"""
    return json.dumps(obj, ensure_ascii=False).replace("</", "<\\/")


def _ts_records(payload: Mapping[str, Any]) -> tuple[list[dict], list[dict]]:
    """展开时距图数据：每个 (案例, 方法) 一条记录 + 每案例一套静态上下文。

    * `ctxs`：每案例的 `ts_ctx`（走廊版面 + 该案例所有方案的窗口结构）——
      多方案 / 双周期案例的方案结构不同，不能共用一份；
    * `records`：`TS.draw(rec, title)` 要的记录（`ts` + `cycle_s/kappa` + 展示信息），
      同时回填 `case["_ts_index"][method]` 供表格按钮引用。
    """
    ctxs: list[dict] = []
    records: list[dict] = []
    for c in payload.get("cases", ()):
        data = c.get("ts_data") or {}
        ctx = c.get("ts_ctx")
        if not data or ctx is None:
            continue
        ctxs.append(ctx)
        ctx_index = len(ctxs) - 1
        idx_map: dict[str, int] = {}
        for method, rec in data.items():
            idx_map[method] = len(records)
            records.append({
                "ts": rec.get("ts"),
                "cycle_s": rec.get("cycle_s"),
                "kappa": rec.get("kappa"),
                "ctx_index": ctx_index,
                "id": c.get("id"),
                "method_label": rec.get("method_label", method),
                "objective": rec.get("objective"),
                "status": rec.get("status"),
                "path": rec.get("path"),
                "wall_s": rec.get("wall_s"),
            })
        c["_ts_index"] = idx_map
    return records, ctxs


_TS_ADAPTER = r"""
const TS_RECORDS = __TS_RECORDS__;
const TS_CTXS = __TS_CTXS__;
function showTS(k){
  const r = TS_RECORDS[k];
  if (!r) return;
  TS.init(TS_CTXS[r.ctx_index], "ts");
  const t = `${r.id} · ${r.method_label} · obj=${tsFmt(r.objective)} · `
          + `${tsFmt(r.wall_s)}s · ${r.status} · ${r.path}`;
  TS.draw(r, t);
  const panel = document.getElementById("ts-panel");
  if (panel) panel.scrollIntoView({behavior: "smooth", block: "center"});
}
"""


def render_report(payload: Mapping[str, Any]) -> str:
    meta = payload.get("meta", {})
    cfg = meta.get("config", {})
    cases = list(payload.get("cases", ()))
    ts_records, ts_ctxs = _ts_records(payload)
    sums = summaries(payload)
    groups = []
    for c in cases:
        gid = c.get("group")
        if gid and gid not in groups:
            groups.append(gid)
    groups.sort()

    blocks = []
    for gid in groups:
        gcases = [c for c in cases if c.get("group") == gid]
        label = next((c.get("group_label") for c in gcases if c.get("group_label")), gid)
        blocks.append(f'<h2>{_esc(label)}</h2>')
        for boundary in ("fixed", "adjustable"):
            t = _matrix_table({"id": gid, "label": label}, boundary, cases, sums)
            if t:
                blocks.append(t)

    scatter = _scatter(payload, sums)
    chart_block = (f'<section class="block"><h3>整模 vs 分阶段（log-log，点越靠下越快）</h3>'
                   f'{scatter}</section>') if scatter else ""

    meta_line = (f"C={cfg.get('cycle_s')}s · κ={cfg.get('kappa')} · 上限 "
                 f"{cfg.get('time_limit_s')}s · {cfg.get('reps')} 次/案例 · "
                 f"{_esc(cfg.get('backend'))} · gap={cfg.get('mip_rel_gap')} · "
                 f"heuristic={'on' if cfg.get('heuristic') else 'off'} · "
                 f"单线程 · 生成于 {_esc(meta.get('generated_at'))} · "
                 f"总墙钟 {_num(meta.get('wall_s'), 1)}s · 口径 {_esc(meta.get('config_hash'))}")
    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>干线绿波/红波 MILP · 全矩阵性能测试</title>
<style>{_CSS}</style></head>
<body>
<header>
  <h1>干线绿波/红波 MILP · 全矩阵性能测试</h1>
  <div class="meta">{meta_line}</div>
</header>
<main>
  {_kpis(payload, sums)}
  {_findings(payload, sums)}
  <h2>矩阵结果（整 = 整模，分 = 分阶段）</h2>
  {''.join(blocks)}
  {_aggregate(payload, sums)}
  {chart_block}
  <h2>明细（可筛选 / 点表头排序）</h2>
  {_detail_table(payload, sums)}
  <div id="ts-panel">
  {ts_canvas.panel_html("时距图（点明细表「时空图」列的按钮就地渲染）", "ts",
                        "尚未选择案例" if ts_records else "本报告没有内嵌时距图数据")}
  </div>
  {_notes_section(payload)}
</main>
<footer>由 <code>examples/benchmark_matrix.py</code> 生成 ·
数据文件 <code>{_esc(os.path.basename(str(payload.get('__path__', 'bench_matrix.json'))))}</code></footer>
<script>{ts_canvas.TS_JS}</script>
<script>{_TS_ADAPTER.replace("__TS_RECORDS__", _ts_json(ts_records)).replace("__TS_CTXS__", _ts_json(ts_ctxs))}</script>
<script>{_JS}</script>
</body></html>"""


def write_report(json_path: str, html_path: str, payload: Mapping[str, Any]) -> None:
    payload = dict(payload)
    payload["__path__"] = json_path
    os.makedirs(os.path.dirname(os.path.abspath(html_path)), exist_ok=True)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(render_report(payload))


def load_payload(json_path: str) -> dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


__all__ = ["summarise", "summaries", "render_report", "write_report", "load_payload"]
