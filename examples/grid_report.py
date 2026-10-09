"""格点搜索结果 → 自包含可交互 HTML 热力图（不依赖任何外部 JS/CSS）。

交互：
* 指标选择器（composite / 带宽 / 均衡组 z / 上下行带宽 / margin 罚 / 求解耗时）；
* 配色方案切换（turbo / viridis / 蓝白红）；
* 鼠标悬停显示该格点全部指标；点击钉住并在下方显示完整记录；
* ★ 标出可行格点里的最优；不可行格点用灰底斜线区分；
* Top-15 表格（Python 侧渲染）。

配色方案与指标都在 `benchmark_grid_search.py` 里定义，这里只负责画。
"""
from __future__ import annotations

import html
import json
import math
import os
from typing import Any, Mapping, Sequence

# ---------------------------------------------------------------------------
# 读写
# ---------------------------------------------------------------------------

def write_json(path: str, payload: Mapping[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def load_payload(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _esc(x: Any) -> str:
    return html.escape("" if x is None else str(x))


def _num(x: Any, digits: int = 3) -> str:
    if x is None:
        return "-"
    try:
        return f"{float(x):.{digits}f}"
    except (TypeError, ValueError):
        return _esc(x)


# ---------------------------------------------------------------------------
# 静态片段
# ---------------------------------------------------------------------------

import ts_canvas  # noqa: E402  （时距图渲染器：两份报告共用同一套画法）

#: 时距图适配层：把共享渲染器接到本报告的 DATA.ts_ctx + 每格点 .ts 上
_TS_ADAPTER = r"""
TS.init(DATA.ts_ctx, "ts");
function tsTitle(rec){
  const r = (rec.composite_objective_ratio !== undefined)
    ? rec.composite_objective_ratio.toFixed(4) : "-";
  return `${DATA.summary.case.spec_id} · C=${rec.cycle_s.toFixed(2)}s · κ=${rec.kappa.toFixed(3)}`
    + ` · composite/C=${r}（择优口径）`
    + ` · β↑=${tsFmt(rec.beta_up_s)} β↓=${tsFmt(rec.beta_down_s)} z=${tsFmt(rec.balanced_z_s)}`;
}
function drawTS(idx){
  if (idx === null || idx === undefined) return;
  TS.draw(P[idx], tsTitle(P[idx]));
}
"""


_CSS = """
:root{--bg:#f5f6f8;--card:#fff;--ink:#1e293b;--muted:#64748b;--line:#e2e8f0;
--accent:#2563eb;--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif}
header{padding:24px 30px 16px;background:linear-gradient(135deg,#1e293b,#334155);color:#fff}
header h1{margin:0 0 6px;font-size:21px;letter-spacing:.3px}
header .meta{font-size:12.5px;color:#cbd5e1;font-family:var(--mono)}
main{padding:18px 30px 60px;max-width:1680px;margin:0 auto}
h2{font-size:16px;margin:26px 0 10px;padding-bottom:6px;border-bottom:2px solid var(--line)}
h3{font-size:14px;margin:0 0 8px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:16px 0 4px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;
box-shadow:0 1px 2px rgba(15,23,42,.04)}
.card .k{font-size:12px;color:var(--muted)}
.card .v{font-size:21px;font-weight:650;font-family:var(--mono);margin:2px 0}
.card .s{font-size:11.5px;color:var(--muted)}
.panel{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;
box-shadow:0 1px 2px rgba(15,23,42,.04);margin:14px 0}
.controls{display:flex;gap:14px;align-items:center;flex-wrap:wrap}
.controls label{font-size:12.5px;color:var(--muted);display:flex;gap:6px;align-items:center}
select,button{padding:5px 9px;border:1px solid var(--line);border-radius:7px;background:#fff;
font:inherit;font-size:12.5px;cursor:pointer}
button:hover{background:#f1f5f9}
.wrap{display:flex;gap:18px;align-items:flex-start;flex-wrap:wrap}
canvas{background:#fff;border:1px solid var(--line);border-radius:8px;cursor:crosshair;display:block}
.legend{display:flex;flex-direction:column;gap:6px;font-size:12px;color:var(--muted);min-width:190px}
.legend .bar{width:26px;height:260px;border:1px solid var(--line);border-radius:6px;overflow:hidden;
position:relative;background:linear-gradient(to top,var(--lo),var(--hi))}
.legend .ticks{position:relative;height:260px;display:flex;flex-direction:column;
justify-content:space-between;font-family:var(--mono);font-size:11.5px}
#tip{position:fixed;pointer-events:none;background:rgba(15,23,42,.93);color:#fff;border-radius:8px;
padding:8px 10px;font-size:12px;font-family:var(--mono);line-height:1.5;opacity:0;transition:opacity .08s;
z-index:50;max-width:340px}
#detail table,.top table{border-collapse:collapse;width:100%;font-size:12.5px}
#detail td,.top td,.top th{border-bottom:1px solid var(--line);padding:5px 8px;text-align:left}
.top th{background:#f8fafc;position:sticky;top:0}
.top td.num,#detail td.num{font-family:var(--mono);text-align:right}
.muted{color:var(--muted)}
.badge{display:inline-block;padding:0 6px;border-radius:999px;font-size:11px;background:#e2e8f0}
.badge.ok{background:#dcfce7;color:#15803d}.badge.warn{background:#fef3c7;color:#92400e}
.badge.bad{background:#fee2e2;color:#b91c1c}
.notes ul{padding-left:18px;margin:6px 0}
.notes li{margin:3px 0;color:#475569;font-size:12.5px}
.notes code{background:#f1f5f9;padding:0 4px;border-radius:4px;font-family:var(--mono);font-size:11.5px}
"""

_JS = r"""
const DATA = __DATA__;
const CYCLES = DATA.grid.cycles, KAPPAS = DATA.grid.kappas;
const NX = CYCLES.length, NY = KAPPAS.length, P = DATA.points;
const METRICS = DATA.summary.metrics;

const CMAPS = {
  turbo: ["#30123b","#4145ab","#4675ed","#39a2fc","#1bcfd4","#24eca6","#61fc6c",
          "#a4fc3b","#d1e834","#f3c63a","#fe9b2d","#f36315","#d93806","#b11901","#7a0402"],
  viridis: ["#440154","#482878","#3e4989","#31688e","#26828e","#1f9e89","#35b779",
            "#6ece58","#b5de2b","#fde725"],
  rdbu: ["#2166ac","#67a9cf","#d1e5f0","#f7f7f7","#fddbc7","#ef8a62","#b2182b"]
};

// 初始指标**以服务端渲染的 <select> 选中项为准**（默认是归一化目标 composite/C），
// 避免"下拉框显示归一化、热力图却画绝对秒"这种口径不一致。
const state = {
  metric: ((document.getElementById("metric") || {}).value)
          || (DATA.summary && DATA.summary.best_metric) || "composite_s",
  cmap: "turbo", pinned: null, grid: true
};
const canvas = document.getElementById("heat");
const ctx = canvas.getContext("2d");
const tip = document.getElementById("tip");
const CELL = 18, PAD_L = 62, PAD_B = 46, PAD_T = 14, PAD_R = 16;
const W = PAD_L + NX * CELL + PAD_R, H = PAD_T + NY * CELL + PAD_B;

function hex2rgb(h){ return [parseInt(h.slice(1,3),16),parseInt(h.slice(3,5),16),parseInt(h.slice(5,7),16)]; }
function cmapColor(t){
  const stops = CMAPS[state.cmap]; t = Math.max(0, Math.min(1, t));
  const x = t * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(x)), f = x - i;
  const a = hex2rgb(stops[i]), b = hex2rgb(stops[i+1]);
  return `rgb(${Math.round(a[0]+(b[0]-a[0])*f)},${Math.round(a[1]+(b[1]-a[1])*f)},${Math.round(a[2]+(b[2]-a[2])*f)})`;
}

function values(){
  const key = state.metric, out = [];
  for (let i = 0; i < P.length; i++){
    const v = P[i][key];
    out.push((typeof v === "number" && isFinite(v)) ? v : null);
  }
  return out;
}
function extent(vals){
  const xs = vals.filter(v => v !== null);
  if (!xs.length) return [0, 1];
  let lo = Math.min(...xs), hi = Math.max(...xs);
  if (hi - lo < 1e-9) { hi = lo + 1e-9; }
  return [lo, hi];
}
const BEST_METRIC = (DATA.summary && DATA.summary.best_metric) || "composite_s";
function bestIndex(){
  // 按**归一化目标**（composite / C）择优，与后端排序口径一致
  let bi = -1, bv = -Infinity;
  for (let i = 0; i < P.length; i++){
    const v = P[i][BEST_METRIC];
    if (typeof v === "number" && v > bv){ bv = v; bi = i; }
  }
  if (bi >= 0) return bi;
  for (let i = 0; i < P.length; i++){
    const v = P[i].composite_s;
    if (typeof v === "number" && v > bv){ bv = v; bi = i; }
  }
  return bi;
}
const BEST = bestIndex();

function draw(){
  const dpr = window.devicePixelRatio || 1;
  canvas.style.width = W + "px"; canvas.style.height = H + "px";
  canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  const vals = values(), [lo, hi] = extent(vals);

  for (let iy = 0; iy < NY; iy++){
    for (let ix = 0; ix < NX; ix++){
      const rec = P[iy * NX + ix];
      const x = PAD_L + ix * CELL, y = PAD_T + (NY - 1 - iy) * CELL;
      const v = vals[iy * NX + ix];
      if (v === null){
        ctx.fillStyle = "#e5e7eb"; ctx.fillRect(x, y, CELL, CELL);
        ctx.strokeStyle = "#cbd5e1"; ctx.lineWidth = 1;
        for (let d = -CELL; d < CELL; d += 5){
          ctx.beginPath(); ctx.moveTo(x + Math.max(0, d), y + Math.max(0, -d));
          ctx.lineTo(x + Math.min(CELL, CELL + d), y + Math.min(CELL, CELL + d));
          ctx.stroke();
        }
      } else {
        ctx.fillStyle = cmapColor((v - lo) / (hi - lo));
        ctx.fillRect(x, y, CELL, CELL);
      }
    }
  }
  if (state.grid){
    ctx.strokeStyle = "rgba(148,163,184,.35)"; ctx.lineWidth = 1;
    for (let ix = 0; ix <= NX; ix += 5){
      ctx.beginPath(); ctx.moveTo(PAD_L + ix * CELL, PAD_T);
      ctx.lineTo(PAD_L + ix * CELL, PAD_T + NY * CELL); ctx.stroke();
    }
    for (let iy = 0; iy <= NY; iy += 5){
      ctx.beginPath(); ctx.moveTo(PAD_L, PAD_T + iy * CELL);
      ctx.lineTo(PAD_L + NX * CELL, PAD_T + iy * CELL); ctx.stroke();
    }
  }
  // 最优格点
  if (BEST >= 0){
    const ix = BEST % NX, iy = Math.floor(BEST / NX);
    const cx = PAD_L + ix * CELL + CELL / 2, cy = PAD_T + (NY - 1 - iy) * CELL + CELL / 2;
    ctx.lineWidth = 2; ctx.strokeStyle = "#fff";
    ctx.beginPath(); ctx.arc(cx, cy, CELL * 0.62, 0, 2 * Math.PI); ctx.stroke();
    ctx.fillStyle = "#fff"; ctx.font = "bold 11px sans-serif";
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText("★", cx, cy + 0.5);
  }
  // 钉住的格点
  if (state.pinned !== null){
    const ix = state.pinned % NX, iy = Math.floor(state.pinned / NX);
    ctx.lineWidth = 2; ctx.strokeStyle = "#0f172a";
    ctx.strokeRect(PAD_L + ix * CELL - 1, PAD_T + (NY - 1 - iy) * CELL - 1, CELL + 2, CELL + 2);
  }
  // 坐标轴
  ctx.fillStyle = "#334155"; ctx.font = "11.5px " + getComputedStyle(document.body).fontFamily;
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  for (let ix = 0; ix < NX; ix += 5){
    ctx.fillText(CYCLES[ix].toFixed(1), PAD_L + ix * CELL + CELL / 2, PAD_T + NY * CELL + 6);
  }
  ctx.textAlign = "right"; ctx.textBaseline = "middle";
  for (let iy = 0; iy < NY; iy += 5){
    ctx.fillText(KAPPAS[iy].toFixed(3), PAD_L - 8, PAD_T + (NY - 1 - iy) * CELL + CELL / 2);
  }
  ctx.fillStyle = "#1e293b"; ctx.font = "12.5px " + getComputedStyle(document.body).fontFamily;
  ctx.textAlign = "center"; ctx.textBaseline = "bottom";
  ctx.fillText("公共周期 C (s)", PAD_L + NX * CELL / 2, H - 6);
  ctx.save(); ctx.translate(14, PAD_T + NY * CELL / 2); ctx.rotate(-Math.PI / 2);
  ctx.textBaseline = "top"; ctx.fillText("速度倍率 κ", 0, 0); ctx.restore();

  // 图例
  const lgTitle = document.getElementById("lg-title");
  if (lgTitle) lgTitle.textContent = state.metric + (METRICS[state.metric] ? "　" + METRICS[state.metric].split("=")[0].trim() : "");
  document.getElementById("lg-lo").textContent = lo.toFixed(3);
  document.getElementById("lg-hi").textContent = hi.toFixed(3);
  document.getElementById("lg-mid").textContent = ((lo + hi) / 2).toFixed(3);
  const bar = document.getElementById("lg-bar");
  bar.style.setProperty("--lo", cmapColor(0));
  bar.style.setProperty("--hi", cmapColor(1));
}

function cellAt(ev){
  const r = canvas.getBoundingClientRect();
  const x = ev.clientX - r.left - PAD_L, y = ev.clientY - r.top - PAD_T;
  const ix = Math.floor(x / CELL), iy = NY - 1 - Math.floor(y / CELL);
  if (ix < 0 || iy < 0 || ix >= NX || iy >= NY) return null;
  return iy * NX + ix;
}

function statusBadge(st){
  const cls = st === "optimal" ? "ok" : (st === "infeasible" ? "bad" : (st === "time_limit" ? "warn" : ""));
  return `<span class="badge ${cls}">${st}</span>`;
}
function tipHtml(i){
  const r = P[i];
  const rows = [
    ["C / κ", `${r.cycle_s.toFixed(3)} s / ${r.kappa.toFixed(4)}`],
    ["状态", statusBadge(r.status)],
  ];
  for (const key in METRICS){
    if (r[key] === undefined || r[key] === null) continue;
    const v = (key === "n_bands" || key === "model_vars" || key === "model_cons")
      ? String(r[key]) : Number(r[key]).toFixed(3);
    rows.push([METRICS[key].split("=")[0].trim().slice(0, 18), v]);
  }
  return rows.map(([k, v]) => `${k}: ${v}`).join("<br>");
}

function detailHtml(i){
  const r = P[i];
  const rows = Object.keys(r).map(k => `<tr><td class="muted">${k}</td><td class="num">${
    (typeof r[k] === "number") ? Number(r[k]).toFixed(4) : r[k]}</td></tr>`).join("");
  return `<h3>格点详情：C = ${r.cycle_s.toFixed(3)} s，κ = ${r.kappa.toFixed(4)}</h3>
          <table>${rows}</table>`;
}

canvas.addEventListener("mousemove", ev => {
  const i = cellAt(ev);
  if (i === null){ tip.style.opacity = 0; return; }
  tip.innerHTML = tipHtml(i);
  tip.style.opacity = 1;
  tip.style.left = Math.min(window.innerWidth - 360, ev.clientX + 14) + "px";
  tip.style.top = Math.min(window.innerHeight - 200, ev.clientY + 14) + "px";
});
canvas.addEventListener("mouseleave", () => { tip.style.opacity = 0; });
canvas.addEventListener("click", ev => {
  const i = cellAt(ev);
  if (i === null) return;
  state.pinned = i; draw();
  document.getElementById("detail").innerHTML = detailHtml(i);
  drawTS(i);
});

document.getElementById("metric").addEventListener("change", e => {
  state.metric = e.target.value; draw();
});
document.getElementById("cmap").addEventListener("change", e => {
  state.cmap = e.target.value; draw();
});
document.getElementById("grid").addEventListener("change", e => {
  state.grid = e.target.checked; draw();
});
document.getElementById("jump").addEventListener("click", () => {
  if (BEST < 0) return;
  state.pinned = BEST; draw();
  document.getElementById("detail").innerHTML = detailHtml(BEST);
  drawTS(BEST);
  document.getElementById("detail").scrollIntoView({behavior: "smooth", block: "nearest"});
});

"""

#: 启动顺序：共享渲染器 → 本报告 JS（含 DATA）→ 适配层 → 首屏绘制
_BOOTSTRAP = """
draw();
if (BEST >= 0){ state.pinned = BEST; draw(); drawTS(BEST); }
"""


def _options(metrics: Mapping[str, str], default: str) -> str:
    """按"归一化 / 绝对秒 / 其它"分组渲染指标下拉（归一化在前，默认选中）。"""
    groups: list[tuple[str, list[tuple[str, str]]]] = [
        ("归一化（除以周期 C，周期比例）", []),
        ("绝对秒", []),
        ("规模与开销", []),
    ]
    for k, v in metrics.items():
        if k.endswith("_ratio"):
            groups[0][1].append((k, v))
        elif k.endswith("_s"):
            groups[1][1].append((k, v))
        else:
            groups[2][1].append((k, v))
    out = []
    for title, items in groups:
        if not items:
            continue
        out.append(f'<optgroup label="{_esc(title)}">')
        for k, v in items:
            out.append(f'<option value="{_esc(k)}"'
                       f'{" selected" if k == default else ""}>'
                       f'{_esc(k)}　{_esc(v)}</option>')
        out.append("</optgroup>")
    return "".join(out)


def _top_table(summary: Mapping[str, Any]) -> str:
    top = summary.get("top") or []
    if not top:
        return '<p class="muted">没有可行格点。</p>'
    head = ("<tr><th>#</th><th>C (s)</th><th>κ</th>"
             "<th>composite / C ★</th><th>composite (s)</th><th>带宽比</th>"
             "<th>z / C</th><th>β↑ / C</th><th>β↓ / C</th><th>耗时 (s)</th></tr>")
    rows = []
    for i, r in enumerate(top, 1):
        rows.append(
            f'<tr><td>{i}</td><td class="num">{_num(r.get("cycle_s"), 2)}</td>'
            f'<td class="num">{_num(r.get("kappa"), 3)}</td>'
            f'<td class="num"><b>{_num(r.get("composite_objective_ratio"), 4)}</b></td>'
            f'<td class="num">{_num(r.get("composite_s"), 2)}</td>'
            f'<td class="num">{_num(r.get("bandwidth_objective_ratio"), 4)}</td>'
            f'<td class="num">{_num(r.get("balanced_z_ratio"), 4)}</td>'
            f'<td class="num">{_num(r.get("beta_up_ratio"), 4)}</td>'
            f'<td class="num">{_num(r.get("beta_down_ratio"), 4)}</td>'
            f'<td class="num">{_num(r.get("wall_s"), 3)}</td></tr>')
    return f'<div class="top" style="max-height:420px;overflow:auto"><table>{head}{"".join(rows)}</table></div>'


def _cards(payload: Mapping[str, Any]) -> str:
    s = payload["summary"]
    cfg = s["config"]
    best = (s.get("top") or [None])[0]
    n_points = cfg["n_cycles"] * cfg["n_kappas"]
    per = s["wall_s"] / n_points if n_points else 0
    cards = [
        ("格点数", f"{n_points}", f"{cfg['n_cycles']}×{cfg['n_kappas']} 个 (C, κ) 组合"),
        ("可行 / 总", f"{s['n_feasible']} / {n_points}",
         "；".join(f"{k}:{v}" for k, v in s["status_count"].items())),
        ("最优 composite / C", _num(best.get("composite_objective_ratio") if best else None, 4),
         (f"C={_num(best.get('cycle_s'), 2)}s，κ={_num(best.get('kappa'), 3)}"
          f"（绝对 {_num(best.get('composite_s'), 2)}s）") if best else "-"),
        ("总用时", f"{s['wall_s']:.1f}s", f"平均每格点 {per * 1000:.0f} ms（含建模）"),
        ("目标", "z + 0.5·(β↑+β↓)", _esc(s["case"]["objective_label"])),
    ]
    return ('<section class="cards">' + "".join(
        f'<div class="card"><div class="k">{_esc(k)}</div><div class="v">{v}</div>'
        f'<div class="s">{sub}</div></div>' for k, v, sub in cards) + "</section>")


def render_report(payload: Mapping[str, Any]) -> str:
    s = payload["summary"]
    cfg, case = s["config"], s["case"]
    metrics = s.get("metrics") or {}
    # 默认展示**归一化**指标（除以周期 C），与 matplotlib 版热力图口径一致
    default_metric = ("composite_objective_ratio" if "composite_objective_ratio" in metrics
                      else ("composite_s" if "composite_s" in metrics
                            else next(iter(metrics), "")))
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    js = (ts_canvas.TS_JS
          + _JS.replace("__DATA__", data)
          + _TS_ADAPTER
          + _BOOTSTRAP)
    meta = (f"{_esc(case['spec_id'])} · {_esc(case['objective_label'])} · 方法 {_esc(cfg['method'])}"
            f" · C ∈ [{cfg['cycle_min']:g}, {cfg['cycle_max']:g}]s × κ ∈ "
            f"[{cfg['kappa_min']:g}, {cfg['kappa_max']:g}] · {cfg['n_cycles']}×{cfg['n_kappas']}"
            f" = {cfg['n_cycles'] * cfg['n_kappas']} 格点 · 单格点时限 {cfg['time_limit_s']:g}s"
            f" · gap={cfg['mip_rel_gap']:g} · {_esc(cfg['backend'])} · heuristic="
            f"{'on' if cfg['heuristic'] else 'off'} · 进程 {cfg['num_workers']}"
            f" · 生成于 {_esc(s['generated_at'])} · 总用时 {s['wall_s']:.1f}s")
    metric_items = "".join(f"<li><code>{_esc(k)}</code>：{_esc(v)}</li>"
                           for k, v in metrics.items())
    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>标准案例格点搜索 · 可交互热力图</title>
<style>{_CSS}</style></head>
<body>
<header>
  <h1>标准案例精细格点搜索 · 可交互热力图</h1>
  <div class="meta">{meta}</div>
</header>
<main>
  {_cards(payload)}
  <div class="panel">
    <div class="controls">
      <label>指标 <select id="metric">{_options(metrics, default_metric)}</select></label>
      <label>配色 <select id="cmap">
        <option value="turbo" selected>turbo</option>
        <option value="viridis">viridis</option>
        <option value="rdbu">蓝白红</option>
      </select></label>
      <label><input type="checkbox" id="grid" checked> 网格线</label>
      <button id="jump">★ 跳到最优格点</button>
      <span class="muted">鼠标悬停看数值，点击钉住并在下方看完整记录</span>
    </div>
  </div>
  <div class="panel wrap">
    <canvas id="heat"></canvas>
    <div class="legend">
      <div style="font-weight:600;color:#334155" id="lg-title">数值</div>
      <div style="display:flex;gap:8px">
        <div class="bar" id="lg-bar"></div>
        <div class="ticks"><span id="lg-hi"></span><span id="lg-mid"></span><span id="lg-lo"></span></div>
      </div>
      <div class="muted">灰底斜线 = 该格点不可行 / 无解<br>★ = <b>归一化目标</b>
    (<code>composite / C</code>) 最优的可行格点</div>
    </div>
  </div>
  {ts_canvas.panel_html("时距图（点击上面任意格点即可就地渲染）", "ts",
                        "尚未选择格点")}
  <div class="panel" id="detail"><p class="muted">点击热力图任意格点，这里显示它的完整记录。</p></div>
  <h2>Top-15 格点（按 <code>composite / C</code> 归一化目标排序）</h2>
  <div class="panel">{_top_table(s)}</div>
  <h2>口径说明</h2>
  <section class="notes panel">
    <div class="wrap" style="align-items:flex-start">
      <div style="flex:1;min-width:320px">
        <h3>案例与目标</h3>
        <ul>
          <li>案例：<code>{_esc(case['spec_id'])}</code> ——
              <b>无双周期 · 单方案 · 可调相位边界</b>，标准案例
              <code>examples/standard_problem.json</code>。</li>
          <li>目标：{_esc(case['objective_label'])}——{_esc(case['objective_desc'])}；
              <code>composite = z·1.0 + (β↑+β↓)·0.5</code>，其中 <code>z = min(β↑, β↓)</code>。</li>
          <li>方法：<b>整模</b>（一次性完整 MILP，δ 自由），每个格点独立求解。</li>
        </ul>
      </div>
      <div style="flex:1;min-width:320px">
        <h3>指标含义</h3>
        <ul>{metric_items}</ul>
      </div>
      <div style="flex:1;min-width:320px">
        <h3>读图提示</h3>
        <ul>
          <li><code>κ</code> 越大表示设计速度越快，行程时间 <code>τ = 距离/(κ·设计速度)</code> 越短；
              所以 κ 小（走得慢）时，绿波更难在有限周期内贯通。</li>
          <li>周期 C 与行程时间是否"整配"（C 能整除/接近覆盖全线行程）会形成明显的条纹结构。</li>
          <li>灰色格点表示该 (C, κ) 下案例不可行（预处理就判定无可行方案或模型无解）。</li>
          <li>耗时指标反映的是"建模 + 求解"，包含 FeasibilityFilter 与 ModelBuilder 的开销。</li>
          <li><b>最优格点按归一化目标判定</b>：★ 标记、"跳到最优"、Top-15 排序
              都用 <code>composite_objective_ratio = composite / C</code>，
              而不是绝对秒 —— 绝对秒会天然偏向大周期（实测两个口径的最优点完全不同）。</li>
          <li><b>指标按周期归一化</b>：热力图默认展示 <code>*_ratio</code> 指标 =
              秒值 ÷ 该格点的周期 C（单位：周期比例，无量纲），横跨不同 C 才能直接比较；
              损失类取负号（<code>negative_*</code>）以保证"越大越好"。
              这与库里的 matplotlib 版热力图 <code>GridHeatmapDiagram</code>
              （<code>bandwidth_objective_ratio / negative_margin_loss_ratio /
              composite_objective_ratio</code>）口径一致；绝对秒的指标在同一下拉框的
              "绝对秒"分组里，<code>wall_s</code> 是时间开销、不做归一化。</li>
          <li><b>时距图</b>：点击任意格点即可就地渲染该 (C, κ) 解的时距图
              （φ 与各相位边界的微调 δ 都内嵌在数据里，画法与 Python 版
              <code>TimeSpaceDiagram</code> 一致），可勾选灯条/绿波带/周期分界，也可下载 PNG。</li>
        </ul>
      </div>
    </div>
  </section>
</main>
<div id="tip"></div>
<script>{js}</script>
</body></html>"""


def write_report(json_path: str, html_path: str, payload: Mapping[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(html_path)), exist_ok=True)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(render_report(payload))


__all__ = ["write_json", "load_payload", "render_report", "write_report"]
