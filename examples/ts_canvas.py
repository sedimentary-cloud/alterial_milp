"""时距图的**共享**渲染资产：Python 侧上下文构造 + 浏览器侧 canvas 渲染。

两份报告都用它，保证口径一致：

* `benchmark_grid_search.py` + `grid_report.py`：点击格点渲染该 (C, κ) 的解；
* `benchmark_matrix.py` + `bench_report.py`：点击明细表里的"整模 / 分阶段"渲染该案例的解。

数据契约
--------
静态上下文（每份报告只嵌一份或每案例一份）::

    "ts_ctx": {
      "nodes": [...], "positions": [...], "bar_h": ...,
      "tau1": {"I1|up": 13.33, ...},                     # κ = 1 的每段行程（秒）
      "plans": {"I2_P1": {"intersection": "I2",
                          "directions": {"up": {"edges": [...], "windows": [[a,b,wrap], ...]},
                                         "down": {...}}}, ...},
      "demands": [{"id": ..., "type": "green"|"red", "direction": ...,
                   "nodes": [...], "cum_tau1": [...]}, ...],
      "delta_order": [["I2_P1", "up", 0], ...],          # 扁平 δ 的顺序
      "colors": {...}
    }

每个点/每次求解::

    "ts": {"plan_by_node": {"I1": "I1_P1", ...},
           "phi": [...], "delta": [...],                # 与 delta_order 对齐
           "bands": [{"id": ..., "u": ..., "beta": ...}, ...]}

画法逐条对齐 `greenwave/report/diagram.py`：红灯底条按全局周期平铺、绿窗按 φ 平移叠加、
绿波带按 `u + cum_T(κ)` 连成斜带；**红波带**用库里的 `_red_band_display_times` 同一套
搜索（±8 个周期副本里找"上游落在真实绿窗、下游落在真实红窗"的那一份）。
"""
from __future__ import annotations

from typing import Any, Mapping


# ---------------------------------------------------------------------------
# Python 侧：静态上下文
# ---------------------------------------------------------------------------

def draw_context(problem) -> dict[str, Any]:
    """时距图的静态部分（与 (C, κ) 无关）：走廊版面 + 所有方案的窗口结构 + 需求链。"""
    corr = problem.corridor
    nodes = list(corr.intersections)
    positions, pos = [0.0], 0.0
    for seg in corr.segments:
        pos += float(seg.distance_for("up"))
        positions.append(pos)

    tau1: dict[str, float] = {}
    for idx, seg in enumerate(corr.segments):
        tau1[f"{nodes[idx]}|up"] = float(seg.distance_for("up")) / float(seg.speed_up_mps)
    for idx in range(1, len(nodes)):
        seg = corr.segments[idx - 1]
        tau1[f"{nodes[idx]}|down"] = (float(seg.distance_for("down"))
                                      / float(seg.speed_down_mps))

    delta_order: list[list[Any]] = []
    plans: dict[str, Any] = {}
    for p in problem.plans:
        per_dir: dict[str, Any] = {}
        for d in ("up", "down"):
            edges: list[dict[str, Any]] = []
            for kind, src, offset in p.expanded_edge_sources(d):
                if kind == "const":
                    edges.append({"const": float(offset)})
                else:
                    j = int(src)
                    nxt = len(delta_order)
                    delta_order.append([p.id, d, j])
                    edges.append({"bnd": nxt,
                                  "nominal": float(p.nominal_boundary_r(d, j)),
                                  "offset": float(offset)})
            mask = p.expanded_mask(d)
            windows: list[list[int]] = []
            i = 0
            while i < len(mask):
                if not mask[i]:
                    i += 1
                    continue
                j = i
                while j + 1 < len(mask) and mask[j + 1]:
                    j += 1
                windows.append([i, j + 1, 0])
                i = j + 1
            per_dir[d] = {"edges": edges, "windows": windows}
        plans[p.id] = {"intersection": p.intersection, "directions": per_dir}

    demands: list[dict[str, Any]] = []
    for b in problem.objective.band_demands:
        cum, total = [0.0], 0.0
        for a_, b_ in zip(b.nodes, b.nodes[1:]):
            total += tau1.get(f"{a_}|{b.direction}", 0.0)
            cum.append(total)
        demands.append({"id": b.id, "type": b.type, "direction": b.direction,
                        "nodes": list(b.nodes), "cum_tau1": cum})

    return {
        "nodes": nodes, "positions": positions,
        "bar_h": max(pos * 0.008, 0.5),
        "tau1": tau1, "plans": plans, "demands": demands,
        "delta_order": delta_order,
        "colors": {"up_window": "#32cd32", "down_window": "#17becf",
                   "red": "#ff0000", "up_band": "#32cd32", "down_band": "#4f86d6",
                   "red_band": "#e05252"},
    }


def draw_payload(solution, ctx_obj: Mapping[str, Any]) -> dict[str, Any]:
    """把一次求解的结果压成 `ts` 载荷（φ、选中方案、δ、带）。"""
    order = ctx_obj.get("delta_order") or []
    delta = [float(solution.adjustments_r.get((k[0], k[1], k[2]), 0.0))
             for k in order]
    return {
        "plan_by_node": dict(solution.plan_selection),
        "phi": [float(solution.offsets_s.get(iid, 0.0))
                for iid in ctx_obj.get("nodes", ())],
        "delta": delta,
        "bands": [{"id": b.demand, "u": float(b.start_s), "width": float(b.width_s)}
                  for b in solution.bands],
    }


# ---------------------------------------------------------------------------
# 浏览器侧：面板 + 渲染器
# ---------------------------------------------------------------------------

def panel_html(title: str = "时距图", canvas_id: str = "ts",
               caption: str = "尚未选择") -> str:
    """可复用的时距图面板（勾选框 id 固定，渲染器会自动接线）。"""
    return f'''  <h2>{title}</h2>
  <div class="panel">
    <div class="controls">
      <label><input type="checkbox" id="ts-bars" checked> 信号灯条</label>
      <label><input type="checkbox" id="ts-bands" checked> 波带</label>
      <label><input type="checkbox" id="ts-ticks" checked> 周期分界</label>
      <button id="ts-download">下载 PNG</button>
      <span class="muted" id="ts-caption">{caption}</span>
    </div>
    <canvas id="{canvas_id}"></canvas>
  </div>'''


TS_JS = r'''
// ======================= 时距图（canvas 就地渲染，绿波 + 红波） =======================
// 画法与 greenwave/report/diagram.py 对齐：
//   * 每路口两条灯条（上行在路口线上方、下行在下方）：红灯底条按全局周期平铺，绿窗按 φ 平移叠上；
//   * 绿波带：用 u（链首到达）与 β 连成斜带，按周期重复铺满可见范围；
//   * 红波带：用库里同一套"展示副本"搜索（±8 个周期里找上游落在真实绿窗、
//     下游落在真实红窗的那一份）。
const TS = (() => {
  const W = 1000, H = 480, L = 70, R = 176, T = 34, B = 46;
  let ctxObj = null, canvas = null, c2d = null, cur = null;

  function init(ctxIn, canvasId) {
    ctxObj = ctxIn;
    canvas = document.getElementById(canvasId || "ts");
    c2d = canvas ? canvas.getContext("2d") : null;
    ["ts-bars", "ts-bands", "ts-ticks"].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.addEventListener("change", () => { if (cur) draw(cur.rec, cur.title); });
    });
    const dl = document.getElementById("ts-download");
    if (dl) dl.addEventListener("click", () => {
      const a = document.createElement("a");
      const tb = (cur && cur.rec && cur.rec.cycle_s !== undefined && cur.rec.kappa !== undefined)
        ? `_C${Number(cur.rec.cycle_s).toFixed(2)}_k${Number(cur.rec.kappa).toFixed(3)}` : "";
      a.download = "timespace" + tb + ".png";
      a.href = canvas.toDataURL("image/png");
      a.click();
    });
  }

  function color(name) { return (ctxObj.colors && ctxObj.colors[name]) || "#888"; }
  function hexA(hex, a) {
    const r = parseInt(hex.slice(1, 3), 16), g = parseInt(hex.slice(3, 5), 16),
          b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r},${g},${b},${a})`;
  }
  function num(v, d) { return (typeof v === "number") ? v.toFixed(d === undefined ? 2 : d) : "-"; }
  function niceStep(t) {
    const raw = t / 6, pow = Math.pow(10, Math.floor(Math.log10(raw))), m = raw / pow;
    return (m >= 5 ? 5 : m >= 2 ? 2 : 1) * pow;
  }

  function edgeRatio(edges, idx, delta) {
    const e = edges[idx];
    if (e.const !== undefined) return e.const;
    return e.nominal + (e.offset || 0) + (delta[e.bnd] || 0);
  }
  function planOf(rec, node) {
    return (rec.ts.plan_by_node || {})[node] || null;
  }
  function greenIntervals(rec, node, dir, phi, C, nMin, nMax) {
    const pid = planOf(rec, node);
    if (!pid || !ctxObj.plans[pid]) return [];
    const spec = ctxObj.plans[pid].directions[dir];
    const out = [];
    for (const win of spec.windows) {
      const s = edgeRatio(spec.edges, win[0], rec.ts.delta) * C + phi;
      const e = (edgeRatio(spec.edges, win[1], rec.ts.delta) + win[2]) * C + phi;
      if (e <= s) continue;
      for (let n = nMin; n <= nMax; n++) out.push([s + n * C, e + n * C]);
    }
    out.sort((a, b) => (a[0] - b[0]) || (a[1] - b[1]));
    return out;
  }
  function redIntervals(greens) {
    const reds = [];
    for (let i = 0; i + 1 < greens.length; i++) {
      if (greens[i][1] < greens[i + 1][0] - 1e-9) reds.push([greens[i][1], greens[i + 1][0]]);
    }
    return reds;
  }
  function contains(intervals, lo, hi, tol) {
    tol = tol === undefined ? 1e-5 : tol;
    for (const [a, b] of intervals) if (lo >= a - tol && hi <= b + tol) return [a, b];
    return null;
  }
  // 与库函数 _red_band_display_times 同一套搜索
  function redDisplayTimes(rec, upNode, downNode, dir, u, width) {
    const C = rec.cycle_s, K = rec.kappa, nodes = ctxObj.nodes;
    const tau = (ctxObj.tau1[upNode + "|" + dir] || 0) / K;
    const phiUp = (rec.ts.phi[nodes.indexOf(upNode)] || 0) % C;
    const phiDown = (rec.ts.phi[nodes.indexOf(downNode)] || 0) % C;
    const ups = greenIntervals(rec, upNode, dir, phiUp, C, -6, 6);
    const reds = redIntervals(greenIntervals(rec, downNode, dir, phiDown, C, -6, 6));
    let best = null;
    for (let n = -8; n <= 8; n++) {
      const dnLo = u + n * C, dnHi = dnLo + width;
      const upLo = dnLo - tau, upHi = upLo + width;
      if (upLo < -1e-6 || dnLo < -1e-6) continue;
      if (!contains(reds, dnLo, dnHi)) continue;
      if (!contains(ups, upLo, upHi)) continue;
      if (best === null || dnLo < best[0]) best = [dnLo, upLo];
    }
    if (best) return best;
    const dnLo0 = u, upLo0 = u - tau;
    const extra = C > 0 ? Math.max(0, Math.ceil((0 - upLo0) / C)) : 0;
    return [dnLo0 + extra * C, upLo0 + extra * C];
  }

  function draw(rec, title) {
    if (!canvas || !rec || !rec.ts) return;
    cur = { rec: rec, title: title };
    const C = rec.cycle_s, K = rec.kappa, nodes = ctxObj.nodes;
    const total = ctxObj.positions[ctxObj.positions.length - 1];
    const greenDemands = ctxObj.demands.filter(d => d.type === "green");
    const maxSpan = Math.max(0, ...greenDemands.map(d => d.cum_tau1[d.cum_tau1.length - 1] / K));
    const widths = (rec.ts.bands || []).map(b => b.width || b.beta || 0);
    const maxBeta = widths.length ? Math.max(...widths) : 0;
    let tMax = maxSpan + maxBeta + C;
    tMax = Math.max(Math.ceil(tMax / C) * C, 2 * C);
    const kMin = -Math.ceil((maxSpan + maxBeta) / C) - 2, kMax = Math.ceil(tMax / C) + 1;
    const dpr = window.devicePixelRatio || 1;
    canvas.style.width = W + "px"; canvas.style.height = H + "px";
    canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
    c2d.setTransform(dpr, 0, 0, dpr, 0, 0);
    c2d.clearRect(0, 0, W, H);
    const plotW = W - L - R, plotH = H - T - B;
    const X = t => L + t / tMax * plotW;
    const Y = y => T + (1 - y / total) * plotH;
    const barHpx = Math.max(2, plotH * ctxObj.bar_h / total);
    const nodeY = nodes.map((_, i) => Y(ctxObj.positions[i]));
    const laneY = (i, dir) => dir === "up" ? nodeY[i] + barHpx / 2 : nodeY[i] - barHpx / 2;
    const showBars = checked("ts-bars"), showBands = checked("ts-bands"), showTicks = checked("ts-ticks");

    c2d.fillStyle = "rgba(248,250,252,.75)";
    c2d.fillRect(L, T, plotW, plotH);
    // 周期竖线
    c2d.strokeStyle = "#eef2f7"; c2d.lineWidth = 1;
    for (let k = 0; k * C <= tMax + 1e-9; k++) {
      c2d.beginPath(); c2d.moveTo(X(k * C), T); c2d.lineTo(X(k * C), T + plotH); c2d.stroke();
    }
    // 信号灯条
    if (showBars) {
      for (let i = 0; i < nodes.length; i++) {
        for (const dir of ["up", "down"]) {
          const cy = laneY(i, dir);
          c2d.fillStyle = color("red");
          for (let k = kMin; k <= kMax; k++) {
            c2d.fillRect(X(k * C), cy - barHpx / 2, X((k + 1) * C) - X(k * C), barHpx);
          }
          const pid = planOf(rec, nodes[i]);
          if (!pid) continue;
          const spec = ctxObj.plans[pid].directions[dir];
          const phi = (rec.ts.phi[i] || 0) % C;
          c2d.fillStyle = dir === "up" ? color("up_window") : color("down_window");
          for (const win of spec.windows) {
            const s0 = edgeRatio(spec.edges, win[0], rec.ts.delta) * C + phi;
            const e0 = (edgeRatio(spec.edges, win[1], rec.ts.delta) + win[2]) * C + phi;
            if (e0 <= s0) continue;
            for (let k = kMin; k <= kMax; k++) {
              c2d.fillRect(X(s0 + k * C), cy - barHpx / 2, X(e0 + k * C) - X(s0 + k * C), barHpx);
            }
          }
        }
      }
    }
    // 波带
    if (showBands) {
      for (const band of (rec.ts.bands || [])) {
        const dm = ctxObj.demands.find(d => d.id === band.id);
        if (!dm) continue;
        const width = band.width !== undefined ? band.width : band.beta;
        if (!(width > 1e-7)) continue;
        if (dm.type === "green") {
          const col = dm.direction === "up" ? color("up_band") : color("down_band");
          const base = dm.cum_tau1.map(t => (band.u % C) + t / K);
          const ys = dm.nodes.map(n => laneY(nodes.indexOf(n), dm.direction));
          c2d.fillStyle = hexA(col, 0.42); c2d.strokeStyle = col; c2d.lineWidth = 1.4;
          for (let k = kMin; k <= kMax; k++) {
            c2d.beginPath();
            for (let j = 0; j < base.length; j++) c2d.lineTo(X(base[j] + k * C), ys[j]);
            for (let j = base.length - 1; j >= 0; j--) c2d.lineTo(X(base[j] + width + k * C), ys[j]);
            c2d.closePath(); c2d.fill(); c2d.stroke();
          }
          const mid = Math.floor(base.length / 2);
          c2d.fillStyle = "#0f172a"; c2d.font = "11px sans-serif";
          c2d.textAlign = "left"; c2d.textBaseline = "middle";
          c2d.fillText(`${band.id} ${width.toFixed(1)}s`, X(base[mid]) + 5, ys[mid] - barHpx * 1.7);
        } else {
          const [upNode, downNode] = dm.nodes;
          const iUp = nodes.indexOf(upNode), iDown = nodes.indexOf(downNode);
          const yUp = laneY(iUp, dm.direction), yDown = laneY(iDown, dm.direction);
          const [downBase, upBase] = redDisplayTimes(rec, upNode, downNode, dm.direction,
                                                     band.u, width);
          const col = color("red_band");
          c2d.fillStyle = hexA(col, 0.38); c2d.strokeStyle = col; c2d.lineWidth = 1.3;
          for (let k = kMin; k <= kMax; k++) {
            const us = upBase + k * C, ue = us + width;
            const ds = downBase + k * C, de = ds + width;
            c2d.beginPath();
            c2d.lineTo(X(us), yUp); c2d.lineTo(X(ue), yUp);
            c2d.lineTo(X(de), yDown); c2d.lineTo(X(ds), yDown);
            c2d.closePath(); c2d.fill(); c2d.stroke();
          }
          const midT = downBase + width / 2;
          c2d.fillStyle = "#7f1d1d"; c2d.font = "11px sans-serif";
          c2d.textAlign = "left"; c2d.textBaseline = "middle";
          c2d.fillText(`${band.id} ${width.toFixed(1)}s`, X(midT) + 5, (yUp + yDown) / 2 - barHpx * 1.7);
        }
      }
    }
    // 路口线与名称
    c2d.font = "11px sans-serif";
    for (let i = 0; i < nodes.length; i++) {
      const y = nodeY[i];
      c2d.strokeStyle = "#94a3b8"; c2d.lineWidth = 0.8;
      c2d.beginPath(); c2d.moveTo(L, y); c2d.lineTo(L + plotW, y); c2d.stroke();
      c2d.fillStyle = "#1e293b"; c2d.textAlign = "right"; c2d.textBaseline = "middle";
      c2d.fillText(nodes[i], L - 6, y);
      if (showTicks) {
        const phi = (rec.ts.phi[i] || 0) % C;
        c2d.strokeStyle = "rgba(15,23,42,.55)"; c2d.lineWidth = 0.7;
        for (let k = kMin; k <= kMax; k++) {
          const t = phi + k * C;
          if (t < 0 || t > tMax) continue;
          c2d.beginPath(); c2d.moveTo(X(t), y - barHpx); c2d.lineTo(X(t), y + barHpx); c2d.stroke();
        }
      }
    }
    // 坐标轴
    c2d.strokeStyle = "#cbd5e1"; c2d.lineWidth = 1;
    c2d.beginPath(); c2d.moveTo(L, T); c2d.lineTo(L, T + plotH); c2d.lineTo(L + plotW, T + plotH); c2d.stroke();
    c2d.fillStyle = "#475569"; c2d.font = "11.5px sans-serif";
    c2d.textAlign = "center"; c2d.textBaseline = "top";
    const step = niceStep(tMax);
    for (let t = 0; t <= tMax + 1e-9; t += step) {
      c2d.fillStyle = "#475569"; c2d.fillText(t.toFixed(0), X(t), T + plotH + 6);
      c2d.strokeStyle = "#e2e8f0";
      c2d.beginPath(); c2d.moveTo(X(t), T); c2d.lineTo(X(t), T + plotH); c2d.stroke();
    }
    c2d.fillStyle = "#1e293b";
    c2d.fillText("时间 (s)", L + plotW / 2, T + plotH + 24);
    c2d.save(); c2d.translate(16, T + plotH / 2); c2d.rotate(-Math.PI / 2);
    c2d.textBaseline = "top"; c2d.fillText("累计距离 (m)", 0, 0); c2d.restore();
    // 标题与图例
    c2d.fillStyle = "#0f172a"; c2d.font = "12.5px sans-serif";
    c2d.textAlign = "left"; c2d.textBaseline = "top";
    c2d.fillText(title, L, 8);
    const items = [[color("up_window"), "上行绿窗"], [color("down_window"), "下行绿窗"],
                   [color("red"), "红灯"], [color("up_band"), "上行绿波带"],
                   [color("down_band"), "下行绿波带"], [color("red_band"), "红波带"]];
    c2d.font = "11.5px sans-serif";
    items.forEach(([col, label], i) => {
      const y = T + 8 + i * 18;
      c2d.fillStyle = col; c2d.fillRect(L + plotW + 14, y, 11, 11);
      c2d.fillStyle = "#475569"; c2d.textBaseline = "middle";
      c2d.fillText(label, L + plotW + 31, y + 6);
    });
    const cap = document.getElementById("ts-caption");
    if (cap) cap.textContent = `${title}　·　C=${num(C, 2)}s，κ=${num(K, 3)}`
      + `　·　色块数 ${(rec.ts.bands || []).length}`;
  }

  function checked(id) { const el = document.getElementById(id); return !el || el.checked; }

  // debug 钩子：供自检脚本把 JS 的几何与 Python 库逐项对账
  return { init, draw,
           debug: { greenIntervals, redDisplayTimes, edgeRatio, contains } };
})();

// 数值格式化助手（各报告的适配层共用，避免各自实现）
function tsFmt(v, d) { return (typeof v === "number") ? v.toFixed(d === undefined ? 2 : d) : "-"; }
'''


__all__ = ["draw_context", "draw_payload", "panel_html", "TS_JS"]
