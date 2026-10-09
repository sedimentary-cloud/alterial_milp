"""时距图与 Pareto 图绘制。

给高中生看的小课堂
------------------
时距图是交通信号协调里最常用的图：
- 横轴是时间；
- 纵轴是距离；
- 每个路口画一条水平灯条，绿色是绿灯，红色是红灯；
- 一条斜着的带子表示车队能连续通过的“绿波带”。

如果带子是左下到右上，表示上行方向；
左上到右下，表示下行方向。

Pareto 图则用来展示“带宽”和“路口损失”之间的权衡关系。

本文件有两部分：
1. TimeSpaceDiagram：画时距图；
2. ParetoFrontierDiagram：画 Pareto 前沿。
"""
from __future__ import annotations

import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Polygon, Rectangle  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

plt.rcParams["axes.unicode_minus"] = False

UP_GREEN_WINDOW_COLOR = "limegreen"
DOWN_GREEN_WINDOW_COLOR = "#17becf"   # lake blue / cyan for down direction
RED_PERIOD_COLOR = "red"
UP_BAND_COLOR = "#4AC52E"
DOWN_BAND_COLOR = "#3077e2"
RED_BAND_COLOR = "#d62728"
BAND_ALPHA = 0.50
BAND_EDGE_LINEWIDTH = 0.4


# 时距图绘制器。
class TimeSpaceDiagram:
    @staticmethod
    def render(solution, problem, ctx, out_path: str, title: str | None = None,
               n_cycles: int | None = None, title_font=None) -> None:
        """画时距图。

        ``title_font``：可选的 ``matplotlib.font_manager.FontProperties``。
        标题要写中文时传它（本机不一定装了 CJK 字体，交给调用方决定）；
        不传就沿用 matplotlib 默认字体。
        """
        C = float(ctx.cycle_s)
        corridor = problem.corridor
        names = list(corridor.intersections)
        pos = _positions(corridor)

        t_min = 0.0
        t_max = _time_max(corridor, ctx, solution, n_cycles)
        k_min = math.floor(t_min / C) - 1
        k_max = math.ceil(t_max / C) + 1
        band_k_min = k_min - math.ceil((_max_corridor_travel(ctx, names) + _max_band_width(solution)) / C) - 1
        band_k_max = k_max + 1

        fig, ax = plt.subplots(figsize=(12.5, 6.8))

        # ------------------------------------------------------------------
        # 1. 不透明信号灯条：
        #    路口线上方 = 上行，下方 = 下行；
        #    绿色 = 绿灯窗口，红色 = 红灯时段。
        # ------------------------------------------------------------------
        bar_h = _bar_height(corridor)

        def lane_y(node: str, direction: str) -> float:
            """Y-centre of the signal bar lane for one direction."""
            base = pos[names.index(node)]
            return base + bar_h / 2.0 if direction == "up" else base - bar_h / 2.0

        for i, node in enumerate(names):
            pid = solution.plan_selection.get(node)
            if pid is None:
                continue
            plan = problem.plan(pid)
            y0 = pos[i]
            up_y = y0
            down_y = y0 - bar_h
            phi = solution.offsets_s.get(node, 0.0) % C
            # Pass 1: opaque red base bars for every visible cycle.
            for k in range(k_min, k_max + 1):
                shift = k * C
                ax.add_patch(Rectangle((shift, up_y), C, bar_h,
                                       facecolor=RED_PERIOD_COLOR, edgecolor="none",
                                       alpha=1.0, zorder=6))
                ax.add_patch(Rectangle((shift, down_y), C, bar_h,
                                       facecolor=RED_PERIOD_COLOR, edgecolor="none",
                                       alpha=1.0, zorder=6))
            # Pass 2: green windows on top, with the intersection offset applied.
            # Drawing all green windows after all red bases prevents a green
            # window that wraps across a cycle boundary from being covered by
            # the next cycle's red base rectangle.
            for d, base_y, green_color in (("up", up_y, UP_GREEN_WINDOW_COLOR),
                                           ("down", down_y, DOWN_GREEN_WINDOW_COLOR)):
                for widx in range(plan.window_count(d)):
                    ws_r = _window_endpoint_ratio(plan, d, widx, "start",
                                                  solution.adjustments_r, pid)
                    we_r = _window_endpoint_ratio(plan, d, widx, "end",
                                                  solution.adjustments_r, pid)
                    ws = ws_r * C + phi
                    we = we_r * C + phi
                    if we <= ws:
                        continue
                    for k in range(k_min, k_max + 1):
                        ax.add_patch(Rectangle((k * C + ws, base_y), we - ws, bar_h,
                                               facecolor=green_color, edgecolor="none",
                                               alpha=1.0, zorder=7))
            # Pass 3: thin black ticks for phase boundaries and cycle boundaries.
            # Cycle boundary: one thin line across both up/down lanes at each
            # visible signal cycle start (offset by this intersection's phi).
            for k in range(k_min, k_max + 1):
                tc = phi + k * C
                if tc < t_min - 1e-9 or tc > t_max + 1e-9:
                    continue
                ax.plot([tc, tc], [y0 - bar_h, y0 + bar_h],
                        color="black", linewidth=0.7, alpha=0.85,
                        zorder=9, solid_capstyle="butt")
            # Phase boundary: short thin ticks on each direction's lane.
            for d, lane_center in (("up", lane_y(node, "up")),
                                   ("down", lane_y(node, "down"))):
                sources = plan.expanded_edge_sources(d)
                half = bar_h * 0.42
                for edge in range(1, len(sources) - 1):
                    ratio = _actual_edge_ratio(plan, d, edge, solution, pid)
                    for k in range(k_min, k_max + 1):
                        tb = phi + k * C + ratio * C
                        if tb < t_min - 1e-9 or tb > t_max + 1e-9:
                            continue
                        ax.plot([tb, tb], [lane_center - half, lane_center + half],
                                color="black", linewidth=0.5, alpha=0.80,
                                zorder=9, solid_capstyle="butt")
            ax.axhline(y0, color="black", linewidth=0.7, zorder=8)
            ax.text(t_min + 0.005 * (t_max - t_min), y0, node,
                    ha="left", va="center", fontsize=9, fontweight="bold",
                    zorder=20, clip_on=True,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.80,
                              boxstyle="round,pad=0.12"))

        # ------------------------------------------------------------------
        # 2. 绿波带/红波带：按周期重复绘制，铺满整个可见时间范围。
        # ------------------------------------------------------------------
        band_handles_added = {"up": False, "down": False, "red": False}
        annotations: list[dict] = []
        for band in solution.bands:
            demand = problem.demand(band.demand)
            if demand.type == "green":
                color = UP_BAND_COLOR if demand.direction == "up" else DOWN_BAND_COLOR
                table = ctx.travel[demand.id]
                node_ys = [lane_y(n, demand.direction) for n in demand.nodes]
                # Use the representative copy whose chain-head start lies in [0, C).
                # The MILP may return an equivalent copy shifted by an integer
                # number of cycles; drawing it verbatim makes the band look
                # detached from the leftmost green windows.
                u_display = band.start_s % C
                base_times = [u_display + table.cum_T_s[j] for j in range(len(demand.nodes))]
                for k in range(band_k_min, band_k_max + 1):
                    shift = k * C
                    t_left = [t + shift for t in base_times]
                    t_right = [t + band.width_s for t in t_left]
                    verts = (list(zip(t_left, node_ys))
                             + list(zip(t_right[::-1], node_ys[::-1])))
                    ax.add_patch(Polygon(verts, closed=True,
                                         facecolor=_band_facecolor(color, BAND_ALPHA),
                                         edgecolor=color, linewidth=BAND_EDGE_LINEWIDTH,
                                         zorder=3))
                band_handles_added[demand.direction] = True
                # one readable label per demand
                mid = len(demand.nodes) // 2
                label_x = base_times[mid] + band.width_s / 2.0
                label_y = node_ys[mid]
                annotations.append({
                    "text": f"{band.demand}\n{band.width_s:.1f}s",
                    "base_x": label_x,
                    "target_y": label_y,
                    "label_y": label_y + (bar_h * 1.8 if demand.direction == "up" else -bar_h * 1.8),
                    "color": color,
                    "min_dx": max(10.0, 0.10 * C),
                })
            else:
                up_node, down_node = demand.nodes
                y_up = lane_y(up_node, demand.direction)
                y_down = lane_y(down_node, demand.direction)
                # Build the red band from the *actual selected plan windows*,
                # not from the raw MILP period indices (k, m).  The MILP may use
                # a periodic copy whose upstream/downstream cycle shifts make the
                # raw polygon slope the wrong way.  Here we search for the
                # representative copy where the upstream green window is exactly
                # one travel time before the downstream red interval.
                down_start_base, up_start_base = _red_band_display_times(
                    solution, problem, ctx, up_node, down_node,
                    demand.direction, band.start_s, band.width_s,
                )
                for k in range(band_k_min, band_k_max + 1):
                    shift = k * C
                    us = up_start_base + shift
                    ue = us + band.width_s
                    ds = down_start_base + shift
                    de = ds + band.width_s
                    verts = [(us, y_up), (ue, y_up), (de, y_down), (ds, y_down)]
                    ax.add_patch(Polygon(verts, closed=True,
                                         facecolor=_band_facecolor(RED_BAND_COLOR, BAND_ALPHA),
                                         edgecolor=RED_BAND_COLOR,
                                         linewidth=BAND_EDGE_LINEWIDTH,
                                         zorder=3))
                    ax.plot([us, ds], [y_up, y_down], color=RED_BAND_COLOR,
                            linewidth=1.1, zorder=4)
                    ax.plot([ue, de], [y_up, y_down], color=RED_BAND_COLOR,
                            linewidth=0.9, linestyle="--", zorder=4)
                band_handles_added["red"] = True
                label_x = down_start_base + band.width_s / 2.0
                label_y = (y_up + y_down) / 2.0
                annotations.append({
                    "text": f"{band.demand}\n{band.width_s:.1f}s",
                    "base_x": label_x,
                    "target_y": label_y,
                    "label_y": label_y + bar_h * 2.2,
                    "color": RED_BAND_COLOR,
                    "min_dx": max(10.0, 0.10 * C),
                })

        # ------------------------------------------------------------------
        # 2b. 波带文字：可移到不同周期副本，避免拥挤。
        # ------------------------------------------------------------------
        _draw_band_annotations(ax, annotations, t_min, t_max, C, bar_h)

        # ------------------------------------------------------------------
        # 3. 坐标轴、图例和标题。
        # ------------------------------------------------------------------
        ax.set_xlim(t_min, t_max)
        ax.set_ylim(-2.5 * bar_h, pos[-1] + 2.5 * bar_h)
        ax.set_yticks(pos)
        ax.set_yticklabels(names)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Distance (m)")
        ax.set_title(title or f"Time-Space Diagram (Cycle {C:.0f}s)", fontsize=12,
                     fontproperties=title_font)
        ax.grid(alpha=0.30, zorder=0)
        ax.xaxis.set_major_locator(MultipleLocator(C))
        ax.xaxis.set_minor_locator(MultipleLocator(C / 4.0))
        ax.grid(which="minor", axis="x", color="lightgray",
                linewidth=0.5, alpha=0.6, zorder=0)

        handles = [
            Patch(facecolor=UP_GREEN_WINDOW_COLOR, label="Up green window"),
            Patch(facecolor=DOWN_GREEN_WINDOW_COLOR, label="Down green window"),
            Patch(facecolor=RED_PERIOD_COLOR, label="Red period"),
            Line2D([0], [0], color="black", linewidth=0.8,
                   label="Phase / cycle boundary"),
        ]
        if band_handles_added["up"]:
            handles.append(Patch(facecolor=_band_facecolor(UP_BAND_COLOR, BAND_ALPHA),
                                 edgecolor=UP_BAND_COLOR,
                                 linewidth=BAND_EDGE_LINEWIDTH, label="Up green band"))
        if band_handles_added["down"]:
            handles.append(Patch(facecolor=_band_facecolor(DOWN_BAND_COLOR, BAND_ALPHA),
                                 edgecolor=DOWN_BAND_COLOR,
                                 linewidth=BAND_EDGE_LINEWIDTH, label="Down green band"))
        if band_handles_added["red"]:
            handles.append(Patch(facecolor=_band_facecolor(RED_BAND_COLOR, BAND_ALPHA),
                                 edgecolor=RED_BAND_COLOR,
                                 linewidth=BAND_EDGE_LINEWIDTH, label="Red band"))
        legend = ax.legend(handles=handles, loc="lower right", fontsize=8,
                           framealpha=0.95)
        legend.set_zorder(40)

        fig.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close(fig)


# Pareto 前沿绘制器。
class ParetoFrontierDiagram:
    """Draw the non-dominated composite-vs-intersection-loss frontier.

    默认把两个目标除以周期 C，得到无量纲的归一化目标，
    这样不同周期的 Pareto 点可以放在同一张图里比较。
    """

    @staticmethod
    def render(result, out_path: str, title: str | None = None,
               normalize_by_cycle: bool = True) -> None:
        cycle_s = None
        if isinstance(result, dict):
            pareto = result.get("pareto")
            if isinstance(pareto, dict):
                points = pareto.get("points", [])
                grid = pareto.get("grid") or {}
                cycle_s = grid.get("cycle_s")
            else:
                points = []
        else:
            points = list(result or [])

        divisor = float(cycle_s) if (normalize_by_cycle and cycle_s) else 1.0
        records = []
        for p in points:
            records.append({
                "composite": float(p.get("composite_actual", p.get("composite_actual_s", 0.0))) / divisor,
                "loss": float(p.get("intersection_loss", p.get("intersection_loss_s", 0.0))) / divisor,
                "epsilon": float(p.get("epsilon", p.get("epsilon_s", 0.0))),
            })
        front = _non_dominated(records)

        fig, ax = plt.subplots(figsize=(8.8, 5.8))
        if not front:
            ax.text(0.5, 0.5, "No Pareto points", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12)
            ax.set_title(title or "Pareto Frontier")
        else:
            ordered = sorted(front, key=lambda rec: (rec["composite"], rec["loss"]))
            xs = [r["composite"] for r in ordered]
            ys = [r["loss"] for r in ordered]
            ax.plot(xs, ys, color="#1f77b4", linewidth=1.6, marker="o",
                    markersize=6, markerfacecolor="white", markeredgewidth=1.6, zorder=3)
            for idx, rec in enumerate(ordered, start=1):
                ax.annotate(f"P{idx}\nε={rec['epsilon']:.1f}s",
                            (rec["composite"], rec["loss"]),
                            textcoords="offset points", xytext=(7, 6),
                            fontsize=8, color="#1f77b4", zorder=4)
            if normalize_by_cycle and cycle_s:
                ax.set_xlabel("Composite band objective / C  →  better")
                ax.set_ylabel("Intersection loss / C  ↓  better")
            else:
                ax.set_xlabel("Composite band objective (s)  →  better")
                ax.set_ylabel("Intersection loss (s)  ↓  better")
            ax.set_title(title or "Pareto Frontier: composite vs intersection loss")
            ax.grid(alpha=0.30, zorder=0)
            ax.margins(x=0.10, y=0.18)
        fig.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close(fig)


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def _band_facecolor(color: str, alpha: float):
    from matplotlib.colors import to_rgba
    r, g, b, _ = to_rgba(color)
    return (r, g, b, alpha)


def _positions(corridor) -> list[float]:
    """Cumulative upstream distance in metres; down bands use the same y-axis."""
    pos = [0.0]
    for seg in corridor.segments:
        pos.append(pos[-1] + float(seg.distance_for("up")))
    return pos


def _bar_height(corridor) -> float:
    total = _positions(corridor)[-1]
    # thin, opaque bars; the two directions touch at the intersection line
    return max(total * 0.008, 0.5)


def _corridor_travel_up(ctx, names) -> float:
    return sum(float(ctx.tau_s.get((names[i], "up"), 0.0) or 0.0)
               for i in range(len(names) - 1))


def _corridor_travel_down(ctx, names) -> float:
    return sum(float(ctx.tau_s.get((names[i], "down"), 0.0) or 0.0)
               for i in range(1, len(names)))


def _max_corridor_travel(ctx, names) -> float:
    return max(_corridor_travel_up(ctx, names), _corridor_travel_down(ctx, names), 0.0)


def _window_endpoint_ratio(plan, direction: str, k: int, side: str,
                           adjustments_r, pid: str) -> float:
    """Derived window endpoint in cycle-ratio domain (v2 boundary model)."""
    w = plan.window_edge_span(direction, k)
    edge = w.start_edge if side == "start" else w.end_edge
    kind, src, offset = plan.expanded_edge_sources(direction)[edge]
    if kind == "const":
        base = float(offset)
    else:
        base = (plan.nominal_boundary_r(direction, src) + offset
                + adjustments_r.get((pid, direction, src), 0.0))
    if side == "end":
        base += w.wrap
    return base


def _actual_edge_ratio(plan, direction: str, edge: int, solution, pid: str) -> float:
    """Actual expanded-edge ratio including declared boundary adjustments."""
    kind, src, offset = plan.expanded_edge_sources(direction)[edge]
    if kind == "const":
        return float(offset)
    delta = solution.adjustments_r.get((pid, direction, src), 0.0)
    return plan.nominal_boundary_r(direction, src) + delta + float(offset)


def _draw_band_annotations(ax, annotations, t_min: float, t_max: float,
                           C: float, bar_h: float) -> None:
    """Place band labels on periodic copies to reduce overlap."""
    if not annotations:
        return
    placed: list[dict] = []
    cycle_load: dict[int, int] = {}

    def vertical_overlap(y1: float, y2: float) -> bool:
        return abs(y1 - y2) < 1.8 * bar_h

    # Process left-most first so early labels get the natural first choice.
    for ann in sorted(annotations, key=lambda a: (a["base_x"], -a["label_y"])):
        base_x = ann["base_x"]
        min_dx = float(ann.get("min_dx", max(10.0, 0.10 * C)))
        # Candidate periodic copies: (cycle_index, absolute_label_time)
        candidates: list[tuple[int, float]] = []
        margin = 0.02 * C
        m = 0
        while True:
            x = base_x + m * C
            if x > t_max - margin + 1e-9 and m > 0:
                break
            if x >= t_min + margin - 1e-9 and x <= t_max - margin + 1e-9:
                candidates.append((m, x))
            if x > t_max + margin:
                break
            m += 1
        if not candidates:
            candidates = [(0, min(max(base_x, t_min + margin), t_max - margin))]

        best_cycle = candidates[0][0]
        best_x = candidates[0][1]
        best_cost = None
        for cycle, x in candidates:
            cost = 0.0
            for p in placed:
                if vertical_overlap(ann["label_y"], p["label_y"]):
                    dx = abs(x - p["x"])
                    if dx < min_dx:
                        cost += (min_dx - dx) ** 2
            # Spread labels evenly across visible cycle copies.
            cost += 1.0 * cycle_load.get(cycle, 0)
            # Very slight preference for earlier copies when everything else ties.
            cost += 0.01 * cycle
            if best_cost is None or cost < best_cost - 1e-12:
                best_cost = cost
                best_cycle = cycle
                best_x = x
        cycle_load[best_cycle] = cycle_load.get(best_cycle, 0) + 1
        placed.append({"x": best_x, "label_y": ann["label_y"]})

        ax.annotate(
            ann["text"],
            xy=(best_x, ann["target_y"]),
            xytext=(best_x, ann["label_y"]),
            ha="center", va="center", fontsize=ann.get("fontsize", 7.5),
            color=ann["color"],
            arrowprops=dict(arrowstyle="-", color=ann["color"], lw=0.7, alpha=0.8),
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75,
                      boxstyle="round,pad=0.25"),
            zorder=30, clip_on=True,
        )


def _green_intervals_for_plan(solution, problem, node: str, direction: str,
                              phi_s: float, C: float,
                              n_min: int = -6, n_max: int = 6):
    """Return tiled absolute green-window intervals for one selected plan."""
    pid = solution.plan_selection[node]
    plan = problem.plan(pid)
    intervals = []
    for widx in range(plan.window_count(direction)):
        ws_r = _window_endpoint_ratio(plan, direction, widx, "start",
                                      solution.adjustments_r, pid)
        we_r = _window_endpoint_ratio(plan, direction, widx, "end",
                                      solution.adjustments_r, pid)
        ws = ws_r * C + phi_s
        we = we_r * C + phi_s
        if we <= ws:
            continue
        for n in range(n_min, n_max + 1):
            intervals.append((ws + n * C, we + n * C))
    intervals.sort(key=lambda item: (item[0], item[1]))
    return intervals


def _red_intervals_from_green(greens):
    """Red intervals are the gaps between consecutive green intervals."""
    reds = []
    for i in range(len(greens) - 1):
        if greens[i][1] < greens[i + 1][0] - 1e-9:
            reds.append((greens[i][1], greens[i + 1][0]))
    return reds


def _find_containing(intervals, lo: float, hi: float, tol: float = 1e-5):
    for s, e in intervals:
        if lo >= s - tol and hi <= e + tol:
            return (s, e)
    return None


def _red_band_display_times(solution, problem, ctx,
                            up_node: str, down_node: str, direction: str,
                            u_s: float, width_s: float):
    """Choose a physical display copy for a red band.

    The downstream interval is ``[u, u+width]``.  The corresponding upstream
    departure interval must be ``[u - tau, u + width - tau]`` plus an integer
    number of cycles.  We select the copy where both intervals fall inside an
    actual selected-plan green/red window and where time is non-negative.
    """
    C = float(ctx.cycle_s)
    tau = float(ctx.tau_s.get((up_node, direction), 0.0) or 0.0)
    phi_up = solution.offsets_s.get(up_node, 0.0)
    phi_down = solution.offsets_s.get(down_node, 0.0)
    ups = _green_intervals_for_plan(solution, problem, up_node, direction, phi_up, C)
    downs = _green_intervals_for_plan(solution, problem, down_node, direction, phi_down, C)
    reds = _red_intervals_from_green(downs)

    best = None
    for n in range(-8, 9):
        down_lo = u_s + n * C
        down_hi = down_lo + width_s
        up_lo = down_lo - tau
        up_hi = up_lo + width_s
        if up_lo < -1e-6 or down_lo < -1e-6:
            continue
        if _find_containing(reds, down_lo, down_hi) is None:
            continue
        if _find_containing(ups, up_lo, up_hi) is None:
            continue
        if best is None or down_lo < best[0]:
            best = (down_lo, up_lo)
    if best is not None:
        return best
    # Fallback: physical relationship, shifted to non-negative time.
    down_lo = u_s
    up_lo = down_lo - tau
    extra = max(0, math.ceil((0.0 - up_lo) / C)) if C > 0 else 0
    return down_lo + extra * C, up_lo + extra * C


def _max_band_width(solution) -> float:
    return max((float(b.width_s) for b in solution.bands), default=0.0)


def _time_max(corridor, ctx, solution, n_cycles: int | None = None) -> float:
    C = float(ctx.cycle_s)
    names = list(corridor.intersections)
    t = _max_corridor_travel(ctx, names) + _max_band_width(solution) + C
    if n_cycles is not None:
        t = max(t, float(n_cycles) * C)
    if C > 0:
        t = math.ceil(t / C) * C
    return max(t, 2.0 * C)


def _non_dominated(records: list[dict]) -> list[dict]:
    """Keep points not dominated in (maximize composite, minimize loss)."""
    tol = 1e-9
    out = []
    for i, p in enumerate(records):
        dominated = False
        for j, q in enumerate(records):
            if i == j:
                continue
            if (q["composite"] >= p["composite"] - tol
                    and q["loss"] <= p["loss"] + tol
                    and (q["composite"] > p["composite"] + tol
                         or q["loss"] < p["loss"] - tol)):
                dominated = True
                break
        if not dominated:
            out.append(p)
    # remove duplicates
    unique = {}
    for p in out:
        key = (round(p["composite"], 6), round(p["loss"], 6))
        unique.setdefault(key, p)
    return list(unique.values())


__all__ = ["TimeSpaceDiagram", "ParetoFrontierDiagram"]
