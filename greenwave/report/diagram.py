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
UP_BAND_HATCH = "..."
DOWN_BAND_HATCH = "ooo"
RED_BAND_HATCH = "///"
BAND_ALPHA = 0.50
BAND_HATCH_LINEWIDTH = 0.25


# 时距图绘制器。
class TimeSpaceDiagram:
    @staticmethod
    def render(solution, problem, ctx, out_path: str, title: str | None = None,
               n_cycles: int | None = None) -> None:
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
                for widx, w in enumerate(plan.effective_windows(d)):
                    adj_s = solution.adjustments_r.get((pid, d, widx, "start"), 0.0)
                    adj_e = solution.adjustments_r.get((pid, d, widx, "end"), 0.0)
                    ws = (w.start_r + adj_s) * C + phi
                    we = (w.end_r + adj_e) * C + phi
                    if we <= ws:
                        continue
                    for k in range(k_min, k_max + 1):
                        ax.add_patch(Rectangle((k * C + ws, base_y), we - ws, bar_h,
                                               facecolor=green_color, edgecolor="none",
                                               alpha=1.0, zorder=7))
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
        for band in solution.bands:
            demand = problem.demand(band.demand)
            if demand.type == "green":
                color = UP_BAND_COLOR if demand.direction == "up" else DOWN_BAND_COLOR
                hatch = UP_BAND_HATCH if demand.direction == "up" else DOWN_BAND_HATCH
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
                                         edgecolor=color, linewidth=BAND_HATCH_LINEWIDTH,
                                         hatch=hatch, zorder=3))
                band_handles_added[demand.direction] = True
                # one readable label per demand
                mid = len(demand.nodes) // 2
                label_x = base_times[mid] + band.width_s / 2.0
                label_y = node_ys[mid]
                ax.annotate(f"{band.demand}\n{band.width_s:.1f}s",
                            xy=(label_x, label_y),
                            xytext=(label_x, label_y + (bar_h * 1.8 if demand.direction == "up" else -bar_h * 1.8)),
                            ha="center", va="center", fontsize=7.5, color=color,
                            arrowprops=dict(arrowstyle="-", color=color, lw=0.7, alpha=0.8),
                            zorder=12, clip_on=True)
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
                                         linewidth=BAND_HATCH_LINEWIDTH,
                                         hatch=RED_BAND_HATCH, zorder=3))
                    ax.plot([us, ds], [y_up, y_down], color=RED_BAND_COLOR,
                            linewidth=1.1, zorder=4)
                    ax.plot([ue, de], [y_up, y_down], color=RED_BAND_COLOR,
                            linewidth=0.9, linestyle="--", zorder=4)
                band_handles_added["red"] = True
                label_x = down_start_base + band.width_s / 2.0
                label_y = (y_up + y_down) / 2.0
                ax.annotate(f"{band.demand}\n{band.width_s:.1f}s",
                            xy=(label_x, label_y), xytext=(label_x, label_y + bar_h * 2.2),
                            ha="center", va="center", fontsize=7.5, color=RED_BAND_COLOR,
                            arrowprops=dict(arrowstyle="-", color=RED_BAND_COLOR, lw=0.7, alpha=0.8),
                            zorder=12, clip_on=True)

        # ------------------------------------------------------------------
        # 3. 坐标轴、图例和标题。
        # ------------------------------------------------------------------
        ax.set_xlim(t_min, t_max)
        ax.set_ylim(-2.5 * bar_h, pos[-1] + 2.5 * bar_h)
        ax.set_yticks(pos)
        ax.set_yticklabels(names)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Distance (m)")
        ax.set_title(title or f"Time-Space Diagram (Cycle {C:.0f}s)", fontsize=12)
        ax.grid(alpha=0.30, zorder=0)
        ax.xaxis.set_major_locator(MultipleLocator(C))
        ax.xaxis.set_minor_locator(MultipleLocator(C / 4.0))
        ax.grid(which="minor", axis="x", color="lightgray",
                linewidth=0.5, alpha=0.6, zorder=0)

        handles = [
            Patch(facecolor=UP_GREEN_WINDOW_COLOR, label="Up green window"),
            Patch(facecolor=DOWN_GREEN_WINDOW_COLOR, label="Down green window"),
            Patch(facecolor=RED_PERIOD_COLOR, label="Red period"),
        ]
        if band_handles_added["up"]:
            handles.append(Patch(facecolor=_band_facecolor(UP_BAND_COLOR, BAND_ALPHA),
                                 edgecolor=UP_BAND_COLOR, hatch=UP_BAND_HATCH,
                                 linewidth=BAND_HATCH_LINEWIDTH, label="Up green band"))
        if band_handles_added["down"]:
            handles.append(Patch(facecolor=_band_facecolor(DOWN_BAND_COLOR, BAND_ALPHA),
                                 edgecolor=DOWN_BAND_COLOR, hatch=DOWN_BAND_HATCH,
                                 linewidth=BAND_HATCH_LINEWIDTH, label="Down green band"))
        if band_handles_added["red"]:
            handles.append(Patch(facecolor=_band_facecolor(RED_BAND_COLOR, BAND_ALPHA),
                                 edgecolor=RED_BAND_COLOR, hatch=RED_BAND_HATCH,
                                 linewidth=BAND_HATCH_LINEWIDTH, label="Red band"))
        ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.95)

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


def _green_intervals_for_plan(solution, problem, node: str, direction: str,
                              phi_s: float, C: float,
                              n_min: int = -6, n_max: int = 6):
    """Return tiled absolute green-window intervals for one selected plan."""
    pid = solution.plan_selection[node]
    plan = problem.plan(pid)
    intervals = []
    for widx, w in enumerate(plan.effective_windows(direction)):
        adj_s = solution.adjustments_r.get((pid, direction, widx, "start"), 0.0)
        adj_e = solution.adjustments_r.get((pid, direction, widx, "end"), 0.0)
        ws = (w.start_r + adj_s) * C + phi_s
        we = (w.end_r + adj_e) * C + phi_s
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
