"""格点搜索热力图：周期 C × 速度倍率 κ。

给高中生看的小课堂
------------------
同一个干线问题，我们可以试不同的公共周期 C，
也可以试不同的整体速度倍率 κ。
每个组合 (C, κ) 叫一个“格点”。

本文件把每个格点的四个指标画成热力图：

1. Bandwidth objective：正带宽目标，越大越好；
2. Negative band-margin loss：绿波带 margin 短缺罚的负值，越接近 0 越好；
3. Negative intersection loss：路口损失的负值，越接近 0 越好；
4. Composite objective：阶段 1 的综合目标 composite，越大越好。

颜色语义统一为：
- 绿色 / 冷色 = 好；
- 红色 / 暖色 = 差；
- 灰色 = 该格点不可行或被预处理跳过（无有效解）。
"""
from __future__ import annotations

import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402


class GridHeatmapDiagram:
    """绘制 2×2 布局的四张格点热力图。

    灰色格子表示该 (C, κ) 不可行或 skipped_infeasible。
    """

    # 不可行格点的统一灰色。
    BAD_FILL = "#bdbdbd"

    # 格点内数值的最小可读字号；小于该字号就不再写数值。
    ANNOTATE_MIN_FONT = 3.5

    # 安全上限：格子太多时不逐格写数值（正常自适应字号也会先触发）。
    ANNOTATE_MAX_CELLS = 2000

    # 统一色阶语义：淡绿 = 好，白色 = 中等，淡红 = 不好。
    # 每个子图再根据自身数据的 min/max 自适应。
    SOFT_DIVERGING = LinearSegmentedColormap.from_list(
        "soft_green_white_red",
        # 低值 = 淡红（不好），中间 = 白色，高值 = 淡绿（好）
        ["#fcbba1", "#ffffff", "#c7e9c0"],
    )

    # 使用“除以周期 C”后的归一化指标，横跨不同周期可直接比较。
    # 单位：周期比例（无量纲）。
    METRICS = (
        ("bandwidth_objective_ratio", "Bandwidth objective / C"),
        ("negative_margin_loss_ratio", "Negative band-margin loss / C"),
        ("negative_intersection_loss_ratio", "Negative intersection loss / C"),
        ("composite_objective_ratio", "Composite objective / C"),
    )

    @staticmethod
    def render(result, out_path: str, title: str | None = None,
               dpi: int = 160) -> None:
        rows = _extract_rows(result)
        if not rows:
            raise ValueError("no grid results to plot")

        cycles = sorted({float(r["cycle_s"]) for r in rows})
        kappas = sorted({float(r["speed_ratio"]) for r in rows})
        if not cycles or not kappas:
            raise ValueError("grid results contain no cycle/speed-ratio values")

        # 先把 (C, κ) -> row 做成字典，方便填矩阵。
        by_key = {(float(r["cycle_s"]), float(r["speed_ratio"])): r for r in rows}

        fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))  # 3:2 长方形
        axes_flat = axes.ravel()

        for idx, (key, label) in enumerate(GridHeatmapDiagram.METRICS):
            ax = axes_flat[idx]
            matrix = np.full((len(cycles), len(kappas)), np.nan, dtype=float)
            for iy, cyc in enumerate(cycles):
                for ix, kap in enumerate(kappas):
                    row = by_key.get((cyc, kap))
                    if row is None:
                        continue
                    value = row.get(key)
                    if value is not None:
                        matrix[iy, ix] = float(value)
            _draw_heatmap(ax, matrix, cycles, kappas, label)

        if title:
            fig.suptitle(title, fontsize=13, y=0.98)
        fig.patch.set_facecolor("#fafafa")
        fig.text(0.5, 0.012, "gray cells = infeasible / skipped",
                 ha="center", va="bottom", fontsize=9, color="#666666")
        fig.tight_layout(rect=(0, 0.035, 1, 0.96))
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)


def _extract_rows(result) -> list[dict]:
    """兼容 report dict 和 GridPointResult 列表两种输入。"""
    if isinstance(result, dict):
        raw = result.get("grid_results", [])
    elif hasattr(result, "results"):
        raw = result.results
    else:
        raw = result
    rows: list[dict] = []
    for item in raw:
        if isinstance(item, dict):
            rows.append(_ensure_ratio_fields(dict(item)))
        else:
            cycle = getattr(item, "cycle_s", None)
            band = getattr(item, "band_reward_s", None)
            margin = getattr(item, "margin_loss_s", None)
            loss = getattr(item, "intersection_loss_s", None)
            composite = getattr(item, "composite_best", None)
            rows.append({
                "cycle_s": cycle,
                "speed_ratio": getattr(item, "kappa", None),
                "status": getattr(item, "status", None),
                "bandwidth_objective_s": band,
                "negative_margin_loss_s": None if margin is None else -float(margin),
                "negative_intersection_loss_s": None if loss is None else -float(loss),
                "composite_objective_s": composite,
                "bandwidth_objective_ratio": (
                    None if band is None or not cycle else float(band) / float(cycle)),
                "negative_margin_loss_ratio": (
                    None if margin is None or not cycle else -float(margin) / float(cycle)),
                "negative_intersection_loss_ratio": (
                    None if loss is None or not cycle else -float(loss) / float(cycle)),
                "composite_objective_ratio": (
                    None if composite is None or not cycle else float(composite) / float(cycle)),
            })
    return [r for r in rows if r.get("cycle_s") is not None and r.get("speed_ratio") is not None]


def _ensure_ratio_fields(row: dict) -> dict:
    """如果只有绝对秒数，就补齐除以周期的归一化字段。"""
    cycle = float(row.get("cycle_s") or 0.0)
    if row.get("composite_objective_s") is None and row.get("composite_best") is not None:
        row["composite_objective_s"] = row["composite_best"]
    if cycle <= 0:
        return row
    pairs = (
        ("bandwidth_objective_s", "bandwidth_objective_ratio"),
        ("negative_margin_loss_s", "negative_margin_loss_ratio"),
        ("negative_intersection_loss_s", "negative_intersection_loss_ratio"),
        ("composite_objective_s", "composite_objective_ratio"),
    )
    for abs_key, ratio_key in pairs:
        if row.get(ratio_key) is None and row.get(abs_key) is not None:
            row[ratio_key] = float(row[abs_key]) / cycle
    return row


def _draw_heatmap(ax, matrix: np.ndarray, cycles, kappas, label: str) -> None:
    finite = matrix[np.isfinite(matrix)]
    cmap = GridHeatmapDiagram.SOFT_DIVERGING.copy()
    cmap.set_bad(GridHeatmapDiagram.BAD_FILL)

    if finite.size == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color="#888888")
        ax.set_title(label, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    # 根据当前子图的实际最小/最大值自适应，避免所有图共用固定量程。
    vmin = float(finite.min())
    vmax = float(finite.max())
    if vmax - vmin < 1e-9:
        delta = max(abs(vmin) * 0.01, 0.5)
        vmin -= delta
        vmax += delta
    norm = Normalize(vmin=vmin, vmax=vmax)

    im = ax.imshow(matrix, origin="lower", aspect="auto", cmap=cmap,
                   vmin=vmin, vmax=vmax, interpolation="nearest")

    tick_fs = 9 if max(len(kappas), len(cycles)) <= 15 else 7
    pos = ax.get_position()
    axis_w_in = ax.figure.get_figwidth() * pos.width
    axis_h_in = ax.figure.get_figheight() * pos.height

    # 轴标签根据可用宽度/高度自动抽稀，避免密集格点时数字全部挤在一起。
    x_ticks = _thin_tick_indices(
        len(kappas), axis_w_in, tick_fs,
        max(len(f"{k:g}") for k in kappas) if kappas else 1)
    y_ticks = _thin_tick_indices(
        len(cycles), axis_h_in, tick_fs,
        max(len(f"{c:g}") for c in cycles) if cycles else 1)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{kappas[i]:g}" for i in x_ticks], fontsize=tick_fs)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{cycles[i]:g}" for i in y_ticks], fontsize=tick_fs)
    ax.set_xlabel("Speed ratio κ", fontsize=10)
    ax.set_ylabel("Cycle C (s)", fontsize=10)
    ax.set_title(label, fontsize=11, pad=8)

    # 根据每个格子可用的物理尺寸自适应字号；密集到不可读时自动省略数字。
    n_cells = len(cycles) * len(kappas)
    cell_w_in = axis_w_in / max(1, len(kappas))
    cell_h_in = axis_h_in / max(1, len(cycles))
    cell_in = min(cell_w_in, cell_h_in)
    annot_fs = min(8.5, max(3.0, 0.50 * cell_in * 72.0))

    if n_cells <= GridHeatmapDiagram.ANNOTATE_MAX_CELLS and annot_fs >= GridHeatmapDiagram.ANNOTATE_MIN_FONT:
        compact = annot_fs < 6.0
        for iy in range(len(cycles)):
            for ix in range(len(kappas)):
                value = matrix[iy, ix]
                if not np.isfinite(value):
                    continue
                rgba = cmap(norm(float(value)))
                luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                text_color = "white" if luminance < 0.48 else "#1a1a1a"
                label_text = _fmt_compact(value) if compact else _fmt(value)
                ax.text(ix, iy, label_text, ha="center", va="center",
                        fontsize=annot_fs, color=text_color)

    for spine in ax.spines.values():
        spine.set_color("#cccccc")
    ax.tick_params(length=0)

    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.035)
    cbar.set_label("ratio of cycle C", fontsize=8)
    cbar.ax.tick_params(labelsize=8, length=0)
    cbar.outline.set_edgecolor("#cccccc")


def _thin_tick_indices(n: int, axis_in: float, fontsize: float,
                      max_label_chars: int) -> list[int]:
    """按轴的实际尺寸和标签字符数，计算需要显示哪几个刻度。

    返回的是 0-based 下标；密集轴只显示间隔 2、3、4… 个的标签。
    """
    if n <= 1:
        return list(range(n))
    # 粗略估算一个标签占用的物理宽度/高度（英寸），再留一点间距。
    label_in = max(0.08, 0.65 * fontsize * max(1, max_label_chars) / 72.0) + 0.06
    max_labels = max(2, int(axis_in / label_in))
    step = max(1, math.ceil(n / max_labels))
    return list(range(0, n, step))


def _fmt(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:.0f}"
    if abs(value) >= 100:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _fmt_compact(value: float) -> str:
    """密集格点下用一个更短的写法，给自适应小字号留空间。"""
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.0f}"
    return f"{value:.1f}"


__all__ = ["GridHeatmapDiagram"]
