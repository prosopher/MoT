from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from matplotlib.patches import Rectangle

from exp_util import (
    AI_PAPER_DOUBLE_COLUMN_TALL_FIGSIZE,
    AI_PAPER_FIGURE_DPI,
    AI_PAPER_LEGEND_EDGE_COLOR,
    AI_PAPER_LINE_WIDTH,
    AI_PAPER_MARKER_EDGE_WIDTH,
    AI_PAPER_MARKER_FACE_COLOR,
    AI_PAPER_MARKERS,
    apply_ai_paper_style,
    require_matplotlib_pyplot,
    save_paper_figure,
    style_axes_common,
)

# ---------------------------------------------------------------------------
# Figure style
# ---------------------------------------------------------------------------

FIGSIZE = AI_PAPER_DOUBLE_COLUMN_TALL_FIGSIZE

OVERLEAF_AXIS_LABEL_SIZE = 28
OVERLEAF_TICK_LABEL_SIZE = 28
OVERLEAF_LEGEND_FONT_SIZE = 28
OVERLEAF_LINE_WIDTH = AI_PAPER_LINE_WIDTH * 2.2
OVERLEAF_MARKER_SIZE = 13.0
OVERLEAF_MARKER_EDGE_WIDTH = AI_PAPER_MARKER_EDGE_WIDTH * 1.70

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

DEFAULT_AGENT_COUNTS = list(range(1, 11))

METHOD_ORDER = [
    "lsc-retain",
    "interlat",
    "mot-retain",
    "mot-free",
]

# method key -> (algorithm, cache_mode)
# Metrics are loaded from:
# outputs/multi_agents/{algorithm}_{cache_mode}_{agent_count}/agent_runner_metrics.json
METHOD_SPECS = {
    "lsc-retain": ("lsc", "retain"),
    "interlat": ("interlat", "retain"),
    "mot-retain": ("mot", "retain"),
    "mot-free": ("mot", "free"),
}

XTICK_LABELS = {
    "lsc-retain": "LSC\n(Retain)",
    "interlat": "Interlat\n(Retain)",
    "mot-retain": "MoT\n(Retain)",
    "mot-free": "MoT\n(Free)",
}

METHOD_COLORS = {
    "lsc-retain": "#C7CDD6",
    "interlat": "#C7CDD6",
    "mot-retain": "#E98C88",
    "mot-free": "#C0504D",
}

F1_COLOR = "#4BACC6"


def _safe_folder_name(algorithm: str, cache_mode: str, agent_count: int) -> str:
    """Return the metrics folder name for one method and one agent count."""

    return f"{algorithm}_{cache_mode}_{agent_count}".replace("-", "_")


def _metrics_path(
    metrics_root: Path,
    algorithm: str,
    cache_mode: str,
    agent_count: int,
) -> Path:
    return (
        metrics_root
        / _safe_folder_name(algorithm, cache_mode, agent_count)
        / "agent_runner_metrics.json"
    )


def _read_metric_value(metrics: dict, key: str, path: Path) -> float:
    if key not in metrics:
        raise KeyError(f"Missing {key!r} in {path}")
    return float(metrics[key])


def load_data_from_metrics(
    metrics_root: Path,
    method_order: list[str] | tuple[str, ...] = METHOD_ORDER,
    agent_counts: list[int] | tuple[int, ...] = tuple(DEFAULT_AGENT_COUNTS),
) -> dict[int, dict[str, dict[str, float]]]:
    """Load metrics for existing folders only.

    The returned structure is:
        data[agent_count][method] = {"f1": ..., "gpu_peak_memory_gib": ...}
    """

    data: dict[int, dict[str, dict[str, float]]] = {}
    scanned_paths: list[Path] = []
    loaded_paths: list[Path] = []

    for agent_count in agent_counts:
        for method in method_order:
            algorithm, cache_mode = METHOD_SPECS[method]
            path = _metrics_path(metrics_root, algorithm, cache_mode, agent_count)
            scanned_paths.append(path)

            if not path.exists():
                continue

            with path.open("r", encoding="utf-8") as f:
                metrics = json.load(f)

            json_algorithm = metrics.get("algorithm")
            json_cache_mode = metrics.get("cache_mode")
            json_agent_count = metrics.get("agent_count")

            if json_algorithm is not None and json_algorithm != algorithm:
                warnings.warn(
                    f"{path} says algorithm={json_algorithm!r}, "
                    f"but expected {algorithm!r} for method {method!r}.",
                    RuntimeWarning,
                )
            if json_cache_mode is not None and json_cache_mode != cache_mode:
                warnings.warn(
                    f"{path} says cache_mode={json_cache_mode!r}, "
                    f"but expected {cache_mode!r} for method {method!r}.",
                    RuntimeWarning,
                )
            if json_agent_count is None:
                raise KeyError(f"Missing 'agent_count' in {path}")
            if int(json_agent_count) != agent_count:
                raise ValueError(
                    f"Agent-count mismatch in {path}: "
                    f"folder implies {agent_count}, json says {json_agent_count}."
                )

            data.setdefault(agent_count, {})[method] = {
                "f1": _read_metric_value(metrics, "f1", path),
                "gpu_peak_memory_gib": _read_metric_value(
                    metrics,
                    "gpu_peak_memory_gib",
                    path,
                ),
            }
            loaded_paths.append(path)

    if not data:
        scanned = "\n".join(f"  - {path}" for path in scanned_paths)
        raise FileNotFoundError(
            f"No metrics were loaded from {metrics_root}.\n"
            "Expected files like:\n"
            "  outputs/multi_agents/mot_free_4/agent_runner_metrics.json\n\n"
            "Scanned paths:\n"
            f"{scanned}"
        )

    for path in loaded_paths:
        print(f"Loaded: {path}")

    return data


def compute_global_axis_limits(
    data: dict[int, dict[str, dict[str, float]]]
) -> tuple[float, float]:
    global_f1_max = (
        max(entry["f1"] for agent_data in data.values() for entry in agent_data.values())
        * 1.25
    )
    global_memory_max = (
        max(
            entry["gpu_peak_memory_gib"]
            for agent_data in data.values()
            for entry in agent_data.values()
        )
        * 1.25
    )
    return global_f1_max, global_memory_max


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def apply_large_text_style(ax) -> None:
    ax.xaxis.label.set_fontsize(OVERLEAF_AXIS_LABEL_SIZE)
    ax.yaxis.label.set_fontsize(OVERLEAF_AXIS_LABEL_SIZE)
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")
    ax.tick_params(axis="both", labelsize=OVERLEAF_TICK_LABEL_SIZE)


def style_method_tick_labels(ax) -> None:
    for label in ax.get_xticklabels():
        label.set_fontweight("normal")
        label.set_fontsize(OVERLEAF_TICK_LABEL_SIZE)
        label.set_rotation(0)
        label.set_ha("center")


def save_both_formats(fig, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    save_paper_figure(
        fig,
        output_base.with_suffix(".png"),
        dpi=AI_PAPER_FIGURE_DPI,
        close=False,
    )
    save_paper_figure(
        fig,
        output_base.with_suffix(".pdf"),
        dpi=AI_PAPER_FIGURE_DPI,
        close=True,
    )


def draw_manual_legend_on_axes(ax) -> None:
    """Draw legend manually so fontsize=28 is applied exactly."""

    box_x = 0.08
    box_y = 1.02
    box_w = 0.84
    box_h = 0.20

    legend_box = Rectangle(
        (box_x, box_y),
        box_w,
        box_h,
        transform=ax.transAxes,
        facecolor="white",
        edgecolor=AI_PAPER_LEGEND_EDGE_COLOR,
        linewidth=1.2,
        clip_on=False,
        zorder=20,
    )
    ax.add_patch(legend_box)

    y = box_y + box_h / 2.0

    mem_box_x = box_x + 0.045
    mem_box_w = 0.055
    mem_box_h = 0.060

    mem_box = Rectangle(
        (mem_box_x, y - mem_box_h / 2.0),
        mem_box_w,
        mem_box_h,
        transform=ax.transAxes,
        facecolor="#C7CDD6",
        edgecolor="black",
        linewidth=1.0,
        clip_on=False,
        zorder=25,
    )
    ax.add_patch(mem_box)

    ax.text(
        mem_box_x + mem_box_w + 0.025,
        y,
        "Peak Memory",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=OVERLEAF_LEGEND_FONT_SIZE,
        fontweight="bold",
        clip_on=False,
        zorder=26,
    )

    f1_x = box_x + 0.64
    handle_half_len = 0.040

    ax.plot(
        [f1_x - handle_half_len, f1_x + handle_half_len],
        [y, y],
        transform=ax.transAxes,
        linestyle="-",
        linewidth=OVERLEAF_LINE_WIDTH,
        color=F1_COLOR,
        clip_on=False,
        zorder=25,
    )

    ax.plot(
        [f1_x],
        [y],
        transform=ax.transAxes,
        linestyle="",
        marker="o",
        markersize=OVERLEAF_MARKER_SIZE * 0.90,
        markerfacecolor=AI_PAPER_MARKER_FACE_COLOR,
        markeredgecolor=F1_COLOR,
        markeredgewidth=OVERLEAF_MARKER_EDGE_WIDTH,
        clip_on=False,
        zorder=26,
    )

    ax.text(
        f1_x + handle_half_len + 0.025,
        y,
        "F1",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=OVERLEAF_LEGEND_FONT_SIZE,
        fontweight="bold",
        clip_on=False,
        zorder=26,
    )


def draw_agent_plot(
    *,
    agent_count: int,
    agent_data: dict[str, dict[str, float]],
    global_memory_max: float,
    global_f1_max: float,
    output_base: Path,
) -> None:
    plt = require_matplotlib_pyplot()

    plotted_methods = [method for method in METHOD_ORDER if method in agent_data]
    if not plotted_methods:
        raise ValueError(f"No methods available for agent_count={agent_count}")

    fig, ax_mem = plt.subplots(figsize=FIGSIZE)
    ax_f1 = ax_mem.twinx()

    fig.patch.set_facecolor("white")
    ax_mem.set_facecolor("white")
    ax_f1.set_facecolor("white")

    x = np.arange(len(plotted_methods), dtype=float)
    bar_width = 0.56

    memory_values = [
        agent_data[method]["gpu_peak_memory_gib"] for method in plotted_methods
    ]
    f1_values = [agent_data[method]["f1"] for method in plotted_methods]

    for idx, method in enumerate(plotted_methods):
        ax_mem.bar(
            x[idx],
            memory_values[idx],
            width=bar_width,
            color=METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.8,
            zorder=2,
        )

    ax_f1.plot(
        x,
        f1_values,
        linestyle="-",
        linewidth=OVERLEAF_LINE_WIDTH,
        color=F1_COLOR,
        zorder=4,
    )

    for idx, method in enumerate(plotted_methods):
        ax_f1.plot(
            [x[idx]],
            [f1_values[idx]],
            linestyle="",
            marker=AI_PAPER_MARKERS[idx % len(AI_PAPER_MARKERS)],
            markersize=OVERLEAF_MARKER_SIZE,
            markerfacecolor=AI_PAPER_MARKER_FACE_COLOR,
            markeredgecolor=F1_COLOR,
            markeredgewidth=OVERLEAF_MARKER_EDGE_WIDTH,
            color=F1_COLOR,
            zorder=5,
        )

    ax_mem.set_xticks(x)
    ax_mem.set_xticklabels([XTICK_LABELS[m] for m in plotted_methods])

    ax_mem.set_xlabel("")
    ax_mem.set_ylabel("GPU Peak Memory (GiB)")
    ax_f1.set_ylabel("F1")

    ax_mem.set_ylim(0.0, global_memory_max)
    ax_f1.set_ylim(0.0, global_f1_max)

    style_axes_common(ax_mem, grid=True, grid_axis="y", title=False)
    style_axes_common(ax_f1, grid=False, grid_axis="y", title=False)

    apply_large_text_style(ax_mem)
    apply_large_text_style(ax_f1)
    style_method_tick_labels(ax_mem)

    ax_mem.set_zorder(1)
    ax_f1.set_zorder(2)
    ax_f1.patch.set_visible(False)

    draw_manual_legend_on_axes(ax_f1)

    fig.subplots_adjust(left=0.125, right=0.875, bottom=0.155, top=0.80)

    save_both_formats(fig, output_base)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create agent-wise F1 vs GPU Peak Memory plots from metrics json files."
    )
    parser.add_argument(
        "--metrics-root",
        type=Path,
        default=Path("outputs/multi_agents"),
        help=(
            "Root directory containing "
            "{algorithm}_{cache_mode}_{agent_count}/agent_runner_metrics.json folders."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory to save plots.",
    )
    parser.add_argument(
        "--agent-counts",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Optional list of agent counts to scan. If omitted, agent counts 1 through 10 "
            "are scanned and only existing folders are plotted."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_ai_paper_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    agent_counts = args.agent_counts if args.agent_counts is not None else DEFAULT_AGENT_COUNTS
    data = load_data_from_metrics(args.metrics_root, agent_counts=agent_counts)
    global_f1_max, global_memory_max = compute_global_axis_limits(data)

    for agent_count in sorted(data):
        output_base = args.output_dir / f"agents_{agent_count}"
        draw_agent_plot(
            agent_count=agent_count,
            agent_data=data[agent_count],
            global_memory_max=global_memory_max,
            global_f1_max=global_f1_max,
            output_base=output_base,
        )
        print(output_base.with_suffix(".png"))
        print(output_base.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
