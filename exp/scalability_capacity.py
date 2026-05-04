from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from exp_util import (
    ACCENT_AQUA,
    ACCENT_BLACK,
    ACCENT_BLUE,
    ACCENT_GREEN,
    ACCENT_ORANGE,
    AI_PAPER_AXIS_LABEL_SIZE,
    AI_PAPER_BAR_X_MARGIN,
    AI_PAPER_GRID_ALPHA,
    AI_PAPER_GRID_LINESTYLE,
    AI_PAPER_GRID_LINE_WIDTH,
    AI_PAPER_LEGEND_EDGE_COLOR,
    AI_PAPER_LEGEND_FONT_SIZE,
    AI_PAPER_LEGEND_HANDLE_LENGTH,
    AI_PAPER_LINE_WIDTH,
    AI_PAPER_MARKER_EDGE_WIDTH,
    AI_PAPER_MARKER_FACE_COLOR,
    AI_PAPER_MARKER_SIZE,
    AI_PAPER_MARKERS,
    AI_PAPER_TICK_LABEL_SIZE,
    apply_ai_paper_style,
    apply_bold_axis_labels,
    require_matplotlib_colors,
    require_matplotlib_pyplot,
)

# Exact two-column raster size requested for this plot set.
OUTPUT_WIDTH_PX = 1881
OUTPUT_HEIGHT_PX = 1291
OUTPUT_DPI = 300
OUTPUT_FIGSIZE = (OUTPUT_WIDTH_PX / OUTPUT_DPI, OUTPUT_HEIGHT_PX / OUTPUT_DPI)

METHOD_ORDER = ["Native", "C2C-Project", "Interlat", "LSC", "MoT"]
EXCLUDED_METHODS = {"KVComm"}
EXCLUDED_SOURCE_MODELS = {"opt-125m"}

METHOD_COLORS = {
    "Native": ACCENT_BLACK,
    "C2C-Project": ACCENT_BLUE,
    "Interlat": ACCENT_AQUA,
    "LSC": ACCENT_ORANGE,
    "MoT": ACCENT_GREEN,
}

FAMILY_ORDER = ["GPT-2", "OPT", "Qwen2.5"]
CAPACITY_ORDER = {
    "GPT-2": ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
    "OPT": ["opt-1.3b", "opt-2.7b", "opt-6.7b"],
    "Qwen2.5": ["Qwen2.5-0.5B", "Qwen2.5-1.5B", "Qwen2.5-3B", "Qwen2.5-7B"],
}
CAPACITY_LABELS = {
    "gpt2": "base",
    "gpt2-medium": "medium",
    "gpt2-large": "large",
    "gpt2-xl": "xl",
    "opt-1.3b": "1.3B",
    "opt-2.7b": "2.7B",
    "opt-6.7b": "6.7B",
    "Qwen2.5-0.5B": "0.5B",
    "Qwen2.5-1.5B": "1.5B",
    "Qwen2.5-3B": "3B",
    "Qwen2.5-7B": "7B",
}


@dataclass(frozen=True)
class Row:
    family: str
    source_model: str
    target_model: str
    method: str
    f1: float
    latency_ms: float
    throughput_toks: float
    peak_memory_gib: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot capacity scalability from a markdown result file."
    )
    parser.add_argument(
        "markdown_file",
        type=Path,
        help="Markdown file containing capacity scalability result tables.",
    )
    return parser.parse_args()


def normalize_header(name: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    if key == "f1_avg":
        return "f1"
    return key


def split_markdown_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def is_separator_row(cells: Iterable[str]) -> bool:
    return all(re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in cells)


def parse_number(value: str) -> float:
    text = value.strip().replace(",", "")
    if text.upper() == "N/A" or not text:
        return float("nan")
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        return float("nan")
    return float(match.group(0))


def family_for_source(source_model: str) -> str | None:
    source_lower = source_model.lower()
    if source_lower.startswith("gpt2"):
        return "GPT-2"
    if source_lower.startswith("opt-"):
        return "OPT"
    if source_lower.startswith("qwen2.5-"):
        return "Qwen2.5"
    return None


def parse_table(
    table_lines: list[str],
    *,
    source_model: str,
    target_model: str,
) -> list[Row]:
    if len(table_lines) < 3:
        return []

    raw_header = split_markdown_row(table_lines[0])
    separator = split_markdown_row(table_lines[1])
    if not is_separator_row(separator):
        return []

    headers = [normalize_header(h) for h in raw_header]
    header_index = {name: idx for idx, name in enumerate(headers)}
    required = ["method", "f1", "latency", "throughput", "gpu_peak_memory"]
    if any(name not in header_index for name in required):
        return []

    family = family_for_source(source_model)
    if family is None or source_model in EXCLUDED_SOURCE_MODELS:
        return []

    parsed: list[Row] = []
    for line in table_lines[2:]:
        cells = split_markdown_row(line)
        if len(cells) != len(headers):
            continue
        method = cells[header_index["method"]].strip()
        if method in EXCLUDED_METHODS or method not in METHOD_ORDER:
            continue
        parsed.append(
            Row(
                family=family,
                source_model=source_model,
                target_model=target_model,
                method=method,
                f1=parse_number(cells[header_index["f1"]]),
                latency_ms=parse_number(cells[header_index["latency"]]),
                throughput_toks=parse_number(cells[header_index["throughput"]]),
                peak_memory_gib=parse_number(cells[header_index["gpu_peak_memory"]]),
            )
        )
    return parsed


def parse_markdown(path: Path) -> list[Row]:
    text = path.read_text(encoding="utf-8")
    rows: list[Row] = []
    current_source: str | None = None
    current_target: str | None = None
    table_lines: list[str] = []

    def flush_table() -> None:
        nonlocal table_lines
        if current_source and current_target and table_lines:
            rows.extend(
                parse_table(
                    table_lines,
                    source_model=current_source,
                    target_model=current_target,
                )
            )
        table_lines = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        heading = re.match(r"^#+\s*(.+?)\s*->\s*(.+?)\s*$", line)
        if heading:
            flush_table()
            current_source = heading.group(1).strip()
            current_target = heading.group(2).strip()
            continue
        if current_source and "|" in line:
            table_lines.append(line)
        elif table_lines:
            flush_table()
    flush_table()
    return rows


def darker(color: str, factor: float = 0.72) -> str:
    if color == ACCENT_BLACK:
        return color
    mcolors = require_matplotlib_colors()
    rgb = np.array(mcolors.to_rgb(color))
    return mcolors.to_hex(np.clip(rgb * factor, 0.0, 1.0))


def metric_value(row: Row, metric: str) -> float:
    if metric == "f1":
        return row.f1
    if metric == "latency_ms":
        return row.latency_ms
    if metric == "throughput_toks":
        return row.throughput_toks
    if metric == "peak_memory_gib":
        return row.peak_memory_gib
    raise KeyError(metric)


def finite(values: Iterable[float]) -> list[float]:
    return [v for v in values if np.isfinite(v)]


def set_bar_axis_limits(ax, values: list[float], *, max_fraction: float = 0.55) -> None:
    vals = finite(values)
    if not vals:
        return
    top = max(vals) / max_fraction
    ax.set_ylim(0.0, top * 1.02)


def set_line_axis_limits(
    ax,
    values: list[float],
    *,
    min_fraction: float = 0.64,
    max_fraction: float = 0.95,
) -> None:
    vals = finite(values)
    if not vals:
        return
    data_min = min(vals)
    data_max = max(vals)
    if np.isclose(data_min, data_max):
        span = max(abs(data_max) * 0.08, 1e-3)
        data_min -= span
        data_max += span
    data_range = data_max - data_min
    axis_range = data_range / (max_fraction - min_fraction)
    axis_min = data_min - min_fraction * axis_range
    axis_max = axis_min + axis_range
    if axis_min > 0 and data_min >= 0:
        # Keep a natural positive scale when possible, but preserve the upper band.
        axis_min = max(0.0, axis_min)
    ax.set_ylim(axis_min, axis_max)


def hide_negative_y_tick_labels(ax) -> None:
    from matplotlib.ticker import FuncFormatter

    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _position: "" if value < 0 else f"{value:g}")
    )


def values_by_method_and_capacity(
    rows: list[Row],
    family: str,
    metric: str,
) -> dict[tuple[str, str], float]:
    result: dict[tuple[str, str], float] = {}
    for row in rows:
        if row.family == family:
            result[(row.source_model, row.method)] = metric_value(row, metric)
    return result


def capacity_models_for_family(rows: list[Row], family: str) -> list[str]:
    seen = {row.source_model for row in rows if row.family == family}
    ordered = [model for model in CAPACITY_ORDER.get(family, []) if model in seen]
    extras = sorted(seen - set(ordered))
    return ordered + extras


def draw_dual_axis_plot(
    *,
    rows: list[Row],
    family: str,
    bar_metric: str,
    line_metric: str,
    bar_label: str,
    line_label: str,
    output_path: Path,
) -> None:
    plt = require_matplotlib_pyplot()
    capacities = capacity_models_for_family(rows, family)
    if not capacities:
        return

    bar_values = values_by_method_and_capacity(rows, family, bar_metric)
    line_values = values_by_method_and_capacity(rows, family, line_metric)

    # Left axis: line metric
    # Right axis: bar metric
    fig, ax_line = plt.subplots(figsize=OUTPUT_FIGSIZE)
    ax_bar = ax_line.twinx()

    x = np.arange(len(capacities), dtype=float)
    n_methods = len(METHOD_ORDER)
    total_width = 0.82
    bar_width = total_width / n_methods
    offsets = (np.arange(n_methods) - (n_methods - 1) / 2.0) * bar_width

    all_bar_values: list[float] = []
    all_line_values: list[float] = []

    for method_index, method in enumerate(METHOD_ORDER):
        color = METHOD_COLORS[method]
        values = [bar_values.get((capacity, method), float("nan")) for capacity in capacities]
        all_bar_values.extend(values)
        valid_positions = [
            x_pos + offsets[method_index]
            for x_pos, v in zip(x, values)
            if np.isfinite(v)
        ]
        valid_values = [v for v in values if np.isfinite(v)]
        if valid_values:
            ax_bar.bar(
                valid_positions,
                valid_values,
                width=bar_width * 0.92,
                color=color,
                alpha=0.78 if method != "Native" else 0.88,
                edgecolor="white",
                linewidth=0.6,
                zorder=2,
            )

    for method_index, method in enumerate(METHOD_ORDER):
        color = darker(METHOD_COLORS[method])
        values = [line_values.get((capacity, method), float("nan")) for capacity in capacities]
        all_line_values.extend(values)
        valid_x = [x_pos for x_pos, v in zip(x, values) if np.isfinite(v)]
        valid_y = [v for v in values if np.isfinite(v)]
        if valid_y:
            ax_line.plot(
                valid_x,
                valid_y,
                linestyle="--" if method == "Native" else "-",
                linewidth=AI_PAPER_LINE_WIDTH,
                marker=AI_PAPER_MARKERS[method_index % len(AI_PAPER_MARKERS)],
                markersize=AI_PAPER_MARKER_SIZE * 0.82,
                markerfacecolor=AI_PAPER_MARKER_FACE_COLOR,
                markeredgecolor=color,
                markeredgewidth=AI_PAPER_MARKER_EDGE_WIDTH,
                color=color,
                zorder=4,
            )

    set_line_axis_limits(ax_line, all_line_values)
    hide_negative_y_tick_labels(ax_line)
    set_bar_axis_limits(ax_bar, all_bar_values)

    ax_line.set_xticks(x)
    ax_line.set_xticklabels([CAPACITY_LABELS.get(model, model) for model in capacities])
    ax_line.set_xlabel("Source Model Capacity")

    ax_line.set_ylabel(line_label)
    ax_bar.set_ylabel(bar_label)

    apply_bold_axis_labels(ax_line)
    apply_bold_axis_labels(ax_bar)

    ax_line.grid(
        axis="y",
        linestyle=AI_PAPER_GRID_LINESTYLE,
        linewidth=AI_PAPER_GRID_LINE_WIDTH,
        alpha=AI_PAPER_GRID_ALPHA,
        zorder=1,
    )
    ax_bar.grid(False)

    ax_line.margins(x=AI_PAPER_BAR_X_MARGIN)
    ax_line.tick_params(axis="both", labelsize=AI_PAPER_TICK_LABEL_SIZE)
    ax_bar.tick_params(axis="y", labelsize=AI_PAPER_TICK_LABEL_SIZE)

    # Keep bars visually behind the line plot.
    ax_bar.set_zorder(1)
    ax_line.set_zorder(2)
    ax_line.patch.set_visible(False)

    legend_handles = []
    for method in METHOD_ORDER:
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                color=METHOD_COLORS[method],
                marker="s",
                linestyle="",
                markersize=7.5,
                markerfacecolor=METHOD_COLORS[method],
                markeredgecolor=METHOD_COLORS[method],
                label=method,
            )
        )
    legend = ax_line.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=len(METHOD_ORDER),
        frameon=True,
        edgecolor=AI_PAPER_LEGEND_EDGE_COLOR,
        fontsize=AI_PAPER_LEGEND_FONT_SIZE,
        handlelength=AI_PAPER_LEGEND_HANDLE_LENGTH * 0.42,
        columnspacing=0.85,
        handletextpad=0.35,
        borderaxespad=0.0,
    )
    for text in legend.get_texts():
        text.set_fontweight("bold")

    fig.subplots_adjust(left=0.115, right=0.875, bottom=0.165, top=0.825)
    fig.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches=None, pad_inches=0.0)
    plt.close(fig)


def plot_all(rows: list[Row], output_dir: Path) -> list[Path]:
    outputs: list[Path] = []
    specs = [
        (
            "peak_memory_f1",
            "peak_memory_gib",
            "f1",
            "Peak Memory (Bar, GiB)",
            "F1 (Line)",
        ),
        (
            "latency_throughput",
            "latency_ms",
            "throughput_toks",
            "Latency (Bar, ms)",
            "Throughput (Line, tok/s)",
        ),
    ]
    for family in FAMILY_ORDER:
        safe_family = family.lower().replace(".", "").replace("-", "").replace(" ", "_")
        for suffix, bar_metric, line_metric, bar_label, line_label in specs:
            output_path = output_dir / f"capacity_scalability_{safe_family}_{suffix}.pdf"
            draw_dual_axis_plot(
                rows=rows,
                family=family,
                bar_metric=bar_metric,
                line_metric=line_metric,
                bar_label=bar_label,
                line_label=line_label,
                output_path=output_path,
            )
            if output_path.exists():
                outputs.append(output_path)
    return outputs


def main() -> None:
    args = parse_args()
    markdown_path = args.markdown_file.expanduser().resolve()
    if not markdown_path.is_file():
        raise FileNotFoundError(f"Markdown file not found: {markdown_path}")

    apply_ai_paper_style()
    rows = parse_markdown(markdown_path)
    if not rows:
        raise ValueError(f"No usable rows found in {markdown_path}")

    outputs = plot_all(rows, markdown_path.parent)
    if len(outputs) != 6:
        print(
            f"[warn] generated {len(outputs)} plots; expected 6. "
            "Check whether all three families exist in the markdown."
        )
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
