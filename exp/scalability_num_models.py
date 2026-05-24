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

METHOD_COLORS = {
    "C2C": ACCENT_BLUE,
    "Interlat": ACCENT_AQUA,
    "MoT": ACCENT_GREEN,
    "LSC": ACCENT_ORANGE,
}

METHOD_ALIASES = {
    "c2c": "C2C",
    "c2c_project": "C2C",
    "c2c_project_": "C2C",
    "c2c-project": "C2C",
    "interlat": "Interlat",
    "kvcomm": "KVComm",
    "kv_comm": "KVComm",
    "kv-comm": "KVComm",
    "mot": "MoT",
    "lsc": "LSC",
}


@dataclass(frozen=True)
class Row:
    method: str
    num_models: int
    training_duration_sec: float
    peak_memory_gib: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot scalability by number of models from a markdown result file."
    )
    parser.add_argument(
        "markdown_file",
        type=Path,
        help="Markdown file containing number-of-models scalability result tables.",
    )
    return parser.parse_args()


def normalize_header(name: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    if key in {"method", "number_of_models", "num_models"}:
        return key
    if key in {"training_duration", "training_duration_sec", "duration"}:
        return "training_duration"
    if key in {"gpu_peak_memory", "gpu_peak_memory_gib", "peak_memory", "peak_memory_gib"}:
        return "gpu_peak_memory"
    return key


def normalize_method(name: str) -> str | None:
    key = re.sub(r"[^a-z0-9-]+", "_", name.strip().lower()).strip("_")
    return METHOD_ALIASES.get(key)


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


def parse_int(value: str) -> int | None:
    parsed = parse_number(value)
    if not np.isfinite(parsed):
        return None
    return int(parsed)


def parse_table(table_lines: list[str], *, section_method: str) -> list[Row]:
    if len(table_lines) < 3:
        return []

    raw_header = split_markdown_row(table_lines[0])
    separator = split_markdown_row(table_lines[1])
    if not is_separator_row(separator):
        return []

    headers = [normalize_header(header) for header in raw_header]
    header_index = {name: idx for idx, name in enumerate(headers)}
    required = ["training_duration", "gpu_peak_memory"]
    if any(name not in header_index for name in required):
        return []

    count_header = "number_of_models" if "number_of_models" in header_index else "num_models"
    if count_header not in header_index:
        # Accept the LSC table variant where the first column is named "Method"
        # but contains the number of models.
        count_header = "method"
    if count_header not in header_index:
        return []

    parsed: list[Row] = []
    for line in table_lines[2:]:
        cells = split_markdown_row(line)
        if len(cells) != len(headers):
            continue
        num_models = parse_int(cells[header_index[count_header]])
        if num_models is None:
            continue
        parsed.append(
            Row(
                method=section_method,
                num_models=num_models,
                training_duration_sec=parse_number(cells[header_index["training_duration"]]),
                peak_memory_gib=parse_number(cells[header_index["gpu_peak_memory"]]),
            )
        )
    return parsed


def parse_markdown(path: Path) -> list[Row]:
    text = path.read_text(encoding="utf-8")
    rows: list[Row] = []
    current_method: str | None = None
    table_lines: list[str] = []

    def flush_table() -> None:
        nonlocal table_lines
        if current_method and table_lines:
            rows.extend(parse_table(table_lines, section_method=current_method))
        table_lines = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        heading = re.match(r"^#+\s*(.+?)\s*$", line)
        if heading:
            flush_table()
            current_method = normalize_method(heading.group(1))
            continue
        if current_method and "|" in line:
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


def finite(values: Iterable[float]) -> list[float]:
    return [value for value in values if np.isfinite(value)]


def set_bar_axis_limits(ax, values: list[float], *, max_fraction: float = 0.60) -> None:
    vals = finite(values)
    if not vals:
        return
    # Keep bars mostly in the lower band, while allowing a little more overlap
    # with the line band than scalability_capacity.py.
    top = max(vals) / max_fraction
    ax.set_ylim(0.0, top * 1.02)


def set_line_axis_limits(
    ax,
    values: list[float],
    *,
    min_fraction: float = 0.58,
    max_fraction: float = 0.96,
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


def values_by_method_and_model_count(
    rows: list[Row],
    metric: str,
) -> dict[tuple[int, str], float]:
    result: dict[tuple[int, str], float] = {}
    for row in rows:
        if metric == "training_duration_sec":
            value = row.training_duration_sec
        elif metric == "peak_memory_gib":
            value = row.peak_memory_gib
        else:
            raise KeyError(metric)
        result[(row.num_models, row.method)] = value
    return result


def model_counts_for_rows(rows: list[Row]) -> list[int]:
    model_counts: list[int] = []
    seen: set[int] = set()
    for row in rows:
        if row.num_models not in seen:
            seen.add(row.num_models)
            model_counts.append(row.num_models)
    return model_counts


def methods_for_rows(rows: list[Row]) -> list[str]:
    methods: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if row.method not in seen:
            seen.add(row.method)
            methods.append(row.method)
    return methods


def draw_dual_axis_plot(*, rows: list[Row], output_path: Path) -> None:
    plt = require_matplotlib_pyplot()
    model_counts = model_counts_for_rows(rows)
    method_order = methods_for_rows(rows)
    if not model_counts or not method_order:
        return

    memory_values = values_by_method_and_model_count(rows, "peak_memory_gib")
    duration_values = values_by_method_and_model_count(rows, "training_duration_sec")

    # Left axis: training duration line
    # Right axis: GPU peak memory bar
    fig, ax_line = plt.subplots(figsize=OUTPUT_FIGSIZE)
    ax_bar = ax_line.twinx()

    x = np.arange(len(model_counts), dtype=float)
    n_methods = len(method_order)
    total_width = 0.82
    bar_width = total_width / n_methods
    offsets = (np.arange(n_methods) - (n_methods - 1) / 2.0) * bar_width

    all_memory_values: list[float] = []
    all_duration_values: list[float] = []

    for method_index, method in enumerate(method_order):
        color = METHOD_COLORS.get(method, ACCENT_BLACK)
        values = [memory_values.get((num_models, method), float("nan")) for num_models in model_counts]
        all_memory_values.extend(values)
        valid_positions = [
            x_pos + offsets[method_index]
            for x_pos, value in zip(x, values)
            if np.isfinite(value)
        ]
        valid_values = [value for value in values if np.isfinite(value)]
        if valid_values:
            ax_bar.bar(
                valid_positions,
                valid_values,
                width=bar_width * 0.92,
                color=color,
                alpha=0.78,
                edgecolor="white",
                linewidth=0.6,
                zorder=2,
            )

    for method_index, method in enumerate(method_order):
        color = darker(METHOD_COLORS.get(method, ACCENT_BLACK))
        values = [duration_values.get((num_models, method), float("nan")) for num_models in model_counts]
        all_duration_values.extend(values)
        valid_x = [x_pos for x_pos, value in zip(x, values) if np.isfinite(value)]
        valid_y = [value for value in values if np.isfinite(value)]
        if valid_y:
            ax_line.plot(
                valid_x,
                valid_y,
                linewidth=AI_PAPER_LINE_WIDTH,
                marker=AI_PAPER_MARKERS[method_index % len(AI_PAPER_MARKERS)],
                markersize=AI_PAPER_MARKER_SIZE * 0.82,
                markerfacecolor=AI_PAPER_MARKER_FACE_COLOR,
                markeredgecolor=color,
                markeredgewidth=AI_PAPER_MARKER_EDGE_WIDTH,
                color=color,
                zorder=4,
            )

    set_line_axis_limits(ax_line, all_duration_values)
    hide_negative_y_tick_labels(ax_line)
    set_bar_axis_limits(ax_bar, all_memory_values)

    ax_line.set_xticks(x)
    ax_line.set_xticklabels([str(num_models) for num_models in model_counts])
    ax_line.set_xlabel("Number of Models")

    ax_line.set_ylabel("Training Duration (Line, sec)")
    ax_bar.set_ylabel("GPU Peak Memory (Bar, GiB)")

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
    for method in method_order:
        color = METHOD_COLORS.get(method, ACCENT_BLACK)
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                color=color,
                marker="s",
                linestyle="",
                markersize=7.5,
                markerfacecolor=color,
                markeredgecolor=color,
                label=method,
            )
        )
    legend = ax_line.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=len(method_order),
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
    output_path = output_dir / "scalability_num_models.pdf"
    draw_dual_axis_plot(rows=rows, output_path=output_path)
    return [output_path] if output_path.exists() else []


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
    if len(outputs) != 1:
        print(
            f"[warn] generated {len(outputs)} plots; expected 1. "
            "Check whether the markdown contains number-of-models tables."
        )
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
