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

SERIES_COLORS = [
    ACCENT_BLUE,
    ACCENT_AQUA,
    ACCENT_ORANGE,
    ACCENT_GREEN,
    ACCENT_BLACK,
]


@dataclass(frozen=True)
class Row:
    series: str
    ratio: float
    f1: float
    latency_ms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot channel-ratio results from a markdown result file."
    )
    parser.add_argument(
        "markdown_file",
        type=Path,
        help="Markdown file containing channel-ratio result tables.",
    )
    return parser.parse_args()


def normalize_header(name: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    if key in {"ratio_of_channels", "channel_ratio", "ratio"}:
        return "ratio_of_channels"
    if key == "avg_latency":
        return "latency"
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


def parse_table(table_lines: list[str], *, series: str) -> list[Row]:
    if len(table_lines) < 3:
        return []

    raw_header = split_markdown_row(table_lines[0])
    separator = split_markdown_row(table_lines[1])
    if not is_separator_row(separator):
        return []

    headers = [normalize_header(header) for header in raw_header]
    header_index = {name: idx for idx, name in enumerate(headers)}
    required = ["ratio_of_channels", "f1", "latency"]
    if any(name not in header_index for name in required):
        return []

    parsed: list[Row] = []
    for line in table_lines[2:]:
        cells = split_markdown_row(line)
        if len(cells) != len(headers):
            continue
        parsed.append(
            Row(
                series=series,
                ratio=parse_number(cells[header_index["ratio_of_channels"]]),
                f1=parse_number(cells[header_index["f1"]]),
                latency_ms=parse_number(cells[header_index["latency"]]),
            )
        )
    return parsed


def parse_markdown(path: Path) -> list[Row]:
    text = path.read_text(encoding="utf-8")
    rows: list[Row] = []
    current_series: str | None = None
    table_lines: list[str] = []

    def flush_table() -> None:
        nonlocal table_lines
        if current_series and table_lines:
            rows.extend(parse_table(table_lines, series=current_series))
        table_lines = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        heading = re.match(r"^#+\s*(.+?)\s*$", line)
        if heading:
            flush_table()
            current_series = heading.group(1).strip()
            continue
        if current_series and "|" in line:
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
        axis_min = max(0.0, axis_min)
    ax.set_ylim(axis_min, axis_max)


def hide_negative_y_tick_labels(ax) -> None:
    from matplotlib.ticker import FuncFormatter

    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _position: "" if value < 0 else f"{value:g}")
    )


def ratios_for_rows(rows: list[Row]) -> list[float]:
    ratios: list[float] = []
    seen: set[float] = set()
    for row in rows:
        if row.ratio not in seen:
            seen.add(row.ratio)
            ratios.append(row.ratio)
    return ratios


def series_for_rows(rows: list[Row]) -> list[str]:
    series_names: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if row.series not in seen:
            seen.add(row.series)
            series_names.append(row.series)
    return series_names


def values_by_series_and_ratio(rows: list[Row], metric: str) -> dict[tuple[float, str], float]:
    result: dict[tuple[float, str], float] = {}
    for row in rows:
        if metric == "f1":
            value = row.f1
        elif metric == "latency_ms":
            value = row.latency_ms
        else:
            raise KeyError(metric)
        result[(row.ratio, row.series)] = value
    return result


def format_ratio_tick(value: float) -> str:
    text = f"{value:.1f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def draw_dual_axis_plot(*, rows: list[Row], output_path: Path) -> None:
    plt = require_matplotlib_pyplot()
    ratios = ratios_for_rows(rows)
    series_names = series_for_rows(rows)
    if not ratios or not series_names:
        return

    f1_values = values_by_series_and_ratio(rows, "f1")
    latency_values = values_by_series_and_ratio(rows, "latency_ms")
    color_map = {
        series: SERIES_COLORS[index % len(SERIES_COLORS)]
        for index, series in enumerate(series_names)
    }

    fig, ax_line = plt.subplots(figsize=OUTPUT_FIGSIZE)
    ax_bar = ax_line.twinx()

    x = np.array(ratios, dtype=float)
    if len(x) >= 2:
        min_spacing = float(np.min(np.diff(x)))
    else:
        min_spacing = 0.1
    total_width = min_spacing * 0.82
    n_series = len(series_names)
    bar_width = total_width / n_series
    offsets = (np.arange(n_series) - (n_series - 1) / 2.0) * bar_width

    all_f1_values: list[float] = []
    all_latency_values: list[float] = []

    for series_index, series in enumerate(series_names):
        color = color_map[series]
        values = [latency_values.get((ratio, series), float("nan")) for ratio in ratios]
        all_latency_values.extend(values)
        valid_positions = [
            x_pos + offsets[series_index]
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

    for series_index, series in enumerate(series_names):
        color = darker(color_map[series])
        values = [f1_values.get((ratio, series), float("nan")) for ratio in ratios]
        all_f1_values.extend(values)
        valid_x = [x_pos for x_pos, value in zip(x, values) if np.isfinite(value)]
        valid_y = [value for value in values if np.isfinite(value)]
        if valid_y:
            ax_line.plot(
                valid_x,
                valid_y,
                linewidth=AI_PAPER_LINE_WIDTH,
                marker=AI_PAPER_MARKERS[series_index % len(AI_PAPER_MARKERS)],
                markersize=AI_PAPER_MARKER_SIZE * 0.82,
                markerfacecolor=AI_PAPER_MARKER_FACE_COLOR,
                markeredgecolor=color,
                markeredgewidth=AI_PAPER_MARKER_EDGE_WIDTH,
                color=color,
                zorder=4,
            )

    set_line_axis_limits(ax_line, all_f1_values)
    hide_negative_y_tick_labels(ax_line)
    set_bar_axis_limits(ax_bar, all_latency_values)

    ax_line.axvline(0.3, color="black", linestyle="--", linewidth=AI_PAPER_LINE_WIDTH, zorder=3)

    ax_line.set_xticks(x)
    ax_line.set_xticklabels([format_ratio_tick(ratio) for ratio in ratios])
    ax_line.set_xlabel("Ratio of Channels")

    ax_line.set_ylabel("F1 (Line)")
    ax_bar.set_ylabel("Latency (Bar, ms)")

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

    ax_bar.set_zorder(1)
    ax_line.set_zorder(2)
    ax_line.patch.set_visible(False)

    legend_handles = []
    for series in series_names:
        color = color_map[series]
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
                label=series,
            )
        )
    legend = ax_line.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=len(series_names),
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
    output_path = output_dir / "channel_ratio.pdf"
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
            "Check whether the markdown contains channel-ratio tables."
        )
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
