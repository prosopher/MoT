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
    ACCENT_PURPLE,
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
    style_algorithm_tick_labels,
)

# Exact two-column raster size requested for this plot set.
OUTPUT_WIDTH_PX = 1881
OUTPUT_HEIGHT_PX = 1291
OUTPUT_DPI = 300
OUTPUT_FIGSIZE = (OUTPUT_WIDTH_PX / OUTPUT_DPI, OUTPUT_HEIGHT_PX / OUTPUT_DPI)

METHOD_COLOR_OVERRIDES = {
    "Native": ACCENT_BLACK,
    "Context Reconstruction": ACCENT_BLUE,
    "Prompt LM": ACCENT_AQUA,
    "Context Correction": ACCENT_ORANGE,
    "Context Correction + Prompt LM": ACCENT_GREEN,
}
FALLBACK_COLORS = [ACCENT_PURPLE, ACCENT_BLUE, ACCENT_AQUA, ACCENT_ORANGE, ACCENT_GREEN]
SECTION_GAP = 1.0


@dataclass(frozen=True)
class Row:
    section: str
    method: str
    accuracy_pct: float
    f1: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot ablation results from a markdown result file."
    )
    parser.add_argument(
        "markdown_file",
        type=Path,
        help="Markdown file containing ablation result tables.",
    )
    return parser.parse_args()


def normalize_header(name: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    if key in {"acc", "accuracy", "acc_avg", "accuracy_avg"}:
        return "acc"
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


def parse_table(table_lines: list[str], *, section: str) -> list[Row]:
    if len(table_lines) < 3:
        return []

    raw_header = split_markdown_row(table_lines[0])
    separator = split_markdown_row(table_lines[1])
    if not is_separator_row(separator):
        return []

    headers = [normalize_header(header) for header in raw_header]
    header_index = {name: idx for idx, name in enumerate(headers)}
    required = ["method", "acc", "f1"]
    if any(name not in header_index for name in required):
        return []

    parsed: list[Row] = []
    for line in table_lines[2:]:
        cells = split_markdown_row(line)
        if len(cells) != len(headers):
            continue
        parsed.append(
            Row(
                section=section,
                method=cells[header_index["method"]].strip(),
                accuracy_pct=parse_number(cells[header_index["acc"]]),
                f1=parse_number(cells[header_index["f1"]]),
            )
        )
    return parsed


def parse_markdown(path: Path) -> list[Row]:
    text = path.read_text(encoding="utf-8")
    rows: list[Row] = []
    current_section: str | None = None
    table_lines: list[str] = []

    def flush_table() -> None:
        nonlocal table_lines
        if current_section and table_lines:
            rows.extend(parse_table(table_lines, section=current_section))
        table_lines = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        heading = re.match(r"^#+\s*(.+?)\s*$", line)
        if heading:
            flush_table()
            current_section = heading.group(1).strip()
            continue
        if current_section and "|" in line:
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


def sections_for_rows(rows: list[Row]) -> list[str]:
    sections: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if row.section not in seen:
            seen.add(row.section)
            sections.append(row.section)
    return sections


def methods_for_section(rows: list[Row], section: str) -> list[str]:
    methods: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if row.section == section and row.method not in seen:
            seen.add(row.method)
            methods.append(row.method)
    return methods


def color_map_for_methods(rows: list[Row]) -> dict[str, str]:
    methods: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if row.method not in seen:
            seen.add(row.method)
            methods.append(row.method)

    color_map: dict[str, str] = {}
    fallback_index = 0
    for method in methods:
        if method in METHOD_COLOR_OVERRIDES:
            color_map[method] = METHOD_COLOR_OVERRIDES[method]
        else:
            color_map[method] = FALLBACK_COLORS[fallback_index % len(FALLBACK_COLORS)]
            fallback_index += 1
    return color_map


def build_layout(rows: list[Row]) -> tuple[list[float], list[str], list[float], list[str], list[float]]:
    positions: list[float] = []
    tick_labels: list[str] = []
    section_centers: list[float] = []
    section_labels: list[str] = []
    section_boundaries: list[float] = []

    cursor = 0.0
    for section in sections_for_rows(rows):
        start_index = len(positions)
        methods = methods_for_section(rows, section)
        for method in methods:
            positions.append(cursor)
            tick_labels.append(method)
            cursor += 1.0
        if methods:
            section_positions = positions[start_index:]
            section_centers.append(float(np.mean(section_positions)))
            section_labels.append(section)
            section_boundaries.append(cursor - 0.5 + SECTION_GAP / 2.0)
        cursor += SECTION_GAP
    if section_boundaries:
        section_boundaries.pop()
    return positions, tick_labels, section_centers, section_labels, section_boundaries


def draw_plot(*, rows: list[Row], output_path: Path) -> None:
    plt = require_matplotlib_pyplot()
    if not rows:
        return

    accuracy_by_key = {(row.section, row.method): row.accuracy_pct for row in rows}
    f1_by_key = {(row.section, row.method): row.f1 for row in rows}
    color_map = color_map_for_methods(rows)

    positions, tick_labels, section_centers, section_labels, section_boundaries = build_layout(rows)
    sections = sections_for_rows(rows)

    fig, ax_line = plt.subplots(figsize=OUTPUT_FIGSIZE)
    ax_bar = ax_line.twinx()

    all_accuracy_values = [row.accuracy_pct for row in rows]
    all_f1_values = [row.f1 for row in rows]

    bar_positions: list[float] = []
    bar_values: list[float] = []
    bar_colors: list[str] = []

    for section in sections:
        for method in methods_for_section(rows, section):
            x = positions[len(bar_positions)]
            bar_positions.append(x)
            bar_values.append(f1_by_key.get((section, method), float("nan")))
            bar_colors.append(color_map.get(method, ACCENT_BLACK))

    valid_bar_positions = [x for x, value in zip(bar_positions, bar_values) if np.isfinite(value)]
    valid_bar_values = [value for value in bar_values if np.isfinite(value)]
    valid_bar_colors = [color for color, value in zip(bar_colors, bar_values) if np.isfinite(value)]
    if valid_bar_values:
        ax_bar.bar(
            valid_bar_positions,
            valid_bar_values,
            width=0.82,
            color=valid_bar_colors,
            alpha=0.78,
            edgecolor="white",
            linewidth=0.6,
            zorder=2,
        )

    for method_index, method in enumerate(color_map):
        xs: list[float] = []
        ys: list[float] = []
        offset = 0
        for section in sections:
            methods = methods_for_section(rows, section)
            for section_method in methods:
                x = positions[offset]
                if section_method == method:
                    value = accuracy_by_key.get((section, section_method), float("nan"))
                    if np.isfinite(value):
                        xs.append(x)
                        ys.append(value)
                offset += 1
        if ys:
            color = darker(color_map.get(method, ACCENT_BLACK))
            ax_line.plot(
                xs,
                ys,
                linestyle="--",
                linewidth=AI_PAPER_LINE_WIDTH,
                marker=AI_PAPER_MARKERS[method_index % len(AI_PAPER_MARKERS)],
                markersize=AI_PAPER_MARKER_SIZE * 0.82,
                markerfacecolor=AI_PAPER_MARKER_FACE_COLOR,
                markeredgecolor=color,
                markeredgewidth=AI_PAPER_MARKER_EDGE_WIDTH,
                color=color,
                zorder=4,
            )

    set_line_axis_limits(ax_line, all_accuracy_values)
    hide_negative_y_tick_labels(ax_line)
    set_bar_axis_limits(ax_bar, all_f1_values)

    ax_line.set_xticks(positions)
    ax_line.set_xticklabels(tick_labels, rotation=18, ha="right")
    ax_line.set_xlabel("")

    ax_line.set_ylabel("Accuracy (Line, %)")
    ax_bar.set_ylabel("F1 (Bar)")

    apply_bold_axis_labels(ax_line)
    apply_bold_axis_labels(ax_bar)
    style_algorithm_tick_labels(ax_line, axis="x")

    for boundary in section_boundaries:
        ax_line.axvline(boundary, color="0.80", linestyle="--", linewidth=0.8, zorder=1)

    for center, label in zip(section_centers, section_labels):
        ax_line.text(
            center,
            -0.18,
            label,
            transform=ax_line.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=AI_PAPER_TICK_LABEL_SIZE,
            fontweight="bold",
        )

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
    for method, color in color_map.items():
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
        ncol=len(legend_handles),
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

    fig.subplots_adjust(left=0.095, right=0.895, bottom=0.24, top=0.86)
    fig.set_size_inches(*OUTPUT_FIGSIZE, forward=True)
    with plt.rc_context({"savefig.bbox": None, "savefig.pad_inches": 0.0}):
        fig.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches=None, pad_inches=0.0)
    plt.close(fig)


def plot_all(rows: list[Row], markdown_path: Path) -> list[Path]:
    output_path = markdown_path.with_suffix(".pdf")
    draw_plot(rows=rows, output_path=output_path)
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

    outputs = plot_all(rows, markdown_path)
    if len(outputs) != 1:
        print(
            f"[warn] generated {len(outputs)} plots; expected 1. "
            "Check whether the markdown contains ablation tables."
        )
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
