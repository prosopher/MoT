#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp.exp_util import (
    ACCENT_RED,
    AI_PAPER_PALETTE,
    AI_PAPER_ALGORITHM_LABEL_ROTATION,
    AI_PAPER_ANNOTATION_FONT_SIZE,
    AI_PAPER_BAR_VALUE_FONT_SIZE,
    AI_PAPER_DENSE_WIDTH_SCALE_MAX,
    AI_PAPER_RADAR_FILL_ALPHA,
    AI_PAPER_VALUE_OFFSET_FRACTION,
    AI_PAPER_FIGURE_DPI,
    AI_PAPER_LEGEND_HANDLE_LENGTH,
    AI_PAPER_LINE_WIDTH,
    AI_PAPER_NATIVE_LINESTYLE,
    AI_PAPER_MARKER_EDGE_WIDTH,
    AI_PAPER_REFERENCE_LINE_WIDTH,
    AI_PAPER_CONTROL_LINESTYLE,
    AI_PAPER_TICK_LABEL_SIZE,
    double_column_figsize,
    scaled_double_column_figsize,
    apply_ai_paper_style,
    require_matplotlib_colors,
    require_matplotlib_pyplot,
    save_paper_figure,
    style_algorithm_tick_labels,
    style_axes_common,
)

DEFAULT_CATEGORY_ORDER = [
    "health",
    "other",
    "math",
    "culture",
    "physics",
    "computer science",
    "psychology",
    "philosophy",
    "politics",
    "history",
    "business",
    "biology",
    "economics",
    "law",
    "chemistry",
    "engineering",
    "geography",
]

EVAL_BAR_METRICS = [
    ("Acc", "acc_avg", "Accuracy (%)"),
    ("F1", "gen_f1_avg", "F1"),
    ("GPU Peak Memory", "gpu_peak_memory", "GPU Peak Memory (GiB)"),
]

MOT_COLOR = ACCENT_RED
MOT_H_COLOR = AI_PAPER_PALETTE[2]
OTHER_BAR_COLOR = "#C7CDD6"
NATIVE_COLOR = AI_PAPER_PALETTE[6]
NON_RED_ORANGE_PURPLE_RADAR_PALETTES = (
    "tab20",
    "tab20b",
    "tab20c",
    "Set2",
    "Dark2",
    "Accent",
    "Paired",
    "Set3",
)

ALGORITHM_DISPLAY_NAMES = {
    "c2c": "C2C-Project",
    "interlat": "Interlat",
    "kvcomm": "KVComm",
    "lsc": "LSC",
    "mot": "MoT",
    "mot-h": "MoT-h",
    "mot-single": "MoT (single)",
    "native": "Native",
}


@dataclass
class Series:
    algorithm: str
    filename: str
    edge_id: str
    category_to_accuracy: Dict[str, float]


@dataclass
class LoadResult:
    series_by_edge: Dict[str, List[Series]]
    native_accuracy_by_edge: Dict[str, Dict[str, float]]
    native_source_file_by_edge: Dict[str, str]


@dataclass
class EvalSection:
    edge_id: str
    title: str
    headers: List[str]
    native_row: Dict[str, str]
    method_row: Dict[str, str]


@dataclass
class EvalRecord:
    study_id: str
    method: str
    values: Dict[str, str]
    source_log: Path
    is_native: bool = False


@dataclass
class EvalSummaryResult:
    title_by_edge: Dict[str, str]
    headers_by_edge: Dict[str, List[str]]
    records_by_edge: Dict[str, List[EvalRecord]]
    source_logs: List[Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize redux JSON files into per-subcategory radar charts and "
            "aggregate eval.log markdown summaries into tables and bar charts."
        )
    )
    parser.add_argument(
        "exp_path",
        type=Path,
        help="Experiment directory. Radar-chart JSON input is read from exp_path/*/mmlu_redux_subject_category_accuracy.json.",
    )
    parser.add_argument(
        "--edge-id",
        default=None,
        help="Only draw the specified edge_id. By default, use --directions.",
    )
    parser.add_argument(
        "--directions",
        default="A_to_B",
        help=(
            "Comma-separated direction IDs to include, such as 'A_to_B' or 'A_to_B,B_to_A'. "
            "Use 'all' to include every discovered direction. Default: A_to_B"
        ),
    )
    parser.add_argument(
        "--min-series",
        type=int,
        default=1,
        help=(
            "Minimum number of algorithm series required to draw an edge. "
            "Set to 2 if you only want comparison charts. Default: 1"
        ),
    )
    parser.add_argument(
        "--title-prefix",
        default="Per-subcategory accuracy",
        help="Title prefix for each radar chart.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=AI_PAPER_FIGURE_DPI,
        help=f"Saved figure DPI. Default: {AI_PAPER_FIGURE_DPI}",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show charts interactively in addition to saving them.",
    )
    parser.add_argument(
        "--max-radius",
        type=float,
        default=None,
        help=(
            "Optional fixed max radius (e.g. 0.8 or 1.0). "
            "If omitted, the script auto-computes it from the data."
        ),
    )
    parser.add_argument(
        "--disable-native",
        action="store_true",
        help=(
            "Do not draw the black 'native' line derived from the first JSON file's "
            "native_accuracy values."
        ),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def category_order(categories: Sequence[str]) -> List[str]:
    categories = list(categories)
    category_set = set(categories)

    ordered = [c for c in DEFAULT_CATEGORY_ORDER if c in category_set]
    leftover = sorted(category_set - set(ordered))
    return ordered + leftover


def extract_native_accuracy(data: dict) -> Dict[str, Dict[str, float]]:
    native_accuracy_by_edge: Dict[str, Dict[str, float]] = {}
    edges = data.get("edges", {})
    if not isinstance(edges, dict):
        return native_accuracy_by_edge

    for edge_id, edge_payload in edges.items():
        if not isinstance(edge_payload, dict):
            continue
        subject_category_accuracy = edge_payload.get("subject_category_accuracy", {})
        if not isinstance(subject_category_accuracy, dict):
            continue

        category_to_native: Dict[str, float] = {}
        for category, payload in subject_category_accuracy.items():
            if not isinstance(payload, dict):
                continue
            native_accuracy = payload.get("native_accuracy")
            if native_accuracy is None:
                continue
            try:
                category_to_native[str(category)] = float(native_accuracy)
            except (TypeError, ValueError):
                continue

        if category_to_native:
            native_accuracy_by_edge[str(edge_id)] = category_to_native

    return native_accuracy_by_edge


def list_subject_category_json_files(exp_path: Path) -> List[Path]:
    if not exp_path.exists() or not exp_path.is_dir():
        return []
    return sorted(
        path
        for path in exp_path.glob("*/mmlu_redux_subject_category_accuracy.json")
        if path.is_file()
    )


def load_series(exp_path: Path) -> LoadResult:
    json_files = list_subject_category_json_files(exp_path)
    if not json_files:
        raise FileNotFoundError(
            f"No mmlu_redux_subject_category_accuracy.json files found in {exp_path}/*/"
        )

    native_accuracy_by_edge: Dict[str, Dict[str, float]] = {}
    native_source_file_by_edge: Dict[str, str] = {}
    series_by_edge: Dict[str, List[Series]] = {}

    for json_file in json_files:
        data = read_json(json_file)

        native_candidates = extract_native_accuracy(data)
        for edge_id, category_to_native in native_candidates.items():
            if edge_id not in native_accuracy_by_edge:
                native_accuracy_by_edge[edge_id] = category_to_native
                native_source_file_by_edge[edge_id] = json_file.name

        algorithm = str(data["algorithm"]).strip()
        edges = data["edges"]
        if not isinstance(edges, dict):
            raise ValueError(f"'edges' must be a dict in {json_file}")

        for edge_id, edge_payload in edges.items():
            if not isinstance(edge_payload, dict):
                continue

            subject_category_accuracy = edge_payload.get("subject_category_accuracy", {})
            if not isinstance(subject_category_accuracy, dict):
                continue

            category_to_accuracy: Dict[str, float] = {}
            for category, payload in subject_category_accuracy.items():
                if not isinstance(payload, dict):
                    continue
                accuracy = payload.get("accuracy")
                if accuracy is None:
                    continue
                try:
                    category_to_accuracy[str(category)] = float(accuracy)
                except (TypeError, ValueError):
                    continue

            if not category_to_accuracy:
                continue

            series_by_edge.setdefault(str(edge_id), []).append(
                Series(
                    algorithm=algorithm,
                    filename=json_file.name,
                    edge_id=str(edge_id),
                    category_to_accuracy=category_to_accuracy,
                )
            )

    if not series_by_edge:
        raise ValueError(
            "No usable 'edges.<edge_id>.subject_category_accuracy.<category>.accuracy' "
            "entries were found."
        )

    return LoadResult(
        series_by_edge=series_by_edge,
        native_accuracy_by_edge=native_accuracy_by_edge,
        native_source_file_by_edge=native_source_file_by_edge,
    )


def display_name_for_algorithm(algorithm: str) -> str:
    normalized = algorithm.strip().lower()
    return ALGORITHM_DISPLAY_NAMES.get(normalized, algorithm.strip())


def display_name_for_subcategory(subcategory: str) -> str:
    value = subcategory.strip()
    if not value:
        return value
    return value[0].upper() + value[1:]


def dedupe_legend_names(series_list: List[Series]) -> List[str]:
    counts: Dict[str, int] = {}
    labels: List[str] = []
    for s in series_list:
        counts[s.algorithm] = counts.get(s.algorithm, 0) + 1

    for s in series_list:
        display_name = display_name_for_algorithm(s.algorithm)
        if counts[s.algorithm] == 1:
            labels.append(display_name)
        else:
            labels.append(f"{display_name} ({s.filename})")
    return labels


def average_accuracy_for_series(series: Series, categories: Sequence[str]) -> float:
    values = [
        series.category_to_accuracy[category]
        for category in categories
        if category in series.category_to_accuracy
    ]
    if not values:
        return float("inf")
    return float(sum(values) / len(values))


def order_series_by_average_accuracy(
    series_list: Sequence[Series],
    categories: Sequence[str],
) -> List[Series]:
    return sorted(
        series_list,
        key=lambda series: (
            average_accuracy_for_series(series, categories),
            series.algorithm.lower(),
            series.filename.lower(),
        ),
    )


def is_excluded_radar_palette_color(color: tuple[float, float, float, float]) -> bool:
    mcolors = require_matplotlib_colors()
    r, g, b, _ = color
    h, s, v = mcolors.rgb_to_hsv((r, g, b))

    if v < 0.22:
        return True
    if s < 0.16:
        return False

    is_red = h < 0.05 or h >= 0.97
    is_red_to_orange = 0.05 <= h <= 0.17
    is_purple_or_magenta = 0.68 <= h <= 0.96
    return is_red or is_red_to_orange or is_purple_or_magenta


def non_red_orange_purple_radar_palette(sample_count: int) -> List[tuple[float, float, float, float]]:
    plt = require_matplotlib_pyplot()
    mcolors = require_matplotlib_colors()
    if sample_count <= 0:
        return []

    colors: List[tuple[float, float, float, float]] = []
    seen_hex: set[str] = set()
    sample_grid = np.linspace(0.03, 0.97, max(sample_count * 8, 96))

    for palette_name in NON_RED_ORANGE_PURPLE_RADAR_PALETTES:
        cmap = plt.get_cmap(palette_name)
        palette_colors = [mcolors.to_rgba(cmap(x)) for x in sample_grid]

        for color in palette_colors:
            if is_excluded_radar_palette_color(color):
                continue
            color_hex = mcolors.to_hex(color, keep_alpha=False)
            if color_hex in seen_hex:
                continue
            seen_hex.add(color_hex)
            colors.append(color)
            if len(colors) >= sample_count:
                return colors

    raise ValueError(
        f"Unable to sample {sample_count} radar colors while excluding red, orange, and purple hues."
    )


def radar_colors_for_series(series_list: Sequence[Series]) -> List[tuple[float, float, float, float] | str]:
    other_count = sum(1 for series in series_list if series.algorithm.strip().lower() not in {"mot", "mot-h"})
    other_colors = non_red_orange_purple_radar_palette(other_count)
    other_color_iter = iter(other_colors)

    colors: List[tuple[float, float, float, float] | str] = []
    for series in series_list:
        normalized = series.algorithm.strip().lower()
        if normalized == "mot":
            colors.append(MOT_COLOR)
        elif normalized == "mot-h":
            colors.append(MOT_H_COLOR)
        else:
            colors.append(next(other_color_iter))
    return colors


def compute_radius_max(
    series_list: List[Series],
    categories: Sequence[str],
    fixed: Optional[float],
    native_accuracy: Optional[Dict[str, float]] = None,
) -> float:
    if fixed is not None:
        return fixed

    values = []
    for s in series_list:
        for c in categories:
            if c in s.category_to_accuracy:
                values.append(s.category_to_accuracy[c])

    if native_accuracy:
        for c in categories:
            if c in native_accuracy:
                values.append(native_accuracy[c])

    if not values:
        return 1.0

    max_value = max(values)
    if max_value <= 0.4:
        return 0.4
    if max_value <= 0.6:
        return 0.6
    if max_value <= 0.8:
        return 0.8
    return min(1.0, math.ceil(max_value * 10) / 10)


def plot_edge_radar(
    edge_id: str,
    series_list: List[Series],
    categories: Sequence[str],
    output_path: Path,
    title_prefix: str = "Per-subcategory accuracy",
    dpi: int = 220,
    max_radius: Optional[float] = None,
    show: bool = False,
    native_accuracy: Optional[Dict[str, float]] = None,
) -> None:
    if not series_list:
        raise ValueError(f"No series to plot for edge_id={edge_id!r}")

    apply_ai_paper_style()
    plt = require_matplotlib_pyplot()

    n = len(categories)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    radius_max = compute_radius_max(
        series_list, categories, max_radius, native_accuracy=native_accuracy
    )
    ordered_series_list = order_series_by_average_accuracy(series_list, categories)
    legend_labels = dedupe_legend_names(ordered_series_list)
    series_colors = radar_colors_for_series(ordered_series_list)

    fig = plt.figure(figsize=double_column_figsize(square=True))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    radar_label_fontsize = AI_PAPER_TICK_LABEL_SIZE + 3
    radar_rtick_fontsize = AI_PAPER_TICK_LABEL_SIZE + 2
    radar_legend_fontsize = AI_PAPER_TICK_LABEL_SIZE + 2

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
        [display_name_for_subcategory(category) for category in categories],
        fontsize=radar_label_fontsize,
        fontweight="bold",
    )
    for tick_label in ax.get_xticklabels():
        tick_label.set_clip_on(False)

    if radius_max <= 0.4:
        rticks = [0.1, 0.2, 0.3, 0.4]
    elif radius_max <= 0.6:
        rticks = [0.2, 0.4, 0.6]
    elif radius_max <= 0.8:
        rticks = [0.2, 0.5, 0.8]
    else:
        rticks = [0.2, 0.5, 0.8, 1.0]

    rticks = [t for t in rticks if t <= radius_max + 1e-9]
    ax.set_rlabel_position(0)
    ax.set_yticks(rticks)
    ax.set_yticklabels([f"{t:.1f}" for t in rticks], fontsize=radar_rtick_fontsize)
    ax.set_ylim(0, radius_max)
    style_axes_common(ax, grid=True, grid_axis="both")

    if native_accuracy:
        native_values = [native_accuracy.get(cat, np.nan) for cat in categories]
        native_values += native_values[:1]
        ax.plot(
            angles,
            native_values,
            color=NATIVE_COLOR,
            linewidth=AI_PAPER_LINE_WIDTH,
            linestyle=AI_PAPER_NATIVE_LINESTYLE,
            label=display_name_for_algorithm("native"),
            zorder=10,
        )

    for series, label, color in zip(ordered_series_list, legend_labels, series_colors):
        values = [series.category_to_accuracy.get(cat, np.nan) for cat in categories]
        values += values[:1]
        ax.plot(angles, values, linewidth=AI_PAPER_LINE_WIDTH, label=label, color=color)
        ax.fill(angles, values, alpha=AI_PAPER_RADAR_FILL_ALPHA, color=color)
    legend = fig.legend(
        loc="upper right",
        bbox_to_anchor=(0.998, 0.998),
        bbox_transform=fig.transFigure,
        frameon=True,
        fontsize=radar_legend_fontsize,
        handlelength=AI_PAPER_LEGEND_HANDLE_LENGTH,
        borderaxespad=0.0,
    )
    for text in legend.get_texts():
        text.set_fontweight("bold")

    ax.set_position([0.04, 0.20, 0.92, 0.76])

    save_paper_figure(fig, output_path, dpi=dpi, show=show)
    plt.close(fig)


def split_markdown_row(line: str) -> List[str]:
    stripped = line.strip()
    if not (stripped.startswith("|") and stripped.endswith("|")):
        raise ValueError(f"Invalid markdown table row: {line!r}")
    return [cell.strip() for cell in stripped.strip("|").split("|")]


def is_markdown_separator_row(line: str) -> bool:
    cells = split_markdown_row(line)
    return all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def parse_eval_log_sections(eval_log_path: Path) -> List[EvalSection]:
    lines = eval_log_path.read_text(encoding="utf-8").splitlines()
    sections: List[EvalSection] = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line.startswith("### "):
            i += 1
            continue

        title = line[4:].strip()
        edge_id = title.split(" 방향", 1)[0].strip()
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1

        if j >= len(lines) or not lines[j].lstrip().startswith("| Method |"):
            i += 1
            continue

        header_cells = split_markdown_row(lines[j])
        j += 1
        if j >= len(lines) or not lines[j].lstrip().startswith("|"):
            raise ValueError(f"Malformed markdown table in {eval_log_path}")
        if not is_markdown_separator_row(lines[j]):
            raise ValueError(f"Expected markdown separator row in {eval_log_path}")
        j += 1

        raw_rows: List[List[str]] = []
        while j < len(lines):
            candidate = lines[j].strip()
            if not candidate.startswith("|"):
                break
            row_cells = split_markdown_row(candidate)
            if len(row_cells) != len(header_cells):
                break
            raw_rows.append(row_cells)
            j += 1

        if len(raw_rows) < 2:
            raise ValueError(
                f"Expected native row and method row in {eval_log_path} for section {title!r}"
            )

        native_row = dict(zip(header_cells, raw_rows[0]))
        method_row = dict(zip(header_cells, raw_rows[1]))
        sections.append(
            EvalSection(
                edge_id=edge_id,
                title=title,
                headers=header_cells,
                native_row=native_row,
                method_row=method_row,
            )
        )
        i = j

    return sections


def load_eval_summaries(exp_path: Path) -> EvalSummaryResult:
    eval_logs = sorted(path for path in exp_path.glob("*/eval.log") if path.is_file())
    title_by_edge: Dict[str, str] = {}
    headers_by_edge: Dict[str, List[str]] = {}
    records_by_edge: Dict[str, List[EvalRecord]] = {}

    for eval_log_path in eval_logs:
        study_id = eval_log_path.parent.name
        sections = parse_eval_log_sections(eval_log_path)
        if not sections:
            continue

        for section in sections:
            title_by_edge.setdefault(section.edge_id, section.title)
            existing_headers = headers_by_edge.setdefault(section.edge_id, section.headers)
            if existing_headers != section.headers:
                raise ValueError(
                    f"Mismatched eval.log table header for edge_id={section.edge_id!r}: {eval_log_path}"
                )

            record_list = records_by_edge.setdefault(section.edge_id, [])
            if not any(record.is_native for record in record_list):
                native_values = dict(section.native_row)
                native_values["Method"] = "native"
                record_list.append(
                    EvalRecord(
                        study_id="all",
                        method="native",
                        values=native_values,
                        source_log=eval_log_path,
                        is_native=True,
                    )
                )

            record_list.append(
                EvalRecord(
                    study_id=study_id,
                    method=section.method_row.get("Method", study_id),
                    values=section.method_row,
                    source_log=eval_log_path,
                    is_native=False,
                )
            )

    for edge_id, records in records_by_edge.items():
        native_records = [record for record in records if record.is_native]
        method_records = sorted(
            (record for record in records if not record.is_native),
            key=lambda record: record.study_id,
        )
        records_by_edge[edge_id] = native_records + method_records

    return EvalSummaryResult(
        title_by_edge=title_by_edge,
        headers_by_edge=headers_by_edge,
        records_by_edge=records_by_edge,
        source_logs=eval_logs,
    )


def sanitize_filename_component(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    slug = slug.strip("._")
    return slug or "summary"


def parse_metric_number(raw_value: str) -> Optional[float]:
    value = raw_value.strip()
    if not value or value.upper() == "N/A":
        return None

    patterns = [
        r"^([-+]?\d*\.?\d+)\s*%$",
        r"^([-+]?\d*\.?\d+)\s*ms$",
        r"^([-+]?\d*\.?\d+)\s*tok/s$",
        r"^([-+]?\d*\.?\d+)\s*GiB$",
        r"^([-+]?\d*\.?\d+)$",
    ]
    for pattern in patterns:
        match = re.match(pattern, value)
        if match:
            return float(match.group(1))
    return None


def write_eval_summary_markdown(summary: EvalSummaryResult, output_path: Path) -> None:
    lines: List[str] = ["# Eval Log Summary", ""]

    for edge_id in sorted(summary.records_by_edge):
        title = summary.title_by_edge[edge_id]
        headers = summary.headers_by_edge[edge_id]
        records = summary.records_by_edge[edge_id]

        lines.append(f"## {title}")
        lines.append("")
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join("---:" if i > 0 else "---" for i in range(len(headers))) + "|")
        for record in records:
            row_values = []
            for header in headers:
                value = record.values.get(header, "")
                if header == "Method":
                    value = display_name_for_algorithm(value)
                row_values.append(value)
            lines.append("| " + " | ".join(row_values) + " |")
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_eval_summary_csvs(summary: EvalSummaryResult, output_dir: Path) -> List[Path]:
    generated: List[Path] = []
    for edge_id in sorted(summary.records_by_edge):
        headers = summary.headers_by_edge[edge_id]
        records = summary.records_by_edge[edge_id]
        output_path = output_dir / f"eval_summary_{sanitize_filename_component(edge_id)}.csv"
        with output_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Study ID"] + headers)
            for record in records:
                row_values = []
                for header in headers:
                    value = record.values.get(header, "")
                    if header == "Method":
                        value = display_name_for_algorithm(value)
                    row_values.append(value)
                writer.writerow([record.study_id] + row_values)
        generated.append(output_path)
    return generated


def chart_title_text(title: str) -> str:
    return title.replace(" 방향 ", " direction ")


def bar_color_for_method(method: str) -> str:
    normalized = method.strip().lower()
    if normalized == "native":
        return NATIVE_COLOR
    if normalized == "mot-h":
        return MOT_H_COLOR
    if normalized == "mot":
        return MOT_COLOR
    return OTHER_BAR_COLOR


def plot_eval_metric_bars(
    edge_id: str,
    title: str,
    records: List[EvalRecord],
    metric_name: str,
    metric_slug: str,
    ylabel: str,
    output_dir: Path,
    dpi: int,
    show: bool,
) -> Optional[Path]:
    plot_items = []
    native_values = []
    for record in records:
        metric_value = parse_metric_number(record.values.get(metric_name, ""))
        if metric_value is None:
            continue
        if record.method.strip().lower() == "native":
            if "peak memory" not in metric_name.lower():
                native_values.append(metric_value)
            continue
        label = display_name_for_algorithm(record.method.strip() or record.study_id)
        plot_items.append((metric_value, label, record.method))

    native_value = native_values[0] if native_values else None

    if not plot_items and native_value is None:
        return None

    plot_items.sort(key=lambda item: (item[0], item[1]))
    values = [item[0] for item in plot_items]
    labels = [item[1] for item in plot_items]
    colors = [bar_color_for_method(item[2]) for item in plot_items]

    apply_ai_paper_style()
    plt = require_matplotlib_pyplot()

    width_scale = min(AI_PAPER_DENSE_WIDTH_SCALE_MAX, max(1.0, (1.2 * max(1, len(labels)) + 2.0) / 7.16))
    fig, ax = plt.subplots(figsize=scaled_double_column_figsize(width_scale=width_scale, height=4.80))
    x_positions = list(range(len(labels)))
    bars = ax.bar(x_positions, values, color=colors) if values else []
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=AI_PAPER_ALGORITHM_LABEL_ROTATION, ha="right")
    ax.set_ylabel(ylabel)
    style_axes_common(ax, grid=True, grid_axis="y")
    style_algorithm_tick_labels(ax, axis="x")

    value_format = "{:.1f}" if metric_name == "Acc" else "{:.3f}"
    plotted_values = values + ([native_value] if native_value is not None else [])
    offset = max(plotted_values) * AI_PAPER_VALUE_OFFSET_FRACTION if plotted_values and max(plotted_values) > 0 else AI_PAPER_VALUE_OFFSET_FRACTION
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            value_format.format(value),
            ha="center",
            va="bottom",
            fontsize=AI_PAPER_BAR_VALUE_FONT_SIZE,
        )

    if native_value is not None:
        ax.axhline(
            native_value,
            color=NATIVE_COLOR,
            linestyle=AI_PAPER_NATIVE_LINESTYLE,
            linewidth=AI_PAPER_LINE_WIDTH,
            label=f"{display_name_for_algorithm('native')}={value_format.format(native_value)}",
            zorder=5,
        )
        ax.legend(loc="best", handlelength=AI_PAPER_LEGEND_HANDLE_LENGTH)

    output_path = output_dir / f"eval_bar_{sanitize_filename_component(edge_id)}_{metric_slug}.pdf"
    save_paper_figure(fig, output_path, dpi=dpi, show=show)
    plt.close(fig)
    return output_path


def resolve_target_edge_ids(
    available_edge_ids: Sequence[str],
    edge_id: Optional[str],
    directions: str,
) -> List[str]:
    available = sorted(dict.fromkeys(available_edge_ids))
    if edge_id is not None:
        if edge_id not in available:
            raise KeyError(
                f"edge_id {edge_id!r} not found. Available edge_ids: {', '.join(available)}"
            )
        return [edge_id]

    directions_value = str(directions).strip()
    if not directions_value or directions_value.lower() == "all":
        return available

    requested = [item.strip() for item in directions_value.split(",") if item.strip()]
    if not requested:
        return available

    missing = [item for item in requested if item not in available]
    if missing:
        raise KeyError(
            f"directions not found: {', '.join(missing)}. Available edge_ids: {', '.join(available)}"
        )
    return requested


def generate_eval_artifacts(
    args: argparse.Namespace,
    exp_path: Path,
    output_dir: Path,
    dpi: int,
    show: bool,
) -> List[Path]:
    summary = load_eval_summaries(exp_path)
    if not summary.records_by_edge:
        return []

    target_edges = resolve_target_edge_ids(
        available_edge_ids=summary.records_by_edge.keys(),
        edge_id=args.edge_id,
        directions=args.directions,
    )
    summary.records_by_edge = {edge_id: summary.records_by_edge[edge_id] for edge_id in target_edges}
    summary.title_by_edge = {edge_id: summary.title_by_edge[edge_id] for edge_id in target_edges}
    summary.headers_by_edge = {edge_id: summary.headers_by_edge[edge_id] for edge_id in target_edges}

    generated: List[Path] = []
    markdown_path = output_dir / "eval_summary.md"
    write_eval_summary_markdown(summary, markdown_path)
    generated.append(markdown_path)

    generated.extend(write_eval_summary_csvs(summary, output_dir))

    for edge_id in sorted(summary.records_by_edge):
        title = summary.title_by_edge[edge_id]
        records = summary.records_by_edge[edge_id]
        for metric_name, metric_slug, ylabel in EVAL_BAR_METRICS:
            chart_path = plot_eval_metric_bars(
                edge_id=edge_id,
                title=title,
                records=records,
                metric_name=metric_name,
                metric_slug=metric_slug,
                ylabel=ylabel,
                output_dir=output_dir,
                dpi=dpi,
                show=show,
            )
            if chart_path is not None:
                generated.append(chart_path)

    return generated


def generate_redux_radar_charts(args: argparse.Namespace, exp_path: Path, output_dir: Path) -> List[Path]:
    json_files = list_subject_category_json_files(exp_path)
    if not json_files:
        print(
            f"[info] no mmlu_redux_subject_category_accuracy.json files found under {exp_path}/*/; skipping radar charts."
        )
        return []

    result = load_series(exp_path)
    series_by_edge = result.series_by_edge
    target_edges = resolve_target_edge_ids(
        available_edge_ids=series_by_edge.keys(),
        edge_id=args.edge_id,
        directions=args.directions,
    )

    if not args.disable_native:
        print(
            "[info] native is taken per edge_id from the first study JSON file "
            "(sorted order) under exp_path/*/ that actually contains that edge."
        )

    generated: List[Path] = []
    for edge_id in target_edges:
        series_list = series_by_edge[edge_id]
        if len(series_list) < args.min_series:
            print(
                f"[skip] edge_id={edge_id!r}: found {len(series_list)} series, "
                f"but --min-series={args.min_series}"
            )
            continue

        categories = category_order(
            {
                category
                for s in series_list
                for category in s.category_to_accuracy.keys()
            }
        )
        output_path = output_dir / f"subcategory_radar_{edge_id}.pdf"
        native_accuracy = None
        native_source = None
        if not args.disable_native:
            native_accuracy = result.native_accuracy_by_edge.get(edge_id)
            native_source = result.native_source_file_by_edge.get(edge_id)

        plot_edge_radar(
            edge_id=edge_id,
            series_list=series_list,
            categories=categories,
            output_path=output_path,
            title_prefix=args.title_prefix,
            dpi=args.dpi,
            max_radius=args.max_radius,
            show=args.show,
            native_accuracy=native_accuracy,
        )
        generated.append(output_path)
        if native_source:
            print(f"[saved] {output_path} (native source: {native_source})")
        else:
            print(f"[saved] {output_path}")

    return generated


def main() -> None:
    args = parse_args()
    exp_path = args.exp_path
    output_dir = exp_path
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_paths: List[Path] = []
    generated_paths.extend(generate_redux_radar_charts(args, exp_path, output_dir))

    eval_artifacts = generate_eval_artifacts(
        args=args,
        exp_path=exp_path,
        output_dir=output_dir,
        dpi=args.dpi,
        show=args.show,
    )
    if eval_artifacts:
        for artifact_path in eval_artifacts:
            print(f"[saved] {artifact_path}")
    else:
        print(f"[info] no eval.log files found under {exp_path}/*/eval.log; skipping eval summaries.")

    generated_paths.extend(eval_artifacts)
    if not generated_paths:
        raise RuntimeError(
            "No artifacts were generated. Expected mmlu_redux_subject_category_accuracy.json in exp_path/*/ or eval.log files in exp_path/*/."
        )


if __name__ == "__main__":
    main()
