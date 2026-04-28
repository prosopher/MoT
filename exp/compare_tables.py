#!/usr/bin/env python3

import re
import math
import argparse
import textwrap
from pathlib import Path
from dataclasses import dataclass
from typing import Literal
from collections import OrderedDict

import matplotlib.pyplot as plt


DEFAULT_Y_COL = "Gen F1 Avg"


# Colorblind-friendly palette, commonly used in academic plots
AI_PAPER_PALETTE = [
    "#C0504D",  # Accent Red
    "#4BACC6",  # Accent Aqua
    "#8064A2",  # Accent Purple
    "#4F81BD",  # Accent Blue
    "#9BBB59",  # Accent Green
    "#F79646",  # Accent Orange
    "#000000",  # black
]


AI_PAPER_MARKERS = [
    "o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">"
]


AI_PAPER_LINESTYLES = [
    "-", "--", "-.", ":"
]


ChartMode = Literal["line", "bar"]


@dataclass
class ExtractedData:
    chart_mode: ChartMode
    x_col_name: str
    line_series_data: OrderedDict[str, list[tuple[float, float]]]
    bar_group_data: OrderedDict[str, OrderedDict[str, float]]
    table_count: int
    matched_table_count: int

    @property
    def has_data(self) -> bool:
        if self.chart_mode == "line":
            return bool(self.line_series_data)
        return bool(self.bar_group_data)


def normalize_col_name(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().strip("`")).lower()


def normalize_group_label(text: str) -> str:
    """
    섹션 제목에 literal '\\n' 문자열이 들어 있으면
    실제 줄바꿈으로 변환합니다.

    예:
    'gpt2-medium→gpt2\\nRatio=0.5'
    ->
    'gpt2-medium→gpt2
    Ratio=0.5'
    """
    return text.replace("\\n", "\n").strip()


def split_markdown_row(line: str) -> list[str]:
    s = line.strip()

    if not s.startswith("|"):
        return []

    s = s[1:]
    if s.endswith("|"):
        s = s[:-1]

    cells = []
    cur = []
    in_code = False
    i = 0

    while i < len(s):
        ch = s[i]

        if ch == "\\" and i + 1 < len(s):
            cur.append(s[i + 1])
            i += 2
            continue

        if ch == "`":
            in_code = not in_code
            cur.append(ch)
        elif ch == "|" and not in_code:
            cells.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)

        i += 1

    cells.append("".join(cur).strip())
    return cells


def is_separator_row(cells: list[str]) -> bool:
    if not cells:
        return False

    for cell in cells:
        compact = cell.replace(" ", "")
        if not re.fullmatch(r":?-{3,}:?", compact):
            return False

    return True


def clean_cell_text(value: str) -> str:
    text = value.strip()
    text = text.strip("`")
    return text


def parse_number(value: str) -> float:
    """
    Y값처럼 단위가 붙을 수 있는 값을 숫자로 파싱합니다.

    예:
    - "0.129" -> 0.129
    - "18.9%" -> 18.9
    - "436.811 ms" -> 436.811
    - "586 tok/s" -> 586
    """
    text = clean_cell_text(value).replace(",", "")

    match = re.search(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
        text,
    )

    if not match:
        raise ValueError(f"숫자를 찾을 수 없습니다: {value!r}")

    return float(match.group(0))


def parse_x_number(value: str) -> float:
    """
    X축이 진짜 숫자형인지 엄격하게 판정합니다.

    parse_number()처럼 문자열 중간의 숫자를 뽑아내면
    "Layer 1", "Block 3" 같은 문자열도 숫자형으로 오판할 수 있으므로,
    첫 번째 컬럼 전체가 숫자 또는 퍼센트 형태일 때만 숫자로 인정합니다.
    """
    text = clean_cell_text(value).replace(",", "").strip()

    if not re.fullmatch(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?\s*%?",
        text,
    ):
        raise ValueError(f"X축 숫자가 아닙니다: {value!r}")

    return parse_number(text)


def make_unique_name(name: str, used: set[str]) -> str:
    base = normalize_group_label(name or "Table")
    candidate = base
    idx = 2

    while candidate in used:
        candidate = f"{base} ({idx})"
        idx += 1

    used.add(candidate)
    return candidate


def extract_tables(
    md_text: str,
    y_col: str,
) -> ExtractedData:
    lines = md_text.splitlines()

    current_heading = None
    used_group_names = set()

    line_series_data: OrderedDict[str, list[tuple[float, float]]] = OrderedDict()
    bar_group_data: OrderedDict[str, OrderedDict[str, float]] = OrderedDict()

    x_col_name = None
    y_key = normalize_col_name(y_col)

    i = 0
    table_count = 0
    matched_table_count = 0

    all_valid_x_are_numeric = True
    found_valid_point = False

    while i < len(lines):
        line = lines[i]

        heading_match = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", line)
        if heading_match:
            current_heading = normalize_group_label(heading_match.group(1))
            i += 1
            continue

        if not line.strip().startswith("|"):
            i += 1
            continue

        table_lines = []
        while i < len(lines) and lines[i].strip().startswith("|"):
            table_lines.append(lines[i])
            i += 1

        if len(table_lines) < 2:
            continue

        header = split_markdown_row(table_lines[0])
        separator = split_markdown_row(table_lines[1])

        if not header or not is_separator_row(separator):
            continue

        table_count += 1

        normalized_header = [normalize_col_name(cell) for cell in header]

        if y_key not in normalized_header:
            continue

        x_idx = 0
        y_idx = normalized_header.index(y_key)

        if x_col_name is None:
            x_col_name = header[x_idx]

        matched_table_count += 1

        group_name = make_unique_name(
            current_heading or f"Table {table_count}",
            used_group_names,
        )

        numeric_points: list[tuple[float, float]] = []
        bar_values: OrderedDict[str, float] = OrderedDict()

        for row_line in table_lines[2:]:
            cells = split_markdown_row(row_line)

            if max(x_idx, y_idx) >= len(cells):
                continue

            x_raw = clean_cell_text(cells[x_idx])
            if not x_raw:
                continue

            try:
                y = parse_number(cells[y_idx])
            except ValueError:
                continue

            found_valid_point = True

            # 문자열 X축이어도 bar chart에서 쓸 수 있도록 항상 저장합니다.
            bar_values[x_raw] = y

            # 숫자 X축인지도 별도로 검사합니다.
            try:
                x = parse_x_number(x_raw)
                numeric_points.append((x, y))
            except ValueError:
                all_valid_x_are_numeric = False

        if numeric_points:
            line_series_data[group_name] = numeric_points

        if bar_values:
            bar_group_data[group_name] = bar_values

    if x_col_name is None:
        x_col_name = "First Column"

    if not found_valid_point:
        chart_mode: ChartMode = "bar"
    elif all_valid_x_are_numeric:
        chart_mode = "line"
    else:
        chart_mode = "bar"

    print(f"Found markdown tables: {table_count}")
    print(f"Matched tables with '{y_col}': {matched_table_count}")
    print(f"Chart mode: {chart_mode}")

    if chart_mode == "line":
        print(f"Plotted series: {len(line_series_data)}")
    else:
        print(f"Plotted groups: {len(bar_group_data)}")

    return ExtractedData(
        chart_mode=chart_mode,
        x_col_name=x_col_name,
        line_series_data=line_series_data,
        bar_group_data=bar_group_data,
        table_count=table_count,
        matched_table_count=matched_table_count,
    )


def apply_ai_paper_style() -> None:
    """
    AI 논문 figure에 자주 쓰이는 Matplotlib 스타일 설정.

    특징:
    - serif font
    - PDF/SVG 저장 시 텍스트 편집 가능
    - 적당한 linewidth와 tick size
    - 과하지 않은 grid
    """
    plt.rcParams.update(
        {
            # Figure and save quality
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,

            # Font
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Times",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "stix",

            # Editable text in vector outputs
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",

            # Axes
            "axes.labelsize": 13,
            "axes.titlesize": 13,
            "axes.linewidth": 1.0,

            # Ticks
            "xtick.labelsize": 10,
            "ytick.labelsize": 11,
            "xtick.direction": "out",
            "ytick.direction": "out",

            # Legend
            "legend.fontsize": 10,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "legend.fancybox": False,
            "legend.edgecolor": "0.85",

            # Lines
            "lines.linewidth": 2.2,
            "lines.markersize": 6,
        }
    )


def wrap_tick_label(text: str, width: int = 26) -> str:
    """
    x tick label을 적당한 길이로 줄바꿈합니다.

    이미 들어 있는 실제 줄바꿈은 보존하고,
    literal '\\n' 문자열도 실제 줄바꿈으로 변환합니다.
    """
    text = normalize_group_label(text)

    wrapped_lines = []
    for line in text.splitlines():
        if not line.strip():
            wrapped_lines.append("")
            continue

        wrapped_lines.extend(
            textwrap.wrap(
                line,
                width=width,
                break_long_words=False,
                break_on_hyphens=False,
            )
        )

    return "\n".join(wrapped_lines)


def style_axes_common(ax) -> None:
    # 논문형 plot: top/right spine 제거
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

    ax.tick_params(axis="both", which="major", length=4, width=1.0)
    ax.tick_params(axis="both", which="minor", length=2, width=0.8)

    ax.set_axisbelow(True)

    # 과하지 않은 grid
    ax.grid(
        True,
        which="major",
        axis="y",
        linestyle="--",
        linewidth=0.7,
        alpha=0.35,
    )


def plot_line_series(
    series_data: OrderedDict[str, list[tuple[float, float]]],
    output_path: Path,
    x_label: str,
    y_label: str,
    title: str | None = None,
) -> None:
    if not series_data:
        raise RuntimeError("Line chart로 플롯할 데이터가 없습니다.")

    apply_ai_paper_style()

    num_series = len(series_data)

    if num_series <= 5:
        fig_width = 6.4
        fig_height = 4.2
    else:
        fig_width = 7.4
        fig_height = 4.8

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    for idx, (series_name, points) in enumerate(series_data.items()):
        points = sorted(points, key=lambda pair: pair[0])
        xs = [x for x, _ in points]
        ys = [y for _, y in points]

        color = AI_PAPER_PALETTE[idx % len(AI_PAPER_PALETTE)]
        marker = AI_PAPER_MARKERS[idx % len(AI_PAPER_MARKERS)]
        linestyle = AI_PAPER_LINESTYLES[
            (idx // len(AI_PAPER_MARKERS)) % len(AI_PAPER_LINESTYLES)
        ]

        ax.plot(
            xs,
            ys,
            label=series_name,
            color=color,
            linestyle=linestyle,
            marker=marker,
            linewidth=2.2,
            markersize=6,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.4,
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if title:
        ax.set_title(title, pad=8)

    style_axes_common(ax)

    ax.minorticks_on()
    ax.margins(x=0.03, y=0.08)

    if num_series > 5:
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
            handlelength=2.6,
        )
    else:
        ax.legend(
            loc="best",
            handlelength=2.6,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def collect_bar_categories(
    group_data: OrderedDict[str, OrderedDict[str, float]],
) -> list[str]:
    categories = []
    seen = set()

    for values in group_data.values():
        for category in values.keys():
            if category not in seen:
                seen.add(category)
                categories.append(category)

    return categories


def plot_grouped_bar(
    group_data: OrderedDict[str, OrderedDict[str, float]],
    output_path: Path,
    category_label: str,
    y_label: str,
    title: str | None = None,
) -> None:
    if not group_data:
        raise RuntimeError("Bar chart로 플롯할 데이터가 없습니다.")

    apply_ai_paper_style()

    group_names = list(group_data.keys())
    categories = collect_bar_categories(group_data)

    num_groups = len(group_names)
    num_categories = len(categories)

    fig_width = max(6.4, min(14.0, 1.35 * num_groups + 0.65 * num_categories + 2.0))
    fig_height = 4.8 if num_groups <= 6 else 5.4

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    x_positions = list(range(num_groups))
    total_width = 0.82
    bar_width = total_width / max(1, num_categories)

    for cat_idx, category in enumerate(categories):
        offset = (cat_idx - (num_categories - 1) / 2) * bar_width

        xs = [x + offset for x in x_positions]
        ys = [
            group_data[group_name].get(category, math.nan)
            for group_name in group_names
        ]

        color = AI_PAPER_PALETTE[cat_idx % len(AI_PAPER_PALETTE)]

        ax.bar(
            xs,
            ys,
            width=bar_width * 0.92,
            label=category,
            color=color,
            edgecolor="black",
            linewidth=0.6,
        )

    # 요청사항:
    # bar chart에서는 X축 label을 출력하지 않습니다.
    # 즉, ax.set_xlabel("Markdown Section")을 호출하지 않습니다.
    ax.set_ylabel(y_label)

    if title:
        ax.set_title(title, pad=8)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [wrap_tick_label(name, width=28) for name in group_names],
        rotation=0,
        ha="center",
    )

    style_axes_common(ax)

    ax.margins(x=0.04, y=0.10)

    if num_categories > 5 or num_groups > 5:
        ax.legend(
            title=category_label,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
            handlelength=1.8,
        )
    else:
        ax.legend(
            title=category_label,
            loc="best",
            handlelength=1.8,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_extracted_data(
    extracted: ExtractedData,
    output_path: Path,
    y_label: str,
    title: str | None = None,
) -> None:
    if extracted.chart_mode == "line":
        plot_line_series(
            series_data=extracted.line_series_data,
            output_path=output_path,
            x_label=extracted.x_col_name,
            y_label=y_label,
            title=title,
        )
    else:
        plot_grouped_bar(
            group_data=extracted.bar_group_data,
            output_path=output_path,
            category_label=extracted.x_col_name,
            y_label=y_label,
            title=title,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Markdown 파일 안의 여러 표에서 특정 컬럼을 추출해 "
            "첫 번째 컬럼이 숫자면 line chart, 문자열이면 grouped bar chart로 출력합니다."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help="입력 Markdown 파일 경로",
    )
    parser.add_argument(
        "--y-col",
        default=DEFAULT_Y_COL,
        help=f"Y축으로 사용할 컬럼명. 기본값: {DEFAULT_Y_COL!r}",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "출력 파일 경로. "
            "생략하면 입력 파일명과 같은 이름의 .png 파일로 저장합니다. "
            "논문용으로는 .pdf 또는 .svg 권장."
        ),
    )
    parser.add_argument(
        "--title",
        default=None,
        help=(
            "그래프 제목. "
            "논문 figure에서는 보통 caption을 사용하므로 기본값은 제목 없음."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = args.input
    output_path = args.output or input_path.with_suffix(".png")

    md_text = input_path.read_text(encoding="utf-8")

    extracted = extract_tables(
        md_text=md_text,
        y_col=args.y_col,
    )

    if extracted.matched_table_count == 0:
        raise SystemExit(
            f"'{args.y_col}' 컬럼을 가진 Markdown 표를 찾지 못했습니다."
        )

    if not extracted.has_data:
        raise SystemExit(
            f"'{args.y_col}' 컬럼에서 플롯할 수 있는 숫자 데이터를 찾지 못했습니다."
        )

    plot_extracted_data(
        extracted=extracted,
        output_path=output_path,
        y_label=args.y_col,
        title=args.title,
    )

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
