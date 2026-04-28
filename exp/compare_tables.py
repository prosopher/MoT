#!/usr/bin/env python3

import sys
import re
import argparse
from pathlib import Path
from collections import OrderedDict

import matplotlib.pyplot as plt


DEFAULT_Y_COL = "Gen F1 Avg"


# Colorblind-friendly palette, commonly used in academic plots
AI_PAPER_PALETTE = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]


AI_PAPER_MARKERS = [
    "o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">"
]


AI_PAPER_LINESTYLES = [
    "-", "--", "-.", ":"
]


def normalize_col_name(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().strip("`")).lower()


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


def parse_number(value: str) -> float:
    text = value.strip().replace(",", "")
    match = re.search(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
        text,
    )

    if not match:
        raise ValueError(f"숫자를 찾을 수 없습니다: {value!r}")

    return float(match.group(0))


def make_unique_name(name: str, used: set[str]) -> str:
    base = name or "Table"
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
) -> tuple[str, OrderedDict[str, list[tuple[float, float]]]]:
    lines = md_text.splitlines()

    current_heading = None
    used_series_names = set()
    series_data = OrderedDict()

    x_col_name = None
    y_key = normalize_col_name(y_col)

    i = 0
    table_count = 0
    matched_table_count = 0

    while i < len(lines):
        line = lines[i]

        heading_match = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", line)
        if heading_match:
            current_heading = heading_match.group(1).strip()
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

        series_name = make_unique_name(
            current_heading or f"Table {table_count}",
            used_series_names,
        )

        points = []

        for row_line in table_lines[2:]:
            cells = split_markdown_row(row_line)

            if max(x_idx, y_idx) >= len(cells):
                continue

            try:
                x = parse_number(cells[x_idx])
                y = parse_number(cells[y_idx])
            except ValueError:
                continue

            points.append((x, y))

        if points:
            series_data[series_name] = points

    if x_col_name is None:
        x_col_name = "First Column"

    print(f"Found markdown tables: {table_count}")
    print(f"Matched tables with '{y_col}': {matched_table_count}")
    print(f"Plotted series: {len(series_data)}")

    return x_col_name, series_data


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
            "xtick.labelsize": 11,
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


def plot_series(
    series_data: OrderedDict[str, list[tuple[float, float]]],
    output_path: Path,
    x_label: str,
    y_label: str,
    title: str | None = None,
) -> None:
    if not series_data:
        raise RuntimeError("플롯할 데이터가 없습니다.")

    apply_ai_paper_style()

    num_series = len(series_data)

    # AI conference paper에서 1-column figure로 쓰기 좋은 비율
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

    # 논문 figure는 보통 title 대신 caption을 사용하므로 기본값은 None
    if title:
        ax.set_title(title, pad=8)

    # 논문형 plot: top/right spine 제거
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

    ax.tick_params(axis="both", which="major", length=4, width=1.0)
    ax.tick_params(axis="both", which="minor", length=2, width=0.8)

    ax.minorticks_on()
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

    # 약간의 여백
    ax.margins(x=0.03, y=0.08)

    # Series가 많으면 legend를 plot 바깥으로 배치
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Markdown 파일 안의 여러 표에서 특정 컬럼을 추출해 "
            "AI 논문 스타일의 그래프로 출력합니다."
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

    x_label, series_data = extract_tables(
        md_text=md_text,
        y_col=args.y_col,
    )

    if not series_data:
        raise SystemExit(
            f"첫 번째 컬럼과 '{args.y_col}' 컬럼을 가진 Markdown 표를 찾지 못했습니다."
        )

    plot_series(
        series_data=series_data,
        output_path=output_path,
        x_label=x_label,
        y_label=args.y_col,
        title=args.title,
    )

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
