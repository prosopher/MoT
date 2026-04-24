#!/usr/bin/env python3

import sys
import re
from pathlib import Path
from collections import OrderedDict

import matplotlib.pyplot as plt


DEFAULT_Y_COL = "Gen F1 Avg"


def normalize_col_name(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().strip("`")).lower()


def split_markdown_row(line: str) -> list[str]:
    s = line.strip()

    if not s.startswith("|"):
        return []

    if s.startswith("|"):
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
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text)

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


def extract_tables(md_text: str, y_col: str) -> tuple[str, OrderedDict[str, list[tuple[float, float]]]]:
    lines = md_text.splitlines()

    current_heading = None
    used_series_names = set()
    series_data = OrderedDict()

    x_col_name = None
    y_key = normalize_col_name(y_col)

    i = 0
    table_count = 0

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

        normalized_header = [normalize_col_name(cell) for cell in header]

        if y_key not in normalized_header:
            continue

        x_idx = 0
        y_idx = normalized_header.index(y_key)

        if x_col_name is None:
            x_col_name = header[x_idx]

        table_count += 1
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

    return x_col_name, series_data


def plot_series(
    series_data: OrderedDict[str, list[tuple[float, float]]],
    output_path: Path,
    x_label: str,
    y_label: str,
) -> None:
    if not series_data:
        raise RuntimeError("플롯할 데이터가 없습니다.")

    plt.figure(figsize=(10, 6))

    for series_name, points in series_data.items():
        points = sorted(points, key=lambda pair: pair[0])
        xs = [x for x, _ in points]
        ys = [y for _, y in points]

        plt.plot(
            xs,
            ys,
            marker="o",
            markersize=5,
            linewidth=1.8,
            label=series_name,
        )

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f"{y_label} by {x_label}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {Path(sys.argv[0]).name} path_to_md/summary.md", file=sys.stderr)
        raise SystemExit(1)

    input_path = Path(sys.argv[1])
    output_path = input_path.with_suffix(".png")

    md_text = input_path.read_text(encoding="utf-8")

    x_label, series_data = extract_tables(
        md_text=md_text,
        y_col=DEFAULT_Y_COL,
    )

    if not series_data:
        raise SystemExit(
            f"첫 번째 컬럼과 '{DEFAULT_Y_COL}' 컬럼을 가진 Markdown 표를 찾지 못했습니다."
        )

    plot_series(
        series_data=series_data,
        output_path=output_path,
        x_label=x_label,
        y_label=DEFAULT_Y_COL,
    )

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
