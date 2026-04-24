#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt


X_COL = "Number of Bottom Layers with Full Attention"
Y_COL = "Gen F1 Avg"

SECTIONS = {
    "Homogeneous: gpt2→gpt2": "Homogeneous: gpt2→gpt2",
    "Heterogeneous: gpt2→gpt2-medium": "Heterogeneous: gpt2→gpt2-medium",
}


def split_md_row(line):
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def is_separator_row(cells):
    return all(
        cell.replace(":", "").replace("-", "").strip() == ""
        for cell in cells
    )


def extract_table_for_section(md_text, section_title):
    lines = md_text.splitlines()

    in_section = False
    table_lines = []

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("## "):
            current_title = stripped[3:].strip()
            in_section = current_title == section_title
            continue

        if in_section:
            if stripped.startswith("|"):
                table_lines.append(stripped)
            elif table_lines:
                break

    if not table_lines:
        raise ValueError(f"Markdown table not found for section: {section_title}")

    header = split_md_row(table_lines[0])

    if X_COL not in header:
        raise ValueError(f"Column not found: {X_COL}")
    if Y_COL not in header:
        raise ValueError(f"Column not found: {Y_COL}")

    x_idx = header.index(X_COL)
    y_idx = header.index(Y_COL)

    values = {}

    for line in table_lines[1:]:
        cells = split_md_row(line)

        if is_separator_row(cells):
            continue

        x = int(cells[x_idx])
        y = float(cells[y_idx])

        values[x] = y

    return values


def main():
    parser = argparse.ArgumentParser(
        description="Extract Gen F1 Avg from markdown tables and plot a line chart."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to markdown file, e.g. path_to_md/summary.md",
    )

    args = parser.parse_args()

    input_path = args.input_path
    output_path = input_path.with_suffix(".png")

    md_text = input_path.read_text(encoding="utf-8")

    extracted = {}

    for section_title, output_col in SECTIONS.items():
        extracted[output_col] = extract_table_for_section(md_text, section_title)

    all_x_values = sorted(
        set().union(*(section_values.keys() for section_values in extracted.values()))
    )

    print(
        f"{X_COL}\t"
        f"Homogeneous: gpt2→gpt2\t"
        f"Heterogeneous: gpt2→gpt2-medium"
    )

    for x in all_x_values:
        homogeneous = extracted["Homogeneous: gpt2→gpt2"].get(x)
        heterogeneous = extracted["Heterogeneous: gpt2→gpt2-medium"].get(x)

        print(f"{x}\t{homogeneous}\t{heterogeneous}")

    plt.figure(figsize=(9, 5))

    for label, section_values in extracted.items():
        xs = sorted(section_values.keys())
        ys = [section_values[x] for x in xs]

        plt.plot(
            xs,
            ys,
            marker="o",
            markersize=4,
            linewidth=1.8,
            label=label,
        )

    plt.xlabel(X_COL)
    plt.ylabel("Gen F1 Avg")
    plt.title("Gen F1 Avg by Number of Bottom Layers with Full Attention")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"\nSaved plot to: {output_path}")


if __name__ == "__main__":
    main()
