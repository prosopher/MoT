#!/usr/bin/env python3
"""
Create Figure-8-style subcategory circle (radar) charts from JSON files.

Expected JSON structure:
{
  "algorithm": "c2c",
  "edges": {
    "A_to_B": {
      "subject_category_accuracy": {
        "math": {"accuracy": 0.176, "native_accuracy": 0.176, ...},
        ...
      }
    }
  }
}

Usage:
  python exp/subcategory_circle_chart.py --input-dir inputs --output-dir outputs
  python exp/subcategory_circle_chart.py --input-dir inputs --edge-id A_to_B
  python exp/subcategory_circle_chart.py --input-dir inputs --show
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

# Paper Figure 8 category order on MMLU-Redux.
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


@dataclass
class Series:
    algorithm: str
    filename: str
    edge_id: str
    category_to_accuracy: Dict[str, float]

    @property
    def legend_name(self) -> str:
        return self.algorithm


@dataclass
class LoadResult:
    series_by_edge: Dict[str, List[Series]]
    native_upperbound_by_edge: Dict[str, Dict[str, float]]
    baseline_source_file_by_edge: Dict[str, str]
    all_categories: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Figure-8-style subcategory radar charts from JSON files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("inputs"),
        help="Directory containing input JSON files. Default: ./inputs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory to save output PNG files. Default: ./outputs",
    )
    parser.add_argument(
        "--pattern",
        default="*.json",
        help="Glob pattern for input files. Default: *.json",
    )
    parser.add_argument(
        "--edge-id",
        default=None,
        help="Only draw the specified edge_id. By default, draw every discovered edge_id.",
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
        help="Title prefix for each chart.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Saved figure DPI. Default: 220",
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
        "--disable-upperbound",
        action="store_true",
        help=(
            "Do not draw the black 'upperbound' line derived from the first input file's "
            "native_accuracy values."
        ),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_algorithm_name(name: Optional[str], fallback: str) -> str:
    name = (name or "").strip()
    return name if name else fallback


def category_order(categories: Sequence[str]) -> List[str]:
    categories = list(categories)
    category_set = set(categories)

    ordered = [c for c in DEFAULT_CATEGORY_ORDER if c in category_set]
    leftover = sorted(category_set - set(ordered))
    return ordered + leftover


def extract_native_upperbound(data: dict) -> Dict[str, Dict[str, float]]:
    native_upperbound_by_edge: Dict[str, Dict[str, float]] = {}
    edges = data.get("edges", {})
    if not isinstance(edges, dict):
        return native_upperbound_by_edge

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
            native_upperbound_by_edge[str(edge_id)] = category_to_native

    return native_upperbound_by_edge


def load_series(input_dir: Path, pattern: str = "*.json") -> LoadResult:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    json_files = sorted(input_dir.glob(pattern))
    if not json_files:
        raise FileNotFoundError(
            f"No JSON files found in {input_dir} with pattern {pattern!r}"
        )

    native_upperbound_by_edge: Dict[str, Dict[str, float]] = {}
    baseline_source_file_by_edge: Dict[str, str] = {}

    series_by_edge: Dict[str, List[Series]] = {}
    category_set = set()

    for json_file in json_files:
        data = read_json(json_file)

        # For each edge_id, use the first file (in sorted input order) that actually
        # contains that edge as the source of the black 'upperbound' line.
        native_candidates = extract_native_upperbound(data)
        for edge_id, category_to_native in native_candidates.items():
            if edge_id not in native_upperbound_by_edge:
                native_upperbound_by_edge[edge_id] = category_to_native
                baseline_source_file_by_edge[edge_id] = json_file.name
        algorithm = normalize_algorithm_name(data.get("algorithm"), json_file.stem)
        edges = data.get("edges", {})
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

            category_set.update(category_to_accuracy.keys())
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

    ordered_categories = category_order(category_set)
    return LoadResult(
        series_by_edge=series_by_edge,
        native_upperbound_by_edge=native_upperbound_by_edge,
        baseline_source_file_by_edge=baseline_source_file_by_edge,
        all_categories=ordered_categories,
    )


def dedupe_legend_names(series_list: List[Series]) -> List[str]:
    counts: Dict[str, int] = {}
    labels: List[str] = []
    for s in series_list:
        counts[s.algorithm] = counts.get(s.algorithm, 0) + 1

    seen: Dict[str, int] = {}
    for s in series_list:
        if counts[s.algorithm] == 1:
            labels.append(s.algorithm)
        else:
            seen[s.algorithm] = seen.get(s.algorithm, 0) + 1
            labels.append(f"{s.algorithm} ({s.filename})")
    return labels


def compute_radius_max(
    series_list: List[Series],
    categories: Sequence[str],
    fixed: Optional[float],
    native_upperbound: Optional[Dict[str, float]] = None,
) -> float:
    if fixed is not None:
        return fixed

    values = []
    for s in series_list:
        for c in categories:
            if c in s.category_to_accuracy:
                values.append(s.category_to_accuracy[c])

    if native_upperbound:
        for c in categories:
            if c in native_upperbound:
                values.append(native_upperbound[c])

    if not values:
        return 1.0

    max_value = max(values)
    # Choose a visually tidy upper bound, similar to the paper's variable scales.
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
    native_upperbound: Optional[Dict[str, float]] = None,
) -> None:
    if not series_list:
        raise ValueError(f"No series to plot for edge_id={edge_id!r}")

    n = len(categories)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    radius_max = compute_radius_max(
        series_list, categories, max_radius, native_upperbound=native_upperbound
    )
    legend_labels = dedupe_legend_names(series_list)

    fig = plt.figure(figsize=(9, 9))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)

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
    ax.set_yticklabels([f"{t:.1f}" for t in rticks], fontsize=9)
    ax.set_ylim(0, radius_max)
    ax.grid(True, alpha=0.35)

    if native_upperbound:
        upperbound_values = [native_upperbound.get(cat, np.nan) for cat in categories]
        upperbound_values += upperbound_values[:1]
        ax.plot(
            angles,
            upperbound_values,
            color="black",
            linewidth=2.4,
            linestyle="-",
            label="upperbound",
            zorder=10,
        )

    for series, label in zip(series_list, legend_labels):
        values = [series.category_to_accuracy.get(cat, np.nan) for cat in categories]
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=label)
        ax.fill(angles, values, alpha=0.08)

    ax.set_title(f"{title_prefix} ({edge_id})", pad=28, fontsize=14)
    ax.legend(loc="upper left", bbox_to_anchor=(1.08, 1.10), frameon=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = parse_args()
    result = load_series(args.input_dir, pattern=args.pattern)

    series_by_edge = result.series_by_edge
    if args.edge_id is not None:
        if args.edge_id not in series_by_edge:
            available = ", ".join(sorted(series_by_edge))
            raise KeyError(
                f"edge_id {args.edge_id!r} not found. Available edge_ids: {available}"
            )
        target_edges = [args.edge_id]
    else:
        target_edges = sorted(series_by_edge)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.disable_upperbound:
        print(
            "[info] upperbound is taken per edge_id from the first input file "
            "(sorted order) that actually contains that edge."
        )

    generated = []
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
        output_path = args.output_dir / f"subcategory_circle_{edge_id}.png"
        native_upperbound = None
        upperbound_source = None
        if not args.disable_upperbound:
            native_upperbound = result.native_upperbound_by_edge.get(edge_id)
            upperbound_source = result.baseline_source_file_by_edge.get(edge_id)

        plot_edge_radar(
            edge_id=edge_id,
            series_list=series_list,
            categories=categories,
            output_path=output_path,
            title_prefix=args.title_prefix,
            dpi=args.dpi,
            max_radius=args.max_radius,
            show=args.show,
            native_upperbound=native_upperbound,
        )
        generated.append(output_path)
        if upperbound_source:
            print(f"[saved] {output_path} (upperbound source: {upperbound_source})")
        else:
            print(f"[saved] {output_path}")

    if not generated:
        raise RuntimeError("No charts were generated.")


if __name__ == "__main__":
    main()
