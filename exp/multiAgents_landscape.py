#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp.exp_util import (
    apply_ai_paper_style,
    require_matplotlib_pyplot,
    save_paper_figure,
    double_column_figsize,
    style_axes_common,
    AI_PAPER_MARKER_SIZE,
    AI_PAPER_MARKER_EDGE_WIDTH,
)

DEFAULT_OUTPUT_PATH = Path("results/performance_landscape.pdf")
DEFAULT_METRICS_ROOT = Path("outputs/multi_agents")
DEFAULT_AGENT_COUNT = 10

# ---------------------------------------------------------------------------
# Metrics loading
# ---------------------------------------------------------------------------

METHOD_SPECS = [
    {
        "algorithm": "kvcomm",
        "cache_mode": "retain",
        "name": "KVComm",
    },
    {
        "algorithm": "c2c-pr",
        "cache_mode": "retain",
        "name": "C2C-Projection",
    },
    {
        "algorithm": "lsc",
        "cache_mode": "retain",
        "name": "LSC",
    },
    {
        "algorithm": "interlat",
        "cache_mode": "retain",
        "name": "Interlat",
    },
    {
        "algorithm": "mot",
        "cache_mode": "retain",
        "name": "MoT (Retain)",
    },
    {
        "algorithm": "mot",
        "cache_mode": "free",
        "name": "MoT (Free)",
    },
]


def _folder_candidates(
    algorithm: str,
    cache_mode: str,
    agent_count: int,
) -> list[str]:
    raw = f"{algorithm}_{cache_mode}_{agent_count}"
    safe = raw.replace("-", "_")

    candidates: list[str] = []
    for folder in [safe, raw]:
        if folder not in candidates:
            candidates.append(folder)
    return candidates


def find_metrics_path(
    *,
    metrics_root: Path,
    algorithm: str,
    cache_mode: str,
    agent_count: int,
) -> Path:
    for folder in _folder_candidates(algorithm, cache_mode, agent_count):
        path = metrics_root / folder / "agent_runner_metrics.json"
        if path.exists():
            return path

    expected = (
        metrics_root
        / _folder_candidates(algorithm, cache_mode, agent_count)[0]
        / "agent_runner_metrics.json"
    )
    raise FileNotFoundError(
        f"Missing metrics file for {algorithm} ({cache_mode}, agents={agent_count}): "
        f"{expected}"
    )


def _normalize_algorithm_name(name: str | None) -> str:
    if name is None:
        return ""
    return name.strip().lower().replace("_", "-")


def _read_metric_value(payload: dict, key: str, path: Path) -> float:
    if key not in payload:
        raise KeyError(f"{path} does not contain required key: {key}")
    value = float(payload[key])
    if not math.isfinite(value):
        raise ValueError(f"{path} has non-finite {key}: {value}")
    return value


def load_one_method_metrics(
    *,
    metrics_root: Path,
    algorithm: str,
    cache_mode: str,
    display_name: str,
    expected_agent_count: int,
) -> dict:
    path = find_metrics_path(
        metrics_root=metrics_root,
        algorithm=algorithm,
        cache_mode=cache_mode,
        agent_count=expected_agent_count,
    )

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    json_algorithm = _normalize_algorithm_name(payload.get("algorithm"))
    expected_algorithm = _normalize_algorithm_name(algorithm)
    if json_algorithm and json_algorithm != expected_algorithm:
        raise ValueError(
            f"{path} has algorithm={payload.get('algorithm')!r}, "
            f"but expected {algorithm!r}."
        )

    json_cache_mode = str(payload.get("cache_mode", "")).strip().lower()
    if json_cache_mode and json_cache_mode != cache_mode:
        raise ValueError(
            f"{path} has cache_mode={payload.get('cache_mode')!r}, "
            f"but expected {cache_mode!r}."
        )

    agent_count = payload.get("agent_count", payload.get("requested_agent_count"))
    if agent_count is not None:
        agent_count = int(agent_count)

    if expected_agent_count is not None and agent_count != expected_agent_count:
        raise ValueError(
            f"{path} has agent_count={agent_count}, "
            f"but --agent-count={expected_agent_count} was requested."
        )

    gpu_peak_memory = payload.get("gpu_peak_memory_gib")
    if not isinstance(gpu_peak_memory, dict):
        raise KeyError(f"Missing 'gpu_peak_memory_gib' breakdown in {path}")
    memory_components = []
    for key in ("model_gib", "translator_gib", "kv_gib"):
        value = gpu_peak_memory.get(key)
        if value is None:
            raise KeyError(f"Missing 'gpu_peak_memory_gib.{key}' in {path}")
        memory_components.append(float(value))
    gpu_peak_memory_gib = sum(memory_components)

    accuracy = _read_metric_value(payload, "accuracy", path)

    if not math.isfinite(gpu_peak_memory_gib):
        raise ValueError(
            f"{path} has non-finite gpu_peak_memory_gib: {gpu_peak_memory_gib}"
        )

    return {
        "name": display_name,
        "algorithm": algorithm,
        "cache_mode": cache_mode,
        "agent_count": agent_count,
        "accuracy": accuracy,
        "gpu_peak_memory_gib": gpu_peak_memory_gib,
        "metrics_path": str(path),
    }


def load_performance_data(
    *,
    metrics_root: Path,
    expected_agent_count: int,
    allow_missing: bool,
) -> list[dict]:
    data: list[dict] = []
    missing_errors: list[str] = []

    for spec in METHOD_SPECS:
        try:
            item = load_one_method_metrics(
                metrics_root=metrics_root,
                algorithm=spec["algorithm"],
                cache_mode=spec["cache_mode"],
                display_name=spec["name"],
                expected_agent_count=expected_agent_count,
            )
        except FileNotFoundError as exc:
            if allow_missing:
                print(f"Warning: {exc}", file=sys.stderr)
                continue
            missing_errors.append(str(exc))
        else:
            data.append(item)

    if missing_errors:
        joined = "\n".join(f"  - {msg}" for msg in missing_errors)
        raise FileNotFoundError(
            "Some required metrics files are missing:\n"
            f"{joined}\n"
            "Use --allow-missing to plot only available methods."
        )

    if not data:
        raise RuntimeError(f"No metrics were loaded from {metrics_root}")

    agent_counts = sorted(
        {
            item["agent_count"]
            for item in data
            if item.get("agent_count") is not None
        }
    )
    if expected_agent_count is None and len(agent_counts) > 1:
        raise ValueError(
            "Loaded metrics contain mixed agent_count values: "
            f"{agent_counts}. Re-run with --agent-count <N> or make sure the "
            "metrics files under outputs/multi_agents are from the same setting."
        )

    return data


# ---------------------------------------------------------------------------
# Figure style
# ---------------------------------------------------------------------------


def get_color(name: str) -> str:
    if name.lower() == "mot (free)":
        return "#C0504D"
    return "#C7CDD6"


def get_edge_color(name: str) -> str:
    if name.lower() == "mot (free)":
        return "#8E2F2D"
    return "#6B7280"


def get_text_color(name: str) -> str:
    if name.lower() == "mot (free)":
        return "#C0504D"
    return "black"


def get_label_style(name: str):
    name_lower = name.lower()

    if name_lower in ["c2c-pr", "c2c-projection"]:
        return {
            "label": "C2C-Projection",
            "xytext": (0, -16),
            "ha": "center",
            "va": "top",
            "arrowprops": {
                "arrowstyle": "-",
                "color": "#bcbcbc",
                "lw": 1.2,
                "shrinkA": 0,
                "shrinkB": 6,
            },
        }

    if name_lower == "interlat":
        return {
            "label": "Interlat",
            "xytext": (-16, 0),
            "ha": "right",
            "va": "center",
            "arrowprops": {
                "arrowstyle": "-",
                "color": "#bcbcbc",
                "lw": 1.2,
                "shrinkA": 0,
                "shrinkB": 2,
            },
        }

    if name_lower == "kvcomm":
        return {
            "label": name,
            "xytext": (10, 0),
            "ha": "left",
            "va": "center",
            "arrowprops": None,
        }

    if name_lower == "mot (retain)":
        return {
            "label": name,
            "xytext": (-22, 4),
            "ha": "center",
            "va": "bottom",
            "arrowprops": None,
        }

    if name_lower == "mot (free)":
        return {
            "label": name,
            "xytext": (14, 4),
            "ha": "center",
            "va": "bottom",
            "arrowprops": None,
        }

    return {
        "label": name,
        "xytext": (0, 4),
        "ha": "center",
        "va": "bottom",
        "arrowprops": None,
    }


def plot_performance_landscape(data: list[dict], output_path: Path) -> None:
    apply_ai_paper_style()
    plt = require_matplotlib_pyplot()

    fig, ax = plt.subplots(figsize=double_column_figsize())

    for item in data:
        name = item["name"]
        x = item["gpu_peak_memory_gib"]
        y = item["accuracy"]

        ax.scatter(
            x,
            y,
            s=AI_PAPER_MARKER_SIZE * 15,
            color=get_color(name),
            edgecolor=get_edge_color(name),
            linewidth=AI_PAPER_MARKER_EDGE_WIDTH,
            zorder=3,
        )

        label_style = get_label_style(name)
        ax.annotate(
            label_style["label"],
            xy=(x, y),
            xytext=label_style["xytext"],
            textcoords="offset points",
            ha=label_style["ha"],
            va=label_style["va"],
            fontsize=20,
            fontweight="semibold",
            color=get_text_color(name),
            arrowprops=label_style["arrowprops"],
            zorder=4,
        )

    style_axes_common(ax)

    ax.set_xlabel("Peak GPU Memory (GiB)", fontsize=20, fontweight="semibold")
    ax.set_ylabel("Accuracy", fontsize=20, fontweight="semibold")

    ax.tick_params(axis="both", labelsize=20)

    x_values = [
        d["gpu_peak_memory_gib"]
        for d in data
        if math.isfinite(d["gpu_peak_memory_gib"])
    ]
    y_values = [
        d["accuracy"]
        for d in data
        if math.isfinite(d["accuracy"])
    ]

    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)

    x_pad = (x_max - x_min) * 0.12 if x_max > x_min else 0.10
    y_pad = (y_max - y_min) * 0.12 if y_max > y_min else 0.02

    x_left = max(0, x_min - x_pad)
    x_right = x_max + x_pad
    y_bottom = max(0, y_min - y_pad)
    y_top = y_max + y_pad

    ax.set_xlim(x_left, x_right)
    ax.set_ylim(y_bottom, y_top)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_paper_figure(fig, output_path)
    plt.close(fig)

    print(f"Saved: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a performance landscape plot from "
            "outputs/multi_agents/{algorithm}_{cache_mode}_{agent_count}/agent_runner_metrics.json."
        )
    )
    parser.add_argument(
        "--metrics-root",
        type=Path,
        default=DEFAULT_METRICS_ROOT,
        help=(
            "Root directory containing {algorithm}_{cache_mode}_{agent_count}/"
            "agent_runner_metrics.json folders."
        ),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to save the plot.",
    )
    parser.add_argument(
        "--agent-count",
        type=int,
        default=DEFAULT_AGENT_COUNT,
        help=(
            "Agent-count suffix used in the metrics folder name. The default "
            "loads outputs/multi_agents/{algorithm}_{cache_mode}_10/"
            "agent_runner_metrics.json and also checks the JSON agent_count."
        ),
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Plot available methods even if some metrics files are missing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data = load_performance_data(
        metrics_root=args.metrics_root,
        expected_agent_count=args.agent_count,
        allow_missing=args.allow_missing,
    )

    print("Loaded metrics:")
    for item in data:
        print(
            f"  - {item['name']}: "
            f"Accuracy={item['accuracy']:.6f}, "
            f"GPU={item['gpu_peak_memory_gib']:.6f} GiB, "
            f"agent_count={item.get('agent_count')}, "
            f"path={item['metrics_path']}"
        )

    plot_performance_landscape(data, args.output_path)


if __name__ == "__main__":
    main()
