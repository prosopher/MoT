#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
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
DEFAULT_METRICS_ROOT = Path("outputs_7a8f1fa")
DEFAULT_AGENT_COUNT = 10

# ---------------------------------------------------------------------------
# Metrics loading
# ---------------------------------------------------------------------------

METHOD_SPECS = [
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


def _successful_turn_tokens(example: dict, *, offset: int) -> tuple[int, int]:
    """Return completion tokens and count for successfully verified turns."""

    messages = example.get("agent_messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"examples[{offset}].agent_messages is missing or empty")

    total = 0
    successful_turn_count = 0
    for message_offset, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(
                f"examples[{offset}].agent_messages[{message_offset}] is not an object"
            )
        if (
            message.get("verification_passed") is not True
            or message.get("agent_failed") is not False
        ):
            continue

        value = message.get("tokens_completion")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"invalid examples[{offset}].agent_messages[{message_offset}]"
                f".tokens_completion: {value!r}"
            )
        total += value
        successful_turn_count += 1

    return total, successful_turn_count


def _token_normalized_example_kv_mean(
    examples: list,
    *,
    kv_cache_mean_gib: float,
    path: Path,
) -> tuple[float, int, int]:
    """Average mean KV GiB normalized by each example's generated tokens."""

    normalized_kv_memory: list[float] = []
    successful_turn_count = 0
    for offset, example in enumerate(examples):
        if not isinstance(example, dict):
            raise ValueError(f"examples[{offset}] is not an object")
        generated_tokens, example_successful_turns = _successful_turn_tokens(
            example, offset=offset
        )
        if generated_tokens <= 0 or example_successful_turns <= 0:
            warnings.warn(
                f"Skipping examples[{offset}] in {path}: no successful turn"
            )
            continue
        successful_turn_count += example_successful_turns
        normalized_kv_memory.append(
            kv_cache_mean_gib * 1000.0 / generated_tokens
        )

    if not normalized_kv_memory:
        raise ValueError("run has no examples containing a successful turn")

    return (
        math.fsum(normalized_kv_memory) / len(normalized_kv_memory),
        len(normalized_kv_memory),
        successful_turn_count,
    )


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

    kv_cache_stats = payload.get("kv_cache_memory_stats_gib")
    if not isinstance(kv_cache_stats, dict):
        raise KeyError(f"Missing 'kv_cache_memory_stats_gib' in {path}")
    kv_cache_mean_gib = _read_metric_value(kv_cache_stats, "mean", path)
    if kv_cache_mean_gib < 0:
        raise ValueError(
            f"{path} has negative kv_cache_memory_stats_gib.mean: "
            f"{kv_cache_mean_gib}"
        )

    count = payload.get("count")
    examples = payload.get("examples")
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count <= 0
        or not isinstance(examples, list)
        or len(examples) != count
    ):
        raise ValueError(
            f"Run is incomplete in {path}: count={count}, examples="
            f"{len(examples) if isinstance(examples, list) else 'invalid'}"
        )
    (
        kv_memory_gib_per_1k_tokens,
        example_count,
        successful_turn_count,
    ) = _token_normalized_example_kv_mean(
        examples,
        kv_cache_mean_gib=kv_cache_mean_gib,
        path=path,
    )

    accuracy = _read_metric_value(payload, "accuracy", path)

    return {
        "name": display_name,
        "algorithm": algorithm,
        "cache_mode": cache_mode,
        "agent_count": agent_count,
        "accuracy": accuracy,
        "kv_memory_gib_per_1k_tokens": kv_memory_gib_per_1k_tokens,
        "example_count": example_count,
        "successful_turn_count": successful_turn_count,
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
            f"metrics files under {metrics_root} are from the same setting."
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
            "xytext": (-34, 20),
            "ha": "right",
            "va": "bottom",
            "arrowprops": {
                "arrowstyle": "-",
                "color": "#bcbcbc",
                "lw": 1.2,
                "shrinkA": 0,
                "shrinkB": 6,
            },
        }

    if name_lower == "lsc":
        return {
            "label": "LSC",
            "xytext": (0, 20),
            "ha": "center",
            "va": "bottom",
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
            "xytext": (0, -12),
            "ha": "center",
            "va": "top",
            "arrowprops": None,
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
        x = item["kv_memory_gib_per_1k_tokens"]
        y = item["accuracy"] * 100.0

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

    ax.set_xlabel("KV Memory (GiB / 1k tokens)", fontsize=20, fontweight="semibold")
    ax.set_ylabel("Accuracy (%)", fontsize=20, fontweight="semibold")

    ax.tick_params(axis="both", labelsize=20)

    x_values = [
        d["kv_memory_gib_per_1k_tokens"]
        for d in data
        if math.isfinite(d["kv_memory_gib_per_1k_tokens"])
    ]
    y_values = [
        d["accuracy"] * 100.0
        for d in data
        if math.isfinite(d["accuracy"])
    ]

    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)

    x_pad = (x_max - x_min) * 0.12 if x_max > x_min else 0.10
    y_pad = (y_max - y_min) * 0.17 if y_max > y_min else 0.02

    x_left = max(0, x_min - x_pad)
    x_right = x_max + x_pad
    y_top = y_max + y_pad
    y_bottom = -0.14 * y_top

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
            "outputs_7a8f1fa/{algorithm}_{cache_mode}_{agent_count}/agent_runner_metrics.json."
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
            "loads outputs_7a8f1fa/{algorithm}_{cache_mode}_10/"
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
            f"KV={item['kv_memory_gib_per_1k_tokens']:.6f} GiB / 1k tokens, "
            f"agent_count={item.get('agent_count')}, "
            f"path={item['metrics_path']}"
        )

    plot_performance_landscape(data, args.output_path)


if __name__ == "__main__":
    main()
