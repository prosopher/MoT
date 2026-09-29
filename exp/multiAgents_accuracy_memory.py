#!/usr/bin/env python3
"""Plot accuracy and token-normalized KV-cache memory for multi-agent runs."""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from exp_util import (
    AI_PAPER_DOUBLE_COLUMN_TALL_FIGSIZE,
    AI_PAPER_FIGURE_DPI,
    AI_PAPER_LEGEND_EDGE_COLOR,
    AI_PAPER_LINE_WIDTH,
    AI_PAPER_MARKER_EDGE_WIDTH,
    apply_ai_paper_style,
    style_axes_common,
)


RUN_DIR_RE = re.compile(r"^(?P<method>.+)_(?P<agents>\d+)$")
PLOTTED_AGENT_COUNTS = frozenset(range(2, 17, 2))

FIGSIZE = (
    AI_PAPER_DOUBLE_COLUMN_TALL_FIGSIZE[0] * 2.0,
    AI_PAPER_DOUBLE_COLUMN_TALL_FIGSIZE[1] * 3.1,
)
OVERLEAF_AXIS_LABEL_SIZE = 56
OVERLEAF_TICK_LABEL_SIZE = 56
OVERLEAF_LEGEND_FONT_SIZE = 50
OVERLEAF_TICK_LENGTH = 12.0
OVERLEAF_TICK_WIDTH = 3.0
OVERLEAF_LINE_WIDTH = AI_PAPER_LINE_WIDTH * 3.0
OVERLEAF_MARKER_SIZE = 28.0
OVERLEAF_MARKER_EDGE_WIDTH = AI_PAPER_MARKER_EDGE_WIDTH * 3.2

METHOD_LABELS = {
    "mot-free": "MoT (Free)",
    "mot-retain": "MoT (Retain)",
    "c2c-pr-retain": "C2C-Project",
    "lsc-retain": "LSC",
    "interlat-retain": "Interlat",
}

METHOD_ORDER = (
    "mot-free",
    "mot-retain",
    "c2c-pr-retain",
    "lsc-retain",
    "interlat-retain",
)

METHOD_STYLES = {
    "c2c-pr-retain": {
        "color": "#4BACC6",
        "marker": "s",
        "linestyle": "-",
        "linewidth": OVERLEAF_LINE_WIDTH,
        "markersize": OVERLEAF_MARKER_SIZE,
        "markerfacecolor": "none",
        "markeredgecolor": "#4BACC6",
        "zorder": 6,
    },
    "lsc-retain": {
        "color": "#8064A2",
        "marker": "D",
        "linestyle": "-",
        "linewidth": OVERLEAF_LINE_WIDTH,
        "markersize": OVERLEAF_MARKER_SIZE,
        "markerfacecolor": "#8064A2",
        "markeredgecolor": "white",
        "zorder": 4,
    },
    "interlat-retain": {
        "color": "#4F81BD",
        "marker": "P",
        "linestyle": "-",
        "linewidth": OVERLEAF_LINE_WIDTH,
        "markersize": OVERLEAF_MARKER_SIZE,
        "markerfacecolor": "#4F81BD",
        "markeredgecolor": "white",
        "zorder": 3,
    },
    "mot-retain": {
        "color": "#C0504D",
        "marker": "X",
        "linestyle": "--",
        "linewidth": OVERLEAF_LINE_WIDTH,
        "markersize": OVERLEAF_MARKER_SIZE,
        "markerfacecolor": "#C0504D",
        "markeredgecolor": "white",
        "zorder": 2,
    },
    "mot-free": {
        "color": "#C0504D",
        "marker": "o",
        "linestyle": "-",
        "linewidth": OVERLEAF_LINE_WIDTH,
        "markersize": OVERLEAF_MARKER_SIZE,
        "markerfacecolor": "none",
        "markeredgecolor": "#C0504D",
        "zorder": 3,
    },
}

MEMORY_PLOT = {
    "ylabel": "KV Memory (GiB / 1k tokens)",
    "filename": "kv_mean.png",
    "pdf_filename": "multi_agents_memory.pdf",
}

ACCURACY_PLOT = {
    "ylabel": "Accuracy (%)",
    "filename": "accuracy.png",
    "pdf_filename": "multi_agents_accuracy.pdf",
}


def _finite_float(value: object, *, field: str, path: Path) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {field}: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"non-finite {field}: {number}")
    return number


def _successful_turn_tokens(
    example: dict[str, object], *, offset: int
) -> tuple[int, int]:
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
    examples: list[object], *, kv_cache_mean_gib: float, path: Path
) -> tuple[float, int, int]:
    """Average the run's mean KV GiB per each example's 1K generated tokens.

    Only turns with ``verification_passed=True`` and ``agent_failed=False``
    contribute completion tokens. Examples with no successful turn are omitted.
    """

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


def load_memory_runs(
    homogeneity_metrics_root: Path,
    baseline_metrics_root: Path,
) -> dict[str, list[dict[str, float]]]:
    """Load and combine accuracy and token-normalized memory for each run."""

    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)

    mot_methods = frozenset({"mot-free", "mot-retain"})
    baseline_methods = frozenset(
        {"c2c-pr-retain", "lsc-retain", "interlat-retain"}
    )
    sources = (
        (homogeneity_metrics_root, mot_methods, PLOTTED_AGENT_COUNTS),
        (
            homogeneity_metrics_root,
            baseline_methods,
            frozenset({2, 6, 10, 14}),
        ),
        (
            baseline_metrics_root,
            baseline_methods,
            frozenset({4, 8, 12, 16}),
        ),
    )
    paths = (
        (path, allowed_methods, allowed_agent_counts)
        for metrics_root, allowed_methods, allowed_agent_counts in sources
        for path in sorted(metrics_root.glob("*/agent_runner_metrics.json"))
    )
    for path, allowed_methods, allowed_agent_counts in paths:
        match = RUN_DIR_RE.match(path.parent.name)
        if match is None:
            warnings.warn(f"Skipping unrecognized run directory: {path.parent}")
            continue
        if int(match.group("agents")) not in allowed_agent_counts:
            continue

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            algorithm = str(payload["algorithm"]).strip().lower().replace("_", "-")
            cache_mode = str(payload["cache_mode"]).strip().lower().replace("_", "-")
            method = f"{algorithm}-{cache_mode}"
            if method not in allowed_methods:
                continue
            agents = int(payload["agent_count"])
            folder_agents = int(match.group("agents"))
            count = int(payload["count"])
            examples = payload["examples"]
            kv_cache_stats = payload["kv_cache_memory_stats_gib"]
            if not isinstance(kv_cache_stats, dict):
                raise TypeError("kv_cache_memory_stats_gib is not an object")
            kv_cache_mean_gib = _finite_float(
                kv_cache_stats["mean"],
                field="kv_cache_memory_stats_gib.mean",
                path=path,
            )
            if kv_cache_mean_gib < 0:
                raise ValueError(
                    "negative kv_cache_memory_stats_gib.mean: "
                    f"{kv_cache_mean_gib}"
                )
            accuracy = _finite_float(
                payload["accuracy"],
                field="accuracy",
                path=path,
            )
            if not 0.0 <= accuracy <= 1.0:
                raise ValueError(f"accuracy is outside [0, 1]: {accuracy}")

            if agents != folder_agents:
                raise ValueError(
                    f"agent_count={agents} does not match directory ({folder_agents})"
                )
            if (
                count <= 0
                or not isinstance(examples, list)
                or len(examples) != count
            ):
                raise ValueError(
                    f"run is incomplete: count={count}, examples="
                    f"{len(examples) if isinstance(examples, list) else 'invalid'}"
                )
            (
                kv_mean,
                example_count,
                successful_turn_count,
            ) = _token_normalized_example_kv_mean(
                examples,
                kv_cache_mean_gib=kv_cache_mean_gib,
                path=path,
            )

            grouped[method].append(
                {
                    "agents": agents,
                    "accuracy": accuracy,
                    "kv_mean": kv_mean,
                    "example_count": example_count,
                    "successful_turn_count": successful_turn_count,
                }
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            warnings.warn(f"Skipping incomplete/invalid run {path}: {exc}")

    for runs in grouped.values():
        runs.sort(key=lambda run: run["agents"])

    if not grouped:
        raise RuntimeError(
            "No completed runs found under "
            f"{homogeneity_metrics_root} or {baseline_metrics_root}"
        )

    return dict(grouped)


def validate_mot_free_is_minimum(
    grouped: dict[str, list[dict[str, float]]],
) -> None:
    """Fail instead of drawing a graph that violates the expected invariant."""

    mot_free = {
        int(run["agents"]): run["kv_mean"]
        for run in grouped.get("mot-free", [])
    }
    if not mot_free:
        raise RuntimeError("MoT (Free) memory results are missing")

    violations: list[str] = []
    for method, runs in grouped.items():
        if method == "mot-free":
            continue
        for run in runs:
            agents = int(run["agents"])
            free_value = mot_free.get(agents)
            if free_value is not None and run["kv_mean"] < free_value:
                violations.append(
                    f"agents={agents}: {METHOD_LABELS.get(method, method)}="
                    f"{run['kv_mean']:.6f} < MoT (Free)={free_value:.6f}"
                )

    if violations:
        raise RuntimeError(
            "MoT (Free) is not the minimum token-normalized memory:\n  "
            + "\n  ".join(violations)
        )


def apply_large_text_style(ax) -> None:
    """Match the typography used by ``multiAgents_figure.py``."""

    ax.xaxis.label.set_fontsize(OVERLEAF_AXIS_LABEL_SIZE)
    ax.yaxis.label.set_fontsize(OVERLEAF_AXIS_LABEL_SIZE)
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")
    ax.tick_params(
        axis="both",
        labelsize=OVERLEAF_TICK_LABEL_SIZE,
        length=OVERLEAF_TICK_LENGTH,
        width=OVERLEAF_TICK_WIDTH,
    )


def plot_metric(
    grouped: dict[str, list[dict[str, float]]],
    output_dir: Path,
    *,
    metric: str,
    plot_config: dict[str, str],
) -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.20, top=0.96)

    # Explicit z-orders keep hollow markers visible when memory values overlap.
    order = {method: index for index, method in enumerate(METHOD_ORDER)}
    methods = sorted(grouped, key=lambda name: (order.get(name, len(order)), name))
    value_scale = 100.0 if metric == "accuracy" else 1.0

    for method in methods:
        runs = [run for run in grouped[method] if math.isfinite(run[metric])]
        style = METHOD_STYLES.get(
            method,
            {
                "color": None,
                "marker": "o",
                "linestyle": "-",
                "linewidth": OVERLEAF_LINE_WIDTH,
                "markersize": OVERLEAF_MARKER_SIZE,
                "markerfacecolor": "none",
                "markeredgecolor": "black",
                "zorder": 2,
            },
        )
        if not runs:
            # Keep methods with no valid points visible in the legend.
            ax.plot(
                [],
                [],
                label=METHOD_LABELS.get(method, method),
                markeredgewidth=OVERLEAF_MARKER_EDGE_WIDTH,
                **style,
            )
            continue
        ax.plot(
            [run["agents"] for run in runs],
            [run[metric] * value_scale for run in runs],
            label=METHOD_LABELS.get(method, method),
            markeredgewidth=OVERLEAF_MARKER_EDGE_WIDTH,
            **style,
        )

    ax.set_xticks(sorted(PLOTTED_AGENT_COUNTS))
    ax.set_xlabel("Number of Agents")
    ax.set_ylabel(plot_config["ylabel"])
    style_axes_common(ax, grid=True, grid_axis="y", title=False)
    apply_large_text_style(ax)
    if metric == "kv_mean":
        ax.yaxis.label.set_fontsize(OVERLEAF_LEGEND_FONT_SIZE)

    if metric == "accuracy":
        ax.set_ylim(-7.0, 100.0)
        ax.set_yticks(range(0, 101, 20))
    else:
        ax.set_ylim(bottom=-0.03 * ax.get_ylim()[1])
    legend = ax.legend(
        loc="upper left",
        bbox_to_anchor=(-0.01, 1.02),
        frameon=True,
        framealpha=0.75,
        fancybox=False,
        edgecolor=AI_PAPER_LEGEND_EDGE_COLOR,
        fontsize=OVERLEAF_LEGEND_FONT_SIZE,
        ncols=2,
        handlelength=1.2,
        handletextpad=0.35,
        columnspacing=0.6,
        borderpad=0.3,
    )
    for text in legend.get_texts():
        text.set_fontweight("bold")

    output_path = output_dir / plot_config["filename"]
    pdf_output_path = output_dir / plot_config["pdf_filename"]
    fig.savefig(
        output_path,
        dpi=AI_PAPER_FIGURE_DPI,
        bbox_inches="tight",
        pad_inches=0.12,
        facecolor="white",
    )
    fig.savefig(
        pdf_output_path,
        dpi=AI_PAPER_FIGURE_DPI,
        bbox_inches="tight",
        pad_inches=0.12,
        facecolor="white",
    )
    plt.close(fig)
    return output_path, pdf_output_path


def plot_accuracy(
    grouped: dict[str, list[dict[str, float]]],
    output_dir: Path,
) -> tuple[Path, Path]:
    return plot_metric(
        grouped,
        output_dir,
        metric="accuracy",
        plot_config=ACCURACY_PLOT,
    )


def plot_memory(
    grouped: dict[str, list[dict[str, float]]],
    output_dir: Path,
) -> tuple[Path, Path]:
    return plot_metric(
        grouped,
        output_dir,
        metric="kv_mean",
        plot_config=MEMORY_PLOT,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--homogeneity-metrics-root",
        "--mot-metrics-root",
        dest="homogeneity_metrics_root",
        type=Path,
        default=Path("multi_agents_homogeneity"),
    )
    parser.add_argument(
        "--baseline-metrics-root",
        type=Path,
        default=Path("outputs/multi_agents"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/multi_agents"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_ai_paper_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    grouped = load_memory_runs(
        args.homogeneity_metrics_root,
        args.baseline_metrics_root,
    )
    validate_mot_free_is_minimum(grouped)
    print("Validated: MoT (Free) has minimum memory at every agent count")

    order = {method: index for index, method in enumerate(METHOD_ORDER)}
    for method, runs in sorted(
        grouped.items(), key=lambda item: (order.get(item[0], len(order)), item[0])
    ):
        counts = ", ".join(
            f"{int(run['agents'])} "
            f"({int(run['successful_turn_count'])} successful turns / "
            f"{int(run['example_count'])} examples)"
            for run in runs
        )
        print(f"{METHOD_LABELS.get(method, method)}: agents={counts}")

    output_paths = (
        *plot_accuracy(grouped, args.output_dir),
        *plot_memory(grouped, args.output_dir),
    )
    for output_path in output_paths:
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
