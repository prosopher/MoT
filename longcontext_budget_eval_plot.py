from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from core.exp_util import (
    ACCENT_BLACK,
    AI_PAPER_DOUBLE_COLUMN_TALL_FIGSIZE,
    AI_PAPER_FIGURE_DPI,
    AI_PAPER_GRID_ALPHA,
    AI_PAPER_GRID_LINESTYLE,
    AI_PAPER_GRID_LINE_WIDTH,
    AI_PAPER_LEGEND_EDGE_COLOR,
    AI_PAPER_LEGEND_HANDLE_LENGTH,
    AI_PAPER_LINE_WIDTH,
    AI_PAPER_MARKER_EDGE_WIDTH,
    AI_PAPER_MARKER_FACE_COLOR,
    AI_PAPER_MARKERS,
    AI_PAPER_NATIVE_LINESTYLE,
    apply_ai_paper_style,
    apply_bold_axis_labels,
    require_matplotlib_pyplot,
    save_paper_figure,
    style_axes_common,
)

# ---------------------------------------------------------------------------
# Figure style for Overleaf 1-column / 1x3 placement
# ---------------------------------------------------------------------------

# Keep the original paper figure size. Overleaf will scale/arrange the images.
FIGSIZE = AI_PAPER_DOUBLE_COLUMN_TALL_FIGSIZE

# Larger text/markers than the default exp_util template so labels remain readable
# after Overleaf scales three figures into a single row.
OVERLEAF_AXIS_LABEL_SIZE = 28
OVERLEAF_TICK_LABEL_SIZE = 28
OVERLEAF_LEGEND_FONT_SIZE = 28
OVERLEAF_LINE_WIDTH = AI_PAPER_LINE_WIDTH * 2.2
OVERLEAF_MARKER_SIZE = 13.0
OVERLEAF_MARKER_EDGE_WIDTH = AI_PAPER_MARKER_EDGE_WIDTH * 1.70

DEFAULT_RAW_MODEL_ORDER = ["upperbound", "c2c", "interlat", "mot128"]

DISPLAY_NAME = {
    "upperbound": "Native",
    "c2c": "C2C-Project",
    "interlat": "Interlat",
    "mot128": "MoT",
}

# User-specified palette.
DEFAULT_METHOD_COLORS = {
    "Native": "#000000",
    "C2C-Project": "#8064A2",
    "Interlat": "#4BACC6",
    "MoT": "#C0504D",
}

DEFAULT_METHOD_XTICK_LABELS = {
    "Native": "Native",
    "C2C-Project": "C2C-Project",
    "Interlat": "Interlat",
    "MoT": "MoT",
}

FALLBACK_METHOD_COLORS = [
    "#1F77B4",
    "#FF7F0E",
    "#2CA02C",
    "#D62728",
    "#9467BD",
    "#8C564B",
    "#E377C2",
    "#7F7F7F",
]

# Optional embedded input data.
# This lets the plotting script run without external result CSV files.
# Replace or extend this block when you want to regenerate figures from
# a different run, then run with: `--input-source embedded`.
EMBEDDED_BUDGET_ROWS: list[dict[str, object]] = [
    {
        "model": "upperbound",
        "budget": 4096,
        "f1": 0.215,
        "kv_size": "40.032 KiB",
        "ttft_source_o_ms": 181.447,
        "ttft_source_x_ms": 181.447,
        "throughput_tok_s": 171.0,
        "cosine": "N/A",
        "gpu_peak_gib": 3.125,
        "count": 300,
        "source": "avg_of_6_runs",
    },
    {
        "model": "upperbound",
        "budget": 8192,
        "f1": 0.22,
        "kv_size": "40.032 KiB",
        "ttft_source_o_ms": 462.913,
        "ttft_source_x_ms": 462.913,
        "throughput_tok_s": 85.7,
        "cosine": "N/A",
        "gpu_peak_gib": 3.399,
        "count": 300,
        "source": "avg_of_6_runs",
    },
    {
        "model": "upperbound",
        "budget": 16384,
        "f1": 0.205,
        "kv_size": "40.032 KiB",
        "ttft_source_o_ms": 1073.993,
        "ttft_source_x_ms": 1073.993,
        "throughput_tok_s": 44.3,
        "cosine": "N/A",
        "gpu_peak_gib": 3.979,
        "count": 300,
        "source": "avg_of_6_runs",
    },
    {
        "model": "upperbound",
        "budget": 24576,
        "f1": 0.21,
        "kv_size": "40.032 KiB",
        "ttft_source_o_ms": 1115.14,
        "ttft_source_x_ms": 1115.14,
        "throughput_tok_s": 43.2,
        "cosine": "N/A",
        "gpu_peak_gib": 4.043,
        "count": 300,
        "source": "avg_of_6_runs",
    },
    {
        "model": "c2c",
        "budget": 4096,
        "f1": 0.099,
        "kv_size": "46.644 MiB",
        "ttft_source_o_ms": 323.917,
        "ttft_source_x_ms": 183.249,
        "throughput_tok_s": 302.0,
        "cosine": "0.987",
        "gpu_peak_gib": 2.161,
        "count": 300,
        "source": "c2c_hotpotqa_e_ctx=4096",
    },
    {
        "model": "c2c",
        "budget": 8192,
        "f1": 0.12,
        "kv_size": "81.349 MiB",
        "ttft_source_o_ms": 881.138,
        "ttft_source_x_ms": 466.626,
        "throughput_tok_s": 122.0,
        "cosine": "0.992",
        "gpu_peak_gib": 2.419,
        "count": 300,
        "source": "c2c_hotpotqa_e_ctx=8192",
    },
    {
        "model": "c2c",
        "budget": 16384,
        "f1": 0.109,
        "kv_size": "115.583 MiB",
        "ttft_source_o_ms": 2089.961,
        "ttft_source_x_ms": 1066.719,
        "throughput_tok_s": 56.0,
        "cosine": "0.991",
        "gpu_peak_gib": 2.932,
        "count": 300,
        "source": "c2c_hotpotqa_e_ctx=16384",
    },
    {
        "model": "c2c",
        "budget": 24576,
        "f1": 0.113,
        "kv_size": "116.209 MiB",
        "ttft_source_o_ms": 2204.917,
        "ttft_source_x_ms": 1128.043,
        "throughput_tok_s": 53.0,
        "cosine": "0.991",
        "gpu_peak_gib": 2.988,
        "count": 300,
        "source": "c2c_hotpotqa_e_ctx=24576",
    },
    {
        "model": "interlat",
        "budget": 4096,
        "f1": 0.137,
        "kv_size": "6.753 MiB",
        "ttft_source_o_ms": 327.618,
        "ttft_source_x_ms": 182.393,
        "throughput_tok_s": 349.0,
        "cosine": "0.987",
        "gpu_peak_gib": 2.184,
        "count": 300,
        "source": "interlat_hotpotqa_e_ctx=4096",
    },
    {
        "model": "interlat",
        "budget": 8192,
        "f1": 0.148,
        "kv_size": "11.830 MiB",
        "ttft_source_o_ms": 891.284,
        "ttft_source_x_ms": 470.005,
        "throughput_tok_s": 136.0,
        "cosine": "0.983",
        "gpu_peak_gib": 2.438,
        "count": 300,
        "source": "interlat_hotpotqa_e_ctx=8192",
    },
    {
        "model": "interlat",
        "budget": 16384,
        "f1": 0.128,
        "kv_size": "16.849 MiB",
        "ttft_source_o_ms": 2122.188,
        "ttft_source_x_ms": 1089.008,
        "throughput_tok_s": 59.0,
        "cosine": "0.984",
        "gpu_peak_gib": 2.944,
        "count": 300,
        "source": "interlat_hotpotqa_e_ctx=16384",
    },
    {
        "model": "interlat",
        "budget": 24576,
        "f1": 0.134,
        "kv_size": "16.947 MiB",
        "ttft_source_o_ms": 2230.957,
        "ttft_source_x_ms": 1147.695,
        "throughput_tok_s": 56.0,
        "cosine": "0.985",
        "gpu_peak_gib": 2.995,
        "count": 300,
        "source": "interlat_hotpotqa_e_ctx=24576",
    },
    {
        "model": "mot128",
        "budget": 4096,
        "f1": 0.182,
        "kv_size": "46.644 MiB",
        "ttft_source_o_ms": 693.761,
        "ttft_source_x_ms": 405.254,
        "throughput_tok_s": 76.0,
        "cosine": "0.996",
        "gpu_peak_gib": 3.601,
        "count": 300,
        "source": "mot_topk128_hotpotqa_e_ctx=4096",
    },
    {
        "model": "mot128",
        "budget": 8192,
        "f1": 0.207,
        "kv_size": "81.349 MiB",
        "ttft_source_o_ms": 1593.127,
        "ttft_source_x_ms": 737.151,
        "throughput_tok_s": 49.0,
        "cosine": "0.998",
        "gpu_peak_gib": 3.882,
        "count": 300,
        "source": "mot_topk128_hotpotqa_e_ctx=8192",
    },
    {
        "model": "mot128",
        "budget": 16384,
        "f1": 0.215,
        "kv_size": "115.583 MiB",
        "ttft_source_o_ms": 3394.85,
        "ttft_source_x_ms": 1234.944,
        "throughput_tok_s": 38.0,
        "cosine": "0.999",
        "gpu_peak_gib": 4.499,
        "count": 300,
        "source": "mot_topk128_hotpotqa_e_ctx=16384",
    },
    {
        "model": "mot128",
        "budget": 24576,
        "f1": 0.215,
        "kv_size": "116.209 MiB",
        "ttft_source_o_ms": 3513.817,
        "ttft_source_x_ms": 1246.496,
        "throughput_tok_s": 38.0,
        "cosine": "0.999",
        "gpu_peak_gib": 4.561,
        "count": 300,
        "source": "mot_topk128_hotpotqa_e_ctx=24576",
    },
]


def parse_numeric_prefix(text: object) -> float:
    if pd.isna(text):
        return float("nan")
    if isinstance(text, (int, float)):
        return float(text)
    raw = str(text).strip()
    if raw.upper() == "N/A" or raw == "":
        return float("nan")
    return float(raw.split()[0])


def parse_size_to_mib(text: object) -> float:
    if pd.isna(text):
        return float("nan")
    if isinstance(text, (int, float)):
        return float(text)

    raw = str(text).strip()
    if raw.upper() == "N/A" or raw == "":
        return float("nan")

    parts = raw.split()
    value = float(parts[0])
    unit = parts[1].lower() if len(parts) > 1 else "mib"

    if unit in {"mib", "mb"}:
        return value
    if unit in {"kib", "kb"}:
        return value / 1024.0
    if unit in {"gib", "gb"}:
        return value * 1024.0

    raise ValueError(f"Unsupported size unit in value: {text}")


def applied_ttft_basis(raw_model: str) -> str:
    """TTFT rule for paper figures.

    - upperbound (native): source_o
    - others: source_x
    """
    if raw_model == "upperbound":
        return "source_o"
    return "source_x"


def build_dataframe_from_raw(raw: pd.DataFrame, *, source_name: str) -> pd.DataFrame:
    required = {
        "model",
        "budget",
        "f1",
        "kv_size",
        "ttft_source_o_ms",
        "ttft_source_x_ms",
        "throughput_tok_s",
    }
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"{source_name} missing required columns: {sorted(missing)}")

    df = pd.DataFrame(
        {
            "Budget": raw["budget"].astype(int),
            "RawModel": raw["model"].astype(str),
            "F1": raw["f1"].astype(float),
            "Translator_Input_Size_MiB": raw["kv_size"].apply(parse_size_to_mib),
            "TTFT_Source_O_ms": raw["ttft_source_o_ms"].astype(float),
            "TTFT_Source_X_ms": raw["ttft_source_x_ms"].astype(float),
            "Throughput_tok_s": raw["throughput_tok_s"].astype(float),
        }
    )

    if "cosine" in raw.columns:
        df["Cosine"] = raw["cosine"].apply(parse_numeric_prefix)
    if "gpu_peak_gib" in raw.columns:
        df["GPU_Peak_GiB"] = raw["gpu_peak_gib"].astype(float)
    if "count" in raw.columns:
        df["Count"] = raw["count"].astype(int)
    if "source" in raw.columns:
        df["Source"] = raw["source"].astype(str)

    df["TTFT_Basis"] = df["RawModel"].map(applied_ttft_basis)
    df["TTFT_ms"] = np.where(
        df["TTFT_Basis"].eq("source_o"),
        df["TTFT_Source_O_ms"],
        df["TTFT_Source_X_ms"],
    )
    return df.sort_values(["Budget", "RawModel"]).reset_index(drop=True)


def build_dataframe_from_csv(budget_csv_path: Path) -> pd.DataFrame:
    if not budget_csv_path.exists():
        raise FileNotFoundError(f"Budget CSV not found: {budget_csv_path}")
    raw = pd.read_csv(budget_csv_path)
    return build_dataframe_from_raw(raw, source_name=f"CSV({budget_csv_path})")


def build_dataframe_from_embedded_rows() -> pd.DataFrame:
    if not EMBEDDED_BUDGET_ROWS:
        raise ValueError(
            "EMBEDDED_BUDGET_ROWS is empty. Fill it at the top of this script or use --budget-csv."
        )
    raw = pd.DataFrame(EMBEDDED_BUDGET_ROWS)
    return build_dataframe_from_raw(raw, source_name="EMBEDDED_BUDGET_ROWS")


def split_cli_tokens(values: list[str] | None) -> list[str]:
    tokens: list[str] = []
    for value in values or []:
        for part in str(value).split(","):
            token = part.strip()
            if token:
                tokens.append(token)
    return tokens


def ordered_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def resolve_selected_raw_models(df: pd.DataFrame, model_tokens: list[str] | None) -> list[str]:
    available_raw_models = ordered_unique(df["RawModel"].astype(str).tolist())
    if not available_raw_models:
        raise ValueError("No models found in input data.")

    token_list = split_cli_tokens(model_tokens)
    if not token_list:
        defaults = [m for m in DEFAULT_RAW_MODEL_ORDER if m in available_raw_models]
        return defaults if defaults else available_raw_models

    raw_lookup = {raw.lower(): raw for raw in available_raw_models}
    display_lookup = {
        display.lower(): raw
        for raw, display in DISPLAY_NAME.items()
        if raw in available_raw_models
    }

    selected: list[str] = []
    unresolved: list[str] = []
    for token in token_list:
        key = token.lower()
        raw = raw_lookup.get(key) or display_lookup.get(key)
        if raw is None:
            unresolved.append(token)
            continue
        selected.append(raw)

    selected = ordered_unique(selected)
    if unresolved:
        raise ValueError(
            f"Unknown model(s): {unresolved}. Available raw models: {available_raw_models}. "
            f"Known display aliases: {sorted(display_lookup.keys())}"
        )
    if not selected:
        raise ValueError("No valid models selected after parsing --models.")
    return selected


def parse_selected_budgets(budget_tokens: list[str] | None) -> list[int] | None:
    token_list = split_cli_tokens(budget_tokens)
    if not token_list:
        return None

    budgets: list[int] = []
    for token in token_list:
        try:
            budgets.append(int(token))
        except ValueError as exc:
            raise ValueError(f"Invalid budget value: {token}") from exc
    return sorted(set(budgets))


def build_method_style_maps(method_order: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    method_colors: dict[str, str] = {}
    method_xtick_labels: dict[str, str] = {}

    fallback_idx = 0
    for method in method_order:
        if method in DEFAULT_METHOD_COLORS:
            method_colors[method] = DEFAULT_METHOD_COLORS[method]
        else:
            method_colors[method] = FALLBACK_METHOD_COLORS[fallback_idx % len(FALLBACK_METHOD_COLORS)]
            fallback_idx += 1
        method_xtick_labels[method] = DEFAULT_METHOD_XTICK_LABELS.get(method, method)
    return method_colors, method_xtick_labels


def apply_model_budget_filters(
    df: pd.DataFrame,
    *,
    selected_raw_models: list[str],
    selected_budgets: list[int] | None,
    max_budgets: int | None,
) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    if max_budgets is not None and max_budgets <= 0:
        raise ValueError("--max-budgets must be >= 1.")

    filtered = df[df["RawModel"].isin(selected_raw_models)].copy()
    if filtered.empty:
        raise ValueError(f"No rows left after model filter: {selected_raw_models}")

    if selected_budgets is not None:
        available_budget_set = set(filtered["Budget"].astype(int).tolist())
        missing = [b for b in selected_budgets if b not in available_budget_set]
        if missing:
            print(f"[warn] missing requested budgets in data: {missing}")
        filtered = filtered[filtered["Budget"].isin(selected_budgets)].copy()
        if filtered.empty:
            raise ValueError(f"No rows left after budget filter: {selected_budgets}")

    if max_budgets is not None:
        budget_candidates = sorted(filtered["Budget"].dropna().astype(int).unique().tolist())
        keep_budgets = budget_candidates[:max_budgets]
        filtered = filtered[filtered["Budget"].isin(keep_budgets)].copy()
        if filtered.empty:
            raise ValueError(f"No rows left after --max-budgets={max_budgets}.")

    present_raw_models = set(filtered["RawModel"].astype(str).tolist())
    raw_order = [raw for raw in selected_raw_models if raw in present_raw_models]

    display_name_map = {raw: DISPLAY_NAME.get(raw, raw) for raw in raw_order}
    method_order = [display_name_map[raw] for raw in raw_order]
    order_map = {method: idx for idx, method in enumerate(method_order)}

    filtered["Model"] = filtered["RawModel"].map(display_name_map)
    filtered["Model_Order"] = filtered["Model"].map(order_map)
    filtered = filtered.sort_values(["Budget", "Model_Order"]).reset_index(drop=True)
    return filtered, method_order, display_name_map


def build_average_table(df: pd.DataFrame, *, method_order: list[str]) -> pd.DataFrame:
    agg_spec = {
        "F1": ("F1", "mean"),
        "TTFT_ms": ("TTFT_ms", "mean"),
        "Translator_Input_Size_MiB": ("Translator_Input_Size_MiB", "mean"),
        "Throughput_tok_s": ("Throughput_tok_s", "mean"),
        "TTFT_Basis": ("TTFT_Basis", "first"),
    }
    if "Cosine" in df.columns:
        agg_spec["Cosine"] = ("Cosine", "mean")
    if "GPU_Peak_GiB" in df.columns:
        agg_spec["GPU_Peak_GiB"] = ("GPU_Peak_GiB", "mean")

    return (
        df.groupby(["Model", "RawModel"], as_index=False)
        .agg(**agg_spec)
        .sort_values("Model", key=lambda s: s.map({m: i for i, m in enumerate(method_order)}))
        .reset_index(drop=True)
    )


def finite(values: Iterable[float]) -> list[float]:
    return [float(v) for v in values if np.isfinite(v)]


def apply_large_text_style(ax) -> None:
    ax.xaxis.label.set_fontsize(OVERLEAF_AXIS_LABEL_SIZE)
    ax.yaxis.label.set_fontsize(OVERLEAF_AXIS_LABEL_SIZE)
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")
    ax.tick_params(axis="both", labelsize=OVERLEAF_TICK_LABEL_SIZE)


def style_method_tick_labels(ax, *, rotation: int = 0) -> None:
    for label in ax.get_xticklabels():
        label.set_fontweight("bold")
        label.set_fontsize(OVERLEAF_TICK_LABEL_SIZE)
        label.set_rotation(rotation)
        label.set_ha("center")


def set_compact_y_limits(
    ax,
    values: list[float],
    *,
    metric: str,
    pad_fraction: float = 0.14,
    force_zero: bool = False,
) -> None:
    vals = finite(values)
    if not vals:
        return

    ymin = min(vals)
    ymax = max(vals)

    if force_zero:
        ax.set_ylim(0.0, ymax * (1.0 + pad_fraction))
        return

    if np.isclose(ymin, ymax):
        pad = max(abs(ymax) * 0.08, 1e-3)
    else:
        pad = (ymax - ymin) * pad_fraction

    lower = ymin - pad
    upper = ymax + pad

    if ymin >= 0:
        lower = max(0.0, lower)
    if metric == "F1":
        upper = min(1.0, upper)

    ax.set_ylim(lower, upper)


def save_both_formats(fig, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    save_paper_figure(fig, output_base.with_suffix(".png"), dpi=AI_PAPER_FIGURE_DPI, close=False)
    save_paper_figure(fig, output_base.with_suffix(".pdf"), dpi=AI_PAPER_FIGURE_DPI, close=True)



def draw_manual_legend_on_axes(ax, *, method_order: list[str], method_colors: dict[str, str]) -> None:
    """Draw a manually controlled legend above the axes.

    Matplotlib's automatic ``ax.legend`` can compress the text/handle layout
    when the legend is expanded across the figure. This manual legend draws
    the box, short handle, marker, and text directly in axes coordinates so
    the text size is controlled by ``ax.text(..., fontsize=...)``.
    """

    # Legend box in axes coordinates: (left, bottom, width, height).
    # This keeps the legend box aligned to the plot width.
    box_x = -0.17
    box_y = 1.02
    box_w = 1.165
    box_h = 0.18

    legend_box = Rectangle(
        (box_x, box_y),
        box_w,
        box_h,
        transform=ax.transAxes,
        facecolor="white",
        edgecolor=AI_PAPER_LEGEND_EDGE_COLOR,
        linewidth=1.2,
        clip_on=False,
        zorder=20,
    )
    ax.add_patch(legend_box)

    if not method_order:
        return

    if len(method_order) > 4:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.17),
            ncol=min(len(method_order), 4),
            frameon=True,
            fontsize=OVERLEAF_LEGEND_FONT_SIZE * 0.55,
        )
        return

    # Keep spacing close to the original for 4 models and distribute for fewer.
    if len(method_order) == 4:
        item_x = [-0.14, 0.105, 0.52, 0.80]
    else:
        item_x = np.linspace(-0.08, 0.80, len(method_order)).tolist()
    y = box_y + box_h / 2.0

    # Shorter handle than ax.legend can reliably produce.
    handle_half_len = 0.020
    text_dx = 0.030

    for idx, method in enumerate(method_order):
        x = item_x[idx]
        linestyle = AI_PAPER_NATIVE_LINESTYLE if method == "Native" else "-"
        marker = AI_PAPER_MARKERS[idx % len(AI_PAPER_MARKERS)]
        color = method_colors[method]

        # Draw handle line without markers so only one marker appears.
        ax.plot(
            [x - handle_half_len, x + handle_half_len],
            [y, y],
            transform=ax.transAxes,
            linestyle=linestyle,
            linewidth=OVERLEAF_LINE_WIDTH,
            color=color,
            clip_on=False,
            zorder=25,
        )

        # Draw a single marker at the center of the handle.
        ax.plot(
            [x],
            [y],
            transform=ax.transAxes,
            linestyle="",
            marker=marker,
            markersize=OVERLEAF_MARKER_SIZE * 0.90,
            markerfacecolor=AI_PAPER_MARKER_FACE_COLOR,
            markeredgecolor=color,
            markeredgewidth=OVERLEAF_MARKER_EDGE_WIDTH,
            clip_on=False,
            zorder=26,
        )

        ax.text(
            x + text_dx,
            y,
            method,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=OVERLEAF_LEGEND_FONT_SIZE,
            fontweight="bold",
            clip_on=False,
            zorder=26,
        )

def draw_average_f1_ttft(
    *,
    avg_df: pd.DataFrame,
    method_order: list[str],
    method_colors: dict[str, str],
    method_xtick_labels: dict[str, str],
    output_base: Path,
) -> None:
    """Average F1 & TTFT.

    Left y-axis: F1 line.
    Right y-axis: TTFT bar.
    Legend is intentionally removed. Axis labels encode line/bar semantics.
    """
    plt = require_matplotlib_pyplot()
    plot_df = avg_df.set_index("Model").reindex(method_order).reset_index()

    fig, ax_line = plt.subplots(figsize=FIGSIZE)
    ax_bar = ax_line.twinx()

    x = np.arange(len(method_order), dtype=float)
    bar_width = 0.56

    f1_values: list[float] = []
    ttft_values: list[float] = []

    for idx, method in enumerate(method_order):
        row = plot_df[plot_df["Model"] == method]
        if row.empty:
            continue
        ttft = float(row.iloc[0]["TTFT_ms"])
        ttft_values.append(ttft)
        ax_bar.bar(
            x[idx],
            ttft,
            width=bar_width,
            color=method_colors[method],
            alpha=0.80 if method != "Native" else 0.90,
            edgecolor="white",
            linewidth=0.8,
            zorder=2,
        )

    valid_x: list[float] = []
    valid_y: list[float] = []
    for idx, method in enumerate(method_order):
        row = plot_df[plot_df["Model"] == method]
        if row.empty:
            continue
        f1 = float(row.iloc[0]["F1"])
        f1_values.append(f1)
        valid_x.append(x[idx])
        valid_y.append(f1)

    ax_line.plot(
        valid_x,
        valid_y,
        linestyle="-",
        linewidth=OVERLEAF_LINE_WIDTH,
        color=ACCENT_BLACK,
        zorder=4,
    )

    for idx, method in enumerate(method_order):
        row = plot_df[plot_df["Model"] == method]
        if row.empty:
            continue
        f1 = float(row.iloc[0]["F1"])
        ax_line.plot(
            [x[idx]],
            [f1],
            linestyle="",
            marker=AI_PAPER_MARKERS[idx % len(AI_PAPER_MARKERS)],
            markersize=OVERLEAF_MARKER_SIZE,
            markerfacecolor=AI_PAPER_MARKER_FACE_COLOR,
            markeredgecolor=method_colors[method],
            markeredgewidth=OVERLEAF_MARKER_EDGE_WIDTH,
            color=method_colors[method],
            zorder=5,
        )

    set_compact_y_limits(ax_line, f1_values, metric="F1", pad_fraction=0.18)
    set_compact_y_limits(ax_bar, ttft_values, metric="TTFT", pad_fraction=0.18, force_zero=True)

    ax_line.set_xticks(x)
    ax_line.set_xticklabels([method_xtick_labels[m] for m in method_order])
    ax_line.set_xlabel("")
    ax_line.set_ylabel("F1 (Line)")
    ax_bar.set_ylabel("TTFT (Bar, ms)")

    style_axes_common(ax_line, grid=True, grid_axis="y", title=False)
    style_axes_common(ax_bar, grid=False, grid_axis="y", title=False)
    apply_large_text_style(ax_line)
    apply_large_text_style(ax_bar)
    style_method_tick_labels(ax_line)

    ax_bar.set_zorder(1)
    ax_line.set_zorder(2)
    ax_line.patch.set_visible(False)

    fig.subplots_adjust(left=0.125, right=0.875, bottom=0.155, top=0.955)
    save_both_formats(fig, output_base)


def draw_budget_line_plot(
    *,
    df: pd.DataFrame,
    metric: str,
    y_label: str,
    method_order: list[str],
    method_colors: dict[str, str],
    output_base: Path,
) -> None:
    """Budget-wise line plot.

    Native is drawn as a dashed reference line.
    """
    plt = require_matplotlib_pyplot()
    fig, ax = plt.subplots(figsize=FIGSIZE)

    all_values: list[float] = []
    for idx, method in enumerate(method_order):
        sub = df[df["Model"] == method].sort_values("Budget")
        if sub.empty:
            continue

        x = sub["Budget"].astype(int).to_numpy()
        y = sub[metric].astype(float).to_numpy()
        all_values.extend([float(v) for v in y if np.isfinite(v)])

        ax.plot(
            x,
            y,
            linestyle=AI_PAPER_NATIVE_LINESTYLE if method == "Native" else "-",
            linewidth=OVERLEAF_LINE_WIDTH,
            color=method_colors[method],
            marker=AI_PAPER_MARKERS[idx % len(AI_PAPER_MARKERS)],
            markersize=OVERLEAF_MARKER_SIZE,
            markerfacecolor=AI_PAPER_MARKER_FACE_COLOR,
            markeredgecolor=method_colors[method],
            markeredgewidth=OVERLEAF_MARKER_EDGE_WIDTH,
            zorder=4,
            label=method,
        )

    style_axes_common(ax, grid=True, grid_axis="y", title=False)
    apply_large_text_style(ax)

    ax.set_xlabel("Budget")
    ax.set_ylabel(y_label)

    budgets = sorted(df["Budget"].dropna().astype(int).unique().tolist())
    ax.set_xticks(budgets)
    ax.set_xticklabels([f"{b // 1024}K" if b >= 1024 else str(b) for b in budgets])

    set_compact_y_limits(ax, all_values, metric=metric, pad_fraction=0.14)

    # Match the c-graph TTFT y-axis tick layout in the reference figure.
    # Keep this only for the TTFT-by-budget plot; F1 uses the compact automatic scale.
    if metric == "TTFT_ms":
        ax.set_yticks([250, 500, 750, 1000, 1250])

    draw_manual_legend_on_axes(ax, method_order=method_order, method_colors=method_colors)

    fig.subplots_adjust(left=0.125, right=0.965, bottom=0.155, top=0.835)
    save_both_formats(fig, output_base)


def save_tables(avg_df: pd.DataFrame, output_dir: Path) -> None:
    table = avg_df.copy()
    for col in table.columns:
        if pd.api.types.is_float_dtype(table[col]):
            table[col] = table[col].round(4)

    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "eval_plot_2paper_average_table.csv", index=False)
    with open(output_dir / "eval_plot_2paper_average_table.md", "w", encoding="utf-8") as f:
        f.write(table.to_markdown(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Overleaf-ready paper plots for HeteroCache long-context case study."
    )
    parser.add_argument(
        "--budget-csv",
        type=Path,
        default=None,
        help="Optional budget-level CSV. If omitted, uses EMBEDDED_BUDGET_ROWS in this file.",
    )
    parser.add_argument(
        "--input-source",
        type=str,
        default="auto",
        choices=["auto", "csv", "embedded"],
        help=(
            "Input source selector. "
            "'auto': use CSV when --budget-csv is set, else embedded rows. "
            "'csv': force --budget-csv. "
            "'embedded': force EMBEDDED_BUDGET_ROWS."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/ojungii/HeteroCache/longcontext_ttft_128/paper_plots_overleaf"),
    )
    parser.add_argument(
        "--plot-types",
        nargs="+",
        default=["avg", "budget"],
        choices=["avg", "budget"],
        help="Select plot groups to generate. Use: --plot-types avg budget",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=(
            "Optional model selection (raw model ids or display aliases). "
            "Examples: --models upperbound c2c interlat mot128, "
            "--models Native,C2C-Project,Interlat,MoT, "
            "--models mot16 mot32 mot64 mot128"
        ),
    )
    parser.add_argument(
        "--budgets",
        nargs="+",
        default=None,
        help="Optional budget filter. Examples: --budgets 4096 8192 or --budgets 4096,8192,16384",
    )
    parser.add_argument(
        "--max-budgets",
        type=int,
        default=None,
        help="Optional cap on number of budgets (keeps lowest budgets after filtering).",
    )
    parser.add_argument("--avg-name", type=str, default="avg_f1_ttft")
    parser.add_argument("--budget-f1-name", type=str, default="budget_f1")
    parser.add_argument("--budget-ttft-name", type=str, default="budget_ttft")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_ai_paper_style()

    if args.input_source == "csv":
        if args.budget_csv is None:
            raise ValueError("--input-source csv requires --budget-csv.")
        df = build_dataframe_from_csv(args.budget_csv)
        print(f"[info] input source: csv ({args.budget_csv})")
    elif args.input_source == "embedded":
        df = build_dataframe_from_embedded_rows()
        print("[info] input source: embedded (EMBEDDED_BUDGET_ROWS)")
    else:
        if args.budget_csv is not None:
            df = build_dataframe_from_csv(args.budget_csv)
            print(f"[info] input source: csv ({args.budget_csv})")
        else:
            df = build_dataframe_from_embedded_rows()
            print("[info] input source: embedded (EMBEDDED_BUDGET_ROWS)")

    selected_raw_models = resolve_selected_raw_models(df, args.models)
    selected_budgets = parse_selected_budgets(args.budgets)
    df, method_order, _display_name_map = apply_model_budget_filters(
        df,
        selected_raw_models=selected_raw_models,
        selected_budgets=selected_budgets,
        max_budgets=args.max_budgets,
    )
    method_colors, method_xtick_labels = build_method_style_maps(method_order)

    print(f"[info] selected raw models: {selected_raw_models}")
    print(f"[info] selected display order: {method_order}")
    if selected_budgets is not None:
        print(f"[info] requested budgets: {selected_budgets}")
    if args.max_budgets is not None:
        print(f"[info] max budgets: {args.max_budgets}")
    print(f"[info] effective budgets: {sorted(df['Budget'].astype(int).unique().tolist())}")

    avg_df = build_average_table(df, method_order=method_order)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_tables(avg_df, args.output_dir)

    if "avg" in args.plot_types:
        draw_average_f1_ttft(
            avg_df=avg_df,
            method_order=method_order,
            method_colors=method_colors,
            method_xtick_labels=method_xtick_labels,
            output_base=args.output_dir / args.avg_name,
        )
        print(args.output_dir / f"{args.avg_name}.png")
        print(args.output_dir / f"{args.avg_name}.pdf")

    if "budget" in args.plot_types:
        draw_budget_line_plot(
            df=df,
            metric="F1",
            y_label="F1",
            method_order=method_order,
            method_colors=method_colors,
            output_base=args.output_dir / args.budget_f1_name,
        )
        draw_budget_line_plot(
            df=df,
            metric="TTFT_ms",
            y_label="TTFT (ms)",
            method_order=method_order,
            method_colors=method_colors,
            output_base=args.output_dir / args.budget_ttft_name,
        )
        print(args.output_dir / f"{args.budget_f1_name}.png")
        print(args.output_dir / f"{args.budget_f1_name}.pdf")
        print(args.output_dir / f"{args.budget_ttft_name}.png")
        print(args.output_dir / f"{args.budget_ttft_name}.pdf")

    print(args.output_dir / "eval_plot_2paper_average_table.csv")
    print(args.output_dir / "eval_plot_2paper_average_table.md")


if __name__ == "__main__":
    main()
