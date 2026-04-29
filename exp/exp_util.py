from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence


# Colorblind-friendly palette for paper figures.
# Important/highlighted results should use Accent Red.
AI_PAPER_PALETTE = [
    "#C0504D",  # Accent Red
    "#4BACC6",  # Accent Aqua
    "#8064A2",  # Accent Purple
    "#4F81BD",  # Accent Blue
    "#9BBB59",  # Accent Green
    "#F79646",  # Accent Orange
    "#000000",  # black
]

ACCENT_RED = AI_PAPER_PALETTE[0]
ACCENT_AQUA = AI_PAPER_PALETTE[1]
ACCENT_PURPLE = AI_PAPER_PALETTE[2]
ACCENT_BLUE = AI_PAPER_PALETTE[3]
ACCENT_GREEN = AI_PAPER_PALETTE[4]
ACCENT_ORANGE = AI_PAPER_PALETTE[5]
ACCENT_BLACK = AI_PAPER_PALETTE[6]

AI_PAPER_MARKERS = [
    "o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">"
]

AI_PAPER_LINESTYLES = [
    "-", "--", "-.", ":"
]


def require_matplotlib_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required only for plotting. Install matplotlib to generate figures."
        ) from exc
    return plt


def require_matplotlib_colors():
    try:
        import matplotlib.colors as mcolors
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required only for plotting. Install matplotlib to generate figures."
        ) from exc
    return mcolors


def apply_ai_paper_style() -> None:
    """Apply the shared paper-figure Matplotlib template."""
    plt = require_matplotlib_pyplot()
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
            "axes.spines.top": True,
            "axes.spines.right": True,

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


def clear_figure_titles(fig) -> None:
    """Use captions instead of in-plot titles for paper figures."""
    if hasattr(fig, "_suptitle") and fig._suptitle is not None:
        fig._suptitle.set_text("")
    for ax in getattr(fig, "axes", []):
        try:
            ax.set_title("")
        except Exception:
            pass


def style_axes_common(
    ax,
    *,
    grid: bool = True,
    grid_axis: str = "y",
    title: bool = False,
) -> None:
    """Apply shared axes styling: boxed border, no title, and light grid."""
    if not title:
        try:
            ax.set_title("")
        except Exception:
            pass

    try:
        ax.set_frame_on(True)
    except Exception:
        pass

    for spine in getattr(ax, "spines", {}).values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color("0.20")

    try:
        ax.tick_params(axis="both", which="major", length=4, width=1.0)
        ax.tick_params(axis="both", which="minor", length=2, width=0.8)
    except Exception:
        pass

    try:
        ax.set_axisbelow(True)
    except Exception:
        pass

    if grid:
        try:
            ax.grid(
                True,
                which="major",
                axis=grid_axis,
                linestyle="--",
                linewidth=0.7,
                alpha=0.35,
            )
        except Exception:
            ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)


def style_paper_axes(
    ax,
    *,
    x_values: Optional[Sequence[int | float]] = None,
    margins: tuple[float, float] | None = (0.03, 0.08),
    minorticks: bool = True,
    grid: bool = True,
    grid_axis: str = "y",
) -> None:
    style_axes_common(ax, grid=grid, grid_axis=grid_axis)
    if minorticks:
        try:
            ax.minorticks_on()
        except Exception:
            pass
    if margins is not None:
        try:
            ax.margins(x=margins[0], y=margins[1])
        except Exception:
            pass
    if x_values is not None:
        ax.set_xticks(list(x_values))


def save_paper_figure(
    fig,
    output_path: Path,
    *,
    dpi: int | None = None,
    show: bool = False,
    close: bool = False,
) -> None:
    """Save a paper-style figure using shared defaults and no in-plot title."""
    clear_figure_titles(fig)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {}
    if dpi is not None:
        save_kwargs["dpi"] = dpi
    fig.savefig(output_path, **save_kwargs)

    plt = require_matplotlib_pyplot()
    if show:
        plt.show()
    if close:
        plt.close(fig)
