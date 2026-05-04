from __future__ import annotations

import os
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Optional, Sequence


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
AI_PAPER_NATIVE_LINESTYLE = ":"
AI_PAPER_CONTROL_LINESTYLE = "--"

# Common double-column figure template for AI conference papers.
AI_PAPER_DOUBLE_COLUMN_WIDTH = 7.16
AI_PAPER_DOUBLE_COLUMN_HEIGHT = 4.20
AI_PAPER_DOUBLE_COLUMN_FIGSIZE = (
    AI_PAPER_DOUBLE_COLUMN_WIDTH,
    AI_PAPER_DOUBLE_COLUMN_HEIGHT,
)
AI_PAPER_DOUBLE_COLUMN_TALL_FIGSIZE = (AI_PAPER_DOUBLE_COLUMN_WIDTH, 4.80)
AI_PAPER_DOUBLE_COLUMN_SHORT_FIGSIZE = (AI_PAPER_DOUBLE_COLUMN_WIDTH, 3.80)
AI_PAPER_DOUBLE_COLUMN_SQUARE_FIGSIZE = (AI_PAPER_DOUBLE_COLUMN_WIDTH, AI_PAPER_DOUBLE_COLUMN_WIDTH)
AI_PAPER_FIGURE_DPI = 300
AI_PAPER_PREVIEW_DPI = 150

# Double-column-friendly typography requested for this project.
# Major text uses 12 pt; minor text uses 10 pt.
AI_PAPER_MAJOR_FONT_SIZE = 12
AI_PAPER_MINOR_FONT_SIZE = 10
AI_PAPER_AXIS_LABEL_SIZE = AI_PAPER_MAJOR_FONT_SIZE
AI_PAPER_LEGEND_FONT_SIZE = AI_PAPER_MAJOR_FONT_SIZE
AI_PAPER_COLORBAR_LABEL_SIZE = AI_PAPER_MAJOR_FONT_SIZE
AI_PAPER_TICK_LABEL_SIZE = AI_PAPER_MINOR_FONT_SIZE
AI_PAPER_ANNOTATION_FONT_SIZE = AI_PAPER_MINOR_FONT_SIZE
AI_PAPER_COLORBAR_TICK_SIZE = AI_PAPER_MINOR_FONT_SIZE

TIMES_NEW_ROMAN_FONT_FAMILY = "Times New Roman"
TIMES_NEW_ROMAN_FONT_ENV_PATHS = (
    "MOT_TIMES_NEW_ROMAN_FONT_PATH",
    "TIMES_NEW_ROMAN_FONT_PATH",
)
TIMES_NEW_ROMAN_FONT_URLS_ENV = "MOT_TIMES_NEW_ROMAN_FONT_URLS"
TIMES_NEW_ROMAN_ARCHIVE_URL_ENV = "MOT_TIMES_NEW_ROMAN_ARCHIVE_URL"
TIMES_NEW_ROMAN_CACHE_SUBDIR = "mot_times_new_roman"
TIMES_NEW_ROMAN_FONT_SUFFIXES = (".ttf", ".otf", ".ttc")
# The direct URLs are used only when the font is absent from the system and no
# local font path is provided. Override them with MOT_TIMES_NEW_ROMAN_FONT_URLS
# in environments that mirror fonts internally. Font files are cached locally
# and are not bundled in this repository.
DEFAULT_TIMES_NEW_ROMAN_FONT_URLS = (
    "https://raw.githubusercontent.com/justrajdeep/fonts/master/Times%20New%20Roman.ttf",
    "https://raw.githubusercontent.com/justrajdeep/fonts/master/Times%20New%20Roman%20Bold.ttf",
    "https://raw.githubusercontent.com/justrajdeep/fonts/master/Times%20New%20Roman%20Italic.ttf",
    "https://raw.githubusercontent.com/justrajdeep/fonts/master/Times%20New%20Roman%20Bold%20Italic.ttf",
)
DEFAULT_TIMES_NEW_ROMAN_ARCHIVE_URL = (
    "https://downloads.sourceforge.net/project/corefonts/the%20fonts/final/times32.exe"
)
TIMES_NEW_ROMAN_ARCHIVE_EXTRACTORS = ("cabextract", "7z", "bsdtar")
AI_PAPER_MATH_FONTSET = "stix"

AI_PAPER_LINE_WIDTH = 1.8
AI_PAPER_REFERENCE_LINE_WIDTH = 1.0
AI_PAPER_MARKER_SIZE = 8.5
AI_PAPER_SCATTER_SIZE = 90.0
AI_PAPER_MARKER_EDGE_WIDTH = 1.5
AI_PAPER_BAR_VALUE_FONT_SIZE = AI_PAPER_MINOR_FONT_SIZE
AI_PAPER_HEATMAP_VALUE_FONT_SIZE = AI_PAPER_MINOR_FONT_SIZE
AI_PAPER_ALGORITHM_LABEL_ROTATION = 35
AI_PAPER_LAYER_TICK_ROTATION = 45
AI_PAPER_LEGEND_HANDLE_LENGTH = 2.4
AI_PAPER_GRID_LINE_WIDTH = 0.6
AI_PAPER_GRID_ALPHA = 0.32
AI_PAPER_GRID_LINESTYLE = "--"
AI_PAPER_REFERENCE_COLOR = "0.35"
AI_PAPER_BORDER_COLOR = "0.20"
AI_PAPER_LEGEND_EDGE_COLOR = "0.85"
AI_PAPER_MARKER_FACE_COLOR = "white"
AI_PAPER_DENSE_WIDTH_SCALE_MAX = 1.0
AI_PAPER_GROUPED_BAR_TOTAL_WIDTH = 0.82
AI_PAPER_GROUPED_BAR_WIDTH_SCALE = 0.92
AI_PAPER_VALUE_OFFSET_FRACTION = 0.01
AI_PAPER_RADAR_FILL_ALPHA = 0.08
AI_PAPER_ANNOTATION_OFFSET = (5, 4)
AI_PAPER_DEFAULT_X_MARGIN = 0.03
AI_PAPER_DEFAULT_Y_MARGIN = 0.08
AI_PAPER_BAR_X_MARGIN = 0.04
AI_PAPER_BAR_Y_MARGIN = 0.10
AI_PAPER_WRAP_TICK_WIDTH = 24
AI_PAPER_LEGEND_OUTSIDE_ANCHOR = (1.02, 0.5)


def double_column_figsize(*, height: float | None = None, square: bool = False) -> tuple[float, float]:
    """Return the shared double-column figure size in inches."""
    if square:
        return AI_PAPER_DOUBLE_COLUMN_SQUARE_FIGSIZE
    return (AI_PAPER_DOUBLE_COLUMN_WIDTH, height or AI_PAPER_DOUBLE_COLUMN_HEIGHT)


def scaled_double_column_figsize(
    *,
    width_scale: float = 1.0,
    height: float | None = None,
    height_scale: float = 1.0,
) -> tuple[float, float]:
    """Return a double-column-derived size, capped at the shared double-column width."""
    bounded_width_scale = min(max(width_scale, 1.0), AI_PAPER_DENSE_WIDTH_SCALE_MAX)
    width = AI_PAPER_DOUBLE_COLUMN_WIDTH * bounded_width_scale
    fig_height = (height or AI_PAPER_DOUBLE_COLUMN_HEIGHT) * height_scale
    return (width, fig_height)


def apply_bold_axis_labels(ax) -> None:
    """Ensure axis labels stay bold even when labels are assigned after rcParams."""
    try:
        ax.xaxis.label.set_fontweight("bold")
        ax.yaxis.label.set_fontweight("bold")
        ax.xaxis.label.set_fontsize(AI_PAPER_AXIS_LABEL_SIZE)
        ax.yaxis.label.set_fontsize(AI_PAPER_AXIS_LABEL_SIZE)
    except Exception:
        pass


def style_algorithm_tick_labels(ax, *, axis: str = "x") -> None:
    """Bold algorithm/category tick labels used as method names in bar charts."""
    labels = ax.get_xticklabels() if axis == "x" else ax.get_yticklabels()
    for label in labels:
        label.set_fontweight("bold")
        label.set_fontsize(AI_PAPER_TICK_LABEL_SIZE)



def _font_cache_dir() -> Path:
    cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache_root / TIMES_NEW_ROMAN_CACHE_SUBDIR


def _iter_font_files(path: Path):
    if path.is_file() and path.suffix.lower() in TIMES_NEW_ROMAN_FONT_SUFFIXES:
        yield path
        return
    if path.is_dir():
        for suffix in TIMES_NEW_ROMAN_FONT_SUFFIXES:
            yield from path.rglob(f"*{suffix}")


def _register_font_files(path: Path) -> int:
    try:
        from matplotlib import font_manager
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required only for plotting. Install matplotlib to generate figures."
        ) from exc

    registered = 0
    for font_path in _iter_font_files(path):
        try:
            font_manager.fontManager.addfont(str(font_path))
            registered += 1
        except Exception:
            continue
    return registered


def _find_times_new_roman_font() -> str | None:
    try:
        from matplotlib import font_manager
        return font_manager.findfont(
            TIMES_NEW_ROMAN_FONT_FAMILY,
            fallback_to_default=False,
            rebuild_if_missing=True,
        )
    except Exception:
        return None


def _font_available() -> bool:
    return _find_times_new_roman_font() is not None


def _split_font_url_env(value: str) -> list[str]:
    pieces: list[str] = []
    for chunk in value.replace("\n", ",").replace(";", ",").split(","):
        chunk = chunk.strip()
        if chunk:
            pieces.append(chunk)
    return pieces


def _download_file(url: str, dst: Path) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=45) as response, dst.open("wb") as fh:
            shutil.copyfileobj(response, fh)
        return dst.exists() and dst.stat().st_size > 0
    except Exception:
        try:
            dst.unlink(missing_ok=True)
        except Exception:
            pass
        return False


def _download_times_new_roman_ttf_files(cache_dir: Path) -> int:
    url_env = os.environ.get(TIMES_NEW_ROMAN_FONT_URLS_ENV, "")
    urls = _split_font_url_env(url_env) if url_env else list(DEFAULT_TIMES_NEW_ROMAN_FONT_URLS)
    downloaded = 0
    for idx, url in enumerate(urls):
        guessed_name = url.rsplit("/", 1)[-1].split("?", 1)[0].replace("%20", "_")
        suffix = Path(guessed_name).suffix.lower()
        if suffix not in TIMES_NEW_ROMAN_FONT_SUFFIXES:
            guessed_name = f"times_new_roman_{idx}.ttf"
        dst = cache_dir / guessed_name
        if dst.exists() and dst.stat().st_size > 0:
            downloaded += 1
            continue
        if _download_file(url, dst):
            downloaded += 1
    if downloaded:
        _register_font_files(cache_dir)
    return downloaded


def _extract_corefonts_archive(archive_path: Path, output_dir: Path) -> bool:
    output_dir.mkdir(parents=True, exist_ok=True)
    extractor = next((name for name in TIMES_NEW_ROMAN_ARCHIVE_EXTRACTORS if shutil.which(name)), None)
    if extractor is None:
        return False
    try:
        if extractor == "cabextract":
            cmd = [extractor, "-q", "-d", str(output_dir), str(archive_path)]
        elif extractor == "7z":
            cmd = [extractor, "x", "-y", f"-o{output_dir}", str(archive_path)]
        else:
            cmd = [extractor, "-xf", str(archive_path), "-C", str(output_dir)]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return any(_iter_font_files(output_dir))
    except Exception:
        return False


def _download_and_extract_corefonts_times(cache_dir: Path) -> bool:
    archive_url = os.environ.get(TIMES_NEW_ROMAN_ARCHIVE_URL_ENV, DEFAULT_TIMES_NEW_ROMAN_ARCHIVE_URL)
    archive_path = cache_dir / "times32.exe"
    if not archive_path.exists() or archive_path.stat().st_size == 0:
        if not _download_file(archive_url, archive_path):
            return False
    extracted_dir = cache_dir / "corefonts"
    if not _extract_corefonts_archive(archive_path, extracted_dir):
        return False
    _register_font_files(extracted_dir)
    return True


def ensure_times_new_roman_font() -> str:
    """Ensure Times New Roman is registered with matplotlib before plotting.

    Lookup order:
    1. system matplotlib/fontconfig fonts,
    2. local paths from MOT_TIMES_NEW_ROMAN_FONT_PATH or TIMES_NEW_ROMAN_FONT_PATH,
    3. previously cached fonts under XDG_CACHE_HOME or ~/.cache,
    4. download direct TTF URLs, overridable with MOT_TIMES_NEW_ROMAN_FONT_URLS,
    5. download the Microsoft core-fonts Times archive and extract it when an
       extractor such as cabextract, 7z, or bsdtar is available.
    """
    existing = _find_times_new_roman_font()
    if existing:
        return existing

    for env_name in TIMES_NEW_ROMAN_FONT_ENV_PATHS:
        raw_path = os.environ.get(env_name)
        if not raw_path:
            continue
        for part in raw_path.split(os.pathsep):
            part = part.strip()
            if part:
                _register_font_files(Path(part).expanduser())
        existing = _find_times_new_roman_font()
        if existing:
            return existing

    cache_dir = _font_cache_dir()
    if cache_dir.exists():
        _register_font_files(cache_dir)
        existing = _find_times_new_roman_font()
        if existing:
            return existing

    _download_times_new_roman_ttf_files(cache_dir)
    existing = _find_times_new_roman_font()
    if existing:
        return existing

    _download_and_extract_corefonts_times(cache_dir)
    existing = _find_times_new_roman_font()
    if existing:
        return existing

    raise RuntimeError(
        "Times New Roman could not be found or downloaded. "
        "Provide a local .ttf/.ttc path via MOT_TIMES_NEW_ROMAN_FONT_PATH, "
        "or provide a reachable URL via MOT_TIMES_NEW_ROMAN_FONT_URLS."
    )

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
    ensure_times_new_roman_font()
    plt = require_matplotlib_pyplot()
    plt.rcParams.update(
        {
            # Figure and save quality
            "figure.figsize": AI_PAPER_DOUBLE_COLUMN_FIGSIZE,
            "figure.dpi": AI_PAPER_PREVIEW_DPI,
            "savefig.dpi": AI_PAPER_FIGURE_DPI,
            "savefig.format": "pdf",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,

            # Font: fixed to Times New Roman; no fallback family list.
            "font.family": TIMES_NEW_ROMAN_FONT_FAMILY,
            "font.size": AI_PAPER_MINOR_FONT_SIZE,
            # Math text is intentionally separated from Times New Roman.
            "mathtext.fontset": AI_PAPER_MATH_FONTSET,

            # Editable text in vector outputs
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",

            # Axes
            "axes.labelsize": AI_PAPER_AXIS_LABEL_SIZE,
            "axes.labelweight": "bold",
            "axes.titlesize": AI_PAPER_AXIS_LABEL_SIZE,
            "axes.linewidth": 1.0,
            "axes.spines.top": True,
            "axes.spines.right": True,

            # Ticks
            "xtick.labelsize": AI_PAPER_TICK_LABEL_SIZE,
            "ytick.labelsize": AI_PAPER_TICK_LABEL_SIZE,
            "xtick.direction": "out",
            "ytick.direction": "out",

            # Legend
            "legend.fontsize": AI_PAPER_LEGEND_FONT_SIZE,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "legend.fancybox": False,
            "legend.edgecolor": AI_PAPER_LEGEND_EDGE_COLOR,

            # Lines
            "lines.linewidth": AI_PAPER_LINE_WIDTH,
            "lines.markersize": AI_PAPER_MARKER_SIZE,
            "lines.markeredgewidth": AI_PAPER_MARKER_EDGE_WIDTH,
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


def style_figure_legends(fig) -> None:
    """Bold legend entries; method/algorithm names are commonly rendered there."""
    for ax in getattr(fig, "axes", []):
        legend = None
        try:
            legend = ax.get_legend()
        except Exception:
            legend = None
        if legend is None:
            continue
        for text in legend.get_texts():
            text.set_fontweight("bold")
            text.set_fontsize(AI_PAPER_LEGEND_FONT_SIZE)
        title = legend.get_title()
        if title is not None:
            title.set_fontweight("bold")
            title.set_fontsize(AI_PAPER_LEGEND_FONT_SIZE)


def apply_times_new_roman_to_figure(fig) -> None:
    """Force all regular Matplotlib text objects to Times New Roman before save.

    Some artists are created after rcParams are set, or by colorbar/legend helper
    code that may retain a backend default font. Applying this at save time makes
    every ordinary text object in the figure use the registered Times New Roman
    family while leaving math rendering controlled by mathtext.fontset.
    """
    ensure_times_new_roman_font()
    try:
        from matplotlib.text import Text
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required only for plotting. Install matplotlib to generate figures."
        ) from exc

    try:
        text_objects = fig.findobj(match=Text)
    except Exception:
        text_objects = []
    for text in text_objects:
        try:
            text.set_fontfamily(TIMES_NEW_ROMAN_FONT_FAMILY)
        except Exception:
            pass

    for ax in getattr(fig, "axes", []):
        for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            try:
                label.set_fontfamily(TIMES_NEW_ROMAN_FONT_FAMILY)
            except Exception:
                pass
        try:
            ax.xaxis.label.set_fontfamily(TIMES_NEW_ROMAN_FONT_FAMILY)
            ax.yaxis.label.set_fontfamily(TIMES_NEW_ROMAN_FONT_FAMILY)
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
        spine.set_color(AI_PAPER_BORDER_COLOR)

    try:
        ax.tick_params(axis="both", which="major", length=4, width=1.0)
        ax.tick_params(axis="both", which="minor", length=2, width=0.8)
    except Exception:
        pass

    apply_bold_axis_labels(ax)

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
                linestyle=AI_PAPER_GRID_LINESTYLE,
                linewidth=AI_PAPER_GRID_LINE_WIDTH,
                alpha=AI_PAPER_GRID_ALPHA,
            )
        except Exception:
            ax.grid(True, linestyle=AI_PAPER_GRID_LINESTYLE, linewidth=AI_PAPER_GRID_LINE_WIDTH, alpha=AI_PAPER_GRID_ALPHA)


def style_paper_axes(
    ax,
    *,
    x_values: Optional[Sequence[int | float]] = None,
    margins: tuple[float, float] | None = (AI_PAPER_DEFAULT_X_MARGIN, AI_PAPER_DEFAULT_Y_MARGIN),
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
    ensure_times_new_roman_font()
    clear_figure_titles(fig)
    style_figure_legends(fig)
    apply_times_new_roman_to_figure(fig)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {"dpi": AI_PAPER_FIGURE_DPI if dpi is None else dpi}
    fig.savefig(output_path, **save_kwargs)

    plt = require_matplotlib_pyplot()
    if show:
        plt.show()
    if close:
        plt.close(fig)
