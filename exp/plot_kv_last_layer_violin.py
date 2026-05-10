import json
import zipfile
import tempfile
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 스크립트 실행용, notebook이면 없어도 됨
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib import font_manager


# =========================================================
# 1) exp_util.py 템플릿 내용을 직접 옮겨서 사용 (import 금지)
# =========================================================

# Color palette
AI_PAPER_PALETTE = [
    "#C71F19",  # Accent Red
    "#4BACC6",  # Accent Aqua
    "#8064A2",  # Accent Purple
    "#4F81BD",  # Accent Blue
    "#9BBB59",  # Accent Green
    "#F79646",  # Accent Orange
    "#000000",  # Black
]

CONTEXT_COLOR = "#C71F19"
PROMPT_COLOR = "#4BACC6"
COMPLETION_COLOR = "#8064A2"

ACCENT_BLACK = "#000000"
AI_PAPER_BORDER_COLOR = "0.20"
AI_PAPER_LEGEND_EDGE_COLOR = "0.85"
AI_PAPER_GRID_ALPHA = 0.32
AI_PAPER_GRID_LINESTYLE = "--"
AI_PAPER_GRID_LINE_WIDTH = 0.6
AI_PAPER_MARKER_EDGE_WIDTH = 1.5

# Typography
AI_PAPER_MAJOR_FONT_SIZE = 12
AI_PAPER_MINOR_FONT_SIZE = 10
AI_PAPER_AXIS_LABEL_SIZE = AI_PAPER_MAJOR_FONT_SIZE
AI_PAPER_LEGEND_FONT_SIZE = AI_PAPER_MAJOR_FONT_SIZE
AI_PAPER_TICK_LABEL_SIZE = AI_PAPER_MINOR_FONT_SIZE
AI_PAPER_MATH_FONTSET = "stix"

TIMES_NEW_ROMAN_FONT_FAMILY = "Times New Roman"

# Figure size: 1881 x 1291 px at 300 DPI
FIG_DPI = 300
FIG_W_INCH = 1881 / FIG_DPI
FIG_H_INCH = 1291 / FIG_DPI

# Try to use Times New Roman if available
available_fonts = {f.name for f in font_manager.fontManager.ttflist}
font_family = TIMES_NEW_ROMAN_FONT_FAMILY if TIMES_NEW_ROMAN_FONT_FAMILY in available_fonts else "DejaVu Serif"

plt.rcParams.update({
    "font.family": font_family,
    "mathtext.fontset": AI_PAPER_MATH_FONTSET,
    "figure.dpi": FIG_DPI,
    "savefig.dpi": FIG_DPI,

    "axes.labelsize": AI_PAPER_AXIS_LABEL_SIZE,
    "axes.titlesize": AI_PAPER_AXIS_LABEL_SIZE,
    "legend.fontsize": AI_PAPER_LEGEND_FONT_SIZE,
    "xtick.labelsize": AI_PAPER_TICK_LABEL_SIZE,
    "ytick.labelsize": AI_PAPER_TICK_LABEL_SIZE,

    "axes.linewidth": 1.0,
})


# =========================================================
# 2) 데이터 로딩
# =========================================================
def load_last_layer_similarity_from_zip(zip_path: str):
    """
    zip 내부의 각 injection_layer_start_idx_xxx 폴더에서
    A_to_B_full_mix_vs_native_kv_similarity_metadata.json을 읽고,
    mean_similarity_matrix의 마지막 row만 취합한다.

    반환 형식:
    {
        0: {"Context": np.array(...), "Prompt": np.array(...), "Completion": np.array(...)},
        1: {...},
        ...
        6: {...}
    }
    """
    results = {}

    with zipfile.ZipFile(zip_path, "r") as zf:
        with tempfile.TemporaryDirectory() as tmpdir:
            zf.extractall(tmpdir)
            root = Path(tmpdir)

            meta_paths = sorted(
                root.glob("injection_layer_start_idx_*/A_to_B_full_mix_vs_native_kv_similarity_metadata.json")
            )

            if len(meta_paths) == 0:
                raise FileNotFoundError("metadata json 파일을 찾지 못했습니다.")

            for meta_path in meta_paths:
                folder_name = meta_path.parent.name  # injection_layer_start_idx_000
                layer_idx = int(folder_name.split("_")[-1])

                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)

                sim_matrix = np.asarray(meta["mean_similarity_matrix"], dtype=float)
                last_layer_values = sim_matrix[-1]   # 마지막 layer index만 사용

                seg_counts = meta["segment_group_counts"]
                n_prefix = seg_counts["prefix"]
                n_suffix = seg_counts["suffix"]
                n_generated = seg_counts["generated"]

                # Prefix / Observed Suffix / Generated Suffix 분리
                # 요청 용어로 변경: Context / Prompt / Completion
                idx0 = 0
                idx1 = idx0 + n_prefix
                idx2 = idx1 + n_suffix
                idx3 = idx2 + n_generated

                context_vals = last_layer_values[idx0:idx1]
                prompt_vals = last_layer_values[idx1:idx2]
                completion_vals = last_layer_values[idx2:idx3]

                results[layer_idx] = {
                    "Context": context_vals,
                    "Prompt": prompt_vals,
                    "Completion": completion_vals,
                }

    results = dict(sorted(results.items(), key=lambda x: x[0]))
    return results


# =========================================================
# 3) Violin plot 그리기
# =========================================================
def draw_grouped_violin_plot(data_dict, save_path: str):
    """
    x축: 7개의 figure (injection_layer_start_idx 0~6)
    각 x tick마다 Context / Prompt / Completion 3개의 violin을 나란히 그림
    legend는 figure 상단에 고정 배치하고, plot과 겹치지 않도록
    figure 레벨에서 관리한다.
    """
    fig, ax = plt.subplots(figsize=(FIG_W_INCH, FIG_H_INCH), dpi=FIG_DPI)

    layer_indices = list(data_dict.keys())   # [0,1,2,3,4,5,6]
    x_base = np.arange(len(layer_indices))

    categories = ["Context", "Prompt", "Completion"]
    category_colors = {
        "Context": CONTEXT_COLOR,
        "Prompt": PROMPT_COLOR,
        "Completion": COMPLETION_COLOR,
    }

    # 한 tick 안에서 3개 violin 위치
    offsets = {
        "Context": -0.24,
        "Prompt": 0.00,
        "Completion": 0.24,
    }
    violin_width = 0.22

    for category in categories:
        values_per_layer = [data_dict[idx][category] for idx in layer_indices]
        positions = x_base + offsets[category]

        vp = ax.violinplot(
            dataset=values_per_layer,
            positions=positions,
            widths=violin_width,
            showmeans=False,
            showmedians=False,
            showextrema=False,
            bw_method=0.65,
        )

        for body in vp["bodies"]:
            body.set_facecolor(category_colors[category])
            body.set_edgecolor(AI_PAPER_BORDER_COLOR)
            body.set_linewidth(0.8)
            body.set_alpha(0.85)

        if category == "Context":
            means = [np.mean(v) for v in values_per_layer]
            
            # ax.scatter(
            #     positions,
            #     means,
            #     s=10,                 # 원 크기 더 작게
            #     facecolors="white",   # 빈 원
            #     edgecolors="black",   # 테두리 검정
            #     linewidths=0.8,       # 선을 얇게
            #     zorder=4,
            # )
            
            ax.plot(
                positions,
                means,
                linestyle="--",
                color=CONTEXT_COLOR,    # "#BFBFBF", CONTEXT_COLOR
                linewidth=1.5,
                zorder=3,
            )

        # # mean: 가운데가 빈 원(hollow circle)
        # means = [np.mean(v) for v in values_per_layer]
        # ax.scatter(
        #     positions,
        #     means,
        #     s=18,                  # 원 크기 축소
        #     facecolors="white",    # 가운데는 비움
        #     edgecolors="black",    # 원 테두리를 검정색으로 변경
        #     linewidths=1.2,        # 테두리 두께 (원하면 더 줄이거나 키울 수 있음)
        #     zorder=3,
        # )

    # Axes
    ax.set_xticks(x_base)
    ax.set_xticklabels([str(idx) for idx in layer_indices])

    ax.set_xlabel("First Layer Index of Translation Channels", fontweight="bold")
    ax.set_ylabel("Cosine Similarity(Native vs Translation)", fontweight="bold")

    # Grid / spine
    ax.grid(
        axis="y",
        linestyle=AI_PAPER_GRID_LINESTYLE,
        linewidth=AI_PAPER_GRID_LINE_WIDTH,
        alpha=AI_PAPER_GRID_ALPHA
    )
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_color(AI_PAPER_BORDER_COLOR)
        spine.set_linewidth(1.0)

    ax.margins(x=0.03, y=0.08)

    # -----------------------------
    # figure-level legend (stable)
    # -----------------------------
    legend_handles = [
        Patch(facecolor=CONTEXT_COLOR, edgecolor=AI_PAPER_BORDER_COLOR, label="Context", alpha=0.85),
        Patch(facecolor=PROMPT_COLOR, edgecolor=AI_PAPER_BORDER_COLOR, label="Prompt", alpha=0.85),
        Patch(facecolor=COMPLETION_COLOR, edgecolor=AI_PAPER_BORDER_COLOR, label="Completion", alpha=0.85),
        # Line2D(
        #     [0], [0],
        #     marker="o",
        #     linestyle="None",
        #     markerfacecolor="white",
        #     markeredgecolor="black",   # 검정색 테두리
        #     markeredgewidth=1.2,
        #     markersize=4.5,            # legend 원 크기도 조금 축소
        #     label="Mean"
        # ),
    ]

    leg = fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),   # figure 기준 상단 중앙
        ncol=3,                       # 가로 배치
        frameon=True,
        columnspacing=1.2,
        handletextpad=0.5,
        borderaxespad=0.2,
    )

    leg.get_frame().set_edgecolor(AI_PAPER_LEGEND_EDGE_COLOR)
    leg.get_frame().set_linewidth(0.8)

    # -----------------------------
    # 상단 공간을 legend용으로 예약
    # -----------------------------
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.90])

    # exact output size 유지
    fig.savefig(save_path, dpi=FIG_DPI)
    plt.close(fig)

    print(f"Saved figure to: {save_path}")
    print(f"Figure size (inch): {FIG_W_INCH:.4f} x {FIG_H_INCH:.4f}")
    print(f"Expected pixel size: 1881 x 1291 @ {FIG_DPI} DPI")


# =========================================================
# 4) 실행
# =========================================================
if __name__ == "__main__":
    zip_path = "./layer_position_outputs.zip"   # 필요시 절대경로로 수정
    save_path = "./kv_last_layer_violin_plot.pdf"

    data_dict = load_last_layer_similarity_from_zip(zip_path)
    draw_grouped_violin_plot(data_dict, save_path)