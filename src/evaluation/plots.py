import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

from src.data_loader import load_task_labels
from src.evaluation.metrics import CLASSES
from src.utils.logger import FIGURE_DIR, ensure_output_dirs, logger

CLASS_TICKS = [
    "1\nSevere",
    "2\nModerate",
    "3\nMild",
    "4\nNo dysarthria",
    "5\nHealthy",
]
CLASS_LABELS = [
    "1 Severe",
    "2 Moderate",
    "3 Mild",
    "4 No dysarthria",
    "5 Healthy",
]
BAR_COLOR = "#4C72B0"


def _blue_gradient(n, base=BAR_COLOR):
    rgb = np.array(to_rgb(base))
    white = np.ones(3)
    mix = np.linspace(0.35, 1.0, n)
    return [tuple(white * (1.0 - a) + rgb * a) for a in mix]


def _baseline_train_counts():
    df = load_task_labels(1, "train")
    train = df.loc[df["baseline_split"] == "train"]
    return train["Class"].value_counts().reindex(CLASSES, fill_value=0).astype(int)


def _save_figure(fig, name: str):
    ensure_output_dirs()
    pdf_path = FIGURE_DIR / f"{name}.pdf"
    png_path = FIGURE_DIR / f"{name}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    logger.info("Saved %s and %s", pdf_path, png_path)
    plt.close(fig)


def plot_class_distribution():
    counts = _baseline_train_counts()
    x = np.arange(len(CLASSES))
    heights = [int(counts[c]) for c in CLASSES]
    n = int(counts.sum())

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    bars = ax.bar(
        x,
        heights,
        width=0.62,
        color=_blue_gradient(len(CLASSES)),
        edgecolor="white",
        linewidth=0.4,
    )
    for bar, h in zip(bars, heights):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 1.2,
            str(h),
            ha="center",
            va="bottom",
            fontsize=9,
            color="#222222",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_TICKS)
    ax.set_xlabel("Severity class")
    ax.set_ylabel("Number of subjects")
    ax.set_ylim(0, max(heights) * 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)
    logger.info("Baseline training class counts (N = %s): %s", n, heights)
    fig.tight_layout()
    _save_figure(fig, "class_distribution")
    return counts


def plot_class_distribution_pie():
    counts = _baseline_train_counts()
    sizes = [int(counts[c]) for c in CLASSES]
    n = int(sum(sizes))
    colors = _blue_gradient(len(CLASSES))

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    wedges, _, autotexts = ax.pie(
        sizes,
        colors=colors,
        startangle=90,
        counterclock=False,
        autopct=lambda pct: f"{pct:.1f}%",
        pctdistance=0.62,
        wedgeprops={"edgecolor": "white", "linewidth": 0.8},
        textprops={"fontsize": 9, "color": "#222222"},
    )
    for text in autotexts:
        text.set_color("#222222")
    ax.legend(
        wedges,
        CLASS_LABELS,
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
    )
    ax.set_aspect("equal")
    logger.info("Baseline training class percentages (N = %s): %s", n, sizes)
    fig.tight_layout()
    _save_figure(fig, "class_distribution_pie")
    return counts


if __name__ == "__main__":
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )
    plot_class_distribution()
    plot_class_distribution_pie()
