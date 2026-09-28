import numpy as np
import matplotlib.pyplot as plt

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
BAR_COLOR = "#4C72B0"


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
        color=BAR_COLOR,
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
