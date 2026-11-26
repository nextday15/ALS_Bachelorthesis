import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.evaluation.plots import BAR_COLOR, _save_figure
from src.utils.logger import TABLE_DIR

MODELS = ("svm", "rf", "xgb", "ordinal")
MODEL_LABELS = {
    "svm": "SVM",
    "rf": "RF",
    "xgb": "XGBoost",
    "ordinal": "Ordinal",
}
SCORE_PATH = TABLE_DIR / "4_hparam" / "hparam_seed_macro_f1.csv"


def plot_seed_stability(score_df: pd.DataFrame | None = None):
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )
    if score_df is None:
        score_df = pd.read_csv(SCORE_PATH)
    missing = [name for name in MODELS if name not in set(score_df["model"])]
    if missing:
        raise ValueError(f"seed scores missing models: {missing}")

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    values = [
        score_df.loc[score_df["model"] == name, "macro_f1"].to_numpy(dtype=float)
        for name in MODELS
    ]
    boxes = ax.boxplot(
        values,
        positions=np.arange(len(MODELS)),
        widths=0.55,
        patch_artist=True,
        medianprops={"color": "white", "linewidth": 1.6},
        whiskerprops={"color": BAR_COLOR, "linewidth": 1.2},
        capprops={"color": BAR_COLOR, "linewidth": 1.2},
        flierprops={
            "marker": "o",
            "markerfacecolor": BAR_COLOR,
            "markeredgecolor": "white",
            "markersize": 5,
        },
        boxprops={"facecolor": BAR_COLOR, "edgecolor": BAR_COLOR, "linewidth": 1.0},
    )
    del boxes

    ax.set_xticks(np.arange(len(MODELS)))
    ax.set_xticklabels([MODEL_LABELS[name] for name in MODELS])
    ax.set_xlabel("Model")
    ax.set_ylabel("Macro-F1")
    ax.set_xlim(-0.55, len(MODELS) - 0.45)
    y_min = float(score_df["macro_f1"].min())
    y_max = float(score_df["macro_f1"].max())
    ax.set_ylim(y_min - 0.02, y_max + 0.02)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save_figure(fig, "hparam_seed_stability")


if __name__ == "__main__":
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )
    plot_seed_stability()
