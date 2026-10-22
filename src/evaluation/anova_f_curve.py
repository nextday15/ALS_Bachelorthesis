import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

from src.data_loader import person_level_group_kfold
from src.evaluation.plots import BAR_COLOR, _save_figure
from src.utils.logger import PROCESSED_DIR, TABLE_DIR, ensure_output_dirs, logger

META_COLS = ["ID", "Age", "Sex", "Class", "baseline_split"]
N_SUBJECTS = 219
BASELINE_TRAIN = "train"
MATRIX_NAME = "fused_egemaps_ddk_vsa_train"
N_SPLITS = 5


def _finite_or_nan(X):
    arr = np.asarray(X, dtype=float)
    return np.where(np.isfinite(arr), arr, np.nan)


def load_matrix() -> pd.DataFrame:
    path = PROCESSED_DIR / f"{MATRIX_NAME}.csv"
    if not path.exists():
        raise FileNotFoundError(f"did not find {path}")
    df = pd.read_csv(path)
    if "baseline_split" not in df.columns:
        raise ValueError("baseline_split missing")
    if (df["baseline_split"] != BASELINE_TRAIN).any():
        raise ValueError("validation rows leaked into baseline train matrix")
    if df["ID"].nunique() != N_SUBJECTS or len(df) != N_SUBJECTS:
        raise ValueError(
            f"expected {N_SUBJECTS} unique subjects, got "
            f"{df['ID'].nunique()} IDs / {len(df)} rows"
        )
    if df["ID"].duplicated().any():
        raise ValueError("duplicate IDs in fused matrix")
    return df


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS]


def _prepare_fold(X_train: np.ndarray, names: np.ndarray):
    X = _finite_or_nan(X_train)
    X = SimpleImputer(strategy="median").fit_transform(X)
    vt = VarianceThreshold(threshold=0.0)
    X = vt.fit_transform(X)
    kept_names = names[vt.get_support()]
    return X, kept_names


def run_anova_curve(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feat_cols = np.asarray(_feature_cols(df))
    logger.info("ANOVA F-score vs feature rank")
    logger.info("Matrix: %s", MATRIX_NAME)
    logger.info("Subjects: %s", len(df))
    logger.info("Input features: %s", len(feat_cols))
    logger.info("n_splits: %s", N_SPLITS)

    rank_rows = []
    feature_rows = []
    for fold, train_df, _val_df in person_level_group_kfold(df, n_splits=N_SPLITS):
        X_train = train_df[feat_cols].to_numpy(dtype=float)
        y_train = train_df["Class"].to_numpy(dtype=int)
        X_ready, names = _prepare_fold(X_train, feat_cols)
        F, p = f_classif(X_ready, y_train)
        F = np.asarray(F, dtype=float)
        p = np.asarray(p, dtype=float)
        F = np.where(np.isfinite(F), F, np.nan)
        p = np.where(np.isfinite(p), p, np.nan)
        order = np.argsort(-np.nan_to_num(F, nan=-np.inf), kind="mergesort")
        logger.info(
            "fold %s  n_train=%s  n_feat=%s  F_max=%.3f  F_median=%.3f",
            fold,
            len(train_df),
            X_ready.shape[1],
            float(np.nanmax(F)),
            float(np.nanmedian(F)),
        )
        for rank, idx in enumerate(order, start=1):
            rank_rows.append(
                {
                    "fold": fold,
                    "rank": rank,
                    "n_features_kept": rank,
                    "f_score": F[idx],
                    "p_value": p[idx],
                    "feature": names[idx],
                }
            )
            feature_rows.append(
                {
                    "fold": fold,
                    "feature": names[idx],
                    "f_score": F[idx],
                    "p_value": p[idx],
                    "rank": rank,
                }
            )

    rank_df = pd.DataFrame(rank_rows)
    feature_fold_df = pd.DataFrame(feature_rows)
    summary_rows = []
    for rank, sub in rank_df.groupby("rank", sort=True):
        summary_rows.append(
            {
                "rank": int(rank),
                "n_features_kept": int(rank),
                "f_score_mean": float(sub["f_score"].mean()),
                "f_score_std": float(sub["f_score"].std(ddof=1)),
                "p_value_mean": float(sub["p_value"].mean()),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    feature_summary = (
        feature_fold_df.groupby("feature", sort=False)
        .agg(
            f_score_mean=("f_score", "mean"),
            f_score_std=("f_score", "std"),
            p_value_mean=("p_value", "mean"),
            rank_mean=("rank", "mean"),
        )
        .reset_index()
        .sort_values("f_score_mean", ascending=False)
    )
    return rank_df, summary_df, feature_summary


def save_tables(rank_df, summary_df, feature_summary):
    ensure_output_dirs()
    rank_path = TABLE_DIR / "anova_f_by_rank_fold.csv"
    summary_path = TABLE_DIR / "anova_f_by_rank_summary.csv"
    feature_path = TABLE_DIR / "anova_f_by_feature.csv"
    rank_df.to_csv(rank_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    feature_summary.to_csv(feature_path, index=False)
    logger.info("Saved %s", rank_path)
    logger.info("Saved %s", summary_path)
    logger.info("Saved %s", feature_path)


def plot_anova_curve(rank_df: pd.DataFrame, summary_df: pd.DataFrame):
    del summary_df
    f_max = float(np.nanmax(rank_df["f_score"]))
    grid = np.linspace(0.0, f_max, 250)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    remaining_rows = []
    for fold, sub in rank_df.groupby("fold"):
        F = sub["f_score"].to_numpy(dtype=float)
        remaining = np.array([(F >= t).sum() for t in grid], dtype=float)
        remaining_rows.append(remaining)
        ax.plot(
            grid,
            remaining,
            color="#BBBBBB",
            linewidth=0.8,
            alpha=0.9,
            label="Training fold" if fold == 0 else None,
        )
    remaining_mat = np.vstack(remaining_rows)
    mean = remaining_mat.mean(axis=0)
    std = remaining_mat.std(axis=0, ddof=1)
    ax.plot(grid, mean, color=BAR_COLOR, linewidth=2.0, label="Fold mean")
    ax.fill_between(
        grid,
        mean - std,
        mean + std,
        color=BAR_COLOR,
        alpha=0.18,
        linewidth=0,
        label="Fold SD",
    )
    n_feat = int(rank_df.groupby("fold").size().max())
    y_top = int(np.ceil(n_feat / 50) * 50)
    ax.set_xlabel("ANOVA F-score")
    ax.set_ylabel("Number of remaining features")
    ax.set_xlim(0.0, f_max)
    ax.set_ylim(0, y_top)
    ax.set_yticks(np.arange(0, y_top + 50, 50))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    _save_figure(fig, "anova_f_curve")


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
    df = load_matrix()
    rank_df, summary_df, feature_summary = run_anova_curve(df)
    save_tables(rank_df, summary_df, feature_summary)
    plot_anova_curve(rank_df, summary_df)
    top = feature_summary.head(5)
    logger.info("Top 5 features by mean ANOVA F:")
    for _, row in top.iterrows():
        logger.info(
            "  %s  F=%.3f  rank_mean=%.1f",
            row["feature"],
            row["f_score_mean"],
            row["rank_mean"],
        )
