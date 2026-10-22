import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

from src.data_loader import person_level_group_kfold
from src.evaluation.plots import BAR_COLOR, _save_figure
from src.features.correlation_filter import CorrelationFilter
from src.utils.logger import PROCESSED_DIR, TABLE_DIR, ensure_output_dirs, logger

META_COLS = ["ID", "Age", "Sex", "Class", "baseline_split"]
N_SUBJECTS = 219
BASELINE_TRAIN = "train"
MATRIX_NAME = "fused_egemaps_ddk_vsa_train"
N_SPLITS = 5
THRESHOLDS = np.round(np.linspace(0.0, 1.0, 21), 2)


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


def _prepare_fold(X_train: np.ndarray):
    X = _finite_or_nan(X_train)
    X = SimpleImputer(strategy="median").fit_transform(X)
    n_after_impute = X.shape[1]
    vt = VarianceThreshold(threshold=0.0)
    X = vt.fit_transform(X)
    n_after_var = X.shape[1]
    return X, n_after_impute, n_after_var


def run_nfeat_curve(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    feat_cols = _feature_cols(df)
    n_input = len(feat_cols)
    logger.info("Pearson |r| remaining-feature curve")
    logger.info("Matrix: %s", MATRIX_NAME)
    logger.info("Subjects: %s", len(df))
    logger.info("Input features: %s", n_input)
    logger.info("n_splits: %s", N_SPLITS)
    logger.info("thresholds: %s", THRESHOLDS.tolist())

    rows = []
    for fold, train_df, _val_df in person_level_group_kfold(df, n_splits=N_SPLITS):
        X_train = train_df[feat_cols].to_numpy(dtype=float)
        X_ready, n_after_impute, n_after_var = _prepare_fold(X_train)
        logger.info(
            "fold %s  n_train=%s  after_impute=%s  after_var=%s",
            fold,
            len(train_df),
            n_after_impute,
            n_after_var,
        )
        for tau in THRESHOLDS:
            filt = CorrelationFilter(threshold=float(tau))
            filt.fit(X_ready)
            n_kept = int(np.sum(filt.keep_mask_))
            rows.append(
                {
                    "fold": fold,
                    "threshold": float(tau),
                    "n_input": n_input,
                    "n_after_impute": n_after_impute,
                    "n_after_var": n_after_var,
                    "n_features": n_kept,
                }
            )

    fold_df = pd.DataFrame(rows)
    summary_rows = []
    for tau, sub in fold_df.groupby("threshold", sort=True):
        summary_rows.append(
            {
                "threshold": float(tau),
                "n_features_mean": float(sub["n_features"].mean()),
                "n_features_std": float(sub["n_features"].std(ddof=1)),
                "n_features_min": int(sub["n_features"].min()),
                "n_features_max": int(sub["n_features"].max()),
                "n_input": int(sub["n_input"].iloc[0]),
                "n_after_impute_mean": float(sub["n_after_impute"].mean()),
                "n_after_var_mean": float(sub["n_after_var"].mean()),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    return fold_df, summary_df


def save_tables(fold_df: pd.DataFrame, summary_df: pd.DataFrame):
    ensure_output_dirs()
    fold_path = TABLE_DIR / "corr_nfeat_curve_fold.csv"
    summary_path = TABLE_DIR / "corr_nfeat_curve_summary.csv"
    fold_df.to_csv(fold_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    logger.info("Saved %s", fold_path)
    logger.info("Saved %s", summary_path)


def plot_nfeat_curve(fold_df: pd.DataFrame, summary_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for fold, sub in fold_df.groupby("fold"):
        ax.plot(
            sub["threshold"],
            sub["n_features"],
            color="#BBBBBB",
            linewidth=0.9,
            alpha=0.9,
            label="Training fold" if fold == 0 else None,
        )
    ax.plot(
        summary_df["threshold"],
        summary_df["n_features_mean"],
        color=BAR_COLOR,
        linewidth=2.0,
        label="Fold mean",
    )
    ax.fill_between(
        summary_df["threshold"],
        summary_df["n_features_mean"] - summary_df["n_features_std"],
        summary_df["n_features_mean"] + summary_df["n_features_std"],
        color=BAR_COLOR,
        alpha=0.18,
        linewidth=0,
        label="Fold SD",
    )
    n_input = int(summary_df["n_input"].iloc[0])
    ax.set_xlabel("Pearson |r| threshold")
    ax.set_ylabel("Number of remaining features")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0, n_input * 1.05)
    ax.set_xticks(np.round(np.linspace(0.0, 1.0, 11), 1))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    _save_figure(fig, "corr_nfeat_curve")


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
    fold_df, summary_df = run_nfeat_curve(df)
    save_tables(fold_df, summary_df)
    plot_nfeat_curve(fold_df, summary_df)
    logger.info(
        "n_features at 0.80: mean=%.1f  at 0.90: mean=%.1f  at 1.00: mean=%.1f",
        float(summary_df.loc[summary_df["threshold"] == 0.80, "n_features_mean"].iloc[0]),
        float(summary_df.loc[summary_df["threshold"] == 0.90, "n_features_mean"].iloc[0]),
        float(summary_df.loc[summary_df["threshold"] == 1.00, "n_features_mean"].iloc[0]),
    )
