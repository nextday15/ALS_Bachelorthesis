"""Twenty person-level CV partitions for the latest searched hyperparameters.

SVM uses the searched linear kernel and gamma=scale, with C set to 0.01.
RF, XGBoost, and ordinal logistic regression use their latest searched values.
The seed is passed only to StratifiedGroupKFold(shuffle=True). Classifier
random_state stays at 42. ANOVA and Pearson stay inside each training fold.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold

from src.evaluation.metrics import CLASSES, classification_scores
from src.evaluation.seed_stability_plot import plot_seed_stability
from src.models.pipelines import RANDOM_STATE, build_pipelines
from src.models.train_baseline import (
    ANOVA_F,
    CORR_RANK,
    CORR_THRESHOLD,
    N_SPLITS,
    N_SUBJECTS,
    USE_CORR_FILTER,
    _feature_cols,
    _fit_predict,
    _xy,
    load_matrix,
)
from src.utils.logger import TABLE_DIR, ensure_output_dirs, logger

N_SEEDS = 20
OUT_DIR = TABLE_DIR / "4_hparam"
SCORE_PATH = OUT_DIR / "hparam_seed_macro_f1.csv"
SUMMARY_PATH = OUT_DIR / "hparam_seed_summary.csv"

MODELS = ("svm", "rf", "xgb", "ordinal")
PARAM_OVERRIDES = {
    "svm": {"clf__kernel": "linear", "clf__C": 0.01, "clf__gamma": "scale"},
    "rf": {
        "clf__n_estimators": 500,
        "clf__max_depth": 5,
        "clf__min_samples_leaf": 5,
    },
    "xgb": {
        "clf__n_estimators": 100,
        "clf__max_depth": 8,
        "clf__learning_rate": 0.05,
        "clf__subsample": 1.0,
        "clf__colsample_bytree": 0.5,
    },
    "ordinal": {"clf__alpha": 100},
}
METRIC_COLS = ["n_features", "macro_f1", "accuracy", "macro_recall", "mae"]
for _class in CLASSES:
    METRIC_COLS.extend([f"recall_{_class}", f"f1_{_class}"])


def _iter_folds(df: pd.DataFrame, seed: int):
    y = df["Class"]
    groups = df["ID"]
    sgkf = StratifiedGroupKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=seed
    )
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(df, y, groups=groups)):
        yield fold, df.iloc[train_idx].copy(), df.iloc[val_idx].copy()


def _partition(df: pd.DataFrame, seed: int) -> tuple:
    return tuple(
        tuple(sorted(val_df["ID"].astype(str)))
        for _, _, val_df in _iter_folds(df, seed)
    )


def _templates():
    base = build_pipelines(
        random_state=RANDOM_STATE,
        anova_f=ANOVA_F,
        use_corr_filter=USE_CORR_FILTER,
        corr_threshold=CORR_THRESHOLD,
        corr_rank=CORR_RANK,
        tuned=True,
    )
    templates = []
    for name in MODELS:
        pipe = clone(base[name])
        pipe.set_params(**PARAM_OVERRIDES[name])
        templates.append((name, pipe))
    return templates


def _check_templates(templates):
    by_name = {name: pipe for name, pipe in templates}
    svm = by_name["svm"].get_params()
    ordinal = by_name["ordinal"].get_params()
    rf = by_name["rf"].get_params()
    xgb = by_name["xgb"].get_params()
    if (
        svm["clf__kernel"] != "linear"
        or svm["clf__C"] != 0.01
        or svm["clf__gamma"] != "scale"
        or svm["clf__random_state"] != RANDOM_STATE
    ):
        raise RuntimeError("SVM seed test is not linear C=0.01, gamma=scale")
    if ordinal["clf__alpha"] != 100:
        raise RuntimeError("ordinal alpha is not 100")
    if (
        rf["clf__n_estimators"] != 500
        or rf["clf__max_depth"] != 5
        or rf["clf__min_samples_leaf"] != 5
        or rf["clf__random_state"] != RANDOM_STATE
    ):
        raise RuntimeError("RF is not the latest searched model")
    if (
        xgb["clf__n_estimators"] != 100
        or xgb["clf__max_depth"] != 8
        or xgb["clf__learning_rate"] != 0.05
        or xgb["clf__subsample"] != 1.0
        or xgb["clf__colsample_bytree"] != 0.5
    ):
        raise RuntimeError("XGBoost is not the latest searched model")


def _choose_seeds(n: int = N_SEEDS) -> tuple[int, ...]:
    chosen = np.random.default_rng().choice(
        np.arange(1, 1_000_000), size=n, replace=False
    )
    return tuple(int(seed) for seed in chosen)


def _check_partitions(df: pd.DataFrame, seeds: tuple[int, ...]):
    parts = {seed: _partition(df, seed) for seed in seeds}
    if len(set(parts.values())) != len(seeds):
        raise RuntimeError("CV seeds did not produce distinct partitions")
    unshuffled = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=False)
    fixed = tuple(
        tuple(sorted(df.iloc[val_idx]["ID"].astype(str)))
        for _, val_idx in unshuffled.split(df, df["Class"], groups=df["ID"])
    )
    if fixed in parts.values():
        raise RuntimeError("a shuffled seed reproduced the unshuffled split")


def run_seed_stability(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(df) != N_SUBJECTS or (df["baseline_split"] != "train").any():
        raise RuntimeError("seed stability must use the 219 training subjects")
    feat_cols = _feature_cols(df)
    seeds = _choose_seeds()
    templates = _templates()
    _check_templates(templates)
    _check_partitions(df, seeds)
    logger.info("Seeds: %s", seeds)
    logger.info("Classifier random_state: %s", RANDOM_STATE)
    logger.info("anova_f: %s", ANOVA_F)
    logger.info("corr_threshold: %s", CORR_THRESHOLD)

    rows = []
    for seed in seeds:
        folds = list(_iter_folds(df, seed))
        for name, template in templates:
            fold_rows = []
            for _, train_df, val_df in folds:
                X_train, y_train = _xy(train_df, feat_cols)
                X_val, y_val = _xy(val_df, feat_cols)
                y_pred, n_kept = _fit_predict(
                    clone(template), name, X_train, y_train, X_val
                )
                fold_rows.append(
                    {
                        "n_features": n_kept,
                        **classification_scores(y_val, y_pred),
                    }
                )
            seed_row = {"model": name, "seed": seed}
            for col in METRIC_COLS:
                seed_row[col] = float(np.mean([row[col] for row in fold_rows]))
            rows.append(seed_row)
            logger.info(
                "%s seed=%s macro_f1=%.3f acc=%.3f recall=%.3f mae=%.3f",
                name,
                seed,
                seed_row["macro_f1"],
                seed_row["accuracy"],
                seed_row["macro_recall"],
                seed_row["mae"],
            )

    score_df = pd.DataFrame(rows)
    summary_rows = []
    for name in MODELS:
        sub = score_df.loc[score_df["model"] == name]
        summary = {"model": name}
        for col in METRIC_COLS:
            summary[f"{col}_mean"] = float(sub[col].mean())
            summary[f"{col}_std"] = float(sub[col].std(ddof=1))
        summary_rows.append(summary)
    return score_df[["model", "seed", "macro_f1"]], pd.DataFrame(summary_rows)


def save_seed_stability(score_df: pd.DataFrame, summary_df: pd.DataFrame):
    ensure_output_dirs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    score_df.to_csv(SCORE_PATH, index=False)
    summary_df.to_csv(SUMMARY_PATH, index=False)
    logger.info("Saved %s", SCORE_PATH)
    logger.info("Saved %s", SUMMARY_PATH)


if __name__ == "__main__":
    matrix = load_matrix("fused_egemaps_ddk_vsa_train")
    scores, summary = run_seed_stability(matrix)
    save_seed_stability(scores, summary)
    plot_seed_stability(scores)
    for row in summary.itertuples(index=False):
        logger.info(
            "%s  macro_f1=%.3f  acc=%.3f  recall=%.3f  mae=%.3f",
            row.model,
            row.macro_f1_mean,
            row.accuracy_mean,
            row.macro_recall_mean,
            row.mae_mean,
        )
