"""Locked hyperparameters on five person-level CV partitions.

The seed is passed only to StratifiedGroupKFold(shuffle=True). Classifier
random_state stays at 42. ANOVA and Pearson stay inside each training fold.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold

from src.evaluation.metrics import classification_scores
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

SEEDS = (6, 42, 75, 715, 2004)
OUT_DIR = TABLE_DIR / "4_hparam"
SCORE_PATH = OUT_DIR / "hparam_seed_macro_f1.csv"
SUMMARY_PATH = OUT_DIR / "hparam_seed_summary.csv"

MODELS = ("svm", "rf", "xgb", "ordinal")


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
    return [(name, clone(base[name])) for name in MODELS]


def _check_templates(templates):
    by_name = {name: pipe for name, pipe in templates}
    svm = by_name["svm"].get_params()
    ordinal = by_name["ordinal"].get_params()
    rf = by_name["rf"].get_params()
    xgb = by_name["xgb"].get_params()
    if svm["clf__kernel"] != "linear" or svm["clf__C"] != 0.1:
        raise RuntimeError("locked SVM is not linear C=0.1")
    if ordinal["clf__alpha"] != 100:
        raise RuntimeError("locked ordinal alpha is not 100")
    if rf["clf__max_depth"] != 10 or rf["clf__random_state"] != RANDOM_STATE:
        raise RuntimeError("RF is not the locked depth-10 model at seed 42")
    if xgb["clf__max_depth"] != 6 or xgb["clf__learning_rate"] != 0.05:
        raise RuntimeError("locked XGBoost hyperparameters changed")


def _check_partitions(df: pd.DataFrame):
    parts = {seed: _partition(df, seed) for seed in SEEDS}
    if len(set(parts.values())) != len(SEEDS):
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
    templates = _templates()
    _check_templates(templates)
    _check_partitions(df)
    logger.info("Seeds: %s", SEEDS)
    logger.info("Classifier random_state: %s", RANDOM_STATE)
    logger.info("anova_f: %s", ANOVA_F)
    logger.info("corr_threshold: %s", CORR_THRESHOLD)

    rows = []
    for seed in SEEDS:
        folds = list(_iter_folds(df, seed))
        for name, template in templates:
            fold_scores = []
            for _, train_df, val_df in folds:
                X_train, y_train = _xy(train_df, feat_cols)
                X_val, y_val = _xy(val_df, feat_cols)
                y_pred, n_kept = _fit_predict(
                    clone(template), name, X_train, y_train, X_val
                )
                fold_scores.append(classification_scores(y_val, y_pred)["macro_f1"])
                logger.info(
                    "  %s seed=%s fold features kept=%s",
                    name,
                    seed,
                    n_kept,
                )
            macro_f1 = float(np.mean(fold_scores))
            rows.append({"model": name, "seed": seed, "macro_f1": macro_f1})
            logger.info("%s seed=%s macro_f1=%.3f", name, seed, macro_f1)

    score_df = pd.DataFrame(rows)
    summary_rows = []
    for name in MODELS:
        values = score_df.loc[score_df["model"] == name, "macro_f1"]
        summary_rows.append(
            {
                "model": name,
                "macro_f1_mean": float(values.mean()),
                "macro_f1_std": float(values.std(ddof=1)),
            }
        )
    return score_df, pd.DataFrame(summary_rows)


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
    for row in summary.itertuples(index=False):
        logger.info(
            "%s  mean=%.3f  std=%.3f",
            row.model,
            row.macro_f1_mean,
            row.macro_f1_std,
        )
