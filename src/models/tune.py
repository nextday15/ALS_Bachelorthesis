import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedGroupKFold

from src.evaluation.metrics import CLASSES, classification_scores
from src.models.experiment_record import build_hparam_record, write_hparam_record
from src.models.param_grids import param_grid
from src.models.pipelines import RANDOM_STATE, SEARCH_MODELS, build_search_pipeline
from src.models.train_baseline import (
    BASELINE_TRAIN,
    EXPERIMENTS,
    N_SUBJECTS,
    _feature_cols,
    _xy,
    load_matrix,
)
from src.utils.logger import OUTPUT_ROOT, TABLE_DIR, logger

BASE_EXPERIMENT = "ddk_vsa_anova_f4_filter_09"
EXPECTED = {
    "matrix": "fused_egemaps_ddk_vsa_train",
    "n_splits": 5,
    "anova_f": 4.0,
    "corr_filter": True,
    "corr_threshold": 0.9,
    "corr_rank": "anova_f",
    "n_features_input": 714,
}
HPARAM_TABLE_DIR = TABLE_DIR / "hparam"
LOG_DIR = OUTPUT_ROOT / "logs"


def macro_f1_scorer(estimator, X, y):
    pred = estimator.predict(X)
    return classification_scores(y, pred)["macro_f1"]


def _experiment_config():
    cfg = EXPERIMENTS[BASE_EXPERIMENT]
    checks = {
        "matrix": EXPECTED["matrix"],
        "n_splits": EXPECTED["n_splits"],
        "anova_f": EXPECTED["anova_f"],
        "corr_filter": True,
        "corr_threshold": EXPECTED["corr_threshold"],
        "corr_rank": EXPECTED["corr_rank"],
    }
    for key, expected in checks.items():
        actual = cfg.get(key)
        if actual != expected:
            raise ValueError(
                f"{BASE_EXPERIMENT} {key} is {actual}, expected {expected}"
            )
    return cfg


def _attach_log_file(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    return handler


def _finite_or_none(value):
    value = float(value)
    if not np.isfinite(value):
        return None
    return value


def run_search(model: str, n_jobs: int = 1):
    if model not in SEARCH_MODELS:
        raise ValueError(f"unknown model {model}")
    cfg = _experiment_config()
    grid = param_grid(model)
    n_candidates = len(ParameterGrid(grid))
    experiment = f"{BASE_EXPERIMENT}_hparam_{model}"
    log_path = LOG_DIR / f"{experiment}.log"
    _attach_log_file(log_path)

    df = load_matrix(cfg["matrix"])
    feat_cols = _feature_cols(df)
    if len(df) != N_SUBJECTS or df["ID"].nunique() != N_SUBJECTS:
        raise ValueError(f"expected {N_SUBJECTS} subjects")
    if len(feat_cols) != EXPECTED["n_features_input"]:
        raise ValueError(
            f"expected {EXPECTED['n_features_input']} unfiltered features, "
            f"got {len(feat_cols)}"
        )
    if set(df["Class"].astype(int)) != set(CLASSES):
        raise ValueError("Class labels are not 1-5")
    if (df["baseline_split"] != BASELINE_TRAIN).any():
        raise ValueError("validation rows leaked into the search matrix")

    X, y = _xy(df, feat_cols)
    groups = df["ID"].to_numpy()
    n_splits = int(cfg["n_splits"])
    logger.info("Hyperparameter search")
    logger.info("Experiment: %s", experiment)
    logger.info("Base experiment: %s", BASE_EXPERIMENT)
    logger.info("Model: %s", model)
    logger.info("Subjects: %s", len(df))
    logger.info("Unfiltered features: %s", len(feat_cols))
    logger.info("n_splits: %s", n_splits)
    logger.info("anova_f: %s", cfg["anova_f"])
    logger.info("corr_threshold: %s", cfg["corr_threshold"])
    logger.info("corr_rank: %s", cfg["corr_rank"])
    logger.info("random_state: %s", RANDOM_STATE)
    logger.info("Candidates: %s", n_candidates)
    logger.info("GridSearch n_jobs: %s", n_jobs)

    pipe = build_search_pipeline(
        model,
        random_state=RANDOM_STATE,
        anova_f=cfg["anova_f"],
        use_corr_filter=True,
        corr_threshold=cfg["corr_threshold"],
        corr_rank=cfg["corr_rank"],
        estimator_n_jobs=1,
    )
    search = GridSearchCV(
        estimator=pipe,
        param_grid=grid,
        scoring=macro_f1_scorer,
        cv=StratifiedGroupKFold(n_splits=n_splits, shuffle=False),
        refit=True,
        n_jobs=n_jobs,
        verbose=1,
        error_score=np.nan,
        return_train_score=False,
    )
    search.fit(X, y, groups=groups)

    best_index = int(search.best_index_)
    results = search.cv_results_
    HPARAM_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    cv_path = HPARAM_TABLE_DIR / f"{experiment}_cv_results.csv"
    pd.DataFrame(results).to_csv(cv_path, index=False)
    logger.info("Saved %s", cv_path)
    logger.info("Best params: %s", search.best_params_)
    logger.info("Best mean macro-F1: %s", search.best_score_)

    record = build_hparam_record(
        experiment=experiment,
        model=model,
        matrix=cfg["matrix"],
        n_splits=n_splits,
        n_subjects=N_SUBJECTS,
        n_features_input=len(feat_cols),
        anova_f=float(cfg["anova_f"]),
        corr_threshold=float(cfg["corr_threshold"]),
        corr_rank=cfg["corr_rank"],
        random_state=RANDOM_STATE,
        n_jobs=int(n_jobs),
        param_grid=grid,
        best_params=dict(search.best_params_),
        best_score_mean=_finite_or_none(search.best_score_),
        best_score_std=_finite_or_none(results["std_test_score"][best_index]),
        cv_results=cv_path,
    )
    write_hparam_record(record)
    return record


def main():
    parser = argparse.ArgumentParser(
        description="Grid search on the ANOVA + Pearson training split."
    )
    parser.add_argument("--model", required=True, choices=SEARCH_MODELS)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()
    if args.n_jobs < 1:
        raise SystemExit("--n-jobs must be at least 1")
    run_search(args.model, n_jobs=args.n_jobs)


if __name__ == "__main__":
    main()
