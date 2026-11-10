import json
from pathlib import Path

from src.models.pipelines import RANDOM_STATE
from src.utils.logger import EXPERIMENT_DIR, PROJECT_ROOT, ensure_output_dirs, logger

RECORD_KEYS = [
    "experiment",
    "matrix",
    "n_splits",
    "n_features_input",
    "anova_f",
    "corr_filter",
    "corr_threshold",
    "corr_rank",
    "n_subjects",
    "split",
    "random_state",
    "models",
    "primary_metric",
    "n_features_kept_mean",
    "mean_macro_f1",
    "paths",
]


def relative_output_path(path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def build_record(
    *,
    experiment,
    matrix,
    n_splits,
    n_features_input,
    n_subjects,
    split,
    mean_macro_f1,
    paths,
    anova_f=None,
    corr_filter=False,
    corr_threshold=None,
    corr_rank=None,
    n_features_kept_mean=None,
    random_state=RANDOM_STATE,
    primary_metric="macro_f1_mean",
) -> dict:
    if n_features_kept_mean is None:
        n_features_kept_mean = n_features_input
    record = {
        "experiment": experiment,
        "matrix": matrix,
        "n_splits": int(n_splits),
        "n_features_input": int(n_features_input),
        "anova_f": anova_f,
        "corr_filter": bool(corr_filter),
        "corr_threshold": corr_threshold if corr_filter else None,
        "corr_rank": corr_rank if corr_filter else None,
        "n_subjects": int(n_subjects),
        "split": split,
        "random_state": random_state,
        "models": list(mean_macro_f1),
        "primary_metric": primary_metric,
        "n_features_kept_mean": n_features_kept_mean,
        "mean_macro_f1": {k: float(v) for k, v in mean_macro_f1.items()},
        "paths": {k: relative_output_path(v) for k, v in paths.items()},
    }
    return {key: record[key] for key in RECORD_KEYS}


def write_record(record: dict) -> Path:
    ensure_output_dirs()
    path = EXPERIMENT_DIR / f"{record['experiment']}.json"
    path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("Saved %s", path)
    return path
