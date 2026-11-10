import json
from pathlib import Path

import numpy as np

from src.models.pipelines import RANDOM_STATE
from src.utils.logger import EXPERIMENT_DIR, PROJECT_ROOT, ensure_output_dirs, logger

RECORD_KEYS = [
    "experiment",
    "matrix",
    "n_splits",
    "n_subjects",
    "n_features_input",
    "n_features_kept_mean",
    "anova_f",
    "corr_threshold",
    "corr_rank",
    "random_state",
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
    mean_macro_f1,
    paths,
    anova_f=None,
    corr_threshold=None,
    corr_rank=None,
    n_features_kept_mean=None,
    random_state=RANDOM_STATE,
) -> dict:
    if n_features_kept_mean is None:
        n_features_kept_mean = n_features_input
    if corr_threshold is None:
        corr_rank = None
    record = {
        "experiment": experiment,
        "matrix": matrix,
        "n_splits": int(n_splits),
        "n_subjects": int(n_subjects),
        "n_features_input": int(n_features_input),
        "n_features_kept_mean": n_features_kept_mean,
        "anova_f": anova_f,
        "corr_threshold": corr_threshold,
        "corr_rank": corr_rank,
        "random_state": random_state,
        "mean_macro_f1": {k: float(v) for k, v in mean_macro_f1.items()},
        "paths": {k: relative_output_path(v) for k, v in paths.items()},
    }
    return {key: record[key] for key in RECORD_KEYS}


HPARAM_KEYS = [
    "experiment",
    "model",
    "matrix",
    "n_splits",
    "n_subjects",
    "n_features_input",
    "anova_f",
    "corr_threshold",
    "corr_rank",
    "random_state",
    "n_jobs",
    "param_grid",
    "best_params",
    "best_score_mean",
    "best_score_std",
    "cv_results",
]


def _json_ready(value):
    if isinstance(value, Path):
        return relative_output_path(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def build_hparam_record(**fields) -> dict:
    missing = [key for key in HPARAM_KEYS if key not in fields]
    if missing:
        raise ValueError(f"hparam record missing {missing}")
    if not str(fields["experiment"]).startswith("hparam_"):
        raise ValueError("hparam record would overwrite a completed experiment")
    return {key: _json_ready(fields[key]) for key in HPARAM_KEYS}


def write_hparam_record(record: dict) -> Path:
    ensure_output_dirs()
    path = EXPERIMENT_DIR / f"{record['experiment']}.json"
    if not path.stem.startswith("hparam_"):
        raise ValueError(f"refusing to overwrite {path}")
    path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("Saved %s", path)
    return path


def write_record(record: dict) -> Path:
    ensure_output_dirs()
    path = EXPERIMENT_DIR / f"{record['experiment']}.json"
    path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("Saved %s", path)
    return path
