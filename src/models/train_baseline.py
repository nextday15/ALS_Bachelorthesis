import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.utils.class_weight import compute_sample_weight

from src.data_loader import person_level_group_kfold
from src.evaluation.metrics import CLASSES, classification_scores, confusion
from src.models.pipelines import ENCODED_MODELS, build_pipelines
from src.utils.logger import (
    PROCESSED_DIR,
    TABLE_DIR,
    ensure_output_dirs,
    logger,
)

META_COLS = ["ID", "Age", "Sex", "Class", "baseline_split"]
N_SUBJECTS = 219
BASELINE_TRAIN = "train"

EXPERIMENTS = {
    "rhythm": {
        "matrix": "fused_egemaps_rhythm_train",
        "n_splits": 5,
        "table_prefix": "rhythm",
        "oof_name": "rhythm_oof_predictions",
        "log": "Training split - fused eGeMAPS + rhythm CV",
    },
    "egemaps_fold4": {
        "matrix": "fused_egemaps_task1_baseline_train",
        "n_splits": 4,
        "table_prefix": "egemaps_fold4",
        "oof_name": "egemaps_fold4_oof_predictions",
        "log": "Training split - fused eGeMAPS, 4-fold CV",
    },
}

EXPERIMENT = "egemaps_fold4"
_CFG = EXPERIMENTS[EXPERIMENT]
MATRIX_NAME = _CFG["matrix"]
N_SPLITS = _CFG["n_splits"]
TABLE_PREFIX = _CFG["table_prefix"]
OOF_NAME = _CFG["oof_name"]
CV_LOG = _CFG["log"]


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


def _encode_labels(y, classes=CLASSES):
    mapping = {c: i for i, c in enumerate(classes)}
    encoded = np.array([mapping[int(v)] for v in y], dtype=int)
    return encoded, mapping


def _decode_labels(y_enc, mapping):
    inv = {i: c for c, i in mapping.items()}
    return np.array([inv[int(v)] for v in y_enc], dtype=int)


def _xy(df: pd.DataFrame, feat_cols: list[str]):
    X = df[feat_cols].to_numpy(dtype=float)
    y = df["Class"].to_numpy(dtype=int)
    return X, y


def _fit_predict(pipe, name, X_train, y_train, X_val):
    y_fit = y_train
    mapping = None
    if name in ENCODED_MODELS:
        y_fit, mapping = _encode_labels(y_train)
    fit_kwargs = {}
    if name == "xgb":
        fit_kwargs["clf__sample_weight"] = compute_sample_weight(
            "balanced", y_train
        )
    pipe.fit(X_train, y_fit, **fit_kwargs)
    y_pred = pipe.predict(X_val)
    if mapping is not None:
        y_pred = _decode_labels(y_pred, mapping)
    return y_pred


def run_cv(df: pd.DataFrame, n_splits: int = N_SPLITS) -> tuple[pd.DataFrame, pd.DataFrame]:
    feat_cols = _feature_cols(df)
    logger.info("%s", CV_LOG)
    logger.info("Experiment: %s", EXPERIMENT)
    logger.info("Subjects: %s", len(df))
    logger.info("Feature columns: %s", len(feat_cols))
    logger.info("n_splits: %s", n_splits)
    logger.info("Class counts: %s", df["Class"].value_counts().sort_index().to_dict())

    templates = build_pipelines()
    fold_rows = []
    oof_rows = []

    for name, template in templates.items():
        logger.info("Model: %s", name)
        for fold, train_df, val_df in person_level_group_kfold(df, n_splits=n_splits):
            X_train, y_train = _xy(train_df, feat_cols)
            X_val, y_val = _xy(val_df, feat_cols)
            pipe = clone(template)
            y_pred = _fit_predict(pipe, name, X_train, y_train, X_val)
            scores = classification_scores(y_val, y_pred)
            scores["model"] = name
            scores["fold"] = fold
            scores["n_val"] = len(val_df)
            fold_rows.append(scores)
            logger.info(
                "  fold %s  macro_f1=%.3f  acc=%.3f  recall=%.3f  mae=%.3f  n_val=%s",
                fold,
                scores["macro_f1"],
                scores["accuracy"],
                scores["macro_recall"],
                scores["mae"],
                scores["n_val"],
            )
            for speaker_id, y_t, y_p in zip(val_df["ID"], y_val, y_pred):
                oof_rows.append(
                    {
                        "model": name,
                        "fold": fold,
                        "ID": speaker_id,
                        "y_true": int(y_t),
                        "y_pred": int(y_p),
                    }
                )

    return pd.DataFrame(fold_rows), pd.DataFrame(oof_rows)


def _pooled_scores(oof_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, sub in oof_df.groupby("model", sort=False):
        scores = classification_scores(sub["y_true"], sub["y_pred"])
        scores["model"] = name
        rows.append(scores)
    return pd.DataFrame(rows)


def _confusion_tables(oof_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, sub in oof_df.groupby("model", sort=False):
        cm = confusion(sub["y_true"], sub["y_pred"])
        for i, true_c in enumerate(CLASSES):
            row = {"model": name, "true_class": true_c}
            for j, pred_c in enumerate(CLASSES):
                row[f"pred_{pred_c}"] = int(cm[i, j])
            rows.append(row)
    return pd.DataFrame(rows)


def _mean_std_table(fold_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        c
        for c in fold_df.columns
        if c not in {"model", "fold", "n_val"}
    ]
    rows = []
    for name, sub in fold_df.groupby("model", sort=False):
        row = {"model": name}
        for col in metric_cols:
            row[f"{col}_mean"] = float(sub[col].mean())
            row[f"{col}_std"] = float(sub[col].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def save_results(fold_df, oof_df):
    ensure_output_dirs()
    summary_df = _mean_std_table(fold_df)
    pooled_df = _pooled_scores(oof_df)
    cm_df = _confusion_tables(oof_df)

    fold_path = TABLE_DIR / f"{TABLE_PREFIX}_cv_fold_metrics.csv"
    summary_path = TABLE_DIR / f"{TABLE_PREFIX}_cv_summary.csv"
    pooled_path = TABLE_DIR / f"{TABLE_PREFIX}_cv_pooled.csv"
    cm_path = TABLE_DIR / f"{TABLE_PREFIX}_confusion.csv"
    oof_path = PROCESSED_DIR / f"{OOF_NAME}.csv"

    fold_df.to_csv(fold_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    pooled_df.to_csv(pooled_path, index=False)
    cm_df.to_csv(cm_path, index=False)
    oof_df.to_csv(oof_path, index=False)
    logger.info("Saved %s", fold_path)
    logger.info("Saved %s", summary_path)
    logger.info("Saved %s", pooled_path)
    logger.info("Saved %s", cm_path)
    logger.info("Saved %s", oof_path)
    return summary_df, pooled_df


if __name__ == "__main__":
    df = load_matrix()
    fold_df, oof_df = run_cv(df)
    save_results(fold_df, oof_df)
