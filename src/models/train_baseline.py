import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.utils.class_weight import compute_sample_weight

from src.data_loader import person_level_group_kfold
from src.evaluation.metrics import CLASSES, classification_scores, confusion
from src.models.experiment_record import build_record, write_record
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
    "egemaps_fold4": {
        "matrix": "fused_egemaps_task1_baseline_train",
        "n_splits": 4,
        "table_prefix": "egemaps_fold4",
        "oof_name": "egemaps_fold4_oof_predictions",
        "log": "Training split - fused eGeMAPS, 4-fold CV",
    },
    "ddk_vsa": {
        "matrix": "fused_egemaps_ddk_vsa_train",
        "n_splits": 5,
        "table_prefix": "ddk_vsa",
        "oof_name": "ddk_vsa_oof_predictions",
        "log": "Training split - fused eGeMAPS + DDK + VSA CV",
    },
    "ddk_vsa_anova_f4": {
        "matrix": "fused_egemaps_ddk_vsa_train",
        "n_splits": 5,
        "table_prefix": "ddk_vsa_anova_f4",
        "oof_name": "ddk_vsa_anova_f4_oof_predictions",
        "log": "Training split - fused eGeMAPS + DDK + VSA, ANOVA F>=4",
        "anova_f": 4.0,
    },
    "ddk_vsa_anova_f4_filter_09": {
        "matrix": "fused_egemaps_ddk_vsa_train",
        "n_splits": 5,
        "table_prefix": "ddk_vsa_anova_f4_filter_09",
        "oof_name": "ddk_vsa_anova_f4_filter_09_oof_predictions",
        "log": "Training split - fused eGeMAPS + DDK + VSA, ANOVA F>=4 then Pearson |r|>0.9 (keep higher F)",
        "anova_f": 4.0,
        "corr_filter": True,
        "corr_threshold": 0.9,
        "corr_rank": "anova_f",
    },
}

EXPERIMENT = "ddk_vsa_anova_f4_filter_09"
_CFG = EXPERIMENTS[EXPERIMENT]
MATRIX_NAME = _CFG["matrix"]
N_SPLITS = _CFG["n_splits"]
TABLE_PREFIX = _CFG["table_prefix"]
OOF_NAME = _CFG["oof_name"]
CV_LOG = _CFG["log"]
ANOVA_F = _CFG.get("anova_f")
USE_CORR_FILTER = _CFG.get("corr_filter", False)
CORR_THRESHOLD = _CFG.get("corr_threshold")
CORR_RANK = _CFG.get("corr_rank", "column_order")


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


def _n_kept_features(pipe, n_in: int) -> int:
    if "corr" in pipe.named_steps:
        return int(np.sum(pipe.named_steps["corr"].keep_mask_))
    if "anova" in pipe.named_steps:
        return int(np.sum(pipe.named_steps["anova"].keep_mask_))
    clf = pipe.named_steps["clf"]
    return int(getattr(clf, "n_features_in_", n_in))


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
    n_kept = _n_kept_features(pipe, X_train.shape[1])
    return y_pred, n_kept


def run_cv(df: pd.DataFrame, n_splits: int = N_SPLITS) -> tuple[pd.DataFrame, pd.DataFrame]:
    feat_cols = _feature_cols(df)
    logger.info("%s", CV_LOG)
    logger.info("Experiment: %s", EXPERIMENT)
    logger.info("Subjects: %s", len(df))
    logger.info("Feature columns: %s", len(feat_cols))
    logger.info("n_splits: %s", n_splits)
    logger.info("anova_f: %s", ANOVA_F)
    logger.info("corr_filter: %s", USE_CORR_FILTER)
    logger.info("corr_threshold: %s", CORR_THRESHOLD if USE_CORR_FILTER else None)
    logger.info("corr_rank: %s", CORR_RANK if USE_CORR_FILTER else None)
    logger.info("Class counts: %s", df["Class"].value_counts().sort_index().to_dict())

    templates = build_pipelines(
        anova_f=ANOVA_F,
        use_corr_filter=USE_CORR_FILTER,
        corr_threshold=CORR_THRESHOLD,
        corr_rank=CORR_RANK,
    )
    fold_rows = []
    oof_rows = []

    for name, template in templates.items():
        logger.info("Model: %s", name)
        for fold, train_df, val_df in person_level_group_kfold(df, n_splits=n_splits):
            X_train, y_train = _xy(train_df, feat_cols)
            X_val, y_val = _xy(val_df, feat_cols)
            pipe = clone(template)
            y_pred, n_kept = _fit_predict(pipe, name, X_train, y_train, X_val)
            scores = {
                "model": name,
                "fold": fold,
                "n_val": len(val_df),
                "n_features": n_kept,
                **classification_scores(y_val, y_pred),
            }
            fold_rows.append(scores)
            logger.info(
                "  fold %s  macro_f1=%.3f  acc=%.3f  recall=%.3f  mae=%.3f  n_val=%s  n_feat=%s",
                fold,
                scores["macro_f1"],
                scores["accuracy"],
                scores["macro_recall"],
                scores["mae"],
                scores["n_val"],
                n_kept,
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
        scores = {"model": name, **classification_scores(sub["y_true"], sub["y_pred"])}
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


def _n_features_kept(summary_df, n_features_input):
    if "n_features_mean" in summary_df.columns:
        return float(summary_df["n_features_mean"].mean())
    return int(n_features_input)


def save_results(fold_df, oof_df, n_features_input=None):
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

    if n_features_input is None:
        if "n_features" in fold_df.columns:
            n_features_input = int(fold_df["n_features"].max())
        else:
            n_features_input = 0
    write_record(
        build_record(
            experiment=EXPERIMENT,
            matrix=MATRIX_NAME,
            n_splits=N_SPLITS,
            n_features_input=n_features_input,
            n_subjects=N_SUBJECTS,
            split=BASELINE_TRAIN,
            anova_f=ANOVA_F,
            corr_filter=USE_CORR_FILTER,
            corr_threshold=CORR_THRESHOLD,
            corr_rank=CORR_RANK,
            n_features_kept_mean=_n_features_kept(summary_df, n_features_input),
            pooled_macro_f1={
                row["model"]: float(row["macro_f1"])
                for _, row in pooled_df.iterrows()
            },
            paths={
                "fold_metrics": fold_path,
                "summary": summary_path,
                "pooled": pooled_path,
                "confusion": cm_path,
                "oof": oof_path,
            },
        )
    )
    return summary_df, pooled_df


if __name__ == "__main__":
    df = load_matrix()
    fold_df, oof_df = run_cv(df)
    save_results(fold_df, oof_df, n_features_input=len(_feature_cols(df)))
