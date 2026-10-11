import pandas as pd

from src.data_loader import SPEECH_TASK_SUFFIXES
from src.utils.logger import logger, save_checkpoint, PROCESSED_DIR

META_COLS = ["ID", "task_key", "task_suffix", "baseline_split", "Age", "Sex", "Class"]
WIDE_META_COLS = ["ID", "Age", "Sex", "Class", "baseline_split"]
BASELINE_TRAIN = "train"
N_EGEMAPS = 88
RHYTHM_TASK_COLS = [
    "f1_mean",
    "f2_mean",
    "f0_mean",
    "f0_std",
    "intensity_mean",
    "intensity_std",
    "ddk_rate",
    "pause_ratio",
    "pvi",
]
VSA_COL = "vsa"
EGEMAPS_RHYTHM_MATRIX = "fused_egemaps_rhythm_train"


def load_egemaps_task1_train() -> pd.DataFrame:
    path = PROCESSED_DIR / "egemaps_task1_train.csv"
    if not path.exists():
        raise FileNotFoundError(f"did not find {path}")
    return pd.read_csv(path)


def load_rhythm_task1_train() -> pd.DataFrame:
    path = PROCESSED_DIR / "rhythm_task1_train.csv"
    if not path.exists():
        raise FileNotFoundError(f"did not find {path}")
    return pd.read_csv(path)


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS]


def _wide_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in WIDE_META_COLS]


def _require_complete_tasks(df: pd.DataFrame) -> None:
    n_tasks = df.groupby("ID")["task_key"].nunique()
    incomplete = n_tasks[n_tasks != len(SPEECH_TASK_SUFFIXES)]
    if len(incomplete):
        raise ValueError(
            f"{len(incomplete)} subjects missing speech tasks: {list(incomplete.index[:10])}"
        )


def select_baseline_train(df: pd.DataFrame) -> pd.DataFrame:
    if "baseline_split" not in df.columns:
        raise ValueError("baseline_split missing")
    out = df.loc[df["baseline_split"] == BASELINE_TRAIN].copy()
    if out.empty:
        raise ValueError("no baseline train rows")
    if (out["baseline_split"] != BASELINE_TRAIN).any():
        raise ValueError("validation rows leaked into baseline train")
    return out


def _subject_meta(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.drop_duplicates("ID")[WIDE_META_COLS]
        .set_index("ID")
    )


def fuse_egemaps_by_subject(df: pd.DataFrame) -> pd.DataFrame:
    feat_cols = _feature_cols(df)
    if len(feat_cols) != N_EGEMAPS:
        raise ValueError(f"expected {N_EGEMAPS} eGeMAPS columns, got {len(feat_cols)}")
    _require_complete_tasks(df)

    meta = _subject_meta(df)
    parts = []
    for task_key in SPEECH_TASK_SUFFIXES:
        sub = df.loc[df["task_key"] == task_key, ["ID"] + feat_cols]
        if sub.empty:
            raise ValueError(f"no rows for task_key={task_key}")
        renamed = sub.set_index("ID").rename(
            columns={c: f"{task_key}_{c}" for c in feat_cols}
        )
        parts.append(renamed)

    wide = pd.concat(parts, axis=1)
    out = meta.join(wide, how="inner").reset_index()
    feat_wide = _wide_feature_cols(out)
    logger.info("Training split - fused eGeMAPS")
    logger.info("Subjects: %s", len(out))
    logger.info("Speech tasks: %s", len(SPEECH_TASK_SUFFIXES))
    logger.info("Feature columns: %s", len(feat_wide))
    logger.info("Class counts: %s", out["Class"].value_counts().sort_index().to_dict())
    return out


def fuse_rhythm_by_subject(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in RHYTHM_TASK_COLS + [VSA_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"rhythm columns missing: {missing}")
    _require_complete_tasks(df)

    meta = _subject_meta(df)
    vsa = df.drop_duplicates("ID").set_index("ID")[[VSA_COL]]
    parts = []
    for task_key in SPEECH_TASK_SUFFIXES:
        sub = df.loc[df["task_key"] == task_key, ["ID"] + RHYTHM_TASK_COLS]
        if sub.empty:
            raise ValueError(f"no rows for task_key={task_key}")
        renamed = sub.set_index("ID").rename(
            columns={c: f"{task_key}_{c}" for c in RHYTHM_TASK_COLS}
        )
        parts.append(renamed)

    wide = pd.concat(parts, axis=1)
    out = meta.join(wide, how="inner").join(vsa, how="inner").reset_index()
    feat_wide = _wide_feature_cols(out)
    logger.info("Training split - fused rhythm")
    logger.info("Subjects: %s", len(out))
    logger.info("Speech tasks: %s", len(SPEECH_TASK_SUFFIXES))
    logger.info("Feature columns: %s", len(feat_wide))
    logger.info("Class counts: %s", out["Class"].value_counts().sort_index().to_dict())
    return out


def _log_split(long_df: pd.DataFrame, train_df: pd.DataFrame, kind: str) -> None:
    logger.info("%s long-table rows after train filter: %s", kind, len(train_df))
    logger.info("%s unique subjects after train filter: %s", kind, train_df["ID"].nunique())
    val_left = long_df.loc[long_df["baseline_split"] != BASELINE_TRAIN, "ID"].nunique()
    logger.info("Held-out val subjects (excluded): %s", val_left)


def build_task1_baseline_train() -> pd.DataFrame:
    long_df = load_egemaps_task1_train()
    train_df = select_baseline_train(long_df)
    _log_split(long_df, train_df, "eGeMAPS")
    return fuse_egemaps_by_subject(train_df)


def build_egemaps_rhythm_train() -> pd.DataFrame:
    egemaps_long = load_egemaps_task1_train()
    rhythm_long = load_rhythm_task1_train()
    egemaps_train = select_baseline_train(egemaps_long)
    rhythm_train = select_baseline_train(rhythm_long)
    _log_split(egemaps_long, egemaps_train, "eGeMAPS")
    _log_split(rhythm_long, rhythm_train, "rhythm")

    egemaps = fuse_egemaps_by_subject(egemaps_train)
    rhythm = fuse_rhythm_by_subject(rhythm_train)
    rhythm_feats = _wide_feature_cols(rhythm)
    out = egemaps.merge(rhythm[["ID"] + rhythm_feats], on="ID", how="inner")
    if len(out) != len(egemaps) or len(out) != len(rhythm):
        raise ValueError(
            f"ID mismatch after join: egemaps={len(egemaps)}, "
            f"rhythm={len(rhythm)}, merged={len(out)}"
        )
    feat_wide = _wide_feature_cols(out)
    logger.info("Training split - fused eGeMAPS + rhythm")
    logger.info("Subjects: %s", len(out))
    logger.info("Feature columns: %s", len(feat_wide))
    logger.info("Class counts: %s", out["Class"].value_counts().sort_index().to_dict())
    return out


def save_matrix(df: pd.DataFrame, name: str = "fused_egemaps_task1_baseline_train"):
    csv_path, _ = save_checkpoint(df, name)
    return csv_path


if __name__ == "__main__":
    save_matrix(build_egemaps_rhythm_train(), name=EGEMAPS_RHYTHM_MATRIX)
