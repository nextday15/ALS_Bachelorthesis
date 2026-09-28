import pandas as pd

from src.data_loader import SPEECH_TASK_SUFFIXES
from src.utils.logger import logger, save_checkpoint, PROCESSED_DIR

META_COLS = ["ID", "task_key", "task_suffix", "baseline_split", "Age", "Sex", "Class"]
BASELINE_TRAIN = "train"
N_EGEMAPS = 88


def load_egemaps_task1_train() -> pd.DataFrame:
    path = PROCESSED_DIR / "egemaps_task1_train.csv"
    if not path.exists():
        raise FileNotFoundError(f"did not find {path}")
    return pd.read_csv(path)


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS]


def select_baseline_train(df: pd.DataFrame) -> pd.DataFrame:
    if "baseline_split" not in df.columns:
        raise ValueError("baseline_split missing")
    out = df.loc[df["baseline_split"] == BASELINE_TRAIN].copy()
    if out.empty:
        raise ValueError("no baseline train rows")
    if (out["baseline_split"] != BASELINE_TRAIN).any():
        raise ValueError("validation rows leaked into baseline train")
    return out


def fuse_egemaps_by_subject(df: pd.DataFrame) -> pd.DataFrame:
    feat_cols = _feature_cols(df)
    if len(feat_cols) != N_EGEMAPS:
        raise ValueError(f"expected {N_EGEMAPS} eGeMAPS columns, got {len(feat_cols)}")

    n_tasks = df.groupby("ID")["task_key"].nunique()
    incomplete = n_tasks[n_tasks != len(SPEECH_TASK_SUFFIXES)]
    if len(incomplete):
        raise ValueError(
            f"{len(incomplete)} subjects missing speech tasks: {list(incomplete.index[:10])}"
        )

    meta = (
        df.drop_duplicates("ID")
        [["ID", "Age", "Sex", "Class", "baseline_split"]]
        .set_index("ID")
    )
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
    feat_wide = [
        c for c in out.columns if c not in ["ID", "Age", "Sex", "Class", "baseline_split"]
    ]
    logger.info("Task 1 baseline train - fused eGeMAPS")
    logger.info("Subjects: %s", len(out))
    logger.info("Speech tasks: %s", len(SPEECH_TASK_SUFFIXES))
    logger.info("Feature columns: %s", len(feat_wide))
    logger.info("Class counts: %s", out["Class"].value_counts().sort_index().to_dict())
    return out


def build_task1_baseline_train() -> pd.DataFrame:
    long_df = load_egemaps_task1_train()
    train_df = select_baseline_train(long_df)
    logger.info("Long-table rows after baseline filter: %s", len(train_df))
    logger.info("Unique subjects after baseline filter: %s", train_df["ID"].nunique())
    val_left = long_df.loc[long_df["baseline_split"] != BASELINE_TRAIN, "ID"].nunique()
    logger.info("Held-out baseline val subjects (excluded): %s", val_left)
    return fuse_egemaps_by_subject(train_df)


def save_matrix(df: pd.DataFrame, name: str = "fused_egemaps_task1_baseline_train"):
    csv_path, _ = save_checkpoint(df, name)
    return csv_path


if __name__ == "__main__":
    df = build_task1_baseline_train()
    save_matrix(df)
