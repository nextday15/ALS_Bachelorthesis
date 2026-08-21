from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "raw"
# Mapping of speech task shorthand keys to standard filename suffixes in SAND dataset
SPEECH_TASK_SUFFIXES = {
    "a": "phonationA",
    "e": "phonationE",
    "i": "phonationI",
    "o": "phonationO",
    "u": "phonationU",
    "pa": "rhythmPA",
    "ta": "rhythmTA",
    "ka": "rhythmKA",
}
# Task specific label and audio directory paths
TASK_PATHS = {
    1: {
        "train": {
            "labels": DATA_ROOT / "task1_training" / "sand_task_1.xlsx",
            "audio_root": DATA_ROOT / "task1_training" / "training",
        },
        "test": {
            "labels": DATA_ROOT / "task1_test" / "sand_task1_test.xlsx",
            "audio_root": DATA_ROOT / "task1_test" / "test",
        },
    },
    2: {
        "train": {
            "labels": DATA_ROOT / "task2_training" / "sand_task_2.xlsx",
            "audio_root": DATA_ROOT / "task2_training" / "training",
        },
        "test": {
            "labels": DATA_ROOT / "task2_test" / "sand_task2_test.xlsx",
            "audio_root": DATA_ROOT / "task2_test" / "test",
        },
    },
}
# data check
LABEL_COLUMNS = {
    1: ["ID", "Age", "Sex", "Class"],
    2: ["ID", "Age", "Sex", "Months", "ALSFRS--R_start", "ALSFRS--R_end"],
}
# match sheet name
SHEET_KEYWORDS = {
    "val_baseline": ("validation baseline",),
    "train_baseline": ("training baseline",),
    "test": ("testing set",),
    "main": ("training set",),
}

# -----------------------------------------
# Excel Parsing and Label Standardization
# -----------------------------------------
# classify a sheet by name
def _classify_sheet(sheet_name: str) -> str:
    lowered = sheet_name.lower()
    for role, keywords in SHEET_KEYWORDS.items():
        if any(kw in lowered for kw in keywords):
            return role
    return "other"


def load_label_sheets(xlsx_path: Path) -> dict[str, pd.DataFrame]:
    xlsx_path = Path(xlsx_path)
    if not xlsx_path.exists():
        raise FileNotFoundError(f"did not find label file: {xlsx_path}")

    sheets = {}
    xls = pd.ExcelFile(xlsx_path)

    for name in xls.sheet_names:
        role = _classify_sheet(name)
        df = xls.parse(name)
        df.columns = [str(c).strip() for c in df.columns]
        sheets[role] = df
    return sheets


# normalize an ID to the "ID###"
def _normalize_id(value: object) -> str:
    text = str(value).strip()
    return f"ID{int(text):03d}" if text.isdigit() else text


# check columns ensure expected columns are present and in the correct order
def _clean_columns(df: pd.DataFrame, task: int) -> pd.DataFrame:
    expected = LABEL_COLUMNS[task]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(
            f"column {missing} missing; present: {list(df.columns)}"
        )
    ordered_cols = expected + [c for c in df.columns if c not in expected]
    df = df[ordered_cols].copy()
    df["ID"] = df["ID"].apply(_normalize_id)
    return df


# load labels for one task/split
def load_task_labels(task: int, split: str = "train") -> pd.DataFrame:
    xlsx_path = TASK_PATHS[task][split]["labels"]
    sheets = load_label_sheets(xlsx_path)

    if split == "train":
        df = sheets.get("main")
        if df is None:
            raise ValueError(f"did not find training set in {xlsx_path}.")
        df = _clean_columns(df, task)
        # Merge official baseline validation split flag if available
        if "val_baseline" in sheets:
            val_ids = set(sheets["val_baseline"]["ID"].apply(_normalize_id))
            df["baseline_split"] = df["ID"].apply(lambda x: "val" if x in val_ids else "train")
        else:
            df["baseline_split"] = "train"
    else:
        df = sheets["test"] if "test" in sheets else list(sheets.values())[0]
        df = _clean_columns(df, task)

    return df

# -----------------------------------------
# Audio File Inventory and Quality Control (QC)
# -----------------------------------------
def find_wav(audio_root: Path, speaker_id: str, suffix: str) -> Path:
    return audio_root / suffix / f"{speaker_id}_{suffix}.wav"

#check per subject if all 8 recordings exist
def build_audio_inventory(labels_df: pd.DataFrame, audio_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for speaker_id in labels_df["ID"]:
        row: dict[str, object] = {"ID": speaker_id}
        for key, suffix in SPEECH_TASK_SUFFIXES.items():
            row[f"has_{key}"] = find_wav(audio_root, speaker_id, suffix).exists()
        row["is_complete"] = all(row[f"has_{k}"] for k in SPEECH_TASK_SUFFIXES)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_inventory(inventory: pd.DataFrame, task: int, split: str) -> None:
    total = len(inventory)
    complete = int(inventory["is_complete"].sum())
    incomplete = inventory[~inventory["is_complete"]]
    print(f"\n=== Task {task} ({split}) - Audio Inventory ===")
    print(f"Subjects in label file         : {total}")
    print(f"Subjects with all 8 recordings : {complete}")
    print(f"Subjects with missing files    : {len(incomplete)}")
    if len(incomplete):
        print("\nMissing files per subject (showing up to 20):")
        for _, row in incomplete.head(20).iterrows():
            missing_keys = [k for k in SPEECH_TASK_SUFFIXES if not row[f"has_{k}"]]
            print(f"  {row['ID']}: missing {missing_keys}")

# load labels and audio inventory and summarize missing files
def load_and_check_task(task: int, split: str = "train") -> tuple[pd.DataFrame, pd.DataFrame]:
    labels_df = load_task_labels(task, split)
    audio_root = TASK_PATHS[task][split]["audio_root"]
    inventory = build_audio_inventory(labels_df, audio_root)
    summarize_inventory(inventory, task, split)
    merged = labels_df.merge(inventory, on="ID", how="left")
    return merged, inventory

# split on subject ID so the same speaker never appears in both train and val
def person_level_group_kfold(df: pd.DataFrame, n_splits: int = 5, group_col: str = "ID"):
    groups = df[group_col]
    n_subjects = groups.nunique()
    if n_splits > n_subjects:
        raise ValueError(
            f"n_splits={n_splits} larger than number of subjects ({n_subjects})"
        )

    if "Class" in df.columns:
        y = df["Class"]
    elif "ALSFRS--R_end" in df.columns:
        y = df["ALSFRS--R_end"]
    else:
        raise ValueError("need Class or ALSFRS--R_end for stratified group split")

    sgkf = StratifiedGroupKFold(n_splits=n_splits)
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(df, y, groups=groups)):
        yield fold, df.iloc[train_idx].copy(), df.iloc[val_idx].copy()


if __name__ == "__main__":
    for task in (1, 2):
        load_and_check_task(task, split="train")
        load_and_check_task(task, split="test")
