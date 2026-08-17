
from pathlib import Path

import opensmile
import pandas as pd

from src.data_loader import (
    load_task_labels,
    TASK_PATHS,
    SPEECH_TASK_SUFFIXES,
    find_wav,
)

OUT_DIR = Path("data/processed/egemaps")
N_EGEMAPS = 88

# -----------------------------------------
# openSMILE eGeMAPSv02
# -----------------------------------------
def build_smile():
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    n = len(smile.feature_names)
    if n != N_EGEMAPS:
        raise ValueError(f"expected {N_EGEMAPS} eGeMAPS functionals, got {n}")
    return smile

def extract_file(smile, wav_path):
    df = smile.process_file(str(wav_path))
    return df.reset_index(drop=True).iloc[0]


# one row per subject and speech task
def extract_task_split(task: int, split: str = "train", smile=None) -> pd.DataFrame:
    if smile is None:
        smile = build_smile()

    labels_df = load_task_labels(task, split)
    audio_root = TASK_PATHS[task][split]["audio_root"]

    rows = []
    missing = []
    for _, rec in labels_df.iterrows():
        speaker_id = rec["ID"]
        for task_key, task_suffix in SPEECH_TASK_SUFFIXES.items():
            wav_path = find_wav(audio_root, speaker_id, task_suffix)
            if not wav_path.exists():
                missing.append(f"{speaker_id}_{task_suffix}.wav")
                continue
            row = rec.to_dict()
            row["task_key"] = task_key
            row["task_suffix"] = task_suffix
           feats = extract_file(smile, wav_path)
            row.update(feats)
            rows.append(row)

    print(f"\n=== Task {task} ({split}) - eGeMAPSv02 ===")
    print(f"Extracted rows                   : {len(rows)}")
    print(f"Missing recordings               : {len(missing)}")
    if missing:
        print("Missing files (showing up to 20):")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    meta_cols = ["ID", "task_key", "task_suffix"]
    extra_cols = [c for c in ["baseline_split", "fold"] if c in df.columns]
    other_cols = [c for c in df.columns if c not in meta_cols + extra_cols]
    
    return df[meta_cols + extra_cols + other_cols]

def save_features(df: pd.DataFrame, task: int, split: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"task{task}_{split}.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    return out_path


if __name__ == "__main__":
    smile = build_smile()
    for task in (1, 2):
        for split in ("train", "test"):
            df = extract_task_split(task, split, smile=smile)
            save_features(df, task, split)
