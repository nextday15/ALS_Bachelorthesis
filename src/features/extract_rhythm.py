import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call
from scipy.signal import find_peaks

from src.data_loader import (
    load_task_labels,
    TASK_PATHS,
    SPEECH_TASK_SUFFIXES,
    find_wav,
)
from src.utils.logger import logger, save_checkpoint

DDK_TASKS = ("pa", "ta", "ka")
VSA_VOWELS = ("i", "a", "u")

PITCH_FLOOR = 75.0
PITCH_CEILING = 500.0
N_FORMANTS = 4
MAXIMUM_FORMANT = 3800.0
FORMANT_WINDOW = 0.025
FORMANT_PRE_EMPHASIS = 50.0
MIN_PEAK_DISTANCE_S = 0.10
MIN_PAUSE_S = 0.10
PAUSE_REL_DB = 25.0
PEAK_REL_DB = 8.0

FEATURE_NAMES = [
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


# -----------------------------------------
# Acoustic helpers
# -----------------------------------------
def _nan_features():
    return {name: np.nan for name in FEATURE_NAMES}


def _as_float(value):
    try:
        x = float(value)
        return x if np.isfinite(x) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _mean_std(values, positive_only=False):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if positive_only:
        x = x[x > 0]
    if x.size == 0:
        return np.nan, np.nan
    return float(np.mean(x)), float(np.std(x, ddof=0))


#-----------------------------------------
#Rhythm features
#-----------------------------------------
#PVI: Pairwise Variability Index
def pairwise_variability_index(durations):
    d = np.asarray(durations, dtype=float)
    d = d[np.isfinite(d) & (d > 0)]
    if d.size < 2:
        return np.nan
    num = np.abs(d[:-1] - d[1:])
    den = 0.5 * (d[:-1] + d[1:])
    return float(100.0 * np.mean(num / den))

#VSA: Vowel Space Area
def vowel_space_area(f1_i, f2_i, f1_a, f2_a, f1_u, f2_u):
    vals = np.array([f1_i, f2_i, f1_a, f2_a, f1_u, f2_u], dtype=float)
    if np.any(~np.isfinite(vals)):
        return np.nan
    return float(0.5 * abs(
        f1_i * (f2_a - f2_u) + f1_a * (f2_u - f2_i) + f1_u * (f2_i - f2_a)
    ))

#Pause Ratio
def _pause_ratio(intens, dx, duration):
    if duration <= 0 or intens.size == 0 or not np.any(np.isfinite(intens)):
        return np.nan
    peak = np.nanmax(intens)
    mask = np.isfinite(intens) & (intens < (peak - PAUSE_REL_DB))
    total_pause = 0.0
    i = 0
    n = mask.size
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i + 1
        while j < n and mask[j]:
            j += 1
        run = (j - i) * dx
        if run >= MIN_PAUSE_S:
            total_pause += run
        i = j
    return float(total_pause / duration)

#DDK metrics: ddk_rate, pause_ratio, pvi
def _ddk_metrics(intensity, duration):
    feats = {"ddk_rate": np.nan, "pause_ratio": np.nan, "pvi": np.nan}
    if intensity is None or duration <= 0:
        return feats

    intens = np.asarray(intensity.values, dtype=float).reshape(-1)
    dx = intensity.get_time_step()
    times = intensity.xs()
    feats["pause_ratio"] = _pause_ratio(intens, dx, duration)

    if intens.size < 3 or not np.any(np.isfinite(intens)):
        return feats

    intens = np.nan_to_num(intens, nan=np.nanmin(intens[np.isfinite(intens)]))
    height = max(np.median(intens), np.max(intens) - PEAK_REL_DB)
    min_distance = max(1, int(round(MIN_PEAK_DISTANCE_S / dx)))
    peaks, _ = find_peaks(intens, height=height, distance=min_distance)
    if peaks.size < 2:
        return feats

    peak_times = times[peaks]
    span = peak_times[-1] - peak_times[0]
    if span > 0:
        feats["ddk_rate"] = float((peaks.size - 1) / span)
    feats["pvi"] = pairwise_variability_index(np.diff(peak_times))
    return feats


# -----------------------------------------
# Praat: pitch, intensity, formants, (DDK metrics)
# -----------------------------------------
def extract_file(wav_path, task_key, sex=None):
    feats = _nan_features()
    try:
        sound = parselmouth.Sound(str(wav_path))
        if sound.get_number_of_channels() > 1:
            sound = sound.convert_to_mono()
        duration = sound.get_total_duration()
        if duration <= 0 or sound.get_number_of_samples() < 1:
            return feats
    except Exception:
        return feats

    try:
        pitch = sound.to_pitch(
            time_step=None,
            pitch_floor=PITCH_FLOOR,
            pitch_ceiling=PITCH_CEILING,
        )
        f0 = pitch.selected_array["frequency"]
        feats["f0_mean"], feats["f0_std"] = _mean_std(f0, positive_only=True)
    except Exception:
        pass

    intensity = None
    try:
        intensity = sound.to_intensity(
            minimum_pitch=PITCH_FLOOR,
            time_step=None,
            subtract_mean=True,
        )
        intens = np.asarray(intensity.values, dtype=float).reshape(-1)
        feats["intensity_mean"], feats["intensity_std"] = _mean_std(intens)
    except Exception:
        intensity = None

    try:
        formant = sound.to_formant_burg(
            time_step=None,
            max_number_of_formants=N_FORMANTS,
            maximum_formant=MAXIMUM_FORMANT,
            window_length=FORMANT_WINDOW,
            pre_emphasis_from=FORMANT_PRE_EMPHASIS,
        )
        feats["f1_mean"] = _as_float(call(formant, "Get mean", 1, 0, 0, "Hertz"))
        feats["f2_mean"] = _as_float(call(formant, "Get mean", 2, 0, 0, "Hertz"))
    except Exception:
        pass

    if task_key in DDK_TASKS:
        try:
            feats.update(_ddk_metrics(intensity, duration))
        except Exception:
            pass
    return feats


def _vsa_from_group(group):
    pts = {}
    for vowel in VSA_VOWELS:
        rows = group.loc[group["task_key"] == vowel]
        if rows.empty:
            return np.nan
        pts[vowel] = (rows["f1_mean"].iloc[0], rows["f2_mean"].iloc[0])
    (f1_i, f2_i), (f1_a, f2_a), (f1_u, f2_u) = pts["i"], pts["a"], pts["u"]
    return vowel_space_area(f1_i, f2_i, f1_a, f2_a, f1_u, f2_u)


# -----------------------------------------
# Split-level extraction
# -----------------------------------------
def extract_task_split(task: int, split: str = "train") -> pd.DataFrame:
    labels_df = load_task_labels(task, split)
    audio_root = TASK_PATHS[task][split]["audio_root"]

    rows = []
    missing = []
    failed = []
    for _, rec in labels_df.iterrows():
        speaker_id = rec["ID"]
        sex = rec["Sex"]
        for task_key, task_suffix in SPEECH_TASK_SUFFIXES.items():
            wav_path = find_wav(audio_root, speaker_id, task_suffix)
            row = rec.to_dict()
            row["task_key"] = task_key
            row["task_suffix"] = task_suffix
            if not wav_path.exists():
                missing.append(f"{speaker_id}_{task_suffix}.wav")
                row.update(_nan_features())
            else:
                try:
                    row.update(extract_file(wav_path, task_key, sex=sex))
                except Exception as exc:
                    failed.append(f"{speaker_id}_{task_suffix}.wav")
                    logger.warning("Failed %s: %s", wav_path, exc)
                    row.update(_nan_features())
            rows.append(row)

    logger.info("Task %s (%s) - rhythm / VSA", task, split)
    logger.info("Extracted rows: %s", len(rows))
    logger.info("Missing recordings: %s", len(missing))
    if missing:
        logger.info("Missing files (showing up to 20):")
        for name in missing[:20]:
            logger.info("  %s", name)
    if failed:
        logger.warning("Failed recordings: %s", len(failed))
        for name in failed[:20]:
            logger.warning("  %s", name)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    vsa_by_id = {
        speaker_id: _vsa_from_group(group)
        for speaker_id, group in df.groupby("ID", sort=False)
    }
    df["vsa"] = df["ID"].map(vsa_by_id)

    meta_cols = ["ID", "task_key", "task_suffix"]
    extra_cols = [c for c in ["baseline_split", "fold"] if c in df.columns]
    feature_cols = FEATURE_NAMES + ["vsa"]
    other_cols = [
        c for c in df.columns if c not in meta_cols + extra_cols + feature_cols
    ]
    return df[meta_cols + extra_cols + other_cols + feature_cols]


def save_features(df: pd.DataFrame, task: int, split: str):
    csv_path, _ = save_checkpoint(df, f"rhythm_task{task}_{split}")
    return csv_path


if __name__ == "__main__":
    for task in (1, 2):
        for split in ("train", "test"):
            df = extract_task_split(task, split)
            save_features(df, task, split)
