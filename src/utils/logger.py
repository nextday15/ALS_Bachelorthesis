from pathlib import Path
import logging

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
PROCESSED_DIR = OUTPUT_ROOT / "processed_data"
TABLE_DIR = OUTPUT_ROOT / "tables"
FIGURE_DIR = OUTPUT_ROOT / "figures"
EXPERIMENT_DIR = OUTPUT_ROOT / "experiments"


def ensure_output_dirs():
    for path in (PROCESSED_DIR, TABLE_DIR, FIGURE_DIR, EXPERIMENT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def setup_logger():
    log = logging.getLogger("sand")
    log.setLevel(logging.INFO)
    if log.handlers:
        return log

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    log.addHandler(stream_handler)
    log.propagate = False
    return log


logger = setup_logger()


def save_checkpoint(df, name):
    ensure_output_dirs()
    csv_path = PROCESSED_DIR / f"{name}.csv"
    parquet_path = PROCESSED_DIR / f"{name}.parquet"
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)
    logger.info("Saved %s and %s", csv_path, parquet_path)
    return csv_path, parquet_path
