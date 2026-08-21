from pathlib import Path
import logging

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
LOG_DIR = OUTPUT_ROOT / "logs"
PROCESSED_DIR = OUTPUT_ROOT / "processed_data"
TABLE_DIR = OUTPUT_ROOT / "tables"
FIGURE_DIR = OUTPUT_ROOT / "figures"


def ensure_output_dirs():
    for path in (LOG_DIR, PROCESSED_DIR, TABLE_DIR, FIGURE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def setup_logger(log_name="feature_extraction.log"):
    ensure_output_dirs()
    log = logging.getLogger("sand")
    log.setLevel(logging.INFO)
    if log.handlers:
        return log

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(LOG_DIR / log_name, encoding="utf-8")
    file_handler.setFormatter(fmt)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    log.addHandler(file_handler)
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
