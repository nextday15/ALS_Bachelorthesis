from src.models.train_baseline import (
    N_SUBJECTS,
    _confusion_tables,
    _feature_cols,
    _mean_std_table,
    load_matrix,
    run_cv,
)
from src.utils.logger import TABLE_DIR, ensure_output_dirs, logger

OUT_DIR = TABLE_DIR / "4_hparam"
PREFIX = "hparam_cv"


def save_hparam_cv(fold_df, oof_df):
    ensure_output_dirs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_df = _mean_std_table(fold_df)
    cm_df = _confusion_tables(oof_df)
    paths = {
        "fold_metrics": OUT_DIR / f"{PREFIX}_cv_fold_metrics.csv",
        "summary": OUT_DIR / f"{PREFIX}_cv_summary.csv",
        "confusion": OUT_DIR / f"{PREFIX}_confusion.csv",
        "oof": OUT_DIR / f"{PREFIX}_oof_predictions.csv",
    }
    fold_df.to_csv(paths["fold_metrics"], index=False)
    summary_df.to_csv(paths["summary"], index=False)
    cm_df.to_csv(paths["confusion"], index=False)
    oof_df.to_csv(paths["oof"], index=False)
    for path in paths.values():
        logger.info("Saved %s", path)
    return summary_df


if __name__ == "__main__":
    df = load_matrix("fused_egemaps_ddk_vsa_train")
    if len(df) != N_SUBJECTS or (df["baseline_split"] != "train").any():
        raise RuntimeError("CV must use the 219 training subjects")
    logger.info("Feature columns: %s", len(_feature_cols(df)))
    fold_df, oof_df = run_cv(df, tuned=True)
    summary_df = save_hparam_cv(fold_df, oof_df)
    for row in summary_df.itertuples(index=False):
        logger.info(
            "%s  macro_f1=%.3f  std=%.3f",
            row.model,
            row.macro_f1_mean,
            row.macro_f1_std,
        )
