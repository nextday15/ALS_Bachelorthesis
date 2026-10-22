import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from mord import LogisticAT

from src.features.correlation_filter import CorrelationFilter

RANDOM_STATE = 42
CORR_THRESHOLD = 0.9


def _finite_or_nan(X):
    arr = np.asarray(X, dtype=float)
    return np.where(np.isfinite(arr), arr, np.nan)


def _preprocess(use_corr_filter=False, corr_threshold=CORR_THRESHOLD):
    steps = [
        ("clean", FunctionTransformer(_finite_or_nan, validate=False)),
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if use_corr_filter:
        steps.append(("var", VarianceThreshold(threshold=0.0)))
        steps.append(("corr", CorrelationFilter(threshold=corr_threshold)))
    steps.append(("scaler", StandardScaler()))
    return steps


# XGB and ordinal logit expect 0-based consecutive labels
ENCODED_MODELS = {"xgb", "ordinal"}


def build_pipelines(
    random_state=RANDOM_STATE,
    use_corr_filter=False,
    corr_threshold=CORR_THRESHOLD,
):
    return {
        "svm": Pipeline(
            _preprocess(
                use_corr_filter=use_corr_filter,
                corr_threshold=corr_threshold,
            )
            + [
                (
                    "clf",
                    SVC(
                        kernel="rbf",
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                )
            ]
        ),
        "rf": Pipeline(
            _preprocess(
                use_corr_filter=use_corr_filter,
                corr_threshold=corr_threshold,
            )
            + [
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                )
            ]
        ),
        "xgb": Pipeline(
            _preprocess(
                use_corr_filter=use_corr_filter,
                corr_threshold=corr_threshold,
            )
            + [
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=200,
                        max_depth=4,
                        learning_rate=0.1,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective="multi:softprob",
                        eval_metric="mlogloss",
                        n_jobs=1,
                        random_state=random_state,
                        verbosity=0,
                    ),
                )
            ]
        ),
        "ordinal": Pipeline(
            _preprocess(
                use_corr_filter=use_corr_filter,
                corr_threshold=corr_threshold,
            )
            + [("clf", LogisticAT(alpha=1.0))]
        ),
    }
