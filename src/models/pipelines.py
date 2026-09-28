import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from mord import LogisticAT

RANDOM_STATE = 42


def _finite_or_nan(X):
    arr = np.asarray(X, dtype=float)
    return np.where(np.isfinite(arr), arr, np.nan)


def _preprocess():
    return [
        ("clean", FunctionTransformer(_finite_or_nan, validate=False)),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]


# XGB and ordinal logit expect 0-based consecutive labels
ENCODED_MODELS = {"xgb", "ordinal"}


def build_pipelines(random_state=RANDOM_STATE):
    return {
        "svm": Pipeline(
            _preprocess()
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
            _preprocess()
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
            _preprocess()
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
            _preprocess()
            + [("clf", LogisticAT(alpha=1.0))]
        ),
    }
