import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBClassifier
from mord import LogisticAT

from src.evaluation.metrics import CLASSES
from src.features.anova_filter import ANOVAFFilter
from src.features.correlation_filter import CorrelationFilter

RANDOM_STATE = 42
CORR_THRESHOLD = 0.9


def _finite_or_nan(X):
    arr = np.asarray(X, dtype=float)
    return np.where(np.isfinite(arr), arr, np.nan)


def _preprocess(
    anova_f=None,
    use_corr_filter=False,
    corr_threshold=CORR_THRESHOLD,
    corr_rank="column_order",
):
    steps = [
        ("clean", FunctionTransformer(_finite_or_nan, validate=False)),
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if anova_f is not None:
        steps.append(("anova", ANOVAFFilter(threshold=anova_f)))
    if use_corr_filter:
        steps.append(
            (
                "corr",
                CorrelationFilter(threshold=corr_threshold, rank=corr_rank),
            )
        )
    steps.append(("scaler", StandardScaler()))
    return steps


# XGB and ordinal logit expect 0-based consecutive labels
ENCODED_MODELS = {"xgb", "ordinal"}
SEARCH_MODELS = ("svm", "rf", "xgb", "ordinal")


def _encode_fixed(y):
    mapping = {c: i for i, c in enumerate(CLASSES)}
    return np.array([mapping[int(v)] for v in np.asarray(y)], dtype=int)


def _decode_fixed(y_enc):
    inv = {i: c for i, c in enumerate(CLASSES)}
    return np.array([inv[int(v)] for v in np.asarray(y_enc)], dtype=int)


class EncodedXGBClassifier(XGBClassifier):
    def fit(self, X, y, **kwargs):
        y = np.asarray(y)
        sample_weight = compute_sample_weight("balanced", y)
        return super().fit(X, _encode_fixed(y), sample_weight=sample_weight, **kwargs)

    def predict(self, X, **kwargs):
        return _decode_fixed(super().predict(X, **kwargs))


class EncodedLogisticAT(ClassifierMixin, BaseEstimator):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        self.model_ = LogisticAT(alpha=self.alpha)
        self.model_.fit(X, _encode_fixed(y))
        self.classes_ = np.asarray(CLASSES)
        return self

    def predict(self, X):
        check_is_fitted(self, "model_")
        return _decode_fixed(self.model_.predict(X))


def build_pipelines(
    random_state=RANDOM_STATE,
    anova_f=None,
    use_corr_filter=False,
    corr_threshold=CORR_THRESHOLD,
    corr_rank="column_order",
):
    pre_kwargs = dict(
        anova_f=anova_f,
        use_corr_filter=use_corr_filter,
        corr_threshold=corr_threshold,
        corr_rank=corr_rank,
    )
    return {
        "svm": Pipeline(
            _preprocess(**pre_kwargs)
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
            _preprocess(**pre_kwargs)
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
            _preprocess(**pre_kwargs)
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
            _preprocess(**pre_kwargs) + [("clf", LogisticAT(alpha=1.0))]
        ),
    }


def build_search_pipeline(
    model,
    random_state=RANDOM_STATE,
    anova_f=None,
    use_corr_filter=False,
    corr_threshold=CORR_THRESHOLD,
    corr_rank="column_order",
    estimator_n_jobs=1,
):
    if model not in SEARCH_MODELS:
        raise ValueError(f"unknown model {model}")
    pre_kwargs = dict(
        anova_f=anova_f,
        use_corr_filter=use_corr_filter,
        corr_threshold=corr_threshold,
        corr_rank=corr_rank,
    )
    if model == "svm":
        clf = SVC(
            kernel="rbf",
            class_weight="balanced",
            random_state=random_state,
        )
    elif model == "rf":
        clf = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=estimator_n_jobs,
        )
    elif model == "xgb":
        clf = EncodedXGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            eval_metric="mlogloss",
            n_jobs=estimator_n_jobs,
            random_state=random_state,
            verbosity=0,
        )
    else:
        clf = EncodedLogisticAT(alpha=1.0)
    return Pipeline(_preprocess(**pre_kwargs) + [("clf", clf)])
