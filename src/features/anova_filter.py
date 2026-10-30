import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_classif
from sklearn.utils.validation import check_is_fitted


class ANOVAFFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=4.0):
        self.threshold = threshold

    def fit(self, X, y):
        if y is None:
            raise ValueError("ANOVAFFilter requires y")
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        F, _ = f_classif(X, y)
        F = np.asarray(F, dtype=float)
        keep = np.isfinite(F) & (F >= self.threshold)
        if not np.any(keep):
            scores = np.nan_to_num(F, nan=-np.inf)
            keep = np.zeros(X.shape[1], dtype=bool)
            keep[int(np.argmax(scores))] = True
        self.scores_ = F
        self.keep_mask_ = keep
        return self

    def transform(self, X):
        check_is_fitted(self, "keep_mask_")
        X = np.asarray(X, dtype=float)
        return X[:, self.keep_mask_]

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "keep_mask_")
        if input_features is None:
            input_features = np.asarray(
                [f"x{i}" for i in range(self.keep_mask_.size)], dtype=object
            )
        else:
            input_features = np.asarray(input_features, dtype=object)
        return input_features[self.keep_mask_]
