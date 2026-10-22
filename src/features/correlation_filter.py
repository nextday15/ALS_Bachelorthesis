import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.9):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        n_features = X.shape[1]
        var = np.nanvar(X, axis=0)
        candidates = np.flatnonzero(np.isfinite(var) & (var > 0))
        if candidates.size == 0:
            self.keep_mask_ = np.zeros(n_features, dtype=bool)
            if n_features:
                self.keep_mask_[0] = True
            return self

        sub = X[:, candidates]
        corr = np.corrcoef(sub, rowvar=False)
        if corr.ndim == 0:
            corr = np.array([[1.0]])
        corr = np.nan_to_num(corr, nan=0.0)

        kept_local = []
        for loc in range(candidates.size):
            redundant = False
            for kept in kept_local:
                r = corr[loc, kept]
                if abs(r) > self.threshold:
                    redundant = True
                    break
            if not redundant:
                kept_local.append(loc)

        keep = np.zeros(n_features, dtype=bool)
        keep[candidates[kept_local]] = True
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
