import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_classif
from sklearn.utils.validation import check_is_fitted


class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.9, rank="column_order"):
        self.threshold = threshold
        self.rank = rank

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

        if self.rank == "anova_f":
            if y is None:
                raise ValueError("rank='anova_f' requires y")
            F, _ = f_classif(X, y)
            F = np.nan_to_num(np.asarray(F, dtype=float)[candidates], nan=-np.inf)
            visit = np.argsort(-F, kind="mergesort")
        elif self.rank == "column_order":
            visit = np.arange(candidates.size)
        else:
            raise ValueError(f"unknown rank={self.rank}")

        kept_local = []
        for loc in visit:
            redundant = False
            for kept in kept_local:
                r = corr[loc, kept]
                if abs(r) > self.threshold:
                    redundant = True
                    break
            if not redundant:
                kept_local.append(int(loc))

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
