PARAM_GRIDS = {
    "svm": {
        "clf__kernel": ["linear", "poly", "rbf", "sigmoid"],
        "clf__C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "clf__gamma": ["scale", 0.001, 0.01, 0.1],
    },
    "rf": {
        "clf__n_estimators": [100, 200, 300, 400],
        "clf__max_depth": [None, 5, 10],
        "clf__min_samples_leaf": [1, 2, 5],
    },
    "xgb": {
        "clf__n_estimators": [100, 200, 300, 400],
        "clf__max_depth": [3,4,5,6],
        "clf__learning_rate": [0.1, 0.08, 0.05, 0.03],
        "clf__subsample": [0.5, 0.8, 1.0],
        "clf__colsample_bytree": [0.5, 0.8, 1.0],
    },
    "ordinal": {
        "clf__alpha": [0.001, 0.01, 0.1, 1.0, 10, 100],
    },
}


def param_grid(model: str) -> dict:
    if model not in PARAM_GRIDS:
        known = ", ".join(PARAM_GRIDS)
        raise ValueError(f"unknown model {model}; expected one of {known}")
    grid = PARAM_GRIDS[model]
    empty = [key for key, values in grid.items() if len(values) == 0]
    if empty:
        joined = ", ".join(empty)
        raise ValueError(
            f"{model} search space has no values for {joined}. "
            "Fill src/models/param_grids.py before running the search."
        )
    return grid
