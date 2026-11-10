PARAM_GRIDS = {
    "svm": {
        "clf__C": [],
        "clf__gamma": [],
    },
    "rf": {
        "clf__n_estimators": [],
        "clf__max_depth": [],
        "clf__min_samples_leaf": [],
        "clf__max_features": [],
    },
    "xgb": {
        "clf__n_estimators": [],
        "clf__max_depth": [],
        "clf__learning_rate": [],
        "clf__subsample": [],
        "clf__colsample_bytree": [],
        "clf__min_child_weight": [],
        "clf__reg_lambda": [],
    },
    "ordinal": {
        "clf__alpha": [],
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
