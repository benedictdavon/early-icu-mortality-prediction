"""XGBoost parameter spaces and defaults."""

from __future__ import annotations


def xgboost_search_space(base_pos_weight: float) -> dict:
    return {
        "n_estimators": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        "learning_rate": [0.01, 0.05, 0.07, 0.1, 0.2, 0.5, 1, 7],
        "max_depth": [3, 5, 7, 9],
        "min_child_weight": [1, 3, 5],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "gamma": [0, 0.1, 0.2, 0.3],
        "reg_lambda": [1, 2, 5, 10],
        "scale_pos_weight": [base_pos_weight * f for f in (0.5, 1, 2, 4)],
    }


def xgboost_default_params() -> dict:
    return {
        "n_estimators": 1000,
        "learning_rate": 0.01,
        "max_depth": 5,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "gamma": 0.2,
        "reg_lambda": 5,
        "scale_pos_weight": 1.9,
        "objective": "binary:logistic",
        "tree_method": "hist",
        "device": "cuda",
        "eval_metric": "aucpr",
    }


def xgboost_ensemble_default_params(scale_pos_weight: float) -> dict:
    return {
        "n_estimators": 600,
        "learning_rate": 0.01,
        "max_depth": 7,
        "min_child_weight": 1,
        "subsample": 0.6,
        "colsample_bytree": 0.6,
        "gamma": 0.2,
        "reg_lambda": 1,
        "scale_pos_weight": scale_pos_weight,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "verbosity": 0,
        "tree_method": "hist",
        "device": "cuda",
    }


def coerce_xgboost_params(params: dict) -> dict:
    """Normalize JSON/string-loaded parameter values before model construction."""
    coerced = params.copy()
    int_keys = {"n_estimators", "max_depth", "min_child_weight"}
    float_keys = {
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "gamma",
        "reg_lambda",
        "scale_pos_weight",
    }
    for key in int_keys.intersection(coerced):
        coerced[key] = int(coerced[key])
    for key in float_keys.intersection(coerced):
        coerced[key] = float(coerced[key])
    return coerced
