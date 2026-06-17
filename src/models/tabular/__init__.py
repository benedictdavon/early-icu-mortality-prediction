"""Tabular model-suite registry."""

from __future__ import annotations

from models.tabular.catboost_model import CatBoostRiskModel
from models.tabular.ebm_model import EBMRiskModel
from models.tabular.extra_trees import ExtraTreesRiskModel
from models.tabular.lightgbm_model import LightGBMRiskModel
from models.tabular.logistic import LogisticRiskModel
from models.tabular.random_forest import RandomForestRiskModel
from models.tabular.xgboost_model import XGBoostRiskModel


MODEL_REGISTRY = {
    "logistic": LogisticRiskModel,
    "random_forest": RandomForestRiskModel,
    "extra_trees": ExtraTreesRiskModel,
    "xgboost": XGBoostRiskModel,
    "lightgbm": LightGBMRiskModel,
    "catboost": CatBoostRiskModel,
    "ebm": EBMRiskModel,
}


def get_model_class(name: str):
    key = str(name).strip().lower()
    try:
        return MODEL_REGISTRY[key]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown tabular model `{name}`. Available: {available}") from exc


__all__ = [
    "MODEL_REGISTRY",
    "get_model_class",
    "LogisticRiskModel",
    "RandomForestRiskModel",
    "ExtraTreesRiskModel",
    "XGBoostRiskModel",
    "LightGBMRiskModel",
    "CatBoostRiskModel",
    "EBMRiskModel",
]
