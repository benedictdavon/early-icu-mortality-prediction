"""Random forest model package."""

from models.random_forest.model import ICUMortalityRandomForest
from models.random_forest.bagging import ICUMortalityRandomForestBagging

__all__ = ["ICUMortalityRandomForest", "ICUMortalityRandomForestBagging"]
