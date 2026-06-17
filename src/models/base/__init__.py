"""Shared model base classes and utilities."""

from models.base.model import ICUMortalityBaseModel
from models.base.risk_model import BaseRiskModel, OptionalDependencyUnavailable

__all__ = [
    "BaseRiskModel",
    "ICUMortalityBaseModel",
    "OptionalDependencyUnavailable",
]
