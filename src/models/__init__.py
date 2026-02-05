"""
Пакет моделей машинного обучения
"""

from .base_model import BaseModel
from .random_forest_model import RandomForestModel
from .xgboost_model import XGBoostModel

__all__ = ['BaseModel', 'RandomForestModel', 'XGBoostModel']