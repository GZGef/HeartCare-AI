"""
Модель XGBoost для предсказания сердечной недостаточности
"""

import xgboost as xgb

from config.settings import XGBOOST_CONFIG
from src.models.base_model import BaseModel


class XGBoostModel(BaseModel):
    """
    Класс модели XGBoost для бинарной классификации
    
    Наследует функциональность базовой модели и реализует
    специфичные методы для XGBoost
    """

    def __init__(self):
        """
        Инициализация модели XGBoost
        """
        super().__init__(model_name="xgboost")
        
        # Параметры модели из конфигурации
        self.n_estimators = XGBOOST_CONFIG['n_estimators']
        self.learning_rate = XGBOOST_CONFIG['learning_rate']
        self.max_depth = XGBOOST_CONFIG['max_depth']
        self.subsample = XGBOOST_CONFIG['subsample']
        self.colsample_bytree = XGBOOST_CONFIG['colsample_bytree']
        self.min_child_weight = XGBOOST_CONFIG['min_child_weight']
        self.objective = XGBOOST_CONFIG['objective']
        self.random_state = XGBOOST_CONFIG['random_state']

    def train(self, X_train, y_train) -> None:
        """
        Обучение модели XGBoost
        
        Args:
            X_train: Обучающие признаки
            y_train: Обучающие метки
        """
        print(f"\nОбучение модели XGBoost...")
        
        # Инициализация модели
        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            objective=self.objective,
            random_state=self.random_state,
            n_jobs=-1,  # Использовать все доступные ядра
            eval_metric='logloss'  # Метрика для бинарной классификации
        )
        
        # Обучение
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Расчет точности на обучающей выборке
        train_score = self.model.score(X_train, y_train)
        
        print(f"Модель XGBoost обучена успешно!")
        print(f"Точность на обучающей выборке: {train_score*100:.2f}%")
        print(f"Параметры: {self.n_estimators} деревьев, скорость обучения {self.learning_rate}")

    def predict(self, X) -> None:
        """
        Предсказание на новых данных
        
        Args:
            X: Признаки для предсказания
            
        Returns:
            array: Предсказания модели
        """
        if not self.is_trained:
            raise ValueError("Модель не обучена. Сначала вызовите train()")
        
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Предсказание вероятностей классов
        
        Args:
            X: Признаки для предсказания
            
        Returns:
            array: Вероятности классов
        """
        if not self.is_trained:
            raise ValueError("Модель не обучена. Сначала вызовите train()")
        
        return self.model.predict_proba(X)

    def get_feature_importance(self, feature_names=None):
        """
        Получение важности признаков
        
        Args:
            feature_names: Названия признаков (опционально)
            
        Returns:
            dict: Важность каждого признака
        """
        if not self.is_trained:
            raise ValueError("Модель не обучена. Сначала вызовите train()")
        
        importances = self.model.feature_importances_
        
        if feature_names is None:
            # Генерация стандартных названий
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        
        # Сортировка по важности
        feature_importance_dict = dict(zip(feature_names, importances))
        sorted_importance = dict(sorted(
            feature_importance_dict.items(), 
            key=lambda x: x[1], 
            reverse=True
        ))
        
        return sorted_importance

    def evaluate(self, X_test, y_test) -> dict:
        """
        Оценка качества модели
        
        Args:
            X_test: Тестовые признаки
            y_test: Тестовые метки
            
        Returns:
            dict: Метрики качества
        """
        metrics = super().evaluate(X_test, y_test)
        
        # Дополнительные метрики
        if self.is_trained:
            # Точность на обучающей выборке
            train_score = self.model.score(X_test, y_test)  # Временно используем тест для совместимости
            metrics['train_accuracy'] = train_score
            
            # Важность признаков
            metrics['feature_importance'] = self.get_feature_importance()
        
        return metrics