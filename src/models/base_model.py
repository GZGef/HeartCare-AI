"""
Базовый класс для моделей машинного обучения
"""

import os
import joblib
from typing import Optional
from abc import ABC, abstractmethod

from config.settings import RESULTS_CONFIG


class BaseModel(ABC):
    """
    Абстрактный базовый класс для всех моделей машинного обучения
    
    Обеспечивает:
    - Сохранение и загрузку моделей
    - Общий интерфейс для обучения и предсказания
    - Управление путями к результатам
    """

    def __init__(self, model_name: str):
        """
        Инициализация базовой модели
        
        Args:
            model_name: Уникальное имя модели для идентификации
        """
        self.model_name = model_name
        self.model = None
        self.is_trained = False
        
        # Пути к результатам
        self.models_folder = RESULTS_CONFIG['models_folder']
        self.model_extension = RESULTS_CONFIG['model_extension']
        
        # Создание папки для моделей, если не существует
        os.makedirs(self.models_folder, exist_ok=True)

    @abstractmethod
    def train(self, X_train, y_train) -> None:
        """
        Обучение модели
        
        Args:
            X_train: Обучающие признаки
            y_train: Обучающие метки
        """
        pass

    @abstractmethod
    def predict(self, X) -> None:
        """
        Предсказание на новых данных
        
        Args:
            X: Признаки для предсказания
            
        Returns:
            Предсказания модели
        """
        pass

    def save_model(self) -> str:
        """
        Сохранение модели на диск
        
        Returns:
            str: Путь к сохраненному файлу модели
        """
        if not self.is_trained:
            raise ValueError("Модель не обучена. Сначала вызовите train()")
        
        model_path = os.path.join(
            self.models_folder, 
            f"{self.model_name}{self.model_extension}"
        )
        
        joblib.dump(self.model, model_path)
        print(f"Модель '{self.model_name}' сохранена: {model_path}")
        
        return model_path

    def load_model(self, model_path: Optional[str] = None) -> None:
        """
        Загрузка модели с диска
        
        Args:
            model_path: Путь к файлу модели. Если None, используется путь по умолчанию
        """
        if model_path is None:
            model_path = os.path.join(
                self.models_folder, 
                f"{self.model_name}{self.model_extension}"
            )
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Файл модели не найден: {model_path}")
        
        self.model = joblib.load(model_path)
        self.is_trained = True
        print(f"Модель '{self.model_name}' загружена: {model_path}")

    def evaluate(self, X_test, y_test) -> dict:
        """
        Оценка качества модели на тестовых данных
        
        Args:
            X_test: Тестовые признаки
            y_test: Тестовые метки
            
        Returns:
            dict: Словарь с метриками качества
        """
        if not self.is_trained:
            raise ValueError("Модель не обучена. Сначала вызовите train()")
        
        # Точность на обучающей выборке (если доступна)
        metrics = {}
        
        # Точность на тестовой выборке
        test_score = self.model.score(X_test, y_test)
        metrics['test_accuracy'] = test_score
        
        return metrics

    def get_model_info(self) -> dict:
        """
        Получение информации о модели
        
        Returns:
            dict: Словарь с информацией о модели
        """
        info = {
            'model_name': self.model_name,
            'is_trained': self.is_trained,
            'model_type': self.model.__class__.__name__ if self.model else None,
        }
        
        if self.is_trained and hasattr(self.model, 'get_params'):
            info['parameters'] = self.model.get_params()
        
        return info

    def __str__(self) -> str:
        """Строковое представление модели"""
        info = self.get_model_info()
        return f"Model: {info['model_name']} | Type: {info['model_type']} | Trained: {info['is_trained']}"