"""
Модуль предобработки данных для системы предсказания сердечной недостаточности
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

from config.settings import (
    PREPROCESSING_CONFIG,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
    TARGET_MAPPING
)


class DataPreprocessor:
    """
    Класс для предобработки данных
    
    Обеспечивает:
    - Кодирование категориальных признаков
    - Масштабирование числовых признаков
    - Удаление выбросов
    - Разделение на обучающую и тестовую выборки
    """

    def __init__(self):
        """
        Инициализация предобработчика данных
        """
        self.test_size = PREPROCESSING_CONFIG['test_size']
        self.random_state = PREPROCESSING_CONFIG['random_state']
        self.scale_range = PREPROCESSING_CONFIG['scale_range']
        self.drop_first = PREPROCESSING_CONFIG['drop_first']
        
        self.one_hot_encoder = None
        self.scaler = None

    def remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Удаление выбросов из датасета
        
        Args:
            df: Исходный DataFrame
            
        Returns:
            pd.DataFrame: DataFrame без выбросов
        """
        df_cleaned = df.copy()
        
        # Удаление записей с холестерином равным 0
        if 'Cholesterol' in df_cleaned.columns:
            initial_shape = df_cleaned.shape[0]
            df_cleaned = df_cleaned[df_cleaned['Cholesterol'] != 0]
            removed = initial_shape - df_cleaned.shape[0]
            if removed > 0:
                print(f"Удалено {removed} записей с Cholesterol = 0")
        
        return df_cleaned

    def encode_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Кодирование целевой переменной
        
        Args:
            df: DataFrame с данными
            
        Returns:
            Tuple[pd.DataFrame, np.ndarray]: Копия DataFrame и массив меток
        """
        df_encoded = df.copy()
        
        if TARGET_COLUMN in df_encoded.columns:
            # Маппинг целевой переменной
            df_encoded[TARGET_COLUMN] = df_encoded[TARGET_COLUMN].map(TARGET_MAPPING)
        
        return df_encoded

    def preprocess_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Предобработка признаков
        
        Args:
            df: DataFrame с данными
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Обработанные признаки и метки
        """
        # Копируем данные
        df_processed = df.copy()
        
        # Кодирование целевой переменной
        df_processed = self.encode_target(df_processed)
        
        # Разделение на признаки и целевую переменную
        X = df_processed.drop(columns=[TARGET_COLUMN])
        y = df_processed[TARGET_COLUMN].values
        
        # Выделение категориальных и числовых признаков
        categorical_columns = X[CATEGORICAL_FEATURES]
        numerical_columns = X[NUMERICAL_FEATURES]
        
        # One-Hot Encoding для категориальных признаков
        self.one_hot_encoder = OneHotEncoder(
            sparse_output=False, 
            drop='first' if self.drop_first else None
        )
        categorical_encoded = self.one_hot_encoder.fit_transform(categorical_columns)
        
        # Масштабирование числовых признаков
        self.scaler = MinMaxScaler(self.scale_range)
        numerical_scaled = self.scaler.fit_transform(numerical_columns)
        
        # Объединение обработанных признаков
        X_processed = np.hstack([numerical_scaled, categorical_encoded])
        
        print(f"Признаки обработаны:")
        print(f"- Числовые признаки: {numerical_columns.shape[1]}")
        print(f"- Категориальные признаки (после кодирования): {categorical_encoded.shape[1]}")
        print(f"- Итоговая размерность: {X_processed.shape}")
        
        return X_processed, y

    def split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Разделение данных на обучающую и тестовую выборки
        
        Args:
            X: Признаки
            y: Метки
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: X_train, X_test, y_train, y_test
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y  # Стратификация для сохранения распределения классов
        )
        
        print(f"\nРазделение данных:")
        print(f"- Обучающая выборка: {X_train.shape[0]} записей")
        print(f"- Тестовая выборка: {X_test.shape[0]} записей")
        print(f"- Соотношение: {self.test_size*100:.0f}% тест, {(1-self.test_size)*100:.0f}% обучение")
        
        return X_train, X_test, y_train, y_test

    def preprocess(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Полный цикл предобработки данных
        
        Args:
            df: Исходный DataFrame
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: X_train, X_test, y_train, y_test
        """
        print("\n" + "="*60)
        print("ПРЕДОБРАБОТКА ДАННЫХ")
        print("="*60)
        
        # Удаление выбросов
        df_cleaned = self.remove_outliers(df)
        
        # Предобработка признаков
        X_processed, y = self.preprocess_features(df_cleaned)
        
        # Разделение на выборки
        X_train, X_test, y_train, y_test = self.split_data(X_processed, y)
        
        print("\nПредобработка завершена успешно!")
        
        return X_train, X_test, y_train, y_test

    def transform_new_data(self, df: pd.DataFrame) -> np.ndarray:
        """
        Трансформация новых данных с использованием обученных кодировщиков
        
        Args:
            df: DataFrame с новыми данными
            
        Returns:
            np.ndarray: Обработанные признаки
        """
        if self.one_hot_encoder is None or self.scaler is None:
            raise ValueError("Кодировщики не обучены. Сначала вызовите preprocess()")
        
        # Копирование данных
        df_processed = df.copy()
        
        # Выделение категориальных и числовых признаков
        categorical_columns = df_processed[CATEGORICAL_FEATURES]
        numerical_columns = df_processed[NUMERICAL_FEATURES]
        
        # Применение обученных кодировщиков
        categorical_encoded = self.one_hot_encoder.transform(categorical_columns)
        numerical_scaled = self.scaler.transform(numerical_columns)
        
        # Объединение
        X_processed = np.hstack([numerical_scaled, categorical_encoded])
        
        return X_processed