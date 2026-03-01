"""
Вспомогательные функции для системы предсказания сердечной недостаточности
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import random

from config.settings import TARGET_LABELS, TARGET_COLUMN


def set_random_seed(seed: int = 42) -> None:
    """
    Установка случайного seed для воспроизводимости результатов
    
    Args:
        seed: Значение seed
    """
    random.seed(seed)
    np.random.seed(seed)
    print(f"Установлен seed для воспроизводимости: {seed}")


def format_prediction(prediction: int, probability: Optional[float] = None) -> str:
    """
    Форматирование предсказания для вывода
    
    Args:
        prediction: Предсказанная метка (0 или 1)
        probability: Вероятность предсказания (опционально)
        
    Returns:
        str: Отформатированная строка с предсказанием
    """
    label = TARGET_LABELS.get(prediction, 'N/A')
    
    if probability is not None:
        return f"Предсказание: {label} (вероятность: {probability:.2%})"
    else:
        return f"Предсказание: {label}"


def calculate_metrics_summary(metrics: Dict[str, float]) -> str:
    """
    Форматирование метрик в читаемый вид
    
    Args:
        metrics: Словарь с метриками
        
    Returns:
        str: Отформатированная строка с метриками
    """
    summary = "\n" + "="*50 + "\n"
    summary += "СВОДКА МЕТРИК КАЧЕСТВА\n"
    summary += "="*50 + "\n"
    
    for metric_name, metric_value in metrics.items():
        if 'accuracy' in metric_name or 'score' in metric_name:
            summary += f"{metric_name}: {metric_value*100:.2f}%\n"
        else:
            summary += f"{metric_name}: {metric_value}\n"
    
    summary += "="*50 + "\n"
    return summary


def validate_data_structure(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Валидация структуры данных
    
    Args:
        df: DataFrame для валидации
        
    Returns:
        dict: Результаты валидации
    """
    validation_result = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'info': {}
    }
    
    # Проверка наличия необходимых столбцов
    required_columns = ['ExerciseAngina', 'Cholesterol', 'Age', 'RestingBP', 'Oldpeak']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        validation_result['is_valid'] = False
        validation_result['errors'].append(f"Отсутствуют необходимые столбцы: {missing_columns}")
    
    # Проверка пропущенных значений
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        validation_result['warnings'].append(
            f"Обнаружены пропущенные значения: {missing_values[missing_values > 0].to_dict()}"
        )
    
    # Проверка типа данных целевой переменной
    if TARGET_COLUMN in df.columns:
        unique_values = df[TARGET_COLUMN].unique()
        valid_values = ['N', 'Y']
        invalid_values = [val for val in unique_values if val not in valid_values]
        if invalid_values:
            validation_result['errors'].append(
                f"Недопустимые значения в целевой переменной: {invalid_values}"
            )
    
    # Информация о данных
    validation_result['info'] = {
        'shape': df.shape,
        'columns': list(df.columns),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
    }
    
    return validation_result


def print_validation_results(validation_result: Dict[str, Any]) -> None:
    """
    Вывод результатов валидации
    
    Args:
        validation_result: Результаты валидации
    """
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ ВАЛИДАЦИИ ДАННЫХ")
    print("="*60)
    
    if validation_result['is_valid']:
        print("✅ Структура данных валидна")
    else:
        print("❌ Обнаружены ошибки:")
        for error in validation_result['errors']:
            print(f"  - {error}")
    
    if validation_result['warnings']:
        print("\n⚠️ Предупреждения:")
        for warning in validation_result['warnings']:
            print(f"  - {warning}")
    
    print("\n📊 Информация о данных:")
    info = validation_result['info']
    print(f"  - Размерность: {info['shape']}")
    print(f"  - Использование памяти: {info['memory_usage_mb']:.2f} MB")
    
    print("="*60)


def create_test_sample(df: pd.DataFrame, n_samples: int = 5) -> pd.DataFrame:
    """
    Создание тестовой выборки для демонстрации
    
    Args:
        df: Исходный DataFrame
        n_samples: Количество сэмплов
        
    Returns:
        pd.DataFrame: Тестовая выборка
    """
    if len(df) < n_samples:
        n_samples = len(df)
    
    # Случайный выбор индексов
    random_indices = random.sample(range(len(df)), n_samples)
    
    return df.iloc[random_indices].copy()


def save_results_to_file(results: Dict[str, Any], filename: str = "results.txt", indent: int = 0) -> None:
    """
    Сохранение результатов в текстовый файл с поддержкой вложенных словарей
    
    Args:
        results: Словарь с результатами
        filename: Имя файла
        indent: Уровень отступа (для рекурсии)
    """
    import os
    
    os.makedirs('results', exist_ok=True)
    filepath = os.path.join('results', filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА\n")
        f.write("="*50 + "\n\n")
        
        for key, value in results.items():
            prefix = "  " * indent
            if isinstance(value, dict):
                f.write(f"{prefix}{key}:\n")
                for k, v in value.items():
                    if isinstance(v, dict):
                        f.write(f"{prefix}  {k}:\n")
                        for kk, vv in v.items():
                            f.write(f"{prefix}    {kk}: {vv}\n")
                    else:
                        f.write(f"{prefix}  {k}: {v}\n")
            else:
                f.write(f"{prefix}{key}: {value}\n")
            f.write("\n")
    
    print(f"Результаты сохранены в файл: {filepath}")


def get_dataset_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Получение подробной статистики по датасету
    
    Args:
        df: DataFrame с данными
        
    Returns:
        dict: Статистика по датасету
    """
    stats = {
        'basic': {
            'total_samples': len(df),
            'total_features': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
        },
        'target_distribution': {},
        'numerical_features': {},
        'categorical_features': {},
    }
    
    # Распределение целевой переменной
    if 'ExerciseAngina' in df.columns:
        target_counts = df['ExerciseAngina'].value_counts()
        target_percent = df['ExerciseAngina'].value_counts(normalize=True) * 100
        stats['target_distribution'] = {
            'counts': target_counts.to_dict(),
            'percentages': target_percent.to_dict(),
        }
    
    # Статистика по числовым признакам
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        stats['numerical_features'][col] = {
            'mean': df[col].mean(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'median': df[col].median(),
        }
    
    # Статистика по категориальным признакам
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        stats['categorical_features'][col] = {
            'unique_values': df[col].nunique(),
            'most_frequent': df[col].mode().iloc[0] if not df[col].mode().empty else None,
            'distribution': df[col].value_counts().to_dict(),
        }
    
    return stats


def print_dataset_statistics(stats: Dict[str, Any]) -> None:
    """
    Вывод статистики по датасету
    
    Args:
        stats: Статистика по датасету
    """
    print("\n" + "="*60)
    print("СТАТИСТИКА ДАТАСЕТА")
    print("="*60)
    
    print("\n📊 Основная информация:")
    basic = stats['basic']
    print(f"  - Всего сэмплов: {basic['total_samples']}")
    print(f"  - Всего признаков: {basic['total_features']}")
    print(f"  - Использование памяти: {basic['memory_usage_mb']:.2f} MB")
    
    if stats['target_distribution']:
        print("\n🎯 Распределение целевой переменной:")
        target = stats['target_distribution']
        for label, count in target['counts'].items():
            percent = target['percentages'][label]
            print(f"  - {label}: {count} ({percent:.1f}%)")
    
    if stats['numerical_features']:
        print("\n📈 Числовые признаки:")
        for col, values in stats['numerical_features'].items():
            print(f"  - {col}:")
            print(f"    среднее: {values['mean']:.2f}, std: {values['std']:.2f}")
            print(f"    min: {values['min']:.2f}, max: {values['max']:.2f}")
    
    if stats['categorical_features']:
        print("\n📋 Категориальные признаки:")
        for col, values in stats['categorical_features'].items():
            print(f"  - {col}: {values['unique_values']} уникальных значений")
            print(f"    самое частое: {values['most_frequent']}")
    
    print("="*60)