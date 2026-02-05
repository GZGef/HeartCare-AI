"""
Основной скрипт для обучения моделей предсказания сердечной недостаточности
"""

import sys
import os

# Добавление корневой директории в путь для импорта модулей
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import DataLoader
from src.data_preprocessor import DataPreprocessor
from src.models.random_forest_model import RandomForestModel
from src.models.xgboost_model import XGBoostModel
from src.visualization.visualizer import Visualizer
from src.utils.helpers import (
    set_random_seed,
    validate_data_structure,
    print_validation_results,
    get_dataset_statistics,
    print_dataset_statistics,
    save_results_to_file
)


def main():
    """
    Основная функция для обучения моделей и визуализации результатов
    """
    print("\n" + "="*70)
    print("СИСТЕМА ПРЕДСКАЗАНИЯ СЕРДЕЧНОЙ НЕДОСТАТОЧНОСТИ")
    print("="*70)
    
    # 1. Установка seed для воспроизводимости
    print("\n1. Установка seed для воспроизводимости...")
    set_random_seed(42)
    
    # 2. Загрузка данных
    print("\n2. Загрузка данных...")
    try:
        loader = DataLoader()
        df = loader.load_data()
        loader.print_dataset_summary(df)
    except Exception as e:
        print(f"❌ Ошибка при загрузке данных: {e}")
        return
    
    # 3. Валидация данных
    print("\n3. Валидация структуры данных...")
    validation_result = validate_data_structure(df)
    print_validation_results(validation_result)
    
    if not validation_result['is_valid']:
        print("❌ Данные не прошли валидацию. Программа завершена.")
        return
    
    # 4. Получение статистики по датасету
    print("\n4. Получение статистики по датасету...")
    stats = get_dataset_statistics(df)
    print_dataset_statistics(stats)
    
    # 5. Предобработка данных
    print("\n5. Предобработка данных...")
    try:
        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.preprocess(df)
    except Exception as e:
        print(f"❌ Ошибка при предобработке данных: {e}")
        return
    
    # 6. Обучение модели Random Forest
    print("\n6. Обучение модели Random Forest...")
    try:
        rf_model = RandomForestModel()
        rf_model.train(X_train, y_train)
        
        # Оценка на тестовых данных
        rf_test_score = rf_model.model.score(X_test, y_test)
        print(f"Точность Random Forest на тестовых данных: {rf_test_score*100:.2f}%")
        
        # Сохранение модели
        rf_model.save_model()
    except Exception as e:
        print(f"❌ Ошибка при обучении Random Forest: {e}")
        rf_model = None
    
    # 7. Обучение модели XGBoost
    print("\n7. Обучение модели XGBoost...")
    try:
        xgb_model = XGBoostModel()
        xgb_model.train(X_train, y_train)
        
        # Оценка на тестовых данных
        xgb_test_score = xgb_model.model.score(X_test, y_test)
        print(f"Точность XGBoost на тестовых данных: {xgb_test_score*100:.2f}%")
        
        # Сохранение модели
        xgb_model.save_model()
    except Exception as e:
        print(f"❌ Ошибка при обучении XGBoost: {e}")
        xgb_model = None
    
    # 8. Предсказание на тестовых данных (используем XGBoost)
    print("\n8. Предсказание на тестовых данных...")
    if xgb_model and xgb_model.is_trained:
        y_pred = xgb_model.predict(X_test)
        
        # 9. Визуализация результатов
        print("\n9. Создание визуализаций...")
        try:
            visualizer = Visualizer()
            
            # Получение важности признаков
            feature_importance = xgb_model.get_feature_importance()
            
            # Создание всех визуализаций
            visualizer.plot_all_visualizations(
                df=df,
                y_true=y_test,
                y_pred=y_pred,
                feature_importance=feature_importance
            )
        except Exception as e:
            print(f"❌ Ошибка при создании визуализаций: {e}")
        
        # 10. Тестовое прогнозирование
        print("\n10. Тестовое прогнозирование...")
        try:
            import random
            random_idx = random.randint(0, len(X_test) - 1)
            random_sample = X_test[random_idx]
            true_label = y_test[random_idx]
            
            # Предсказание обеими моделями
            if rf_model and rf_model.is_trained:
                rf_pred = rf_model.predict([random_sample])[0]
                print(f"\nСлучайный индекс: {random_idx}")
                print(f"Прогноз Random Forest: {'Y' if rf_pred == 1 else 'N'}")
            
            if xgb_model and xgb_model.is_trained:
                xgb_pred = xgb_model.predict([random_sample])[0]
                print(f"Прогноз XGBoost: {'Y' if xgb_pred == 1 else 'N'}")
            
            print(f"Настоящая метка: {'Y' if true_label == 1 else 'N'}")
        except Exception as e:
            print(f"❌ Ошибка при тестовом прогнозировании: {e}")
    
    # 11. Сохранение результатов
    print("\n11. Сохранение результатов...")
    try:
        results = {
            'Модель': 'XGBoost',
            'Точность на тесте': f"{xgb_test_score*100:.2f}%" if xgb_model else "N/A",
            'Точность на обучении': f"{xgb_model.model.score(X_train, y_train)*100:.2f}%" if xgb_model else "N/A",
            'Параметры XGBoost': {
                'n_estimators': 100,
                'learning_rate': 0.01,
                'max_depth': 3,
            },
            'Размерность данных': df.shape,
            'Размер обучающей выборки': X_train.shape,
            'Размер тестовой выборки': X_test.shape,
        }
        
        save_results_to_file(results, "training_results.txt")
    except Exception as e:
        print(f"❌ Ошибка при сохранении результатов: {e}")
    
    # 12. Итоговый отчет
    print("\n" + "="*70)
    print("ИТОГОВЫЙ ОТЧЕТ")
    print("="*70)
    
    if rf_model and rf_model.is_trained:
        print(f"\n✅ Random Forest:")
        print(f"   - Точность на обучении: {rf_model.model.score(X_train, y_train)*100:.2f}%")
        print(f"   - Точность на тесте: {rf_test_score*100:.2f}%")
    
    if xgb_model and xgb_model.is_trained:
        print(f"\n✅ XGBoost:")
        print(f"   - Точность на обучении: {xgb_model.model.score(X_train, y_train)*100:.2f}%")
        print(f"   - Точность на тесте: {xgb_test_score*100:.2f}%")
    
    print(f"\n📊 Данные:")
    print(f"   - Всего сэмплов: {df.shape[0]}")
    print(f"   - Всего признаков: {df.shape[1]}")
    print(f"   - Обучающая выборка: {X_train.shape[0]} сэмплов")
    print(f"   - Тестовая выборка: {X_test.shape[0]} сэмплов")
    
    print(f"\n📁 Результаты сохранены в папке 'results/'")
    print(f"   - Модели: results/models/")
    print(f"   - Графики: results/plots/")
    print(f"   - Текстовые отчеты: results/")
    
    print("\n" + "="*70)
    print("✅ ПРОГРАММА ЗАВЕРШЕНА УСПЕШНО!")
    print("="*70)


if __name__ == "__main__":
    main()