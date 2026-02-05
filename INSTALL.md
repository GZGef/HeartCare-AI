# Инструкция по установке и использованию

## 📋 Предварительные требования

### Системные требования
- Python 3.8 или выше
- Windows 10/11, Linux или macOS
- 4+ GB RAM (рекомендуется 8+ GB)
- 2+ GB свободного места на диске

### Необходимые инструменты
- Git (для клонирования репозитория)
- pip (менеджер пакетов Python)

## 🚀 Быстрая установка

### 1. Клонирование репозитория

```bash
git clone https://github.com/YOUR_USERNAME/HeartCare-AI.git
cd HeartCare-AI
```

### 2. Создание виртуального окружения

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Установка зависимостей

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Подготовка данных

#### Вариант A: Использовать существующий датасет

Поместите файл `heart.csv` в папку `dataset/`:

```
HeartCare-AI/
├── dataset/
│   └── heart.csv    # Ваш файл данных
├── src/
├── scripts/
└── ...
```

#### Вариант B: Скачать с Kaggle

1. Перейдите на [Kaggle Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
2. Скачайте архив `heart-failure-prediction.zip`
3. Поместите архив в корневую папку проекта
4. Скрипт автоматически разархивирует данные

### 5. Запуск обучения моделей

```bash
python scripts/train.py
```

## 📊 Результаты работы

После выполнения скрипта будут созданы:

### Сохраненные модели
```
results/
├── models/
│   ├── random_forest.joblib    # Модель Random Forest
│   └── xgboost.joblib          # Модель XGBoost
```

### Текстовые отчеты
```
results/
├── training_results.txt        # Результаты обучения
└── ...                         # Другие отчеты
```

### Графики (если успешно созданы)
```
results/
├── plots/
│   ├── pie_chart.png           # Круговая диаграмма
│   ├── distributions.png       # Распределения признаков
│   ├── correlation_matrix.png  # Корреляционная матрица
│   └── confusion_matrix.png    # Матрица ошибок
```

## 🎯 Использование в коде

### Пример 1: Загрузка и обучение моделей

```python
from src.data_loader import DataLoader
from src.data_preprocessor import DataPreprocessor
from src.models.random_forest_model import RandomForestModel
from src.models.xgboost_model import XGBoostModel

# Загрузка данных
loader = DataLoader()
df = loader.load_data()

# Предобработка
preprocessor = DataPreprocessor()
X_train, X_test, y_train, y_test = preprocessor.preprocess(df)

# Обучение моделей
rf_model = RandomForestModel()
rf_model.train(X_train, y_train)

xgb_model = XGBoostModel()
xgb_model.train(X_train, y_train)

# Сохранение моделей
rf_model.save_model()
xgb_model.save_model()
```

### Пример 2: Предсказание на новых данных

```python
from src.models.xgboost_model import XGBoostModel
from src.data_preprocessor import DataPreprocessor
import pandas as pd

# Загрузка обученной модели
model = XGBoostModel()
model.load_model('results/models/xgboost.joblib')

# Подготовка новых данных
new_data = pd.DataFrame({
    'Age': [55],
    'Sex': ['M'],
    'ChestPainType': ['ASY'],
    'RestingBP': [140],
    'Cholesterol': [250],
    'FastingBS': [1],
    'RestingECG': ['Normal'],
    'MaxHR': [150],
    'ExerciseAngina': ['N'],  # Заглушка, будет удалена
    'Oldpeak': [1.5],
    'ST_Slope': ['Flat']
})

# Предобработка
preprocessor = DataPreprocessor()
X_new = preprocessor.transform_new_data(new_data)

# Предсказание
prediction = model.predict(X_new)
probability = model.predict_proba(X_new)

print(f"Предсказание: {'Есть риск' if prediction[0] == 1 else 'Нет риска'}")
print(f"Вероятность: {probability[0][1]:.2%}")
```

### Пример 3: Визуализация результатов

```python
from src.visualization.visualizer import Visualizer
from src.models.xgboost_model import XGBoostModel

# Загрузка модели и данных
model = XGBoostModel()
model.load_model('results/models/xgboost.joblib')

# Создание визуализатора
visualizer = Visualizer()

# Важность признаков
feature_importance = model.get_feature_importance()
visualizer.plot_feature_importance(feature_importance)
```

## 📱 Использование Jupyter Notebook

### Запуск Jupyter

```bash
jupyter notebook
```

### Открытие notebook

1. В браузере откройте `notebooks/analysis.ipynb`
2. Выполняйте ячейки последовательно
3. Изучайте анализ данных и результаты

## 🔧 Устранение проблем

### Проблема: Отсутствует модуль

**Решение:**
```bash
pip install -r requirements.txt
```

### Проблема: Файл данных не найден

**Решение:**
- Убедитесь, что `heart.csv` находится в папке `dataset/`
- Или скачайте архив с Kaggle и поместите в корень проекта

### Проблема: Ошибка при создании графиков

**Решение:**
- Установите дополнительные зависимости для plotly:
```bash
pip install kaleido
```

### Проблема: Недостаточно памяти

**Решение:**
- Уменьшите параметры моделей в `config/settings.py`:
```python
RANDOM_FOREST_CONFIG = {
    'n_estimators': 50,  # Было 100
    'max_depth': 3,      # Было 5
    'random_state': 20,
}
```

## 📈 Производительность

### Время выполнения
- Загрузка данных: ~1-2 секунды
- Предобработка: ~2-3 секунды
- Обучение Random Forest: ~5-10 секунд
- Обучение XGBoost: ~3-5 секунд
- Визуализация: ~2-5 секунд

**Итого:** ~15-25 секунд на полный цикл

### Потребление памяти
- Данные: ~1-2 MB
- Модели: ~1-2 MB каждая
- Итого: ~5-10 MB

## 🎓 Дополнительные возможности

### 1. Кросс-валидация

```python
from sklearn.model_selection import cross_val_score
from src.models.xgboost_model import XGBoostModel

model = XGBoostModel()
model.train(X_train, y_train)

# Кросс-валидация
scores = cross_val_score(model.model, X_train, y_train, cv=5)
print(f"Средняя точность: {scores.mean():.2%}")
```

### 2. Подбор гиперпараметров

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}

grid_search = GridSearchCV(
    XGBoostModel().model,
    param_grid,
    cv=5,
    scoring='accuracy'
)

grid_search.fit(X_train, y_train)
print(f"Лучшие параметры: {grid_search.best_params_}")
```

### 3. Сохранение отчетов

```python
from src.utils.helpers import save_results_to_file

results = {
    'Модель': 'XGBoost',
    'Точность': '75.33%',
    'Параметры': {'n_estimators': 100, 'learning_rate': 0.01}
}

save_results_to_file(results, 'my_report.txt')
```

## 📚 Документация

Для получения дополнительной информации:

1. **README.md** - общее описание проекта
2. **PROJECT_NAME_SUGGESTION.md** - рекомендации по названию
3. **notebooks/analysis.ipynb** - интерактивный анализ
4. **src/** - документация в комментариях кода

## 🤝 Поддержка

Если возникли проблемы:

1. Проверьте раздел "Устранение проблем" выше
2. Изучите вывод скрипта для деталей ошибки
3. Убедитесь, что все зависимости установлены
4. Проверьте наличие файла данных

## 📄 Лицензия

MIT License - свободное использование для образовательных и коммерческих целей.

---

**Версия документации:** 1.0.0  
**Последнее обновление:** 05.02.2026