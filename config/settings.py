"""
Конфигурационные параметры для системы предсказания сердечной недостаточности
"""

# Параметры данных
DATA_CONFIG = {
    'dataset_id': 'fedesoriano/heart-failure-prediction',
    'dataset_name': 'heart.csv',
    'data_folder': 'dataset',
    'separator': ',',
}

# Параметры предобработки
PREPROCESSING_CONFIG = {
    'test_size': 0.2,
    'random_state': 20,
    'scale_range': (-1, 1),
    'drop_first': True,  # Для OneHotEncoder
}

# Параметры модели Random Forest
RANDOM_FOREST_CONFIG = {
    'n_estimators': 100,
    'max_depth': 5,
    'random_state': 20,
}

# Параметры модели XGBoost
XGBOOST_CONFIG = {
    'n_estimators': 100,
    'learning_rate': 0.01,
    'max_depth': 3,
    'subsample': 0.7,
    'colsample_bytree': 0.2,
    'min_child_weight': 1,
    'objective': 'binary:logistic',
    'random_state': 5,
}

# Параметры визуализации
VISUALIZATION_CONFIG = {
    'figsize': (12, 10),
    'heatmap_cmap': 'coolwarm',
    'confusion_matrix_cmap': 'Blues',
    'dpi': 300,
}

# Пути к результатам
RESULTS_CONFIG = {
    'models_folder': 'results/models',
    'plots_folder': 'results/plots',
    'model_extension': '.joblib',
}

# Категориальные признаки
CATEGORICAL_FEATURES = [
    'Sex',
    'ChestPainType',
    'RestingECG',
    'ST_Slope',
]

# Числовые признаки
NUMERICAL_FEATURES = [
    'Age',
    'RestingBP',
    'Cholesterol',
    'FastingBS',
    'MaxHR',
    'Oldpeak',
]

# Целевая переменная
TARGET_COLUMN = 'ExerciseAngina'

# Маппинг целевой переменной
TARGET_MAPPING = {
    'N': 0,  # Нет сердечной недостаточности
    'Y': 1,  # Есть сердечная недостаточность
}

# Маппинг для визуализации
TARGET_LABELS = {
    0: 'N',
    1: 'Y',
}

# Названия графиков
PLOT_TITLES = {
    'pie_chart': 'Круговая диаграмма частоты случаев сердечной недостаточности',
    'distributions': 'Распределение медицинских показателей',
    'correlation': 'График корреляций',
    'confusion_matrix': 'Матрица ошибок',
}

# Описания признаков для документации
FEATURE_DESCRIPTIONS = {
    'Age': 'возраст пациента (лет)',
    'Sex': 'пол (M/F)',
    'ChestPainType': 'тип боли в груди (TA/ATA/NAP/ASY)',
    'RestingBP': 'артериальное давление в покое (mm Hg)',
    'Cholesterol': 'уровень холестерина (mm/dl)',
    'FastingBS': 'уровень сахара натощак (1: >120 mg/dl, 0: иначе)',
    'RestingECG': 'результат электрокардиограммы в покое (Normal/ST/LVH)',
    'MaxHR': 'максимальная частота сердечных сокращений',
    'ExerciseAngina': 'стенокардия при физической нагрузке (Y/N) - целевая переменная',
    'Oldpeak': 'ST-сегмент (числовое значение)',
    'ST_Slope': 'наклон ST-сегмента (Up/Flat/Down)',
}