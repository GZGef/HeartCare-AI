"""
Модуль визуализации данных для системы предсказания сердечной недостаточности
"""

import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plotly import express as px
from plotly.io import renderers
from sklearn.metrics import confusion_matrix, classification_report

# Настройка рендера Plotly: открывать как статическое изображение, а не в браузере
renderers.default = "png"

from config.settings import (
    VISUALIZATION_CONFIG,
    PLOT_TITLES,
    TARGET_COLUMN,
    TARGET_MAPPING,
    TARGET_LABELS,
)

# Отключаем интерактивное отображение графиков plotly (только сохранение)
import plotly.io as pio
pio.renderers.default = "png"

# Выключаем интерактивный режим matplotlib, чтобы окна не всплывали сами
plt.ioff()


class Visualizer:
    """
    Класс для визуализации данных и результатов моделирования

    Обеспечивает:
    - Визуализацию распределения признаков
    - Корреляционные матрицы
    - Матрицы ошибок
    - Сохранение графиков
    """

    def __init__(self):
        """
        Инициализация визуализатора
        """
        self.figsize = VISUALIZATION_CONFIG["figsize"]
        self.heatmap_cmap = VISUALIZATION_CONFIG["heatmap_cmap"]
        self.confusion_matrix_cmap = VISUALIZATION_CONFIG["confusion_matrix_cmap"]
        self.dpi = VISUALIZATION_CONFIG["dpi"]

        self.plots_folder = VISUALIZATION_CONFIG.get("plots_folder", "results/plots")
        os.makedirs(self.plots_folder, exist_ok=True)

    def save_plot(self, fig, filename: str) -> str:
        """
        Сохранение графика на диск

        Args:
            fig: Объект фигуры matplotlib или plotly
            filename: Имя файла для сохранения

        Returns:
            str: Путь к сохраненному файлу
        """
        filepath = os.path.join(self.plots_folder, filename)

        if hasattr(fig, "write_image"):
            # Plotly figure
            # Plotly write_image doesn't support 'dpi' argument directly.
            # We can use scale to increase resolution if needed.
            scale = self.dpi / 72.0 if self.dpi > 72 else 1
            fig.write_image(filepath, scale=scale)
        else:
            # Matplotlib figure
            fig.savefig(filepath, dpi=self.dpi, bbox_inches="tight")

        print(f"График сохранен: {filepath}")
        return filepath

    def plot_pie_chart(self, df: pd.DataFrame, save: bool = True) -> None:
        """
        Круговая диаграмма частоты случаев сердечной недостаточности

        Args:
            df: DataFrame с данными
            save: Сохранять ли график
        """
        if TARGET_COLUMN not in df.columns:
            print(f"Столбец '{TARGET_COLUMN}' не найден в данных")
            return

        # Подсчет случаев
        exercise_angina_counts = df[TARGET_COLUMN].value_counts().reset_index()
        exercise_angina_counts[TARGET_COLUMN] = exercise_angina_counts[
            TARGET_COLUMN
        ].replace(
            {
                "N": "Нет сердечной недостаточности",
                "Y": "Есть сердечная недостаточность",
            }
        )

        # Создание графика
        fig = px.pie(
            data_frame=exercise_angina_counts,
            names=TARGET_COLUMN,
            values="count",
            title=PLOT_TITLES["pie_chart"],
        )

        # Графики сохраняются в файл, без открытия в браузере
        if save:
            self.save_plot(fig, "pie_chart.png")

    def plot_distributions(self, df: pd.DataFrame, save: bool = True) -> None:
        """
        Распределение медицинских показателей

        Args:
            df: DataFrame с данными
            save: Сохранять ли график
        """
        plt.figure(figsize=self.figsize)

        # Список числовых признаков для визуализации
        numerical_features = ["Cholesterol", "Age", "RestingBP", "Oldpeak"]

        for i, feature in enumerate(numerical_features, 1):
            plt.subplot(2, 2, i)
            sns.histplot(df[feature], kde=True, ax=plt.gca())
            plt.title(f'Распределение "{feature}"')
            plt.xlabel("Значение")
            plt.ylabel("Частота")

        plt.suptitle(PLOT_TITLES["distributions"], fontsize=16)
        plt.tight_layout()

        if save:
            self.save_plot(plt.gcf(), "distributions.png")

        plt.close()

    def plot_correlation_matrix(self, df: pd.DataFrame, save: bool = True) -> None:
        """
        Тепловая карта корреляций между числовыми признаками

        Args:
            df: DataFrame с данными
            save: Сохранять ли график
        """
        # Выбор числовых столбцов (исключая категориальные)
        num_columns = df.select_dtypes(include=[np.number])

        plt.figure(figsize=self.figsize)

        # Расчет корреляций
        corr_matrix = num_columns.corr()

        # Визуализация
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap=self.heatmap_cmap,
            fmt=".2f",
            square=True,
            cbar_kws={"shrink": 0.8},
        )

        plt.title(PLOT_TITLES["correlation"], fontsize=16)
        plt.tight_layout()

        if save:
            self.save_plot(plt.gcf(), "correlation_matrix.png")

        plt.close()

    def plot_confusion_matrix(self, y_true, y_pred, save: bool = True) -> None:
        """
        Матрица ошибок классификации

        Args:
            y_true: Истинные метки
            y_pred: Предсказанные метки
            save: Сохранять ли график
        """
        # Преобразование меток в строки для визуализации
        y_true_str = np.where(y_true == 0, "N", "Y")
        y_pred_str = np.where(y_pred == 0, "N", "Y")

        # Вычисление матрицы ошибок
        cm = confusion_matrix(y_true_str, y_pred_str, labels=["N", "Y"])

        # Нормализация
        cm_normalized = cm / cm.sum(axis=1, keepdims=True)

        plt.figure(figsize=(10, 10))

        # Визуализация
        sns.heatmap(
            cm_normalized,
            xticklabels=["N", "Y"],
            yticklabels=["N", "Y"],
            cmap=self.confusion_matrix_cmap,
            annot=True,
            fmt=".4f",
            square=True,
            cbar_kws={"shrink": 0.8},
        )

        plt.title(PLOT_TITLES["confusion_matrix"], fontsize=16)
        plt.ylabel("Фактические метки")
        plt.xlabel("Прогнозируемые метки")
        plt.tight_layout()

        if save:
            self.save_plot(plt.gcf(), "confusion_matrix.png")

        plt.close()

    def plot_classification_report(self, y_true, y_pred) -> None:
        """
        Вывод отчета о качестве классификации

        Args:
            y_true: Истинные метки
            y_pred: Предсказанные метки
        """
        # Преобразование меток в строки
        y_true_str = np.where(y_true == 0, "N", "Y")
        y_pred_str = np.where(y_pred == 0, "N", "Y")

        print("\n" + "=" * 60)
        print("ОТЧЕТ О КАЧЕСТВЕ КЛАССИФИКАЦИИ")
        print("=" * 60)
        print(classification_report(y_true_str, y_pred_str))
        print("=" * 60)

    def plot_feature_importance(
        self, feature_importance: dict, save: bool = True
    ) -> None:
        """
        Визуализация важности признаков

        Args:
            feature_importance: Словарь с важностью признаков
            save: Сохранять ли график
        """
        if not feature_importance:
            print("Нет данных о важности признаков")
            return

        # Преобразование в DataFrame для удобства
        importance_df = pd.DataFrame(
            list(feature_importance.items()), columns=["Feature", "Importance"]
        )

        plt.figure(figsize=(12, 8))

        # Столбчатая диаграмма
        sns.barplot(
            data=importance_df.head(15),  # Топ-15 признаков
            x="Importance",
            y="Feature",
            palette="viridis",
        )

        plt.title("Важность признаков", fontsize=16)
        plt.xlabel("Важность")
        plt.ylabel("Признак")
        plt.tight_layout()

        if save:
            self.save_plot(plt.gcf(), "feature_importance.png")

        plt.close()

    def plot_all_visualizations(
        self, df: pd.DataFrame, y_true, y_pred, feature_importance: dict = None
    ) -> list:
        """
        Создание всех визуализаций

        Args:
            df: DataFrame с данными
            y_true: Истинные метки
            y_pred: Предсказанные метки
            feature_importance: Важность признаков (опционально)

        Returns:
            list: Список путей к сохраненным файлам графиков
        """
        saved_plots = []

        print("\n" + "=" * 60)
        print("СОЗДАНИЕ ВИЗУАЛИЗАЦИЙ")
        print("=" * 60)
        print("\nВНИМАНИЕ: Графики сохраняются в фоновом режиме.")
        print("=" * 60)

        # 1. Круговая диаграмма
        print("\n1. Создание круговой диаграммы...")
        self.plot_pie_chart(df)
        saved_plots.append("pie_chart.png")

        # 2. Распределения
        print("\n2. Создание графиков распределения...")
        self.plot_distributions(df)
        saved_plots.append("distributions.png")

        # 3. Корреляционная матрица
        print("\n3. Создание корреляционной матрицы...")
        self.plot_correlation_matrix(df)
        saved_plots.append("correlation_matrix.png")

        # 4. Матрица ошибок
        print("\n4. Создание матрицы ошибок...")
        self.plot_confusion_matrix(y_true, y_pred)
        saved_plots.append("confusion_matrix.png")

        # 5. Отчет о классификации
        print("\n5. Вывод отчета о классификации...")
        self.plot_classification_report(y_true, y_pred)

        # 6. Важность признаков
        if feature_importance:
            print("\n6. Создание графика важности признаков...")
            self.plot_feature_importance(feature_importance)
            saved_plots.append("feature_importance.png")

        print("\n" + "=" * 60)
        print("ВСЕ ВИЗУАЛИЗАЦИИ СОХРАНЕНЫ УСПЕШНО!")
        print("=" * 60)

        return saved_plots

    def open_plot_file(self, filename: str) -> None:
        """
        Открытие конкретного файла графика в системе

        Args:
            filename: Имя файла в папке plots_folder или полный путь
        """
        if os.path.isabs(filename):
            filepath = filename
        else:
            filepath = os.path.join(self.plots_folder, filename)

        if not os.path.exists(filepath):
            print(f"Файл не найден: {filepath}")
            return

        try:
            # os.startfile работает в Windows для открытия файла приложением по умолчанию
            os.startfile(os.path.abspath(filepath))
            print(f"Открыт график: {filepath}")
        except Exception as e:
            print(f"Не удалось открыть файл {filepath}: {e}")

    def open_plots_folder(self) -> None:
        """
        Открытие папки с графиками в проводнике
        """
        try:
            # Открываем папку в проводнике Windows
            os.startfile(os.path.abspath(self.plots_folder))
            print(f"Папка с графиками открыта: {self.plots_folder}")
        except Exception as e:
            print(f"Не удалось открыть папку: {e}")
