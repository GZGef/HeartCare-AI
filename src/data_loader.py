"""
Модуль загрузки и подготовки данных для системы предсказания сердечной недостаточности
"""

import os
import zipfile
import pandas as pd
from typing import Optional

from config.settings import DATA_CONFIG


class DataLoader:
    """
    Класс для загрузки и подготовки данных
    
    Обеспечивает загрузку датасета с Kaggle, его разархивацию и создание DataFrame
    """

    def __init__(self):
        """
        Инициализация загрузчика данных
        """
        self.dataset_id = DATA_CONFIG['dataset_id']
        self.dataset_name = DATA_CONFIG['dataset_name']
        self.data_folder = DATA_CONFIG['data_folder']
        self.separator = DATA_CONFIG['separator']

    def download_and_extract_dataset(self) -> None:
        """
        Загрузка и разархивация датасета с Kaggle
        
        Ожидает, что архив с датасетом находится в текущей директории
        """
        dataset_zip = f"{self.dataset_id.split('/')[-1]}.zip"
        
        if not os.path.exists(dataset_zip):
            raise FileNotFoundError(
                f"Архив с датасетом '{dataset_zip}' не найден. "
                f"Пожалуйста, скачайте датасет с Kaggle: {self.dataset_id}"
            )
        
        # Создание папки для данных, если не существует
        os.makedirs(self.data_folder, exist_ok=True)
        
        # Разархивация
        with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
            zip_ref.extractall(self.data_folder)
        
        print(f"Датасет успешно скачан и разархивирован в папку '{self.data_folder}'!")

    def load_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Загрузка данных в DataFrame
        
        Args:
            file_path: Путь к файлу с данными. Если None, используется путь по умолчанию
            
        Returns:
            pd.DataFrame: DataFrame с данными
            
        Raises:
            FileNotFoundError: Если файл с данными не найден
        """
        if file_path is None:
            file_path = os.path.join(self.data_folder, self.dataset_name)
        
        if not os.path.exists(file_path):
            # Попытка загрузить и разархивировать датасет
            try:
                self.download_and_extract_dataset()
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"Файл с данными '{file_path}' не найден. {str(e)}"
                )
        
        # Загрузка данных
        df = pd.read_csv(file_path, sep=self.separator)
        
        print(f"Данные успешно загружены. Размер датасета: {df.shape}")
        print(f"Количество строк: {df.shape[0]}")
        print(f"Количество столбцов: {df.shape[1]}")
        
        return df

    def get_dataset_info(self, df: pd.DataFrame) -> dict:
        """
        Получение информации о датасете
        
        Args:
            df: DataFrame с данными
            
        Returns:
            dict: Словарь с информацией о датасете
        """
        info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024 / 1024,  # MB
        }
        
        return info

    def print_dataset_summary(self, df: pd.DataFrame) -> None:
        """
        Вывод сводной информации о датасете
        
        Args:
            df: DataFrame с данными
        """
        print("\n" + "="*60)
        print("ИНФОРМАЦИЯ О ДАТАСЕТЕ")
        print("="*60)
        
        print(f"\nРазмерность: {df.shape}")
        print(f"\nТипы данных:")
        print(df.dtypes)
        
        print(f"\nПропущенные значения:")
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(missing[missing > 0])
        else:
            print("Нет пропущенных значений")
        
        print(f"\nДубликаты строк: {df.duplicated().sum()}")
        
        print(f"\nПервые 5 строк:")
        print(df.head())
        
        print(f"\nПоследние 5 строк:")
        print(df.tail())
        
        print(f"\nСтатистические характеристики:")
        print(df.describe(include='all'))
        
        print("\n" + "="*60)