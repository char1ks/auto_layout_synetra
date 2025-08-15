"""
Этап 1: ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ

Основная задача: Загрузить все необходимые изображения (целевое, позитивные/негативные примеры) 
и подготовить их к обработке.
"""

import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional

from ..utils.helpers import load_image


class DataLoader:
    """Класс для загрузки и подготовки данных."""
    
    def __init__(self):
        """Инициализация загрузчика данных."""
        pass
    
    def load_target_image(self, image_path: str) -> np.ndarray:
        """
        1.1. Загрузка целевого изображения.
        
        Args:
            image_path: Путь к целевому изображению
            
        Returns:
            Изображение в формате RGB (NumPy array)
        """
        print(f"   📸 Загружаем целевое изображение: {image_path}")
        
        # cv2.imread() читает в BGR, конвертируем в RGB
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")
        
        # Конвертация BGR -> RGB для совместимости с другими библиотеками
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        print(f"   ✅ Изображение загружено: {rgb_img.shape}")
        return rgb_img
    
    def load_example_images(self, directory: Optional[str]) -> List[Image.Image]:
        """
        1.2. Загрузка примеров (positive или negative).
        
        Args:
            directory: Путь к директории с примерами
            
        Returns:
            Список изображений в формате PIL.Image
        """
        if not directory or not os.path.exists(directory):
            print(f"   ℹ️ Директория примеров не найдена или не указана: {directory}")
            return []
        
        print(f"   📁 Загружаем примеры из: {directory}")
        
        images = []
        supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
        
        try:
            files = os.listdir(directory)
            for filename in files:
                if filename.lower().endswith(supported_extensions):
                    file_path = os.path.join(directory, filename)
                    try:
                        # PIL.Image.open() и конвертация в RGB
                        img = Image.open(file_path).convert("RGB")
                        images.append(img)
                        print(f"     ✅ Загружен: {filename}")
                    except Exception as e:
                        print(f"     ❌ Ошибка загрузки {filename}: {e}")
                        
        except Exception as e:
            print(f"   ❌ Ошибка чтения директории {directory}: {e}")
            return []
        
        print(f"   ✅ Загружено {len(images)} примеров")
        return images
    
    def prepare_data(self, image_path: str, 
                    positive_dir: Optional[str] = None,
                    negative_dir: Optional[str] = None) -> Tuple[np.ndarray, List[Image.Image], List[Image.Image]]:
        """
        Полная подготовка данных для пайплайна.
        
        Args:
            image_path: Путь к целевому изображению
            positive_dir: Директория с положительными примерами
            negative_dir: Директория с отрицательными примерами
            
        Returns:
            Кортеж (целевое_изображение, positive_примеры, negative_примеры)
        """
        print("\n🔄 ЭТАП 1: ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ")
        print("=" * 60)
        
        # Загрузка целевого изображения
        target_image = self.load_target_image(image_path)
        
        # Загрузка positive примеров
        positive_examples = self.load_example_images(positive_dir)
        
        # Загрузка negative примеров  
        negative_examples = self.load_example_images(negative_dir)
        
        print(f"\n📊 Итого загружено:")
        print(f"   • Целевое изображение: {target_image.shape}")
        print(f"   • Positive примеры: {len(positive_examples)}")
        print(f"   • Negative примеры: {len(negative_examples)}")
        
        return target_image, positive_examples, negative_examples
    
    @staticmethod
    def validate_image_path(image_path: str) -> bool:
        """
        Проверяет корректность пути к изображению.
        
        Args:
            image_path: Путь к изображению
            
        Returns:
            True если путь корректен
        """
        if not image_path:
            return False
        
        path = Path(image_path)
        if not path.exists():
            return False
        
        supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff')
        return path.suffix.lower() in supported_extensions
    
    @staticmethod
    def get_image_info(image: np.ndarray) -> dict:
        """
        Получает информацию об изображении.
        
        Args:
            image: Изображение в формате NumPy
            
        Returns:
            Словарь с информацией об изображении
        """
        height, width = image.shape[:2]
        channels = image.shape[2] if len(image.shape) > 2 else 1
        
        return {
            'width': width,
            'height': height, 
            'channels': channels,
            'dtype': str(image.dtype),
            'total_pixels': height * width,
            'memory_mb': image.nbytes / (1024 * 1024)
        }
