"""
SearchDet Pipeline - Профессиональный пайплайн для детекции объектов на изображениях.

Основные компоненты:
- core: Основная логика пайплайна
- cli: Интерфейс командной строки  
- utils: Вспомогательные функции
- models: Модели и конфигурации
"""

__version__ = "1.0.0"
__author__ = "SearchDet Team"

from .core.detector import SearchDetDetector
from .core.pipeline import PipelineProcessor

__all__ = [
    'SearchDetDetector',
    'PipelineProcessor',
]
