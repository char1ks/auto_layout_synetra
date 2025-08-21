"""
Core модули пайплайна SearchDet.

Содержит основную логику обработки:
- detector: Главный класс детектора
- pipeline: Обработчик пайплайна
"""

from .detector import SearchDetDetector
from .pipeline import PipelineProcessor

__all__ = [
    'SearchDetDetector', 
    'PipelineProcessor',
]
