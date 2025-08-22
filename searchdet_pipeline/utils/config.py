"""
Центр управления конфигурацией для SearchDet Pipeline.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from pathlib import Path


@dataclass
class MaskGenerationConfig:
    """Конфигурация для генерации масок."""
    backend: str = "sam-hq"  # sam-hq, sam2, fastsam
    sam_long_side: int = 1024
    points_per_side: int = 32
    points_per_side_multi: Optional[str] = None
    pred_iou_thresh: float = 0.88
    stability_score_thresh: float = 0.95
    min_mask_region_area: int = 100
    crop_n_layers: int = 0
    crop_nms_thresh: float = 0.7


@dataclass  
class FilteringConfig:
    """Конфигурация для фильтрации масок."""
    perfect_rectangle_iou_threshold: float = 0.99
    ban_border_masks: bool = True
    border_width: int = 10
    border_clip_small: bool = True
    border_clip_max_frac: float = 0.05
    containment_iou_threshold: float = 0.9
    iou_threshold: float = 0.8
    min_area_frac: float = 0.0001
    max_area_frac: float = 0.8


@dataclass
class ScoringConfig:
    """Конфигурация для скоринга."""
    min_confidence: float = 0.3
    margin: float = 0.1
    ratio: float = 1.5
    neg_cap: float = 0.8
    consensus_thr: float = 0.25
    consensus_k: int = 1


@dataclass
class PostProcessingConfig:
    """Конфигурация для постобработки."""
    max_masks: int = 100
    nms_iou_threshold: float = 0.5


@dataclass
class Config:
    """Главная конфигурация пайплайна."""
    # Пути
    sam_hq_checkpoint: Optional[str] = None
    sam2_checkpoint: Optional[str] = None
    sam2_config: Optional[str] = None
    fastsam_checkpoint: Optional[str] = None
    
    # Конфигурации компонентов
    mask_generation: MaskGenerationConfig = field(default_factory=MaskGenerationConfig)
    filtering: FilteringConfig = field(default_factory=FilteringConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    post_processing: PostProcessingConfig = field(default_factory=PostProcessingConfig)
    
    # Настройки сохранения
    save_all: bool = True
    overlay_alpha: float = 0.5
    
    # Технические настройки
    auto_install: bool = True
    device: str = "auto"  # auto, cpu, cuda, mps
    backbone: str = "dinov2_b"  # dinov2_b, dinov2_s, dinov2_l, dinov2_g, resnet101
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'Config':
        """Создает конфигурацию из словаря."""
        # Реализация парсинга конфигурации из словаря
        return cls(**config_dict)
    
    def to_dict(self) -> dict:
        """Конвертирует конфигурацию в словарь."""
        # Реализация конвертации в словарь
        result = {}
        for key, value in self.__dict__.items():
            if hasattr(value, '__dict__'):
                result[key] = value.__dict__
            else:
                result[key] = value
        return result


# Дефолтная конфигурация
DEFAULT_CONFIG = Config()
