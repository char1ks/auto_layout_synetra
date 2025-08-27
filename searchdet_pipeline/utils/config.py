from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from pathlib import Path


@dataclass
class MaskGenerationConfig:
    backend: str = "sam-hq"
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
    perfect_rectangle_iou_threshold: float = 0.99
    ban_border_masks: bool = True
    border_width: int = 10
    border_clip_small: bool = True
    border_clip_max_frac: float = 0.05
    containment_iou_threshold: float = 0.9
    iou_threshold: float = 0.8
    min_area_frac: float = 0.0001
    max_area_frac: float = 0.8
    
    # Параметры умного фильтра прямоугольников
    smart_rectangle_filter: bool = True  # Включить умный анализ прямоугольных форм
    rectangle_bbox_iou_threshold: float = 0.85  # Минимальный IoU с bbox для четких прямоугольников
    rectangle_straight_line_ratio: float = 0.7  # Минимальная доля прямых линий в контуре
    rectangle_area_ratio_threshold: float = 0.9  # Минимальное отношение площади маски к контуру
    rectangle_angle_tolerance: float = 15.0  # Допустимое отклонение углов от 90 градусов
    rectangle_side_ratio_threshold: float = 0.8  # Минимальное отношение противоположных сторон


@dataclass
class ScoringConfig:
    # Основные пороги (ужесточены для фильтрации ложных масок)
    min_pos_score: float = 0.65  # снижен с 0.65 для менее строгой фильтрации
    decision_threshold: float = 0.06  # было 0.04 - увеличен минимальный diff между pos и neg
    class_separation: float = 0.04  # было 0.02 - увеличен для лучшего разделения классов
    neg_cap: float = 0.90  # было 0.95 - снижен для отсечения масок с высоким neg_score
    topk: int = 5
    
    # Параметры агрегации
    pos_trim: float = 0.2  # обрезка краев для позитивной агрегации (20%)
    neg_quantile: float = 0.80  # квантиль для негативной агрегации (снижен с 0.95)
    
    # Консенсус параметры
    consensus_k: int = 0
    consensus_thr: float = 0.45
    
    # Адаптивные пороги (ужесточены)
    adaptive_ratio: float = 0.85  # было 0.90 - снижен для более строгого адаптивного порога
    adaptive_diff_floor: float = 0.04  # было 0.02 - увеличен минимальный diff в адаптивном режиме
    adaptive_trigger_pos_range: float = 0.20
    adaptive_trigger_neg_range: float = 0.20
    
    # Дополнительные параметры скоринга
    margin: float = -0.10
    ratio: float = 1.01
    confidence: float = 0.60
    
    # Флаги
    allow_unknown: bool = True
    verbose: bool = True
    
    # Устаревшие параметры для совместимости
    min_confidence: float = 0.3


@dataclass
class PostProcessingConfig:
    max_masks: int = 100
    nms_iou_threshold: float = 0.5


@dataclass
class Config:
    sam_hq_checkpoint: Optional[str] = None
    sam2_checkpoint: Optional[str] = None
    sam2_config: Optional[str] = None
    fastsam_checkpoint: Optional[str] = None
    
    mask_generation: MaskGenerationConfig = field(default_factory=MaskGenerationConfig)
    filtering: FilteringConfig = field(default_factory=FilteringConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    post_processing: PostProcessingConfig = field(default_factory=PostProcessingConfig)
    
    save_all: bool = True
    overlay_alpha: float = 0.5
    
    auto_install: bool = True
    device: str = "auto"
    backbone: str = "dinov2_b"
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'Config':
        return cls(**config_dict)
    
    def to_dict(self) -> dict:
        result = {}
        for key, value in self.__dict__.items():
            if hasattr(value, '__dict__'):
                result[key] = value.__dict__
            else:
                result[key] = value
        return result


DEFAULT_CONFIG = Config()
