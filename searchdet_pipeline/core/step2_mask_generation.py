"""
Этап 2: ГЕНЕРАЦИЯ МАСОК-КАНДИДАТОВ

Основная задача: Разбить исходное изображение на множество потенциальных объектов (масок) 
с помощью одной из моделей семейства Segment Anything (SAM).
"""

import cv2
import time
import numpy as np
from typing import List, Dict, Any, Optional

from ..utils.config import MaskGenerationConfig
from ..utils.helpers import ensure_ultralytics


class MaskGenerator:
    """Класс для генерации масок с помощью различных SAM моделей."""
    
    def __init__(self, config: MaskGenerationConfig):
        """
        Инициализация генератора масок.
        
        Args:
            config: Конфигурация для генерации масок
        """
        self.config = config
        self.sam_model = None
        self.sam2_model = None  
        self.fastsam_model = None
        
    def initialize_backend(self, backend: str, **kwargs):
        """
        2.1. Выбор и инициализация бэкенда.
        
        Args:
            backend: Тип модели ("sam-hq", "sam2", "fastsam")
            **kwargs: Дополнительные параметры для инициализации
        """
        print(f"   🔧 Инициализация бэкенда: {backend}")
        
        if backend == "sam-hq":
            self._init_sam_hq(**kwargs)
        elif backend == "sam2":
            self._init_sam2(**kwargs)
        elif backend == "fastsam":
            self._init_fastsam(**kwargs)
        else:
            raise ValueError(f"Неизвестный бэкенд: {backend}")
    
    def _init_sam_hq(self, checkpoint_path: Optional[str] = None):
        """Инициализация SAM-HQ модели."""
        try:
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
            
            if not checkpoint_path:
                raise ValueError("Требуется путь к checkpoint для SAM-HQ")
            
            sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
            self.sam_model = sam
            print(f"   ✅ SAM-HQ инициализирован: {checkpoint_path}")
            
        except Exception as e:
            raise RuntimeError(f"Ошибка инициализации SAM-HQ: {e}")
    
    def _init_sam2(self, checkpoint_path: Optional[str] = None, config_path: Optional[str] = None):
        """Инициализация SAM2 модели."""
        try:
            from sam2.build_sam import build_sam2
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            
            if not checkpoint_path or not config_path:
                raise ValueError("Требуется путь к checkpoint и config для SAM2")
            
            sam2_model = build_sam2(config_path, checkpoint_path, device="cuda")
            self.sam2_model = sam2_model
            print(f"   ✅ SAM2 инициализирован: {checkpoint_path}")
            
        except Exception as e:
            raise RuntimeError(f"Ошибка инициализации SAM2: {e}")
    
    def _init_fastsam(self, checkpoint_path: Optional[str] = None):
        """Инициализация FastSAM модели."""
        try:
            if not ensure_ultralytics():
                raise RuntimeError("Не удалось подготовить ultralytics/FastSAM")
            
            from ultralytics import FastSAM
            
            # Используем дефолтный checkpoint если не указан
            model_path = checkpoint_path or "FastSAM-x.pt"
            self.fastsam_model = FastSAM(model_path)
            print(f"   ✅ FastSAM инициализирован: {model_path}")
            
        except Exception as e:
            raise RuntimeError(f"Ошибка инициализации FastSAM: {e}")
    
    def resize_image_if_needed(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        """
        2.2. Опциональный даунскейл изображения.
        
        Args:
            image: Исходное изображение
            
        Returns:
            Кортеж (измененное_изображение, коэффициент_масштабирования)
        """
        original_h, original_w = image.shape[:2]
        scale = 1.0
        
        if self.config.sam_long_side and max(original_h, original_w) > self.config.sam_long_side:
            # Вычисляем коэффициент масштабирования
            if original_h >= original_w:
                scale = self.config.sam_long_side / float(original_h)
            else:
                scale = self.config.sam_long_side / float(original_w)
            
            # Применяем линейную интерполяцию для фотореалистичных изображений
            new_w, new_h = int(original_w * scale), int(original_h * scale)
            resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            print(f"   📏 Изображение изменено: {original_w}x{original_h} → {new_w}x{new_h} (scale={scale:.3f})")
            return resized_image, scale
        
        return image, scale
    
    def generate_masks(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        2.3. Основной процесс генерации масок.
        
        Args:
            image: Изображение для обработки
            
        Returns:
            Список словарей с масками
        """
        print("\n🔄 ЭТАП 2: ГЕНЕРАЦИЯ МАСОК-КАНДИДАТОВ")
        print("=" * 60)
        
        # Даунскейл если необходимо
        run_img, scale = self.resize_image_if_needed(image)
        
        # Генерация масок в зависимости от бэкенда
        if self.config.backend == "sam-hq":
            masks = self._generate_sam_hq_masks(run_img)
        elif self.config.backend == "sam2":
            masks = self._generate_sam2_masks(run_img)
        elif self.config.backend == "fastsam":
            masks = self._generate_fastsam_masks(run_img)
        else:
            raise ValueError(f"Неизвестный бэкенд: {self.config.backend}")
        
        # Рескейл масок к исходному размеру
        if scale != 1.0:
            masks = self._rescale_masks(masks, scale, image.shape[:2])
        
        print(f"   ✅ Сгенерировано {len(masks)} масок")
        return masks
    
    def _generate_sam_hq_masks(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        2.3.A) Генерация масок через SAM-HQ.
        
        Args:
            image: Изображение для обработки
            
        Returns:
            Список масок
        """
        if self.sam_model is None:
            raise RuntimeError("SAM-HQ модель не инициализирована")
        
        from segment_anything import SamAutomaticMaskGenerator
        
        h, w = image.shape[:2]
        min_region = max(self.config.min_mask_region_area, int(0.0001 * (h * w)))
        
        print(f"   🚀 SAM-HQ генерация на {w}x{h}...")
        
        t0 = time.time()
        
        # Создаем генератор с параметрами
        mask_generator = SamAutomaticMaskGenerator(
            model=self.sam_model,
            points_per_side=self.config.points_per_side,
            pred_iou_thresh=self.config.pred_iou_thresh,
            stability_score_thresh=self.config.stability_score_thresh,
            min_mask_region_area=min_region,
            crop_n_layers=self.config.crop_n_layers,
            crop_nms_thresh=self.config.crop_nms_thresh,
        )
        
        masks = mask_generator.generate(image)
        
        print(f"   🔍 SAM-HQ сгенерировал {len(masks)} масок за {time.time()-t0:.2f} сек")
        return masks
    
    def _generate_sam2_masks(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        2.3.B) Генерация масок через SAM2.
        
        Args:
            image: Изображение для обработки
            
        Returns:
            Список масок
        """
        if self.sam2_model is None:
            raise RuntimeError("SAM2 модель не инициализирована")
        
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        
        print(f"   🚀 SAM2 генерация...")
        
        t0 = time.time()
        
        mask_generator = SAM2AutomaticMaskGenerator(self.sam2_model)
        masks = mask_generator.generate(image)
        
        print(f"   🔍 SAM2 сгенерировал {len(masks)} масок за {time.time()-t0:.2f} сек")
        return masks
    
    def _generate_fastsam_masks(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        2.3.C) Генерация масок через FastSAM.
        
        Args:
            image: Изображение для обработки
            
        Returns:
            Список масок в стандартном формате
        """
        if self.fastsam_model is None:
            raise RuntimeError("FastSAM модель не инициализирована")
        
        h, w = image.shape[:2]
        print(f"   🚀 FastSAM генерация на {w}x{h}...")
        
        t0 = time.time()
        
        # Выполняем детекцию через FastSAM
        results = self.fastsam_model(
            source=image,
            conf=0.4,
            iou=0.9,
            device="auto",
            retina_masks=True,
            verbose=False,
        )
        
        if not results:
            print("   ⚠️ FastSAM вернул пустой результат")
            return []
        
        # Конвертируем результаты в стандартный формат
        masks = self._convert_fastsam_results(results[0], image.shape)
        
        print(f"   🔍 FastSAM сгенерировал {len(masks)} масок за {time.time()-t0:.2f} сек")
        return masks
    
    def _convert_fastsam_results(self, result, image_shape: tuple) -> List[Dict[str, Any]]:
        """
        Конвертирует результаты FastSAM в стандартный формат.
        
        Args:
            result: Результат от FastSAM
            image_shape: Размеры изображения (H, W, C)
            
        Returns:
            Список масок в стандартном формате
        """
        if getattr(result, "masks", None) is None:
            return []
        
        masks_tensor = result.masks.data  # [N, H, W]
        try:
            masks_np = masks_tensor.cpu().numpy().astype(np.uint8)
        except Exception:
            masks_np = np.array(masks_tensor).astype(np.uint8)
        
        output_masks = []
        H, W = image_shape[:2]
        
        for mask in masks_np:
            binary_mask = (mask > 0).astype(bool)
            
            # Вычисляем bbox
            ys, xs = np.where(binary_mask)
            if len(ys) > 0 and len(xs) > 0:
                x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
                bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
            else:
                bbox = [0, 0, 0, 0]
            
            output_masks.append({
                "segmentation": binary_mask,
                "bbox": bbox,
                "area": int(binary_mask.sum()),
                "stability_score": 1.0,  # Заполняем значениями по умолчанию
                "predicted_iou": 1.0,
                "crop_box": [0, 0, W, H],
            })
        
        return output_masks
    
    def _rescale_masks(self, masks: List[Dict[str, Any]], scale: float, 
                      original_shape: tuple) -> List[Dict[str, Any]]:
        """
        2.4. Рескейл масок к исходному размеру.
        
        Args:
            masks: Список масок для рескейла
            scale: Коэффициент масштабирования
            original_shape: Исходные размеры (H, W)
            
        Returns:
            Список масок в исходном разрешении
        """
        if scale == 1.0:
            return masks
        
        original_h, original_w = original_shape
        rescaled_masks = []
        
        for mask in masks:
            # Рескейл маски с использованием nearest neighbor для сохранения четких границ
            segmentation = mask['segmentation'].astype(np.uint8)
            rescaled_seg = cv2.resize(
                segmentation, (original_w, original_h), 
                interpolation=cv2.INTER_NEAREST
            ) > 0
            
            # Пересчитываем bbox и area
            rows = np.any(rescaled_seg, axis=1)
            cols = np.any(rescaled_seg, axis=0)
            
            if rows.any() and cols.any():
                yidx = np.where(rows)[0]
                xidx = np.where(cols)[0]
                y1, y2 = yidx[0], yidx[-1]
                x1, x2 = xidx[0], xidx[-1]
                bbox = [x1, y1, x2-x1, y2-y1]
            else:
                bbox = [0, 0, 0, 0]
            
            rescaled_mask = dict(mask)
            rescaled_mask['segmentation'] = rescaled_seg
            rescaled_mask['bbox'] = bbox
            rescaled_mask['area'] = int(rescaled_seg.sum())
            
            rescaled_masks.append(rescaled_mask)
        
        print(f"   📏 Маски отмасштабированы к исходному размеру: {original_w}x{original_h}")
        return rescaled_masks
