"""
Этап 6: ПОСТОБРАБОТКА

Основная задача: Финальная очистка принятых масок через сортировку, 
ограничение количества и Non-Maximum Suppression (NMS).
"""

import numpy as np
from typing import List, Dict, Any, Tuple

from ..utils.config import PostProcessingConfig
from ..utils.helpers import iou_xyxy


class PostProcessor:
    """Класс для постобработки результатов детекции."""
    
    def __init__(self, config: PostProcessingConfig):
        """
        Инициализация постпроцессора.
        
        Args:
            config: Конфигурация постобработки
        """
        self.config = config
    
    def postprocess(self, masks: List[Dict[str, Any]], 
                   accept_flags: List[bool],
                   confidence_scores: List[float]) -> List[Dict[str, Any]]:
        """
        Выполняет полную постобработку детекций.
        
        Args:
            masks: Список всех масок-кандидатов
            accept_flags: Флаги принятия для каждой маски
            confidence_scores: Скоры уверенности для каждой маски
            
        Returns:
            Финальный список принятых и обработанных масок
        """
        print("\n🔄 ЭТАП 6: ПОСТОБРАБОТКА")
        print("=" * 60)
        
        if not masks or not accept_flags:
            print("   ⚠️ Нет данных для постобработки")
            return []
        
        # Фильтруем только принятые маски
        accepted_masks = self._filter_accepted_masks(masks, accept_flags, confidence_scores)
        
        if not accepted_masks:
            print("   ⚠️ Нет принятых масок для постобработки")
            return []
        
        print(f"   📊 Принято масок: {len(accepted_masks)}")
        
        # 6.1. Сортировка по уверенности
        sorted_masks = self._sort_by_confidence(accepted_masks)
        
        # 6.2. Ограничение количества
        limited_masks = self._limit_masks(sorted_masks)
        
        # 6.3. Non-Maximum Suppression
        final_masks = self._apply_nms(limited_masks)
        
        print(f"   ✅ Финальное количество масок: {len(final_masks)}")
        return final_masks
    
    def _filter_accepted_masks(self, masks: List[Dict[str, Any]], 
                              accept_flags: List[bool],
                              confidence_scores: List[float]) -> List[Dict[str, Any]]:
        """
        Фильтрует только принятые маски и добавляет к ним confidence.
        
        Args:
            masks: Список всех масок
            accept_flags: Флаги принятия
            confidence_scores: Скоры уверенности
            
        Returns:
            Список принятых масок с добавленным полем confidence
        """
        accepted_masks = []
        
        for i, (mask, accepted, confidence) in enumerate(zip(masks, accept_flags, confidence_scores)):
            if accepted:
                # Создаем копию маски и добавляем confidence
                mask_with_confidence = dict(mask)
                mask_with_confidence['confidence'] = confidence
                mask_with_confidence['original_index'] = i
                accepted_masks.append(mask_with_confidence)
        
        return accepted_masks
    
    def _sort_by_confidence(self, masks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        6.1. Сортировка масок по уверенности (по убыванию).
        
        Args:
            masks: Список масок с полем confidence
            
        Returns:
            Отсортированный список масок
        """
        sorted_masks = sorted(masks, key=lambda m: m['confidence'], reverse=True)
        
        if sorted_masks:
            print(f"   📈 Сортировка по confidence: {sorted_masks[0]['confidence']:.3f} → {sorted_masks[-1]['confidence']:.3f}")
        
        return sorted_masks
    
    def _limit_masks(self, masks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        6.2. Ограничение количества масок.
        
        Args:
            masks: Отсортированный список масок
            
        Returns:
            Список масок, ограниченный max_masks
        """
        if len(masks) <= self.config.max_masks:
            return masks
        
        limited_masks = masks[:self.config.max_masks]
        
        print(f"   🔢 Ограничение количества: {len(masks)} → {len(limited_masks)} (max={self.config.max_masks})")
        
        return limited_masks
    
    def _apply_nms(self, masks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        6.3. Применение Non-Maximum Suppression.
        
        Args:
            masks: Список масок, отсортированных по confidence
            
        Returns:
            Список масок после NMS
        """
        if len(masks) <= 1:
            return masks
        
        print(f"   🎯 Применение NMS (IoU threshold={self.config.nms_iou_threshold:.2f})...")
        
        # Конвертируем bbox из формата [x, y, w, h] в [x1, y1, x2, y2]
        boxes = []
        confidences = []
        
        for mask in masks:
            x, y, w, h = mask['bbox']
            boxes.append([x, y, x + w, y + h])
            confidences.append(mask['confidence'])
        
        # Применяем NMS алгоритм
        keep_indices = self._nms_algorithm(boxes, confidences, self.config.nms_iou_threshold)
        
        # Фильтруем маски по оставшимся индексам
        nms_masks = [masks[i] for i in keep_indices]
        
        removed_count = len(masks) - len(nms_masks)
        print(f"   🗑️ NMS удалил {removed_count} перекрывающихся масок")
        
        return nms_masks
    
    def _nms_algorithm(self, boxes: List[List[int]], confidences: List[float], 
                      iou_threshold: float) -> List[int]:
        """
        Реализация алгоритма Non-Maximum Suppression.
        
        Args:
            boxes: Список bbox в формате [x1, y1, x2, y2]
            confidences: Список значений confidence
            iou_threshold: Порог IoU для подавления
            
        Returns:
            Список индексов оставшихся детекций
        """
        if not boxes:
            return []
        
        # Индексы, отсортированные по confidence (по убыванию)
        # Маски уже отсортированы, поэтому просто берём по порядку
        indices = list(range(len(boxes)))
        keep = []
        
        while indices:
            # Берём детекцию с наибольшей уверенностью
            current = indices[0]
            keep.append(current)
            indices.remove(current)
            
            # Удаляем все детекции с высоким IoU относительно текущей
            current_box = boxes[current]
            
            remaining_indices = []
            for idx in indices:
                other_box = boxes[idx]
                iou = iou_xyxy(current_box, other_box)
                
                # Оставляем только детекции с низким IoU
                if iou <= iou_threshold:
                    remaining_indices.append(idx)
            
            indices = remaining_indices
        
        return keep
    
    def create_detection_summary(self, final_masks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Создаёт сводку финальных детекций.
        
        Args:
            final_masks: Список финальных масок
            
        Returns:
            Словарь со сводкой детекций
        """
        if not final_masks:
            return {
                "total_detections": 0,
                "mean_confidence": 0,
                "total_area": 0,
                "area_fraction": 0
            }
        
        confidences = [mask['confidence'] for mask in final_masks]
        areas = [mask['area'] for mask in final_masks]
        
        summary = {
            "total_detections": len(final_masks),
            "mean_confidence": np.mean(confidences),
            "min_confidence": np.min(confidences),
            "max_confidence": np.max(confidences),
            "std_confidence": np.std(confidences),
            "total_area": sum(areas),
            "mean_area": np.mean(areas),
            "min_area": np.min(areas),
            "max_area": np.max(areas),
            "area_std": np.std(areas),
        }
        
        # Добавляем информацию о размерах bbox
        bbox_widths = [mask['bbox'][2] for mask in final_masks]
        bbox_heights = [mask['bbox'][3] for mask in final_masks]
        
        summary.update({
            "mean_bbox_width": np.mean(bbox_widths),
            "mean_bbox_height": np.mean(bbox_heights),
            "min_bbox_width": np.min(bbox_widths),
            "max_bbox_width": np.max(bbox_widths),
            "min_bbox_height": np.min(bbox_heights),
            "max_bbox_height": np.max(bbox_heights),
        })
        
        return summary
    
    def validate_final_masks(self, masks: List[Dict[str, Any]]) -> bool:
        """
        Проверяет корректность финальных масок.
        
        Args:
            masks: Список финальных масок
            
        Returns:
            True если все маски корректны
        """
        if not masks:
            return True
        
        required_fields = ['segmentation', 'bbox', 'area', 'confidence']
        
        for i, mask in enumerate(masks):
            # Проверяем наличие обязательных полей
            for field in required_fields:
                if field not in mask:
                    print(f"   ❌ Маска {i}: отсутствует поле '{field}'")
                    return False
            
            # Проверяем корректность segmentation
            seg = mask['segmentation']
            if not isinstance(seg, np.ndarray):
                print(f"   ❌ Маска {i}: segmentation не является numpy array")
                return False
            
            if seg.dtype != bool:
                print(f"   ❌ Маска {i}: segmentation должен быть boolean")
                return False
            
            # Проверяем корректность bbox
            bbox = mask['bbox']
            if len(bbox) != 4:
                print(f"   ❌ Маска {i}: bbox должен содержать 4 элемента")
                return False
            
            x, y, w, h = bbox
            if w <= 0 or h <= 0:
                print(f"   ❌ Маска {i}: некорректные размеры bbox ({w}x{h})")
                return False
            
            # Проверяем корректность area
            if mask['area'] <= 0:
                print(f"   ❌ Маска {i}: area должна быть положительной")
                return False
            
            # Проверяем корректность confidence
            confidence = mask['confidence']
            if not (0 <= confidence <= 1):
                print(f"   ❌ Маска {i}: confidence должен быть в диапазоне [0, 1]")
                return False
        
        print(f"   ✅ Все {len(masks)} финальных масок прошли валидацию")
        return True
