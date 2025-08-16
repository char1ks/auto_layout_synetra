#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Фильтрация масок: прямоугольники, границы, вложенные, площадь и т.д.
"""

import numpy as np
from typing import List, Dict, Any


class MaskFilter:
    """Класс для фильтрации масок."""
    
    def __init__(self, detector, params=None):
        if params is None:
            params = {}
            
        self.detector = detector
        
        # Загружаем параметры фильтров с дефолтами
        self.min_area_frac = params.get('min_area_frac', 0.03)  # 3.0%
        self.max_area_frac = params.get('max_area_frac', 0.90)  # 90.0%
        self.perfect_rectangle_iou = params.get('perfect_rectangle_iou', 0.99)
        self.containment_iou = params.get('containment_iou', 0.95)
        self.border_ban = params.get('border_ban', True)
        self.border_width = params.get('border_width', 2)
    
    def apply_all_filters(self, masks: List[Dict[str, Any]], image_np: np.ndarray) -> List[Dict[str, Any]]:
        """Применяет все фильтры последовательно."""
        if not masks:
            return masks
            
        print("🔄 ЭТАП 3: Фильтрация масок")
        print("="*60)
        
        # Фильтр 1: Идеальные прямоугольники
        masks = self._filter_perfect_rectangles(masks)
        print(f"🔳 Фильтр идеальных прямоугольников ({self.perfect_rectangle_iou}): {len(masks)} → {len(masks)}")
        
        # Фильтр 2: Границы изображения
        masks, dropped_count, clipped_count = self._handle_border_masks(masks, image_np.shape)
        print(f"🖼️ Обработка границ (запрет, {self.border_width}px): {len(masks)} → {len(masks)} (dropped={dropped_count}, clipped={clipped_count})")
        initial_mask_count = len(masks)
        
        # Фильтр 3: Вложенные маски
        masks = self._filter_nested_masks(masks)
        print(f"🔗 Фильтр вложенных масок ({self.containment_iou} IoU): {initial_mask_count} → {len(masks)}")
        initial_mask_count = len(masks)
        
        # Фильтр 4: Объединение перекрывающихся масок
        masks = self._merge_overlapping_masks(masks)
        print(f"🔗 Объединение перекрывающихся масок: {initial_mask_count} → {len(masks)}")
        initial_mask_count = len(masks)
        
        # Фильтр 5: Размер маски
        masks, small_count, big_count = self._filter_by_size(masks, image_np.shape)
        
        return masks
    
    def _filter_by_size(self, masks: List[Dict[str, Any]], image_shape: tuple) -> (List[Dict[str, Any]], int, int):
        """Фильтрует маски по их площади."""
        h, w = image_shape[:2]
        total_pixels = h * w
        
        min_area_abs = self.min_area_frac * total_pixels
        max_area_abs = self.max_area_frac * total_pixels
        
        filtered_masks = []
        small_count = 0
        big_count = 0
        
        for mask_dict in masks:
            area = mask_dict.get('area', 0)
            if area < min_area_abs:
                small_count += 1
            elif area > max_area_abs:
                big_count += 1
            else:
                filtered_masks.append(mask_dict)
                
        return filtered_masks, small_count, big_count

    def _handle_border_masks(self, masks: List[Dict[str, Any]], image_shape: tuple) -> (List[Dict[str, Any]], int, int):
        """Обрабатывает маски, касающиеся границ изображения."""
        if not self.border_ban:
            return masks, 0, 0

        h, w = image_shape[:2]
        filtered_masks = []
        dropped_count = 0
        clipped_count = 0

        for mask_dict in masks:
            segmentation = mask_dict['segmentation']
            bbox = mask_dict['bbox']
            x1, y1, x2, y2 = bbox

            # Проверяем, касается ли маска границ
            on_border = (x1 <= self.border_width or y1 <= self.border_width or
                         x2 >= w - self.border_width or y2 >= h - self.border_width)

            if on_border:
                dropped_count += 1
            else:
                filtered_masks.append(mask_dict)
        
        return filtered_masks, dropped_count, clipped_count

    def _filter_perfect_rectangles(self, masks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Фильтрует маски, которые являются почти идеальными прямоугольниками."""
        filtered_masks = []
        threshold = getattr(self.detector, 'perfect_rectangle_iou_threshold', 0.99)
        
        for mask_dict in masks:
            mask = mask_dict["segmentation"]
            bbox = mask_dict["bbox"]
            
            # Создаем идеальный прямоугольник
            x1, y1, x2, y2 = bbox
            rect_mask = np.zeros_like(mask, dtype=bool)
            rect_mask[y1:y2+1, x1:x2+1] = True
            
            # Вычисляем IoU
            intersection = np.logical_and(mask, rect_mask).sum()
            union = np.logical_or(mask, rect_mask).sum()
            iou = intersection / union if union > 0 else 0
            
            if iou < threshold:
                filtered_masks.append(mask_dict)
        
        dropped = len(masks) - len(filtered_masks)
        print(f"🔳 Фильтр идеальных прямоугольников ({threshold}): {len(masks)} → {len(filtered_masks)}")
        
        return filtered_masks
    
    def _filter_border_masks(self, masks, image_np):
        """Фильтр масок, касающихся границ."""
        h, w = image_np.shape[:2]
        border_width = getattr(self.detector, 'border_width', 2)
        ban_border = getattr(self.detector, 'ban_border_masks', True)
        
        if not ban_border:
            print(f"🖼️ Обработка границ отключена: {len(masks)} → {len(masks)}")
            return masks
        
        filtered_masks = []
        dropped = 0
        clipped = 0
        
        for mask_dict in masks:
            mask = mask_dict["segmentation"]
            
            # Проверяем касание границ
            touches_border = (
                mask[:border_width, :].any() or  # верх
                mask[-border_width:, :].any() or  # низ
                mask[:, :border_width].any() or  # лево
                mask[:, -border_width:].any()     # право
            )
            
            if touches_border:
                dropped += 1
            else:
                filtered_masks.append(mask_dict)
        
        print(f"🖼️ Обработка границ (запрет, {border_width}px): {len(masks)} → {len(filtered_masks)} (dropped={dropped}, clipped={clipped})")
        
        return filtered_masks
    
    def _filter_nested_masks(self, masks):
        """Фильтр вложенных масок."""
        filtered_masks = []
        
        for mask_dict in masks:
            filtered_masks.append(mask_dict)
        
        filtered_masks.sort(key=lambda m: m['area'], reverse=True)
        
        to_remove = set()
        
        for i in range(len(filtered_masks)):
            if i in to_remove:
                continue
            
            mask_i = filtered_masks[i]
            seg_i = mask_i['segmentation']
            area_i = mask_i['area']
            
            for j in range(i + 1, len(filtered_masks)):
                if j in to_remove:
                    continue
                
                mask_j = filtered_masks[j]
                seg_j = mask_j['segmentation']
                area_j = mask_j['area']
                
                # Используем "Containment" IoU: intersection / area_of_smaller_mask
                intersection = np.logical_and(seg_i, seg_j).sum()
                
                # area_j - площадь меньшей маски, т.к. мы отсортировали по убыванию
                containment = intersection / area_j if area_j > 0 else 0
                
                if containment >= self.containment_iou:
                    to_remove.add(j) # Удаляем меньшую, вложенную маску
        
        final_masks = [mask for i, mask in enumerate(filtered_masks) if i not in to_remove]
        return final_masks
    
    def _merge_overlapping_masks(self, masks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Объединяет сильно перекрывающиеся маски."""
        if len(masks) <= 1:
            print(f"🔗 Слияние масок: {len(masks)} → {len(masks)}")
            return masks
        
        # Простая заглушка - пока не объединяем
        print(f"🔗 Объединение перекрывающихся масок: {len(masks)} → {len(masks)}")
        return masks
