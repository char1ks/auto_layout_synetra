#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Фильтрация масок: прямоугольники, границы, вложенные, площадь и т.д.
"""

import numpy as np


class MaskFilter:
    """Фильтрация масок по различным критериям."""
    
    def __init__(self, detector):
        self.detector = detector
    
    def apply_all_filters(self, masks, image_np):
        """Применяет все фильтры последовательно."""
        if not masks:
            return masks
            
        print("🔄 ЭТАП 3: Фильтрация масок")
        print("-" * 60)
        
        # 1. Фильтр идеальных прямоугольников
        masks = self._filter_perfect_rectangles(masks)
        
        # 2. Фильтр границ
        masks = self._filter_border_masks(masks, image_np)
        
        # 3. Фильтр вложенных масок
        masks = self._filter_nested_masks(masks)
        
        # 4. Объединение перекрывающихся
        masks = self._merge_overlapping_masks(masks)
        
        # 5. Фильтр по площади
        masks = self._filter_by_area(masks, image_np)
        
        return masks
    
    def _filter_perfect_rectangles(self, masks):
        """Фильтр идеальных прямоугольников."""
        threshold = getattr(self.detector, 'perfect_rectangle_iou_threshold', 0.99)
        
        filtered_masks = []
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
        threshold = getattr(self.detector, 'containment_iou_threshold', 0.95)
        
        if len(masks) <= 1:
            return masks
        
        # Вычисляем IoU между всеми парами
        keep = set(range(len(masks)))
        
        for i in range(len(masks)):
            if i not in keep:
                continue
            for j in range(i + 1, len(masks)):
                if j not in keep:
                    continue
                
                mask_i = masks[i]["segmentation"]
                mask_j = masks[j]["segmentation"]
                
                intersection = np.logical_and(mask_i, mask_j).sum()
                area_i = mask_i.sum()
                area_j = mask_j.sum()
                
                if area_i == 0 or area_j == 0:
                    continue
                
                # Проверяем, если одна маска содержится в другой
                iou_i_in_j = intersection / area_i
                iou_j_in_i = intersection / area_j
                
                if iou_i_in_j >= threshold:
                    # i содержится в j, удаляем меньшую (i)
                    keep.discard(i)
                elif iou_j_in_i >= threshold:
                    # j содержится в i, удаляем меньшую (j)
                    keep.discard(j)
        
        filtered_masks = [masks[i] for i in sorted(keep)]
        
        print(f"🔗 Фильтр вложенных масок ({threshold} IoU): {len(masks)} → {len(filtered_masks)}")
        
        return filtered_masks
    
    def _merge_overlapping_masks(self, masks, iou_threshold=0.7):
        """Объединение перекрывающихся масок."""
        if len(masks) <= 1:
            print(f"🔗 Слияние масок: {len(masks)} → {len(masks)}")
            return masks
        
        # Простая заглушка - пока не объединяем
        print(f"🔗 Объединение перекрывающихся масок: {len(masks)} → {len(masks)}")
        return masks
    
    def _filter_by_area(self, masks, image_np):
        """Фильтр по площади."""
        h, w = image_np.shape[:2]
        total_pixels = h * w
        
        min_area_frac = getattr(self.detector, 'min_area_frac', 0.03)
        max_area_frac = getattr(self.detector, 'max_area_frac', 0.90)
        
        min_area_abs = min_area_frac * total_pixels
        max_area_abs = max_area_frac * total_pixels
        
        filtered_masks = []
        small_count = 0
        big_count = 0
        
        for mask_dict in masks:
            area = mask_dict["area"]
            
            if area < min_area_abs:
                small_count += 1
            elif area > max_area_abs:
                big_count += 1
            else:
                filtered_masks.append(mask_dict)
        
        print(f"📏 Фильтр размера ({min_area_frac*100:.1f}% - {max_area_frac*100:.1f}%): {len(masks)} → {len(filtered_masks)} (small={small_count}, big={big_count})")
        
        return filtered_masks
