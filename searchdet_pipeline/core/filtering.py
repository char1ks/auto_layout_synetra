#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from typing import List, Dict, Any, Tuple


class MaskFilter:
    
    def __init__(self, detector, params=None):
        if params is None:
            params = {}
            
        self.detector = detector
        
        self.min_area_frac = params.get('min_area_frac', 0.03)
        self.max_area_frac = params.get('max_area_frac', 0.90)
        self.perfect_rectangle_iou = params.get('perfect_rectangle_iou', 0.99)
        self.containment_iou = params.get('containment_iou', 0.95)
        self.border_ban = params.get('border_ban', True)
        self.border_width = params.get('border_width', 2)
        
        # Параметры коррекции масок
        self.enable_mask_correction = params.get('enable_mask_correction', True)
        self.erosion_iterations = params.get('erosion_iterations', 1)
        self.dilation_iterations = params.get('dilation_iterations', 2)
        self.correction_kernel_size = params.get('correction_kernel_size', 3)
    
    def apply_all_filters(self, masks: List[Dict[str, Any]], image_np: np.ndarray) -> List[Dict[str, Any]]:
        if not masks:
            return masks
            
        print("🔄 ЭТАП 3: Фильтрация масок")
        print("="*60)
        
        masks = self._filter_perfect_rectangles(masks)
        print(f"🔳 Фильтр идеальных прямоугольников ({self.perfect_rectangle_iou}): {len(masks)} → {len(masks)}")
        
        before = len(masks)
        masks_border = self._filter_border_masks(masks, image_np)
        dropped_count = before - len(masks_border)
        clipped_count = 0
        print(f"🖼️ Обработка границ (запрет, {self.border_width}px): {before} → {len(masks_border)} (dropped={dropped_count}, clipped={clipped_count})")
        masks = masks_border
        initial_mask_count = len(masks)
        
        masks = self._filter_nested_masks(masks)
        print(f"🔗 Фильтр вложенных масок ({self.containment_iou} IoU): {initial_mask_count} → {len(masks)}")
        initial_mask_count = len(masks)
        
        masks = self._merge_overlapping_masks(masks)
        print(f"🔗 Объединение перекрывающихся масок: {initial_mask_count} → {len(masks)}")
        initial_mask_count = len(masks)
        
        masks, small_count, big_count = self._filter_by_size(masks, image_np.shape)
        
        # Применяем коррекцию масок (erosion/dilation)
        if self.enable_mask_correction:
            masks = self._apply_mask_correction(masks)
            print(f"🔧 Коррекция масок (erosion={self.erosion_iterations}, dilation={self.dilation_iterations}): {len(masks)} масок обработано")
        
        return masks
    
    def _filter_by_size(self, masks: List[Dict[str, Any]], image_shape: tuple) -> Tuple[List[Dict[str, Any]], int, int]:
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

    def _filter_perfect_rectangles(self, masks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        filtered_masks = []
        threshold = getattr(self.detector, 'perfect_rectangle_iou_threshold', 0.99)
        
        for mask_dict in masks:
            mask = mask_dict["segmentation"]
            bbox = mask_dict["bbox"]
            
            x, y, w, h = bbox
            H, W = mask.shape[:2]
            x = int(round(float(x)))
            y = int(round(float(y)))
            w = int(round(float(w)))
            h = int(round(float(h)))
            if w < 0: w = 0
            if h < 0: h = 0
            x0 = max(0, min(x, W))
            y0 = max(0, min(y, H))
            x1 = max(x0, min(x0 + w, W))
            y1 = max(y0, min(y0 + h, H))
            rect_mask = np.zeros_like(mask, dtype=bool)
            if (x1 - x0) > 0 and (y1 - y0) > 0:
                rect_mask[y0:y1, x0:x1] = True
            
            intersection = np.logical_and(mask, rect_mask).sum()
            union = np.logical_or(mask, rect_mask).sum()
            iou = intersection / union if union > 0 else 0
            
            if iou < threshold:
                filtered_masks.append(mask_dict)
        
        dropped = len(masks) - len(filtered_masks)
        print(f"🔳 Фильтр идеальных прямоугольников ({threshold}): {len(masks)} → {len(filtered_masks)}")
        
        return filtered_masks
    
    def _filter_border_masks(self, masks, image_np):
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
            
            touches_border = (
                mask[:border_width, :].any() or
                mask[-border_width:, :].any() or
                mask[:, :border_width].any() or
                mask[:, -border_width:].any()
            )
            
            if touches_border:
                dropped += 1
            else:
                filtered_masks.append(mask_dict)
        
        print(f"🖼️ Обработка границ (запрет, {border_width}px): {len(masks)} → {len(filtered_masks)} (dropped={dropped}, clipped={clipped})")
        
        return filtered_masks
    
    def _filter_nested_masks(self, masks):
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
                
                intersection = np.logical_and(seg_i, seg_j).sum()
                
                containment = intersection / area_j if area_j > 0 else 0
                
                if containment >= self.containment_iou:
                    to_remove.add(j)
        
        final_masks = [mask for i, mask in enumerate(filtered_masks) if i not in to_remove]
        return final_masks
    
    def _merge_overlapping_masks(self, masks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if len(masks) <= 1:
            print(f"🔗 Слияние масок: {len(masks)} → {len(masks)}")
            return masks
        
        print(f"🔗 Объединение перекрывающихся масок: {len(masks)} → {len(masks)}")
        return masks
    
    def _apply_mask_correction(self, masks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Применяет коррекцию масок через erosion/dilation операции для улучшения качества"""
        try:
            import cv2
        except ImportError:
            print("⚠️ OpenCV не найден, пропускаем коррекцию масок")
            return masks
        
        corrected_masks = []
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                         (self.correction_kernel_size, self.correction_kernel_size))
        
        for mask_dict in masks:
            mask = mask_dict["segmentation"]
            
            # Конвертируем boolean маску в uint8
            mask_uint8 = mask.astype(np.uint8) * 255
            
            # Применяем erosion для удаления шума и мелких артефактов
            if self.erosion_iterations > 0:
                mask_uint8 = cv2.erode(mask_uint8, kernel, iterations=self.erosion_iterations)
            
            # Применяем dilation для восстановления размера и заполнения дыр
            if self.dilation_iterations > 0:
                mask_uint8 = cv2.dilate(mask_uint8, kernel, iterations=self.dilation_iterations)
            
            # Конвертируем обратно в boolean
            corrected_mask = (mask_uint8 > 127).astype(bool)
            
            # Пересчитываем bbox и area для скорректированной маски
            ys, xs = np.where(corrected_mask)
            if ys.size > 0 and xs.size > 0:
                x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
                new_bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
                new_area = int(corrected_mask.sum())
                
                # Обновляем mask_dict с скорректированными данными
                corrected_mask_dict = mask_dict.copy()
                corrected_mask_dict["segmentation"] = corrected_mask
                corrected_mask_dict["bbox"] = new_bbox
                corrected_mask_dict["area"] = new_area
                
                # Добавляем только если маска не стала пустой
                if new_area > 0:
                    corrected_masks.append(corrected_mask_dict)
            
        return corrected_masks
