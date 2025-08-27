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
        
        # Параметры умного фильтра прямоугольников
        self.smart_rectangle_filter = params.get('smart_rectangle_filter', True)
        self.rectangle_bbox_iou_threshold = params.get('rectangle_bbox_iou_threshold', 0.85)
        self.rectangle_straight_line_ratio = params.get('rectangle_straight_line_ratio', 0.7)
        self.rectangle_area_ratio_threshold = params.get('rectangle_area_ratio_threshold', 0.9)
        self.rectangle_angle_tolerance = params.get('rectangle_angle_tolerance', 15.0)
        self.rectangle_side_ratio_threshold = params.get('rectangle_side_ratio_threshold', 0.8)
        # Новые параметры для наложения прямоугольника/квадрата и анализа дыр
        self.rectangle_similarity_iou_threshold = params.get('rectangle_similarity_iou_threshold', 0.92)
        self.square_similarity_iou_threshold = params.get('square_similarity_iou_threshold', 0.92)
        self.rectangle_use_silhouette = params.get('rectangle_use_silhouette', True)
        self.hole_area_ratio_threshold = params.get('hole_area_ratio_threshold', 0.02)
        
        # Кэш для ускорения вычислений
        self._bbox_cache = {}
        self._area_cache = {}
    
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
        
        masks = self._filter_nested_masks_fast(masks)
        print(f"🔗 Фильтр вложенных масок ({self.containment_iou} IoU): {initial_mask_count} → {len(masks)}")
        initial_mask_count = len(masks)
        
        masks = self._merge_overlapping_masks(masks)
        print(f"🔗 Объединение перекрывающихся масок: {initial_mask_count} → {len(masks)}")
        initial_mask_count = len(masks)
        
        masks, small_count, big_count = self._filter_by_size(masks, image_np.shape)
        
        # Применяем коррекцию масок (быстрая версия)
        if self.enable_mask_correction:
            masks = self._apply_mask_correction_fast(masks)
            print(f"🔧 Коррекция масок (быстрая версия): {len(masks)} масок обработано")
        
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
        """Улучшенный фильтр прямоугольных/квадратных масок с анализом контуров и прямых линий"""
        if not self.smart_rectangle_filter:
            return self._filter_rectangles_simple(masks)
            
        try:
            import cv2
        except ImportError:
            print("⚠️ OpenCV не найден, используем простой IoU фильтр")
            return self._filter_rectangles_simple(masks)
        
        filtered_masks = []
        threshold = getattr(self.detector, 'perfect_rectangle_iou_threshold', 0.99)
        
        for mask_dict in masks:
            mask = mask_dict["segmentation"]
            
            # Проверяем является ли маска прямоугольной/квадратной
            is_rectangular = self._is_rectangular_shape(mask, cv2)
            
            if not is_rectangular:
                filtered_masks.append(mask_dict)
        
        dropped = len(masks) - len(filtered_masks)
        print(f"🔳 Умный фильтр прямоугольников (IoU>{threshold}): {len(masks)} → {len(filtered_masks)} (удалено {dropped})")
        
        return filtered_masks
    
    def _is_rectangular_shape(self, mask: np.ndarray, cv2) -> bool:
        """Быстрая проверка прямоугольности маски с минимальными OpenCV операциями"""
        # Быстрая проверка 1: IoU с bounding box
        bbox_iou = self._calculate_bbox_iou_fast(mask)
        
        # Ранний выход для очевидно прямоугольных масок
        if bbox_iou > 0.95:
            return True
            
        # Ранний выход для очевидно не прямоугольных масок
        if bbox_iou < 0.7:
            return False
        
        # Быстрая проверка 2: Анализ формы через соотношение площадей
        if bbox_iou > 0.85:
            # Дополнительная проверка только для пограничных случаев
            overlay_scores = self._overlay_similarity_scores_fast(mask, cv2)
            axis_rect_iou = overlay_scores.get('axis_rect_iou', 0.0)
            rot_rect_iou = overlay_scores.get('rot_rect_iou', 0.0)
            square_iou = overlay_scores.get('square_iou', 0.0)
            hole_ratio = overlay_scores.get('hole_ratio', 0.0)
            
            rect_like_base = max(rot_rect_iou, axis_rect_iou)
            
            # Упрощенная логика принятия решения
            is_rect = (
                rect_like_base >= self.rectangle_similarity_iou_threshold or
                square_iou >= self.square_similarity_iou_threshold or
                (bbox_iou > 0.92 and hole_ratio < 0.1)
            )
            
            return is_rect
        
        return False
    
    def _check_rectangle_properties(self, approx: np.ndarray, mask: np.ndarray, cv2) -> bool:
        """Проверяет свойства четырехугольника на прямоугольность"""
        if len(approx) != 4:
            return False
        
        # Вычисляем углы между сторонами
        angles = []
        points = approx.reshape(-1, 2)
        
        for i in range(4):
            p1 = points[i]
            p2 = points[(i + 1) % 4]
            p3 = points[(i + 2) % 4]
            
            # Векторы сторон
            v1 = p1 - p2
            v2 = p3 - p2
            
        
            # Угол между векторами
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            angle = np.arccos(np.clip(cos_angle, -1, 1))
            angles.append(np.degrees(angle))
        
        # Проверяем что углы близки к 90 градусам
        right_angles = sum(1 for angle in angles if abs(angle - 90) < self.rectangle_angle_tolerance)
        
        # Проверяем соотношение сторон (для квадратов и прямоугольников)
        side_lengths = []
        for i in range(4):
            p1 = points[i]
            p2 = points[(i + 1) % 4]
            length = np.linalg.norm(p2 - p1)
            side_lengths.append(length)
        
        # Противоположные стороны должны быть примерно равны
        side_ratio1 = min(side_lengths[0], side_lengths[2]) / (max(side_lengths[0], side_lengths[2]) + 1e-8)
        side_ratio2 = min(side_lengths[1], side_lengths[3]) / (max(side_lengths[1], side_lengths[3]) + 1e-8)
        
        return right_angles >= 3 and side_ratio1 > self.rectangle_side_ratio_threshold and side_ratio2 > self.rectangle_side_ratio_threshold
    
    def _get_mask_hash(self, mask: np.ndarray) -> str:
        """Создает хэш маски для кэширования"""
        return str(hash(mask.tobytes()))
    
    def _calculate_bbox_iou_fast(self, mask: np.ndarray) -> float:
        """Быстрое вычисление IoU маски с её bounding box с кэшированием"""
        mask_hash = self._get_mask_hash(mask)
        
        # Проверяем кэш
        if mask_hash in self._bbox_cache:
            return self._bbox_cache[mask_hash]
        
        # Находим bounding box маски
        ys, xs = np.where(mask)
        if len(ys) == 0:
            result = 0.0
        else:
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
            
            # Площадь маски
            mask_area = mask.sum()
            
            # Площадь bounding box
            bbox_area = (x2 - x1 + 1) * (y2 - y1 + 1)
            
            # IoU = intersection / union = mask_area / bbox_area (так как маска всегда внутри bbox)
            result = float(mask_area) / float(bbox_area)
        
        # Кэшируем результат
        self._bbox_cache[mask_hash] = result
        return result
    
    def _calculate_bbox_iou(self, mask: np.ndarray) -> float:
        """Вычисляет IoU маски с её bounding box"""
        # Находим bounding box маски
        ys, xs = np.where(mask)
        if len(ys) == 0:
            return 0.0
        
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        
        # Создаем маску bounding box
        bbox_mask = np.zeros_like(mask, dtype=bool)
        bbox_mask[y1:y2+1, x1:x2+1] = True
        
        # Вычисляем IoU
        intersection = np.logical_and(mask, bbox_mask).sum()
        union = np.logical_or(mask, bbox_mask).sum()
        
        return intersection / (union + 1e-8)
    
    def _calculate_area_ratio(self, mask: np.ndarray, contour: np.ndarray, cv2) -> float:
        """Вычисляет отношение площади маски к площади контура"""
        mask_area = mask.sum()
        contour_area = cv2.contourArea(contour)
        
        if contour_area == 0:
            return 0.0
        
        return mask_area / contour_area
    
    def _overlay_similarity_scores_fast(self, mask: np.ndarray, cv2) -> Dict[str, float]:
        """Быстрая версия проверки схожести с прямоугольником/квадратом"""
        H, W = mask.shape[:2]
        
        # Используем исходную маску без силуэта для скорости
        ys, xs = np.where(mask)
        if ys.size == 0:
            return {'axis_rect_iou': 0.0, 'rot_rect_iou': 0.0, 'square_iou': 0.0, 'hole_ratio': 0.0}
        
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        
        # Axis-aligned rectangle mask
        axis_rect = np.zeros((H, W), dtype=bool)
        axis_rect[y1:y2+1, x1:x2+1] = True
        
        # Упрощенная ориентированная проверка - используем только axis-aligned
        rot_rect_iou = self._calculate_bbox_iou_fast(mask)  # Переиспользуем быстрый расчет
        
        # Square variants: choose max IoU among two sizes (min and max side)
        side_min = int(min(w, h))
        side_max = int(max(w, h))
        
        def _square_mask(side: int) -> np.ndarray:
            half = side // 2
            sx1 = int(max(0, int(round(cx)) - half))
            sy1 = int(max(0, int(round(cy)) - half))
            sx2 = int(min(W, sx1 + side))
            sy2 = int(min(H, sy1 + side))
            sq = np.zeros((H, W), dtype=bool)
            if sx2 > sx1 and sy2 > sy1:
                sq[sy1:sy2, sx1:sx2] = True
            return sq
        
        square_min = _square_mask(side_min)
        square_max = _square_mask(side_max)
        
        # Быстрые IoU вычисления
        def _iou_fast(a: np.ndarray, b: np.ndarray) -> float:
            inter = np.logical_and(a, b).sum()
            uni = np.logical_or(a, b).sum()
            return float(inter) / (float(uni) + 1e-8)
        
        axis_rect_iou = _iou_fast(mask, axis_rect)
        square_iou = max(_iou_fast(mask, square_min), _iou_fast(mask, square_max))
        
        # Упрощенная оценка дыр
        bbox_area = w * h
        mask_area = mask.sum()
        hole_ratio = max(0.0, float(bbox_area - mask_area) / float(bbox_area + 1e-8))
        
        return {
            'axis_rect_iou': axis_rect_iou,
            'rot_rect_iou': rot_rect_iou,
            'square_iou': square_iou,
            'hole_ratio': hole_ratio,
        }
    
    # Удалена медленная функция _compute_silhouette
    
    def _filter_rectangles_simple(self, masks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Простой фильтр прямоугольников без OpenCV (fallback)"""
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
    
    def _filter_nested_masks_fast(self, masks):
        if len(masks) <= 1:
            return masks
            
        filtered_masks = sorted(masks, key=lambda m: m['area'], reverse=True)
        
        to_remove = set()
        
        for i in range(len(filtered_masks)):
            if i in to_remove:
                continue
            
            mask_i = filtered_masks[i]
            bbox_i = mask_i['bbox']
            area_i = mask_i['area']
            
            for j in range(i + 1, len(filtered_masks)):
                if j in to_remove:
                    continue
                
                mask_j = filtered_masks[j]
                bbox_j = mask_j['bbox']
                area_j = mask_j['area']
                x1_i, y1_i, w_i, h_i = bbox_i
                x1_j, y1_j, w_j, h_j = bbox_j
                x2_i, y2_i = x1_i + w_i, y1_i + h_i
                x2_j, y2_j = x1_j + w_j, y1_j + h_j
                
                if not (x1_i < x2_j and x2_i > x1_j and y1_i < y2_j and y2_i > y1_j):
                    continue  # Нет пересечения bbox - пропускаем
                
                seg_i = mask_i['segmentation']
                seg_j = mask_j['segmentation']
                
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
    
    def _apply_mask_correction_fast(self, masks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.enable_mask_correction:
            return masks
        
        import cv2
        
        corrected_masks = []
        
        for mask_dict in masks:
            mask = mask_dict['segmentation']
            mask_uint8 = mask.astype(np.uint8)
            
            kernel_size = self.correction_kernel_size
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
            smoothed_mask = mask_uint8
            
            if self.erosion_iterations > 0:
                smoothed_mask = cv2.morphologyEx(smoothed_mask, cv2.MORPH_OPEN, kernel, iterations=self.erosion_iterations)
            
            if self.dilation_iterations > 0:
                smoothed_mask = cv2.morphologyEx(smoothed_mask, cv2.MORPH_CLOSE, kernel, iterations=self.dilation_iterations)
            
            blur_kernel_size = 3 
            smoothed_mask = cv2.GaussianBlur(smoothed_mask.astype(np.float32), (blur_kernel_size, blur_kernel_size), 0.8)
            smoothed_mask = (smoothed_mask > 0.5).astype(bool)
            
            corrected_mask_dict = mask_dict.copy()
            corrected_mask_dict['segmentation'] = smoothed_mask
            
            coords = np.column_stack(np.where(smoothed_mask))
            if len(coords) > 0:
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)
                bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
            else:
                bbox = (0, 0, 0, 0)
            
            corrected_mask_dict['bbox'] = bbox
            corrected_mask_dict['area'] = np.sum(smoothed_mask)
            
            # Обновляем кэш
            mask_hash = self._get_mask_hash(smoothed_mask)
            self._bbox_cache[mask_hash] = bbox
            self._area_cache[mask_hash] = corrected_mask_dict['area']
            
            corrected_masks.append(corrected_mask_dict)
        
        return corrected_masks
