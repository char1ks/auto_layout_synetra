"""
Этап 3: ПОСЛЕДОВАТЕЛЬНАЯ ФИЛЬТРАЦИЯ МАСОК

Основная задача: Уменьшить количество масок-кандидатов, удалив заведомо ложные или избыточные.
Фильтры применяются один за другим.
"""

import numpy as np
from typing import List, Dict, Any

from ..utils.config import FilteringConfig
from ..utils.helpers import calculate_mask_bbox_iou, is_contained


class MaskFilter:
    """Класс для фильтрации масок через последовательные этапы."""
    
    def __init__(self, config: FilteringConfig):
        """
        Инициализация фильтра масок.
        
        Args:
            config: Конфигурация фильтрации
        """
        self.config = config
    
    def filter_masks(self, masks: List[Dict[str, Any]], image_shape: tuple) -> List[Dict[str, Any]]:
        """
        Применяет все фильтры последовательно.
        
        Args:
            masks: Список масок для фильтрации
            image_shape: Размеры изображения (H, W)
            
        Returns:
            Отфильтрованный список масок
        """
        print("\n🔄 ЭТАП 3: ПОСЛЕДОВАТЕЛЬНАЯ ФИЛЬТРАЦИЯ МАСОК")
        print("=" * 60)
        
        if not masks:
            print("   ⚠️ Нет масок для фильтрации")
            return []
        
        print(f"   📊 Начальное количество масок: {len(masks)}")
        
        # 3.1. Удаление идеальных прямоугольников
        masks = self._filter_perfect_rectangles(masks)
        
        # 3.2. Удаление или обрезка масок на краях
        masks = self._drop_or_clip_border_masks(masks, image_shape)
        
        # 3.3. Удаление вложенных масок
        masks = self._filter_nested_masks(masks)
        
        # 3.4. Объединение сильно перекрывающихся масок
        masks = self._merge_overlapping_masks(masks)
        
        # 3.5. Фильтр по относительной площади
        masks = self._filter_by_area_only(masks, image_shape)
        
        print(f"   ✅ Финальное количество масок: {len(masks)}")
        return masks
    
    def _filter_perfect_rectangles(self, masks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        3.1. Удаление идеальных прямоугольников.
        
        Args:
            masks: Список масок
            
        Returns:
            Отфильтрованный список без идеальных прямоугольников
        """
        if not masks:
            return masks
        
        filtered_masks = []
        
        for mask in masks:
            segmentation = mask['segmentation']
            bbox = mask['bbox']
            
            # Вычисляем IoU между маской и её bbox
            mask_bbox_iou = calculate_mask_bbox_iou(segmentation, bbox)
            
            # Если IoU слишком высокий, маска слишком прямоугольная
            if mask_bbox_iou <= self.config.perfect_rectangle_iou_threshold:
                filtered_masks.append(mask)
        
        removed_count = len(masks) - len(filtered_masks)
        print(f"   🔳 Фильтр идеальных прямоугольников (IoU>{self.config.perfect_rectangle_iou_threshold:.2f}): {len(masks)} → {len(filtered_masks)} (-{removed_count})")
        
        return filtered_masks
    
    def _drop_or_clip_border_masks(self, masks: List[Dict[str, Any]], 
                                  image_shape: tuple) -> List[Dict[str, Any]]:
        """
        3.2. Удаление или обрезка масок на краях.
        
        Args:
            masks: Список масок
            image_shape: Размеры изображения (H, W)
            
        Returns:
            Отфильтрованный список масок
        """
        if not masks:
            return masks
        
        height, width = image_shape[:2]
        
        # Создаем маску-кольцо по периметру изображения
        border_ring = np.zeros((height, width), dtype=bool)
        w = self.config.border_width
        
        # Границы кольца
        border_ring[:w, :] = True  # верх
        border_ring[-w:, :] = True  # низ
        border_ring[:, :w] = True  # лево
        border_ring[:, -w:] = True  # право
        
        filtered_masks = []
        total_pixels = height * width
        
        for mask in masks:
            segmentation = mask['segmentation']
            
            # Проверяем пересечение с границей
            touches_border = np.any(segmentation & border_ring)
            
            if not touches_border:
                # Маска не касается краев - оставляем
                filtered_masks.append(mask)
            elif self.config.ban_border_masks:
                # Режим запрета - удаляем маски, касающиеся краев
                continue
            elif self.config.border_clip_small:
                # Режим обрезки - проверяем размер маски
                mask_area_fraction = mask['area'] / total_pixels
                
                if mask_area_fraction < self.config.border_clip_max_frac:
                    # Маска маленькая - обрезаем только границы
                    clipped_seg = segmentation & (~border_ring)
                    
                    if clipped_seg.any():
                        # Пересчитываем bbox и area для обрезанной маски
                        rows = np.any(clipped_seg, axis=1)
                        cols = np.any(clipped_seg, axis=0)
                        
                        if rows.any() and cols.any():
                            yidx = np.where(rows)[0]
                            xidx = np.where(cols)[0]
                            y1, y2 = yidx[0], yidx[-1]
                            x1, x2 = xidx[0], xidx[-1]
                            new_bbox = [x1, y1, x2-x1, y2-y1]
                        else:
                            new_bbox = [0, 0, 0, 0]
                        
                        clipped_mask = dict(mask)
                        clipped_mask['segmentation'] = clipped_seg
                        clipped_mask['bbox'] = new_bbox
                        clipped_mask['area'] = int(clipped_seg.sum())
                        
                        filtered_masks.append(clipped_mask)
                else:
                    # Маска большая - удаляем полностью
                    continue
            else:
                # Нет специального поведения - оставляем как есть
                filtered_masks.append(mask)
        
        removed_count = len(masks) - len(filtered_masks)
        mode_str = "запрет" if self.config.ban_border_masks else "обрезка"
        print(f"   🖼️ Фильтр границ ({mode_str}, ширина={self.config.border_width}px): {len(masks)} → {len(filtered_masks)} (-{removed_count})")
        
        return filtered_masks
    
    def _filter_nested_masks(self, masks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        3.3. Удаление вложенных масок.
        
        Args:
            masks: Список масок
            
        Returns:
            Список масок без вложенных
        """
        if not masks or len(masks) <= 1:
            return masks
        
        # Сортируем маски по площади (большие первыми)
        sorted_masks = sorted(masks, key=lambda m: m['area'], reverse=True)
        keep_indices = set(range(len(sorted_masks)))
        
        for i, outer_mask in enumerate(sorted_masks):
            if i not in keep_indices:
                continue
            
            seg_outer = outer_mask['segmentation']
            
            for j, inner_mask in enumerate(sorted_masks):
                if i == j or j not in keep_indices:
                    continue
                
                seg_inner = inner_mask['segmentation']
                area_inner = inner_mask['area']
                
                # Проверяем containment IoU
                if is_contained(seg_inner, seg_outer, area_inner, 
                              self.config.containment_iou_threshold):
                    keep_indices.discard(j)
        
        filtered_masks = [sorted_masks[i] for i in sorted(list(keep_indices))]
        
        removed_count = len(masks) - len(filtered_masks)
        print(f"   🔗 Фильтр вложенных масок (containment IoU>{self.config.containment_iou_threshold:.2f}): {len(masks)} → {len(filtered_masks)} (-{removed_count})")
        
        return filtered_masks
    
    def _merge_overlapping_masks(self, masks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        3.4. Объединение сильно перекрывающихся масок.
        
        Args:
            masks: Список масок
            
        Returns:
            Список масок с объединенными перекрывающимися
        """
        if not masks:
            return []
        
        merged_masks = []
        used_indices = set()
        
        for i, mask1 in enumerate(masks):
            if i in used_indices:
                continue
            
            seg1 = mask1['segmentation']
            group = [mask1]
            group_indices = {i}
            
            # Ищем все перекрывающиеся маски
            for j, mask2 in enumerate(masks[i+1:], i+1):
                if j in used_indices:
                    continue
                
                seg2 = mask2['segmentation']
                
                # Вычисляем обычный IoU
                intersection = np.logical_and(seg1, seg2).sum()
                union = np.logical_or(seg1, seg2).sum()
                
                if union > 0 and intersection / union > self.config.iou_threshold:
                    group.append(mask2)
                    group_indices.add(j)
            
            # Если найдено несколько перекрывающихся масок - объединяем их
            if len(group) > 1:
                # Объединяем маски через логическое ИЛИ
                merged_seg = group[0]['segmentation'].copy()
                for mask in group[1:]:
                    merged_seg = np.logical_or(merged_seg, mask['segmentation'])
                
                # Пересчитываем bbox
                rows = np.any(merged_seg, axis=1)
                cols = np.any(merged_seg, axis=0)
                
                if rows.any() and cols.any():
                    yidx = np.where(rows)[0]
                    xidx = np.where(cols)[0]
                    y1, y2 = yidx[0], yidx[-1]
                    x1, x2 = xidx[0], xidx[-1]
                    bbox = [x1, y1, x2-x1, y2-y1]
                else:
                    bbox = [0, 0, 0, 0]
                
                # Создаем объединенную маску
                merged_mask = {
                    'segmentation': merged_seg,
                    'area': int(merged_seg.sum()),
                    'bbox': bbox,
                    'stability_score': max(m.get('stability_score', 0) for m in group),
                    'predicted_iou': max(m.get('predicted_iou', 0) for m in group),
                    'crop_box': group[0].get('crop_box', [0, 0, merged_seg.shape[1], merged_seg.shape[0]])
                }
                
                merged_masks.append(merged_mask)
            else:
                # Одиночная маска - оставляем как есть
                merged_masks.append(mask1)
            
            # Отмечаем все использованные индексы
            used_indices.update(group_indices)
        
        merged_count = len(masks) - len(merged_masks)
        print(f"   🔄 Объединение перекрывающихся масок (IoU>{self.config.iou_threshold:.2f}): {len(masks)} → {len(merged_masks)} (-{merged_count})")
        
        return merged_masks
    
    def _filter_by_area_only(self, masks: List[Dict[str, Any]], 
                            image_shape: tuple) -> List[Dict[str, Any]]:
        """
        3.5. Фильтр по относительной площади.
        
        Args:
            masks: Список масок
            image_shape: Размеры изображения (H, W)
            
        Returns:
            Отфильтрованный список масок
        """
        if not masks:
            return masks
        
        total_pixels = image_shape[0] * image_shape[1]
        filtered_masks = []
        
        for mask in masks:
            area_fraction = mask['area'] / total_pixels
            
            # Проверяем минимальную и максимальную площадь
            if (area_fraction >= self.config.min_area_frac and 
                area_fraction <= self.config.max_area_frac):
                filtered_masks.append(mask)
        
        removed_count = len(masks) - len(filtered_masks)
        print(f"   📐 Фильтр по площади ({self.config.min_area_frac:.4f}-{self.config.max_area_frac:.2f} от кадра): {len(masks)} → {len(filtered_masks)} (-{removed_count})")
        
        return filtered_masks
