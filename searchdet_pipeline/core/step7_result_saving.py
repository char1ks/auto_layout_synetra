import cv2
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..utils.helpers import save_json, create_output_dir


class ResultSaver:

    def __init__(self, overlay_alpha: float = 0.5):

        self.overlay_alpha = overlay_alpha
        self.colors = self._generate_colors()
    
    def save_all_results(self, image: np.ndarray, final_masks: List[Dict[str, Any]],
                        output_dir: str, image_name: str,
                        pipeline_config: Optional[Dict] = None) -> Dict[str, str]:
        print("\n🔄 ЭТАП 7: СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
        print("=" * 60)
        
        # Создаём директорию для результатов
        result_dir = create_output_dir(output_dir, image_name)
        print(f"   📁 Директория результатов: {result_dir}")
        
        saved_files = {}
        
        if not final_masks:
            print("   ⚠️ Нет детекций для сохранения")
            # Сохраняем только исходное изображение и пустые аннотации
            saved_files = self._save_empty_results(image, result_dir, image_name, pipeline_config)
        else:
            print(f"   💾 Сохранение {len(final_masks)} детекций...")
            
            # 7.1. Сохранение визуализаций
            saved_files.update(self._save_visualizations(image, final_masks, result_dir))
            
            # 7.2. Сохранение отдельных масок
            saved_files.update(self._save_individual_masks(final_masks, result_dir))
            
            # 7.3. Сохранение общей маски
            saved_files.update(self._save_total_mask(final_masks, image.shape, result_dir))
            
            # 7.4. Сохранение JSON аннотаций
            saved_files.update(self._save_annotations(
                image, final_masks, result_dir, image_name, pipeline_config
            ))
        
        print(f"   ✅ Сохранено {len(saved_files)} файлов")
        return saved_files
    
    def _save_empty_results(self, image: np.ndarray, result_dir: Path, 
                           image_name: str, pipeline_config: Optional[Dict]) -> Dict[str, str]:
        saved_files = {}
        
        # Сохраняем исходное изображение
        original_path = result_dir / "original.png"
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(original_path), image_bgr)
        saved_files['original'] = str(original_path)
        
        # Сохраняем пустые аннотации
        empty_annotations = self._build_annotations(
            image, [], image_name, pipeline_config
        )
        annotations_path = result_dir / "annotations.json"
        save_json(empty_annotations, str(annotations_path))
        saved_files['annotations'] = str(annotations_path)
        
        return saved_files
    
    def _save_visualizations(self, image: np.ndarray, masks: List[Dict[str, Any]], 
                           result_dir: Path) -> Dict[str, str]:
        print("   🎨 Создание визуализаций...")
        
        saved_files = {}
        
        # Исходное изображение
        original_path = result_dir / "original.png"
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(original_path), image_bgr)
        saved_files['original'] = str(original_path)
        
        # Наложение масок с прозрачностью
        overlay_path = result_dir / "overlay_masks.png"
        overlay_img = self._create_overlay_visualization(image, masks)
        cv2.imwrite(str(overlay_path), overlay_img)
        saved_files['overlay'] = str(overlay_path)
        
        # Только контуры
        contours_path = result_dir / "contours_only.png"
        contours_img = self._create_contours_visualization(image, masks)
        cv2.imwrite(str(contours_path), contours_img)
        saved_files['contours'] = str(contours_path)
        
        # Семантическая маска (каждый объект своим цветом)
        semantic_path = result_dir / "semantic_mask.png"
        semantic_img = self._create_semantic_visualization(image.shape, masks)
        cv2.imwrite(str(semantic_path), semantic_img)
        saved_files['semantic'] = str(semantic_path)
        
        print(f"     ✅ Создано {len(saved_files)} визуализаций")
        return saved_files
    
    def _create_overlay_visualization(self, image: np.ndarray, 
                                    masks: List[Dict[str, Any]]) -> np.ndarray:
        # Конвертируем в BGR для OpenCV
        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR).copy()
        
        # Создаём словарь цветов для классов
        class_colors = {}
        unique_classes = list(set(mask.get('class', 'unknown') for mask in masks))
        for i, cls in enumerate(unique_classes):
            class_colors[cls] = self.colors[i % len(self.colors)]
        
        for i, mask in enumerate(masks):
            # Получаем цвет для класса
            cls = mask.get('class', 'unknown')
            # Обрабатываем случай, когда класс может быть None
            if cls is None:
                cls = 'unknown'
            color = class_colors[cls]
            segmentation = mask['segmentation']
            bbox = mask['bbox']
            confidence = mask['confidence']
            
            # Создаём цветную маску
            mask_colored = np.zeros_like(result)
            mask_colored[segmentation] = color
            
            # Накладываем с прозрачностью
            result = cv2.addWeighted(result, 1 - self.overlay_alpha, 
                                   mask_colored, self.overlay_alpha, 0)
            
            # Рисуем контур
            contours, _ = cv2.findContours(
                segmentation.astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(result, contours, -1, color, 2)
            
            # Рисуем bbox
            x, y, w, h = bbox
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
            
            # Добавляем текст с классом и confidence
            text = f"{cls}: {confidence:.2f}"
            (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Рисуем подложку для текста, проверяя границы
            y_text = max(text_h + 5, y)  # Убедимся, что текст не выходит за верхнюю границу
            text_bg_rect = (x, y_text - text_h - 5, text_w, text_h + 5)
            
            # Проверяем, что координаты не отрицательные
            if text_bg_rect[1] >= 0 and text_bg_rect[0] >= 0 and text_bg_rect[2] > 0 and text_bg_rect[3] > 0:
                # Проверяем, что не выходим за границы изображения
                max_h, max_w = result.shape[:2]
                if (text_bg_rect[1] + text_bg_rect[3] <= max_h and 
                    text_bg_rect[0] + text_bg_rect[2] <= max_w):
                    sub_img = result[text_bg_rect[1]:text_bg_rect[1]+text_bg_rect[3], 
                                     text_bg_rect[0]:text_bg_rect[0]+text_bg_rect[2]]
                    black_rect = np.zeros(sub_img.shape, dtype=np.uint8)
                    res = cv2.addWeighted(sub_img, 0.5, black_rect, 0.5, 1.0)
                    result[text_bg_rect[1]:text_bg_rect[1]+text_bg_rect[3], 
                           text_bg_rect[0]:text_bg_rect[0]+text_bg_rect[2]] = res

            # Рисуем текст
            cv2.putText(result, text, (x, max(y, text_h + 5) - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return result
    
    def _create_contours_visualization(self, image: np.ndarray,
                                     masks: List[Dict[str, Any]]) -> np.ndarray:
        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR).copy()
        overlay = result.copy()
        
        # Создаём словарь цветов для классов
        class_colors = {}
        unique_classes = list(set(mask.get('class', 'unknown') for mask in masks))
        for i, cls in enumerate(unique_classes):
            class_colors[cls] = self.colors[i % len(self.colors)]
        
        for i, mask in enumerate(masks):
            # Получаем цвет для класса
            cls = mask.get('class', 'unknown')
            color = class_colors[cls]
            segmentation = mask['segmentation']
            
            # Добавляем полупрозрачную заливку
            overlay[segmentation] = color
            
            # Находим и рисуем контуры
            contours, _ = cv2.findContours(
                segmentation.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(result, contours, -1, color, 3)
            
            # Добавляем подпись с классом
            if contours:
                # Находим центр контура для размещения текста
                M = cv2.moments(contours[0])
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(result, cls, (cx-20, cy), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (255, 255, 255), 2)
        
        # Смешиваем оригинал с заливкой
        cv2.addWeighted(overlay, 0.3, result, 0.7, 0, result)
        
        return result
    
    def _create_semantic_visualization(self, image_shape: tuple,
                                     masks: List[Dict[str, Any]]) -> np.ndarray:
        h, w = image_shape[:2]
        semantic_mask = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Создаём словарь цветов для классов
        class_colors = {}
        unique_classes = list(set(mask.get('class', 'unknown') for mask in masks))
        for i, cls in enumerate(unique_classes):
            class_colors[cls] = self.colors[i % len(self.colors)]
        
        for i, mask in enumerate(masks):
            cls = mask.get('class', 'unknown')
            color = class_colors[cls]
            segmentation = mask['segmentation']
            semantic_mask[segmentation] = color
        
        return semantic_mask
    
    def _save_individual_masks(self, masks: List[Dict[str, Any]], 
                             result_dir: Path) -> Dict[str, str]:
        print("   💾 Сохранение отдельных масок...")
        
        saved_files = {}
        masks_dir = result_dir / "masks"
        masks_dir.mkdir(exist_ok=True)
        
        for i, mask in enumerate(masks):
            segmentation = mask['segmentation']
            confidence = mask['confidence']
            
            # Конвертируем boolean маску в uint8
            mask_img = (segmentation * 255).astype(np.uint8)
            
            # Сохраняем маску
            mask_filename = f"mask_{i:03d}_conf_{confidence:.3f}.png"
            mask_path = masks_dir / mask_filename
            cv2.imwrite(str(mask_path), mask_img)
            
            saved_files[f'mask_{i}'] = str(mask_path)
        
        print(f"     ✅ Сохранено {len(masks)} отдельных масок")
        return saved_files
    
    def _save_total_mask(self, masks: List[Dict[str, Any]], image_shape: tuple,
                        result_dir: Path) -> Dict[str, str]:
        print("   🔗 Создание общей маски...")
        
        h, w = image_shape[:2]
        total_mask = np.zeros((h, w), dtype=bool)
        
        # Объединяем все маски через логическое ИЛИ
        for mask in masks:
            total_mask = np.logical_or(total_mask, mask['segmentation'])
        
        # Конвертируем в uint8 и сохраняем
        total_mask_img = (total_mask * 255).astype(np.uint8)
        total_mask_path = result_dir / "total_mask.png"
        cv2.imwrite(str(total_mask_path), total_mask_img)
        
        print(f"     ✅ Общая маска покрывает {np.sum(total_mask)} пикселей")
        return {'total_mask': str(total_mask_path)}
    
    def _save_annotations(self, image: np.ndarray, masks: List[Dict[str, Any]],
                         result_dir: Path, image_name: str,
                         pipeline_config: Optional[Dict]) -> Dict[str, str]:

        print("   📋 Создание JSON аннотаций...")
        
        annotations = self._build_annotations(image, masks, image_name, pipeline_config)
        
        annotations_path = result_dir / "annotations.json"
        save_json(annotations, str(annotations_path))
        
        print(f"     ✅ Аннотации сохранены: {len(masks)} объектов")
        return {'annotations': str(annotations_path)}
    
    def _build_annotations(self, image: np.ndarray, masks: List[Dict[str, Any]],
                          image_name: str, pipeline_config: Optional[Dict]) -> Dict[str, Any]:

        h, w = image.shape[:2]
        
        # Базовая информация
        annotations = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "image_name": image_name,
                "image_width": w,
                "image_height": h,
                "total_pixels": h * w,
                "total_detections": len(masks),
                "pipeline_version": "2.0.0",
            },
            "pipeline_config": pipeline_config or {},
            "detections": []
        }
        
        # Статистики детекций
        if masks:
            confidences = [mask['confidence'] for mask in masks]
            areas = [mask['area'] for mask in masks]
            
            annotations["statistics"] = {
                "mean_confidence": float(np.mean(confidences)),
                "min_confidence": float(np.min(confidences)),
                "max_confidence": float(np.max(confidences)),
                "std_confidence": float(np.std(confidences)),
                "total_detected_area": int(sum(areas)),
                "coverage_fraction": float(sum(areas) / (h * w)),
                "mean_detection_area": float(np.mean(areas)),
                "min_detection_area": int(np.min(areas)),
                "max_detection_area": int(np.max(areas)),
            }
        else:
            annotations["statistics"] = {
                "mean_confidence": 0,
                "min_confidence": 0, 
                "max_confidence": 0,
                "std_confidence": 0,
                "total_detected_area": 0,
                "coverage_fraction": 0,
                "mean_detection_area": 0,
                "min_detection_area": 0,
                "max_detection_area": 0,
            }
        
        # Детальная информация о каждой детекции
        for i, mask in enumerate(masks):
            segmentation = mask['segmentation']
            bbox = mask['bbox']
            
            # Конвертируем маску в полигон (список координат контура)
            contours, _ = cv2.findContours(
                segmentation.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Берём самый большой контур
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                polygon = largest_contour.reshape(-1, 2).tolist()
            else:
                polygon = []
            
            detection = {
                "id": i,
                "bbox": bbox,  # [x, y, width, height]
                "area": int(mask['area']),
                "area_fraction": float(mask['area'] / (h * w)),
                "confidence": float(mask['confidence']),
                "class": mask.get("class"),
                "segmentation": polygon,  # контур как список [x, y] точек
                "center": [
                    int(bbox[0] + bbox[2] / 2),
                    int(bbox[1] + bbox[3] / 2)
                ],
                "original_mask_index": mask.get('original_index', i),
            }
            
            # Дополнительные поля если есть
            for field in ['stability_score', 'predicted_iou']:
                if field in mask:
                    detection[field] = float(mask[field])
            
            annotations["detections"].append(detection)
        
        return annotations
    
    def _generate_colors(self, num_colors: int = 20) -> List[tuple]:
        colors = []
        
        # Используем HSV для равномерного распределения цветов
        # Увеличиваем шаг по hue для большей контрастности
        hue_step = 180 / num_colors
        for i in range(num_colors):
            hue = int(i * hue_step)
            saturation = 255
            value = 220 # Слегка уменьшим яркость для лучшего вида
            
            # Конвертируем HSV в BGR
            hsv_color = np.uint8([[[hue, saturation, value]]])
            bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
            colors.append(tuple(map(int, bgr_color)))
        
        return colors
    
    def create_summary_report(self, saved_files: Dict[str, str], 
                            processing_time: float,
                            final_masks: List[Dict[str, Any]]) -> str:

        summary_lines = [
            "🎯 СВОДКА РЕЗУЛЬТАТОВ ДЕТЕКЦИИ",
            "=" * 50,
            f"⏱️  Время обработки: {processing_time:.2f} сек",
            f"🔍 Найдено объектов: {len(final_masks)}",
        ]
        
        if final_masks:
            confidences = [mask['confidence'] for mask in final_masks]
            areas = [mask['area'] for mask in final_masks]
            
            summary_lines.extend([
                f"📈 Confidence: {np.mean(confidences):.3f} ± {np.std(confidences):.3f}",
                f"📐 Средняя площадь: {np.mean(areas):.0f} пикселей",
                f"📊 Диапазон confidence: {np.min(confidences):.3f} - {np.max(confidences):.3f}",
            ])
        
        summary_lines.extend([
            "",
            "📁 Сохранённые файлы:",
        ])
        
        for file_type, file_path in saved_files.items():
            summary_lines.append(f"   • {file_type}: {Path(file_path).name}")
        
        return "\n".join(summary_lines)
