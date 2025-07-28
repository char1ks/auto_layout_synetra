#!/usr/bin/env python3
"""
Гибридный pipeline: LLaVA (контекст) + SearchDet (отсутствующие элементы) + SAM2 (сегментация)
"""

import os
import sys
import cv2
import numpy as np
import torch
import json
from pathlib import Path
from PIL import Image
import argparse
from datetime import datetime
import time

# Добавляем путь к SearchDet
sys.path.append('./searchdet-main')

# Импортируем наши модули
from llava_sam2_pipeline import MaterialAndDefectAnalyzer, SAM2DefectSegmenter

# Импортируем SearchDet
try:
    from mask_withsearch import (
        initialize_models as init_searchdet,
        get_vector,
        adjust_embedding,
        extract_features_from_masks,
        calculate_attention_weights_softmax
    )
    SEARCHDET_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ SearchDet недоступен: {e}")
    SEARCHDET_AVAILABLE = False


class HybridDefectDetector:
    """Гибридный детектор: LLaVA + SearchDet + SAM2"""
    
    def __init__(self, model_type="detailed"):
        print("🚀 Инициализация гибридного детектора...")
        
        # LLaVA для контекстного анализа
        self.llava_analyzer = MaterialAndDefectAnalyzer()
        if model_type != "detailed":
            self.llava_analyzer.switch_model(model_type)
        
        # SAM2 для финальной сегментации
        self.sam2_segmenter = SAM2DefectSegmenter()
        
        # SearchDet для поиска отсутствующих элементов
        if SEARCHDET_AVAILABLE:
            self.searchdet_resnet, self.searchdet_layer, self.searchdet_transform, self.searchdet_sam = init_searchdet()
            print("✅ SearchDet инициализирован")
        else:
            print("❌ SearchDet недоступен")
        
    def analyze_with_examples(self, image_path, positive_examples_dir, negative_examples_dir, output_dir="./output"):
        """Полный анализ с использованием примеров для SearchDet"""
        
        print(f"\n🔍 ГИБРИДНЫЙ АНАЛИЗ: {image_path}")
        print("=" * 60)
        
        start_time = time.time()
        results = {
            "timestamp": datetime.now().isoformat(),
            "image_path": str(image_path),
            "stages": {}
        }
        
        # ЭТАП 1: LLaVA - контекстный анализ
        print("🧠 ЭТАП 1: LLaVA контекстный анализ...")
        stage1_start = time.time()
        
        material_result = self.llava_analyzer.classify_material(image_path)
        llava_defect_analysis = self.llava_analyzer.analyze_defects(image_path, material_result['material'])
        
        stage1_time = time.time() - stage1_start
        results["stages"]["llava_analysis"] = {
            "duration": stage1_time,
            "material": material_result,
            "defects": llava_defect_analysis
        }
        
        print(f"   ✅ Материал: {material_result['material']}")
        print(f"   🔍 LLaVA дефекты: {llava_defect_analysis.get('defects_found', False)}")
        print(f"   ⏱️ Время: {stage1_time:.2f} сек")
        
        # ЭТАП 2: SearchDet - поиск отсутствующих элементов
        print("\n🔍 ЭТАП 2: SearchDet поиск отсутствующих элементов...")
        stage2_start = time.time()
        
        searchdet_results = None
        if SEARCHDET_AVAILABLE and os.path.exists(positive_examples_dir) and os.path.exists(negative_examples_dir):
            searchdet_results = self._run_searchdet_analysis(
                image_path, positive_examples_dir, negative_examples_dir
            )
        else:
            print("   ⚠️ SearchDet пропущен - нет примеров или модуль недоступен")
            searchdet_results = {"missing_elements": [], "detected_areas": []}
        
        stage2_time = time.time() - stage2_start
        results["stages"]["searchdet_analysis"] = {
            "duration": stage2_time,
            "result": searchdet_results
        }
        
        print(f"   ✅ Найдено отсутствующих элементов: {len(searchdet_results.get('missing_elements', []))}")
        print(f"   ⏱️ Время: {stage2_time:.2f} сек")
        
        # ЭТАП 3: Объединение результатов
        print("\n🔄 ЭТАП 3: Объединение LLaVA + SearchDet...")
        combined_defects = self._combine_llava_searchdet_results(llava_defect_analysis, searchdet_results)
        
        # ЭТАП 4: SAM2 финальная сегментация
        print("\n🎯 ЭТАП 4: SAM2 финальная сегментация...")
        stage4_start = time.time()
        
        sam2_results = self.sam2_segmenter.segment_defects_with_prompts(image_path, combined_defects)
        
        stage4_time = time.time() - stage4_start
        results["stages"]["sam2_segmentation"] = {
            "duration": stage4_time,
            "result": sam2_results
        }
        
        print(f"   ✅ SAM2 маски: {sam2_results.get('num_detections', 0)}")
        print(f"   ⏱️ Время: {stage4_time:.2f} сек")
        
        # Создание финальных аннотаций
        image = cv2.imread(str(image_path))
        annotations = self._create_hybrid_annotations(
            image, sam2_results.get("masks", []), material_result, combined_defects, searchdet_results
        )
        results["annotations"] = annotations
        
        # Сохранение результатов
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        self._save_hybrid_results(image_path, image, sam2_results.get("masks", []), results, output_path)
        
        total_time = time.time() - start_time
        print(f"\n🎉 Гибридный анализ завершен за {total_time:.2f} секунд")
        print(f"📁 Результаты сохранены в: {output_path}")
        
        return results
    
    def _run_searchdet_analysis(self, image_path, positive_dir, negative_dir):
        """Запуск SearchDet для поиска отсутствующих элементов"""
        
        try:
            # Загрузка изображения
            example_img = Image.open(image_path).convert("RGB")
            
            # Загрузка положительных примеров
            positive_imgs = self._load_example_images(positive_dir)
            negative_imgs = self._load_example_images(negative_dir)
            
            if len(positive_imgs) == 0 or len(negative_imgs) == 0:
                print("   ⚠️ Недостаточно примеров для SearchDet")
                return {"missing_elements": [], "detected_areas": []}
            
            print(f"   📁 Положительные примеры: {len(positive_imgs)}")
            print(f"   📁 Отрицательные примеры: {len(negative_imgs)}")
            
            # Извлечение признаков
            pos_embeddings = np.stack([
                get_vector(img, self.searchdet_resnet, self.searchdet_layer, self.searchdet_transform).numpy()
                for img in positive_imgs
            ], axis=0).astype(np.float32)
            
            neg_embeddings = np.stack([
                get_vector(img, self.searchdet_resnet, self.searchdet_layer, self.searchdet_transform).numpy()
                for img in negative_imgs
            ], axis=0).astype(np.float32)
            
            # Корректировка query векторов
            adjusted_queries = np.stack([
                adjust_embedding(q, pos_embeddings, neg_embeddings)
                for q in pos_embeddings
            ], axis=0).astype(np.float32)
            
            # Поиск отсутствующих элементов - более строгий threshold
            missing_elements = self._find_missing_elements(
                example_img, adjusted_queries, similarity_threshold=0.3  # Более строгий: только <70% сходства
            )
            
            return {
                "missing_elements": missing_elements,
                "detected_areas": [],
                "positive_examples_count": len(positive_imgs),
                "negative_examples_count": len(negative_imgs)
            }
            
        except Exception as e:
            print(f"   ❌ Ошибка SearchDet: {e}")
            return {"missing_elements": [], "detected_areas": []}
    
    def _load_example_images(self, directory):
        """Загрузка примеров изображений из директории"""
        images = []
        if not os.path.exists(directory):
            return images
            
        for filename in os.listdir(directory):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                try:
                    img_path = os.path.join(directory, filename)
                    img = Image.open(img_path).convert("RGB")
                    images.append(img)
                except Exception as e:
                    print(f"   ⚠️ Не удалось загрузить {filename}: {e}")
                    continue
        
        return images
    
    def _find_missing_elements(self, image, query_vectors, similarity_threshold=0.4):
        """Поиск отсутствующих элементов с помощью SearchDet"""
        
        from segment_anything_hq import SamAutomaticMaskGenerator
        import faiss
        
        # Генерация масок с помощью SAM (из SearchDet)
        mask_generator = SamAutomaticMaskGenerator(
            model=self.searchdet_sam,
            points_per_side=32,
            points_per_batch=64,
            pred_iou_thresh=0.8,
            stability_score_thresh=0.9,
            min_mask_region_area=1000,  # Увеличили чтобы исключить мелкие шумовые маски
        )
        
        masks = mask_generator.generate(np.array(image))
        
        if not masks:
            return []
        
        # Извлечение признаков масок
        image_np = np.array(image)
        mask_vectors = extract_features_from_masks(
            image_np, masks, self.searchdet_resnet, self.searchdet_layer, self.searchdet_transform
        )
        mask_vectors = np.array(mask_vectors, dtype=np.float32)
        
        # Нормализация для cosine similarity
        query_vectors = query_vectors / np.linalg.norm(query_vectors, axis=1, keepdims=True)
        mask_vectors = mask_vectors / np.linalg.norm(mask_vectors, axis=1, keepdims=True)
        
        # FAISS поиск
        index = faiss.IndexFlatIP(query_vectors.shape[1])
        index.add(query_vectors)
        
        similarities, indices = index.search(mask_vectors, 1)
        normalized_similarities = (similarities + 1) / 2
        
        # Поиск областей с низким сходством (потенциально отсутствующие элементы)
        missing_threshold = 1.0 - similarity_threshold  # Инвертируем логику
        missing_indices = np.where(normalized_similarities.flatten() < missing_threshold)[0]
        
        missing_elements = []
        for idx in missing_indices:
            mask = masks[idx]
            seg = mask['segmentation']
            
            # Вычисление bounding box
            coords = np.column_stack(np.where(seg))
            if len(coords) > 0:
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)
                
                # Вычисляем параметры маски
                h, w = image_np.shape[:2]
                mask_area = int(seg.sum())
                

                
                # Нормализованные координаты
                bbox_norm = (x_min/w, y_min/h, x_max/w, y_max/h)
                
                missing_elements.append({
                    "type": "missing_element",
                    "bbox": bbox_norm,
                    "area": mask_area,
                    "confidence": float(1.0 - normalized_similarities[idx][0]),
                    "description": "Potentially missing component detected by SearchDet"
                })
        
        print(f"   ✅ После фильтрации: {len(missing_elements)} подходящих отсутствующих элементов")
        
        return missing_elements
    

    
    def _combine_llava_searchdet_results(self, llava_results, searchdet_results):
        """Объединение результатов LLaVA и SearchDet"""
        
        combined = llava_results.copy()
        
        # Добавляем отсутствующие элементы от SearchDet
        missing_elements = searchdet_results.get("missing_elements", [])
        if missing_elements:
            # Добавляем новые типы дефектов
            missing_types = ["missing_element", "absent_component"]
            combined["defect_types"] = list(set(combined.get("defect_types", []) + missing_types))
            
            # Обновляем серьезность если найдены отсутствующие элементы
            if combined.get("severity") in ["unknown", "minor"]:
                combined["severity"] = "moderate"
            
            # Обновляем полноту
            combined["completeness"] = "incomplete"
            
            # Добавляем bounding boxes от SearchDet
            searchdet_boxes = [elem["bbox"] for elem in missing_elements]
            existing_boxes = combined.get("bounding_boxes", [])
            combined["bounding_boxes"] = existing_boxes + searchdet_boxes
            
            # Обновляем описание
            combined["description"] += f"\n\nSearchDet detected {len(missing_elements)} potentially missing elements."
            
            # Добавляем информацию о SearchDet
            combined["searchdet_missing"] = missing_elements
        
        return combined
    
    def _create_hybrid_annotations(self, image, masks, material_result, combined_defects, searchdet_results):
        """Создание аннотаций с информацией от обеих систем"""
        
        annotations = {
            "image_info": {
                "height": image.shape[0],
                "width": image.shape[1],
                "channels": image.shape[2]
            },
            "material": material_result,
            "hybrid_analysis": {
                "llava_defects": combined_defects,
                "searchdet_missing": searchdet_results.get("missing_elements", []),
                "detection_methods": ["llava_coordinates", "searchdet_examples", "sam2_segmentation"]
            },
            "defects": []
        }
        
        # Аннотации от масок SAM2
        for i, mask in enumerate(masks):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            
            main_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(main_contour)
            area = cv2.countNonZero(mask)
            polygon = main_contour.flatten().tolist()
            
            # Определение источника детекции
            detection_method = "sam2_segmentation"
            defect_type = "defect"
            confidence = 0.85
            
            # Проверка соответствия SearchDet результатам
            if searchdet_results.get("missing_elements"):
                for missing in searchdet_results["missing_elements"]:
                    bbox = missing["bbox"]
                    # Проверка пересечения (упрощенная)
                    h_img, w_img = image.shape[:2]
                    search_x1, search_y1, search_x2, search_y2 = [int(c * dim) for c, dim in zip(bbox, [w_img, h_img, w_img, h_img])]
                    
                    if (abs(x - search_x1) < 50 and abs(y - search_y1) < 50):
                        detection_method = "searchdet_missing"
                        defect_type = "missing_element"
                        confidence = missing["confidence"]
                        break
            
            defect_annotation = {
                "id": i + 1,
                "category": defect_type,
                "bbox": [x, y, w, h],
                "area": int(area),
                "segmentation": [polygon],
                "confidence": confidence,
                "detection_method": detection_method,
                "severity": combined_defects.get("severity", "unknown"),
                "completeness": combined_defects.get("completeness", "unknown")
            }
            
            annotations["defects"].append(defect_annotation)
        
        return annotations
    
    def _convert_numpy_to_json(self, obj):
        """Рекурсивная конвертация numpy массивов в JSON-совместимые типы"""
        import numpy as np
        
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_to_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_to_json(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_numpy_to_json(item) for item in obj)
        else:
            return obj
    
    def _save_hybrid_results(self, image_path, image, masks, results, output_path):
        """Сохранение результатов гибридного анализа с множественными визуализациями"""
        
        image_name = Path(image_path).stem
        
        # Конвертация numpy массивов перед сериализацией
        json_compatible_results = self._convert_numpy_to_json(results)
        
        # JSON с результатами
        json_path = output_path / f"{image_name}_hybrid_analysis.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_compatible_results, f, ensure_ascii=False, indent=2)
        
        print(f"📁 Сохраняем результаты: {output_path}")
        
        # 1. Основная визуализация с полной информацией
        main_visualization = self._create_hybrid_visualization(image, masks, results)
        cv2.imwrite(str(output_path / f"{image_name}_hybrid_result.jpg"), main_visualization)
        print("   ✅ Основная визуализация сохранена")
        
        # 2. Оригинальное изображение с масками (без панелей)
        clean_overlay = self._create_clean_overlay(image, masks, results)
        cv2.imwrite(str(output_path / f"{image_name}_clean_overlay.jpg"), clean_overlay)
        print("   ✅ Чистое наложение сохранено")
        
        # 3. Только контуры (без заливки)
        contours_only = self._create_contours_visualization(image, masks, results)
        cv2.imwrite(str(output_path / f"{image_name}_contours_only.jpg"), contours_only)
        print("   ✅ Визуализация контуров сохранена")
        
        # 4. Отдельные маски
        for i, mask in enumerate(masks):
            mask_path = output_path / f"{image_name}_mask_{i+1:02d}.png"
            cv2.imwrite(str(mask_path), mask)
        print(f"   ✅ {len(masks)} отдельных масок сохранено")
        
        # 5. Composite mask (все маски в одном изображении)
        if masks:
            composite_mask = self._create_composite_mask(masks, image.shape[:2])
            cv2.imwrite(str(output_path / f"{image_name}_composite_mask.png"), composite_mask)
            print("   ✅ Композитная маска сохранена")
        
        # 6. Side-by-side сравнение
        comparison = self._create_before_after_comparison(image, main_visualization)
        cv2.imwrite(str(output_path / f"{image_name}_comparison.jpg"), comparison)
        print("   ✅ Сравнение до/после сохранено")
    
    def _create_clean_overlay(self, image, masks, results):
        """Создание чистого наложения ТОЛЬКО красных масок SearchDet"""
        
        overlay = image.copy()
        red_color = (0, 0, 255)  # Красный для SearchDet
        
        for i, mask in enumerate(masks):
            if i < len(results["annotations"]["defects"]):
                defect_info = results["annotations"]["defects"][i]
                detection_method = defect_info.get("detection_method", "sam2_segmentation")
                category = defect_info.get("category", "defect")
                
                # ПОКАЗЫВАЕМ ТОЛЬКО SearchDet маски (красные)
                if detection_method == "searchdet_missing" or category == "missing_element":
                    color = red_color
                else:
                    # Пропускаем все остальные маски
                    continue
            else:
                # Пропускаем неизвестные маски
                continue
            
            # Полупрозрачная красная маска
            colored_mask = np.zeros_like(overlay)
            colored_mask[mask > 0] = color
            
            alpha = 0.3
            cv2.addWeighted(overlay, 1-alpha, colored_mask, alpha, 0, overlay)
            
            # Красный контур
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                thickness = 2
                cv2.drawContours(overlay, contours, -1, color, thickness)
        
        # Накладываем 10px рамку от оригинального изображения поверх визуализации
        border_width = 10
        overlay[:border_width, :] = image[:border_width, :]  # верхняя полоса
        overlay[-border_width:, :] = image[-border_width:, :]  # нижняя полоса
        overlay[:, :border_width] = image[:, :border_width]  # левая полоса
        overlay[:, -border_width:] = image[:, -border_width:]  # правая полоса

        return overlay
    
    def _create_contours_visualization(self, image, masks, results):
        """Создание визуализации ТОЛЬКО красных контуров SearchDet"""
        
        contour_image = image.copy()
        red_color = (0, 0, 255)  # Красный для SearchDet
        
        searchdet_mask_count = 0  # Счетчик для нумерации только SearchDet масок
        
        for i, mask in enumerate(masks):
            if i < len(results["annotations"]["defects"]):
                defect_info = results["annotations"]["defects"][i]
                detection_method = defect_info.get("detection_method", "sam2_segmentation")
                category = defect_info.get("category", "defect")
                
                # ПОКАЗЫВАЕМ ТОЛЬКО SearchDet маски (красные)
                if detection_method == "searchdet_missing" or category == "missing_element":
                    color = red_color
                    searchdet_mask_count += 1
                else:
                    # Пропускаем все остальные маски
                    continue
            else:
                # Пропускаем неизвестные маски
                continue
            
            # Только красные контуры
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                thickness = 3  # Толстые контуры для SearchDet
                cv2.drawContours(contour_image, contours, -1, color, thickness)
                
                # Номер SearchDet маски в красном кружке
                main_contour = max(contours, key=cv2.contourArea)
                M = cv2.moments(main_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    cv2.circle(contour_image, (cx, cy), 20, color, -1)
                    cv2.putText(contour_image, str(searchdet_mask_count), (cx-8, cy+8), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
         
        # Накладываем 10px рамку от оригинального изображения поверх визуализации
        border_width = 10
        border_mask = np.zeros_like(image, dtype=np.uint8)
        border_mask[:border_width, :] = 255  # верх
        border_mask[-border_width:, :] = 255  # низ
        border_mask[:, :border_width] = 255  # лево
        border_mask[:, -border_width:] = 255  # право
        cv2.copyTo(image, border_mask, contour_image)  # Накладываем оригинал по маске

        return contour_image
    
    def _create_composite_mask(self, masks, image_shape):
        """Создание композитной маски со всеми дефектами"""
        
        h, w = image_shape
        composite = np.zeros((h, w), dtype=np.uint8)
        
        for i, mask in enumerate(masks):
            # Каждая маска получает уникальное значение (1, 2, 3, ...)
            composite[mask > 0] = i + 1
        
        # Конвертируем в цветную маску
        colored_composite = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Генерируем разные цвета для каждой маски
        for i in range(len(masks)):
            mask_value = i + 1
            color = [(i * 50) % 255, (i * 80 + 100) % 255, (i * 120 + 150) % 255]
            colored_composite[composite == mask_value] = color
        
        return colored_composite
    
    def _create_before_after_comparison(self, original, processed):
        """Создание сравнения до и после"""
        
        # Изменяем размер изображений для равномерного сравнения
        h, w = original.shape[:2]
        
        # Создаем композицию side-by-side
        comparison = np.zeros((h, w * 2 + 20, 3), dtype=np.uint8)
        
        # Оригинал слева
        comparison[:h, :w] = original
        
        # Разделительная линия
        comparison[:, w:w+20] = (100, 100, 100)
        
        # Обработанное изображение справа
        processed_resized = cv2.resize(processed, (w, h))
        comparison[:h, w+20:w*2+20] = processed_resized
        
        # Подписи
        cv2.putText(comparison, "ORIGINAL", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(comparison, "DETECTED DEFECTS", (w + 40, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return comparison
    
    def _create_hybrid_visualization(self, image, masks, results):
        """Создание улучшенной визуализации с масками поверх фото"""
        
        vis_image = image.copy()
        overlay = image.copy()
        
        # Улучшенные цвета для разных типов детекции (в BGR формате)
        colors = {
            "sam2_segmentation": (0, 255, 128),        # Ярко-зеленый  
            "searchdet_missing": (0, 64, 255),        # Красно-оранжевый
            "llava_coordinates": (255, 128, 0),       # Голубой
            "missing_element": (0, 0, 255),           # Красный
            "defect": (0, 255, 0),                    # Зеленый
            "wire_missing": (0, 100, 255),            # Оранжевый
            "scratch": (0, 255, 255),                 # Желтый
            "crack": (128, 0, 255),                   # Фиолетовый
            "corrosion": (0, 165, 255),               # Оранжевый
        }
        
        print(f"🎨 Создаем визуализацию для {len(masks)} масок...")
        
        # Наложение ТОЛЬКО КРАСНЫХ масок от SearchDet
        for i, mask in enumerate(masks):
            if i < len(results["annotations"]["defects"]):
                defect_info = results["annotations"]["defects"][i]
                detection_method = defect_info.get("detection_method", "sam2_segmentation")
                category = defect_info.get("category", "defect")
                severity = defect_info.get("severity", "unknown")
                
                # ПОКАЗЫВАЕМ ТОЛЬКО SearchDet маски (красные)
                if detection_method == "searchdet_missing" or category == "missing_element":
                    color = (0, 0, 255)  # КРАСНЫЙ для SearchDet
                    category = "missing_element"
                else:
                    # Пропускаем все остальные маски (не от SearchDet)
                    continue
                
            else:
                # Пропускаем неизвестные маски
                continue
            
            # Создание цветной маски с градиентом к краям
            colored_mask = np.zeros_like(vis_image)
            
            # Основная цветная область
            colored_mask[mask > 0] = color
            
            # Полупрозрачное наложение маски
            alpha = 0.4 if severity in ["critical", "severe"] else 0.3
            cv2.addWeighted(overlay, 1-alpha, colored_mask, alpha, 0, overlay)
            
            # Контур с переменной толщиной в зависимости от серьезности
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                thickness = 3 if severity in ["critical", "severe"] else 2
                cv2.drawContours(overlay, contours, -1, color, thickness)
                
                # Добавляем номер и тип дефекта на маску
                main_contour = max(contours, key=cv2.contourArea)
                M = cv2.moments(main_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Иконка для SearchDet отсутствующих элементов
                    icon = "❌"
                    
                    # Текст с тенью для лучшей читаемости  
                    text = f"{icon} MISSING {i+1}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    thickness = 2
                    
                    # Тень
                    cv2.putText(overlay, text, (cx-17, cy+7), font, font_scale, (0, 0, 0), thickness+1)
                    # Основной текст
                    cv2.putText(overlay, text, (cx-15, cy+5), font, font_scale, (255, 255, 255), thickness)
        
        # Создание информационной панели
        self._add_info_panel(overlay, results, len(masks))
        
        # Создание легенды
        self._add_legend(overlay, results, colors)
         
         # Накладываем 10px рамку от оригинального изображения поверх визуализации
        border_width = 10
        border_mask = np.zeros_like(image, dtype=np.uint8)
        border_mask[:border_width, :] = 255  # верх
        border_mask[-border_width:, :] = 255  # низ
        border_mask[:, :border_width] = 255  # лево
        border_mask[:, -border_width:] = 255  # право
        cv2.copyTo(image, border_mask, overlay)  # Накладываем оригинал по маске

        return overlay
    
    def _add_info_panel(self, image, results, num_masks):
        """Добавление информационной панели"""
        
        llava_result = results["stages"]["llava_analysis"]["material"]
        searchdet_result = results["stages"]["searchdet_analysis"]["result"]
        defects = results["stages"]["llava_analysis"]["defects"]
        
        # Фон для информационной панели
        h, w = image.shape[:2]
        panel_height = 120
        panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
        panel[:] = (40, 40, 40)  # Темно-серый фон
        
        # Основная информация
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Заголовок
        title = "🔍 SEARCHDET MISSING ELEMENTS (RED MASKS ONLY)"
        cv2.putText(panel, title, (10, 25), font, 0.7, (255, 255, 255), 2)
        
        # Материал
        material_text = f"📦 Material: {llava_result['material'].upper()} (conf: {llava_result['confidence']:.2f})"
        cv2.putText(panel, material_text, (10, 50), font, 0.5, (100, 255, 100), 1)
        
        # Показываем только SearchDet результаты
        missing_count = len(searchdet_result.get('missing_elements', []))
        red_masks_text = f"🔴 RED MASKS (SearchDet): {missing_count} missing elements found"
        cv2.putText(panel, red_masks_text, (10, 70), font, 0.5, (0, 100, 255), 1)
        
        # Статус завершенности
        status_text = "❌ INCOMPLETE ASSEMBLY" if missing_count > 0 else "✅ COMPLETE ASSEMBLY"
        status_color = (0, 100, 255) if missing_count > 0 else (0, 255, 0)
        cv2.putText(panel, status_text, (w-300, 50), font, 0.5, status_color, 1)
        
        # Объединяем панель с изображением
        combined = np.vstack([panel, image])
        image[:] = combined[:h]  # Размещаем панель сверху, обрезая если нужно
    
    def _add_legend(self, image, results, colors):
        """Добавление легенды только для SearchDet (красные маски)"""
        
        h, w = image.shape[:2]
        
        # Проверяем есть ли отсутствующие элементы SearchDet
        searchdet_result = results["stages"]["searchdet_analysis"]["result"]
        missing_count = len(searchdet_result.get('missing_elements', []))
        
        if missing_count > 0:
            # Простая легенда только для SearchDet
            legend_width = 250
            legend_height = 60
            legend_x = w - legend_width - 10
            legend_y = h - legend_height - 10
            
            # Полупрозрачный фон
            overlay = image.copy()
            cv2.rectangle(overlay, (legend_x, legend_y), (legend_x + legend_width, legend_y + legend_height), 
                         (0, 0, 0), -1)
            cv2.addWeighted(image, 0.7, overlay, 0.3, 0, image)
            
            # Заголовок
            cv2.putText(image, "🎨 LEGEND:", (legend_x + 10, legend_y + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Красный квадрат для SearchDet
            cv2.rectangle(image, (legend_x + 10, legend_y + 35), (legend_x + 25, legend_y + 50), (0, 0, 255), -1)
            cv2.rectangle(image, (legend_x + 10, legend_y + 35), (legend_x + 25, legend_y + 50), (255, 255, 255), 1)
            
            # Текст
            cv2.putText(image, "Missing Elements (SearchDet)", (legend_x + 35, legend_y + 47), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)


def main():
    parser = argparse.ArgumentParser(description="Гибридный анализ дефектов: LLaVA + SearchDet + SAM2")
    parser.add_argument("--image", required=True, help="Путь к изображению для анализа")
    parser.add_argument("--positive", required=True, help="Папка с примерами правильных элементов")
    parser.add_argument("--negative", required=True, help="Папка с примерами неправильных/отсутствующих элементов")
    parser.add_argument("--output", default="./output", help="Директория для сохранения результатов")
    parser.add_argument("--model", default="detailed", 
                       choices=["detailed", "standard", "latest", "onevision"],
                       help="Тип модели LLaVA")
    
    args = parser.parse_args()
    
    # Проверка файлов и папок
    if not Path(args.image).exists():
        print(f"❌ Изображение не найдено: {args.image}")
        return
    
    if not Path(args.positive).exists():
        print(f"❌ Папка с положительными примерами не найдена: {args.positive}")
        return
        
    if not Path(args.negative).exists():
        print(f"❌ Папка с отрицательными примерами не найдена: {args.negative}")
        return
    
    # Создание детектора
    detector = HybridDefectDetector(model_type=args.model)
    
    # Запуск анализа
    results = detector.analyze_with_examples(
        args.image, args.positive, args.negative, args.output
    )
    
    if results:
        print(f"\n📊 ИТОГОВАЯ СТАТИСТИКА:")
        print(f"   🧪 Материал: {results['stages']['llava_analysis']['material']['material']}")
        print(f"   🔍 LLaVA дефекты: {results['stages']['llava_analysis']['defects'].get('defects_found', False)}")
        print(f"   🎯 SearchDet отсутствующие: {len(results['stages']['searchdet_analysis']['result'].get('missing_elements', []))}")
        
        # Подсчитаем только SearchDet маски в визуализации
        searchdet_masks_shown = 0
        for defect in results['annotations']['defects']:
            if (defect.get('detection_method') == 'searchdet_missing' or 
                defect.get('category') == 'missing_element'):
                searchdet_masks_shown += 1
        
        print(f"   🔴 КРАСНЫХ масок показано: {searchdet_masks_shown} (только SearchDet)")
        print(f"   📝 Всего аннотаций: {len(results['annotations']['defects'])}")
        total_time = sum(stage['duration'] for stage in results['stages'].values())
        print(f"   ⏱️ Общее время: {total_time:.2f} сек")
        print(f"\n🔴 В визуализации показываются ТОЛЬКО КРАСНЫЕ маски от SearchDet!")


if __name__ == "__main__":
    main() 