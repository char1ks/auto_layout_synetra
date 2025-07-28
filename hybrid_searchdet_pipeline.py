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
        """Интегрированный анализ: LLaVA направляет SearchDet"""
        
        print(f"\n🔍 ИНТЕГРИРОВАННЫЙ АНАЛИЗ: {image_path}")
        print("=" * 60)
        
        start_time = time.time()
        results = {
            "timestamp": datetime.now().isoformat(),
            "image_path": str(image_path),
            "stages": {}
        }
        
        # ЭТАП 1: LLaVA - контекстный анализ и планирование
        print("🧠 ЭТАП 1: LLaVA контекстный анализ и планирование поиска...")
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
        print(f"   🔍 LLaVA видимые дефекты: {llava_defect_analysis.get('defects_found', False)}")
        print(f"   📊 LLaVA области: {len(llava_defect_analysis.get('bounding_boxes', []))}")
        print(f"   ⏱️ Время: {stage1_time:.2f} сек")
        
        # ЭТАП 2: SearchDet с учётом LLaVA информации
        print("\n🎯 ЭТАП 2: SearchDet направленный поиск отсутствующих элементов...")
        stage2_start = time.time()
        
        searchdet_results = None
        if SEARCHDET_AVAILABLE and os.path.exists(positive_examples_dir) and os.path.exists(negative_examples_dir):
            # Передаём LLaVA контекст в SearchDet
            searchdet_results = self._run_guided_searchdet_analysis(
                image_path, positive_examples_dir, negative_examples_dir, 
                llava_context=llava_defect_analysis  # ← НОВОЕ: передаём контекст LLaVA
            )
        else:
            print("   ⚠️ SearchDet пропущен - нет примеров или модуль недоступен")
            searchdet_results = {"missing_elements": [], "detected_areas": []}
        
        stage2_time = time.time() - stage2_start
        results["stages"]["searchdet_analysis"] = {
            "duration": stage2_time,
            "result": searchdet_results
        }
        
        print(f"   ✅ Найдено дефектов: {len(searchdet_results.get('defect_elements', []))}")
        print(f"   ⏱️ Время: {stage2_time:.2f} сек")
        
        # ЭТАП 3: Объединение и финализация (БЕЗ отдельного SAM2)
        print("\n🔄 ЭТАП 3: Финализация результатов...")
        combined_defects = self._combine_llava_searchdet_results(llava_defect_analysis, searchdet_results)
        
        # Используем только маски от SearchDet (которые уже используют SAM внутри)
        final_masks = searchdet_results.get("final_masks", [])
        
        # Создание финальных аннотаций
        image = cv2.imread(str(image_path))
        annotations = self._create_hybrid_annotations(
            image, final_masks, material_result, combined_defects, searchdet_results
        )
        results["annotations"] = annotations
        
        # Сохранение результатов
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        self._save_hybrid_results(image_path, image, final_masks, results, output_path)
        
        total_time = time.time() - start_time
        print(f"\n🎉 Интегрированный анализ завершен за {total_time:.2f} секунд")
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
            
            # Поиск отсутствующих элементов - немного смягчаем threshold
            missing_elements = self._find_missing_elements(
                example_img, adjusted_queries, similarity_threshold=0.4  # Смягчаем с 0.3 до 0.4
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
    
    def _run_guided_searchdet_analysis(self, image_path, positive_dir, negative_dir, llava_context):
        """Направленный SearchDet анализ для поиска ДЕФЕКТОВ (negative examples)"""
        
        try:
            # Загрузка изображения
            example_img = Image.open(image_path).convert("RGB")
            image_np = np.array(example_img)
            
            # Загрузка примеров
            positive_imgs = self._load_example_images(positive_dir)  # Хорошие примеры (норма)
            negative_imgs = self._load_example_images(negative_dir)  # ДЕФЕКТЫ - это наш QUERY!
            
            if len(positive_imgs) == 0 or len(negative_imgs) == 0:
                print("   ⚠️ Недостаточно примеров для SearchDet")
                return {"defect_elements": [], "detected_areas": [], "final_masks": []}
            
            print(f"   📁 Примеры нормы (positive): {len(positive_imgs)}")
            print(f"   🔍 Примеры дефектов (QUERY): {len(negative_imgs)}")
            
            # Извлечение признаков из примеров
            pos_embeddings = np.stack([
                get_vector(img, self.searchdet_resnet, self.searchdet_layer, self.searchdet_transform).numpy()
                for img in positive_imgs
            ], axis=0).astype(np.float32)
            
            # ДЕФЕКТЫ становятся нашими QUERY векторами!
            defect_embeddings = np.stack([
                get_vector(img, self.searchdet_resnet, self.searchdet_layer, self.searchdet_transform).numpy()
                for img in negative_imgs
            ], axis=0).astype(np.float32)
            
            # Корректировка query векторов: усиливаем дефекты, ослабляем норму
            adjusted_defect_queries = np.stack([
                self._adjust_defect_embedding(q, defect_embeddings, pos_embeddings)
                for q in defect_embeddings
            ], axis=0).astype(np.float32)
            
            # НОВОЕ: Используем LLaVA информацию для направленного поиска ДЕФЕКТОВ
            llava_boxes = llava_context.get("bounding_boxes", [])
            llava_defects_found = llava_context.get("defects_found", False)
            
            print(f"   🎯 LLaVA обнаружено дефектов: {llava_defects_found}")
            print(f"   📦 LLaVA областей для анализа: {len(llava_boxes)}")
            
            if llava_defects_found and llava_boxes:
                # Режим 1: Направленный поиск дефектов в областях LLaVA
                defect_elements, final_masks = self._guided_defect_search_in_llava_regions(
                    image_np, adjusted_defect_queries, llava_boxes
                )
                print(f"   🔍 Направленный поиск дефектов в {len(llava_boxes)} LLaVA областях")
            else:
                # Режим 2: Общий поиск дефектов по всему изображению
                defect_elements, final_masks = self._find_defects_enhanced(
                    image_np, adjusted_defect_queries, similarity_threshold=0.6  # Высокое сходство = дефект найден
                )
                print(f"   🔍 Общий поиск дефектов по всему изображению")
            
            return {
                "defect_elements": defect_elements,  # Изменили название
                "detected_areas": [],
                "final_masks": final_masks,
                "llava_guided": bool(llava_boxes),
                "positive_examples_count": len(positive_imgs),
                "defect_examples_count": len(negative_imgs)  # Изменили название
            }
            
        except Exception as e:
            print(f"   ❌ Ошибка направленного SearchDet: {e}")
            return {"defect_elements": [], "detected_areas": [], "final_masks": []}
    
    def _adjust_defect_embedding(self, defect_query, defect_embeddings, normal_embeddings):
        """Корректировка дефектного эмбеддинга: усиливаем дефекты, ослабляем норму"""
        
        # Вычисляем веса для дефектных примеров (чем больше сходство, тем больше вес)
        defect_similarities = np.array([
            np.dot(defect_query, emb) / (np.linalg.norm(defect_query) * np.linalg.norm(emb))
            for emb in defect_embeddings
        ])
        defect_weights = np.exp(defect_similarities) / np.sum(np.exp(defect_similarities))
        
        # Вычисляем веса для нормальных примеров
        normal_similarities = np.array([
            np.dot(defect_query, emb) / (np.linalg.norm(defect_query) * np.linalg.norm(emb))
            for emb in normal_embeddings
        ])
        normal_weights = np.exp(normal_similarities) / np.sum(np.exp(normal_similarities))
        
        # Усиливаем похожие дефекты
        defect_adjustment = np.sum(defect_weights[:, np.newaxis] * defect_embeddings, axis=0)
        
        # Ослабляем нормальные области
        normal_adjustment = np.sum(normal_weights[:, np.newaxis] * normal_embeddings, axis=0)
        
        # Итоговая корректировка: усиливаем дефекты, вычитаем норму
        adjusted_defect = defect_adjustment - 0.3 * normal_adjustment  # 0.3 - коэффициент подавления нормы
        
        return adjusted_defect
    
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
        
        # Конвертируем PIL Image в numpy array если нужно
        if hasattr(image, 'mode'):  # Это PIL Image
            image_np = np.array(image)
        else:
            image_np = image
        
        # Генерация масок с помощью SAM (из SearchDet)
        mask_generator = SamAutomaticMaskGenerator(
            model=self.searchdet_sam,
            points_per_side=16,  # Уменьшаем количество точек для более крупных сегментов
            points_per_batch=32,  # Уменьшаем batch size
            pred_iou_thresh=0.90,  # Немного смягчаем с 0.92 до 0.90
            stability_score_thresh=0.95,  # Немного смягчаем с 0.97 до 0.95
            min_mask_region_area=800,  # Немного уменьшаем с 1000 до 800
            box_nms_thresh=0.5,  # Более строгое удаление перекрывающихся боксов
            crop_nms_thresh=0.5,  # Более строгое удаление перекрывающихся кропов
        )
        
        masks = mask_generator.generate(image_np)
        
        # Постобработка: фильтруем слишком большие маски
        image_area = image_np.shape[0] * image_np.shape[1]
        max_allowed_area = image_area * 0.15  # Уменьшили с 30% до 15% от изображения
        
        original_count = len(masks)
        filtered_masks = []
        for mask in masks:
            mask_area = mask['area']
            if mask_area <= max_allowed_area:
                filtered_masks.append(mask)
        
        masks = filtered_masks
        print(f"   🔍 Отфильтровано масок SAM: {len(masks)} из {original_count}")
        
        if not masks:
            return []
        
        # Извлечение признаков масок
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
        
        print(f"   🔍 Обработано масок: {len(normalized_similarities)}")
        print(f"   📊 Порог отсутствующих: {missing_threshold:.3f}")
        print(f"   📈 Диапазон сходства: {normalized_similarities.min():.3f} - {normalized_similarities.max():.3f}")
        print(f"   🎯 Кандидатов на отсутствующие: {len(missing_indices)}")
        
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
    
    def _guided_search_with_llava_regions(self, image_np, adjusted_queries, llava_boxes):
        """Направленный поиск в областях, указанных LLaVA"""
        
        from segment_anything_hq import SamAutomaticMaskGenerator, SamPredictor
        import faiss
        
        all_missing_elements = []
        all_masks = []
        
        h, w = image_np.shape[:2]
        
        # Создаем SAM predictor для точной сегментации
        sam_predictor = SamPredictor(self.searchdet_sam)
        sam_predictor.set_image(image_np)
        
        print(f"   🎯 Анализируем {len(llava_boxes)} областей от LLaVA...")
        
        for i, bbox_norm in enumerate(llava_boxes):
            # Преобразуем нормализованные координаты в абсолютные
            if len(bbox_norm) == 4:
                x_norm, y_norm, w_norm, h_norm = bbox_norm
                x1 = int(x_norm * w)
                y1 = int(y_norm * h)
                x2 = int((x_norm + w_norm) * w)
                y2 = int((y_norm + h_norm) * h)
                
                # Убеждаемся что координаты в пределах изображения
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                print(f"   📦 Область {i+1}: ({x1},{y1}) -> ({x2},{y2})")
                
                # Используем SAM для точной сегментации в этой области
                input_box = np.array([x1, y1, x2, y2])
                masks_in_region, scores, logits = sam_predictor.predict(
                    box=input_box,
                    multimask_output=True
                )
                
                # Выбираем лучшую маску
                best_mask_idx = np.argmax(scores)
                best_mask = masks_in_region[best_mask_idx]
                
                # Анализируем эту маску с помощью SearchDet эмбеддингов
                if np.sum(best_mask) > 100:  # Минимальный размер маски
                    # Извлекаем признаки из маскированной области
                    masked_region = image_np.copy()
                    masked_region[~best_mask] = 0  # Обнуляем области вне маски
                    
                    region_img = Image.fromarray(masked_region)
                    region_features = get_vector(
                        region_img, self.searchdet_resnet, 
                        self.searchdet_layer, self.searchdet_transform
                    ).numpy().reshape(1, -1).astype(np.float32)
                    
                    # Нормализация для cosine similarity
                    adjusted_queries_norm = adjusted_queries / np.linalg.norm(adjusted_queries, axis=1, keepdims=True)
                    region_features_norm = region_features / np.linalg.norm(region_features, axis=1, keepdims=True)
                    
                    # FAISS поиск
                    index = faiss.IndexFlatIP(adjusted_queries_norm.shape[1])
                    index.add(adjusted_queries_norm)
                    
                    similarities, indices = index.search(region_features_norm, 1)
                    normalized_similarity = (similarities[0][0] + 1) / 2
                    
                    # Проверяем является ли это отсутствующим элементом
                    similarity_threshold = 0.4
                    if normalized_similarity < similarity_threshold:
                        print(f"   ✅ Найден отсутствующий элемент в области {i+1}: сходство {normalized_similarity:.3f}")
                        
                        all_missing_elements.append({
                            "type": "missing_element",
                            "bbox": bbox_norm,
                            "area": int(np.sum(best_mask)),
                            "confidence": float(1.0 - normalized_similarity),
                            "description": f"Missing element in LLaVA region {i+1}",
                            "llava_guided": True
                        })
                        
                        all_masks.append(best_mask.astype(np.uint8) * 255)
                    else:
                        print(f"   ⚪ Область {i+1} в норме: сходство {normalized_similarity:.3f}")
        
        # Дополнительный общий поиск для областей, не покрытых LLaVA
        print(f"   🔍 Дополнительный общий поиск...")
        general_missing, general_masks = self._find_missing_elements_enhanced(
            image_np, adjusted_queries, similarity_threshold=0.35, exclude_boxes=llava_boxes
        )
        
        all_missing_elements.extend(general_missing)
        all_masks.extend(general_masks)
        
        print(f"   📊 Итого найдено: {len(all_missing_elements)} отсутствующих элементов")
        
        return all_missing_elements, all_masks
    
    def _guided_defect_search_in_llava_regions(self, image_np, defect_queries, llava_boxes):
        """Направленный поиск ДЕФЕКТОВ в областях, указанных LLaVA"""
        
        from segment_anything_hq import SamAutomaticMaskGenerator, SamPredictor
        import faiss
        
        all_defect_elements = []
        all_masks = []
        
        h, w = image_np.shape[:2]
        
        # Создаем SAM predictor для точной сегментации
        sam_predictor = SamPredictor(self.searchdet_sam)
        sam_predictor.set_image(image_np)
        
        print(f"   🎯 Анализируем {len(llava_boxes)} областей от LLaVA на наличие дефектов...")
        
        for i, bbox_norm in enumerate(llava_boxes):
            # Преобразуем нормализованные координаты в абсолютные
            if len(bbox_norm) == 4:
                x_norm, y_norm, w_norm, h_norm = bbox_norm
                x1 = int(x_norm * w)
                y1 = int(y_norm * h)
                x2 = int((x_norm + w_norm) * w)
                y2 = int((y_norm + h_norm) * h)
                
                # Убеждаемся что координаты в пределах изображения
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                print(f"   📦 Область {i+1}: ({x1},{y1}) -> ({x2},{y2})")
                
                # Используем SAM для точной сегментации в этой области
                input_box = np.array([x1, y1, x2, y2])
                masks_in_region, scores, logits = sam_predictor.predict(
                    box=input_box,
                    multimask_output=True
                )
                
                # Выбираем лучшую маску
                best_mask_idx = np.argmax(scores)
                best_mask = masks_in_region[best_mask_idx]
                
                # Анализируем эту маску с помощью дефектных эмбеддингов
                if np.sum(best_mask) > 100:  # Минимальный размер маски
                    # Извлекаем признаки из маскированной области
                    masked_region = image_np.copy()
                    masked_region[~best_mask] = 0  # Обнуляем области вне маски
                    
                    region_img = Image.fromarray(masked_region)
                    region_features = get_vector(
                        region_img, self.searchdet_resnet, 
                        self.searchdet_layer, self.searchdet_transform
                    ).numpy().reshape(1, -1).astype(np.float32)
                    
                    # Нормализация для cosine similarity
                    defect_queries_norm = defect_queries / np.linalg.norm(defect_queries, axis=1, keepdims=True)
                    region_features_norm = region_features / np.linalg.norm(region_features, axis=1, keepdims=True)
                    
                    # FAISS поиск - ищем ВЫСОКОЕ сходство с дефектами
                    index = faiss.IndexFlatIP(defect_queries_norm.shape[1])
                    index.add(defect_queries_norm)
                    
                    similarities, indices = index.search(region_features_norm, 1)
                    normalized_similarity = (similarities[0][0] + 1) / 2
                    
                    # Проверяем является ли это дефектом (ВЫСОКОЕ сходство = дефект)
                    defect_threshold = 0.6  # Высокий порог для уверенного обнаружения дефекта
                    if normalized_similarity > defect_threshold:
                        print(f"   🔴 Найден ДЕФЕКТ в области {i+1}: сходство {normalized_similarity:.3f}")
                        
                        all_defect_elements.append({
                            "type": "defect",
                            "bbox": bbox_norm,
                            "area": int(np.sum(best_mask)),
                            "confidence": float(normalized_similarity),
                            "description": f"Defect found in LLaVA region {i+1}",
                            "llava_guided": True
                        })
                        
                        all_masks.append(best_mask.astype(np.uint8) * 255)
                    else:
                        print(f"   ✅ Область {i+1} в норме: сходство с дефектами {normalized_similarity:.3f}")
        
        # Дополнительный общий поиск дефектов в непроанализированных областях
        print(f"   🔍 Дополнительный общий поиск дефектов...")
        general_defects, general_masks = self._find_defects_enhanced(
            image_np, defect_queries, similarity_threshold=0.55, exclude_boxes=llava_boxes
        )
        
        all_defect_elements.extend(general_defects)
        all_masks.extend(general_masks)
        
        print(f"   📊 Итого найдено: {len(all_defect_elements)} дефектов")
        
        return all_defect_elements, all_masks
    
    def _find_missing_elements_enhanced(self, image_np, adjusted_queries, similarity_threshold=0.4, exclude_boxes=None):
        """Улучшенная версия поиска отсутствующих элементов с возможностью исключения областей"""
        
        from segment_anything_hq import SamAutomaticMaskGenerator
        import faiss
        
        # Генерация масок с помощью SAM
        mask_generator = SamAutomaticMaskGenerator(
            model=self.searchdet_sam,
            points_per_side=16,
            points_per_batch=32,
            pred_iou_thresh=0.90,
            stability_score_thresh=0.95,
            min_mask_region_area=800,
            box_nms_thresh=0.5,
            crop_nms_thresh=0.5,
        )
        
        masks = mask_generator.generate(image_np)
        
        # Постобработка: фильтруем слишком большие маски
        image_area = image_np.shape[0] * image_np.shape[1]
        max_allowed_area = image_area * 0.15
        
        filtered_masks = []
        for mask in masks:
            mask_area = mask['area']
            if mask_area <= max_allowed_area:
                # Если есть exclude_boxes, проверяем не перекрывается ли маска с ними
                if exclude_boxes:
                    mask_overlaps = self._check_mask_overlap_with_boxes(mask, exclude_boxes, image_np.shape)
                    if not mask_overlaps:  # Только если не перекрывается
                        filtered_masks.append(mask)
                else:
                    filtered_masks.append(mask)
        
        masks = filtered_masks
        print(f"   🔍 Отфильтровано масок SAM: {len(masks)}")
        
        if not masks:
            return [], []
        
        # Извлечение признаков масок
        mask_vectors = extract_features_from_masks(
            image_np, masks, self.searchdet_resnet, self.searchdet_layer, self.searchdet_transform
        )
        mask_vectors = np.array(mask_vectors, dtype=np.float32)
        
        # Нормализация для cosine similarity
        query_vectors = adjusted_queries / np.linalg.norm(adjusted_queries, axis=1, keepdims=True)
        mask_vectors = mask_vectors / np.linalg.norm(mask_vectors, axis=1, keepdims=True)
        
        # FAISS поиск
        index = faiss.IndexFlatIP(query_vectors.shape[1])
        index.add(query_vectors)
        
        similarities, indices = index.search(mask_vectors, 1)
        normalized_similarities = (similarities + 1) / 2
        
        # Поиск областей с низким сходством
        missing_threshold = 1.0 - similarity_threshold
        missing_indices = np.where(normalized_similarities.flatten() < missing_threshold)[0]
        
        print(f"   🔍 Обработано масок: {len(normalized_similarities)}")
        print(f"   📊 Порог отсутствующих: {missing_threshold:.3f}")
        print(f"   📈 Диапазон сходства: {normalized_similarities.min():.3f} - {normalized_similarities.max():.3f}")
        print(f"   🎯 Кандидатов на отсутствующие: {len(missing_indices)}")
        
        missing_elements = []
        final_masks = []
        
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
                    "description": "Potentially missing component detected by SearchDet",
                    "llava_guided": False
                })
                
                final_masks.append(seg.astype(np.uint8) * 255)
        
        print(f"   ✅ Найдено отсутствующих элементов: {len(missing_elements)}")
        
        return missing_elements, final_masks
    
    def _check_mask_overlap_with_boxes(self, mask, exclude_boxes, image_shape):
        """Проверяет перекрывается ли маска с исключаемыми областями"""
        
        seg = mask['segmentation']
        h, w = image_shape[:2]
        
        # Получаем bounding box маски
        coords = np.column_stack(np.where(seg))
        if len(coords) == 0:
            return False
            
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        # Проверяем пересечение с каждым exclude_box
        for bbox_norm in exclude_boxes:
            if len(bbox_norm) == 4:
                bx_norm, by_norm, bw_norm, bh_norm = bbox_norm
                bx1 = int(bx_norm * w)
                by1 = int(by_norm * h)
                bx2 = int((bx_norm + bw_norm) * w)
                by2 = int((by_norm + bh_norm) * h)
                
                # Проверяем пересечение прямоугольников
                if not (x_max < bx1 or x_min > bx2 or y_max < by1 or y_min > by2):
                    return True  # Есть пересечение
        
        return False  # Нет пересечений
    
    def _find_defects_enhanced(self, image_np, defect_queries, similarity_threshold=0.6, exclude_boxes=None):
        """Улучшенная версия поиска ДЕФЕКТОВ с возможностью исключения областей"""
        
        from segment_anything_hq import SamAutomaticMaskGenerator
        import faiss
        
        # Генерация масок с помощью SAM
        mask_generator = SamAutomaticMaskGenerator(
            model=self.searchdet_sam,
            points_per_side=20,  # Больше точек для поиска дефектов
            points_per_batch=32,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.93,
            min_mask_region_area=200,  # Меньший минимум для мелких дефектов
            box_nms_thresh=0.5,
            crop_nms_thresh=0.5,
        )
        
        masks = mask_generator.generate(image_np)
        
        # Постобработка: фильтруем слишком большие маски
        image_area = image_np.shape[0] * image_np.shape[1]
        max_allowed_area = image_area * 0.1  # Дефекты обычно небольшие
        
        filtered_masks = []
        for mask in masks:
            mask_area = mask['area']
            if mask_area <= max_allowed_area:
                # Если есть exclude_boxes, проверяем не перекрывается ли маска с ними
                if exclude_boxes:
                    mask_overlaps = self._check_mask_overlap_with_boxes(mask, exclude_boxes, image_np.shape)
                    if not mask_overlaps:  # Только если не перекрывается
                        filtered_masks.append(mask)
                else:
                    filtered_masks.append(mask)
        
        masks = filtered_masks
        print(f"   🔍 Отфильтровано масок SAM для поиска дефектов: {len(masks)}")
        
        if not masks:
            return [], []
        
        # Извлечение признаков масок
        mask_vectors = extract_features_from_masks(
            image_np, masks, self.searchdet_resnet, self.searchdet_layer, self.searchdet_transform
        )
        mask_vectors = np.array(mask_vectors, dtype=np.float32)
        
        # Нормализация для cosine similarity
        query_vectors = defect_queries / np.linalg.norm(defect_queries, axis=1, keepdims=True)
        mask_vectors = mask_vectors / np.linalg.norm(mask_vectors, axis=1, keepdims=True)
        
        # FAISS поиск - ищем ВЫСОКОЕ сходство с дефектами
        index = faiss.IndexFlatIP(query_vectors.shape[1])
        index.add(query_vectors)
        
        similarities, indices = index.search(mask_vectors, 1)
        normalized_similarities = (similarities + 1) / 2
        
        # Поиск областей с ВЫСОКИМ сходством (дефекты)
        defect_indices = np.where(normalized_similarities.flatten() > similarity_threshold)[0]
        
        print(f"   🔍 Обработано масок: {len(normalized_similarities)}")
        print(f"   📊 Порог дефектов: {similarity_threshold:.3f}")
        print(f"   📈 Диапазон сходства: {normalized_similarities.min():.3f} - {normalized_similarities.max():.3f}")
        print(f"   🎯 Кандидатов на дефекты: {len(defect_indices)}")
        
        defect_elements = []
        final_masks = []
        
        for idx in defect_indices:
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
                bbox_norm = (x_min/w, y_min/h, (x_max-x_min)/w, (y_max-y_min)/h)
                
                defect_elements.append({
                    "type": "defect",
                    "bbox": bbox_norm,
                    "area": mask_area,
                    "confidence": float(normalized_similarities[idx][0]),
                    "description": "Defect detected by SearchDet similarity matching",
                    "llava_guided": False
                })
                
                final_masks.append(seg.astype(np.uint8) * 255)
        
        print(f"   ✅ Найдено дефектов: {len(defect_elements)}")
        
        return defect_elements, final_masks
    
    def _combine_llava_searchdet_results(self, llava_results, searchdet_results):
        """Объединение результатов LLaVA и SearchDet для дефектов"""
        
        combined = llava_results.copy()
        
        # Добавляем найденные дефекты от SearchDet
        defect_elements = searchdet_results.get("defect_elements", [])
        if defect_elements:
            # Добавляем новые типы дефектов
            defect_types = ["searchdet_defect", "visual_defect"]
            combined["defect_types"] = list(set(combined.get("defect_types", []) + defect_types))
            
            # Обновляем серьезность если найдены дефекты
            if len(defect_elements) > 3:
                combined["severity"] = "critical"
            elif len(defect_elements) > 1:
                combined["severity"] = "moderate"
            elif combined.get("severity") in ["unknown", "minor"]:
                combined["severity"] = "minor"
            
            # Добавляем bounding boxes от SearchDet
            searchdet_boxes = [elem["bbox"] for elem in defect_elements]
            existing_boxes = combined.get("bounding_boxes", [])
            combined["bounding_boxes"] = existing_boxes + searchdet_boxes
            
            # Обновляем описание
            combined["description"] += f"\n\nSearchDet detected {len(defect_elements)} additional defects through similarity matching."
            
            # Добавляем информацию о SearchDet дефектах
            combined["searchdet_defects"] = defect_elements
        
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
        """Сохранение результатов с созданием папки masks и отдельных файлов масок"""
        
        image_name = Path(image_path).stem
        
        # Создаем папку masks
        masks_dir = output_path / "masks"
        masks_dir.mkdir(exist_ok=True)
        
        # JSON с результатами
        json_compatible_results = self._convert_numpy_to_json(results)
        json_path = output_path / f"{image_name}_analysis.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_compatible_results, f, ensure_ascii=False, indent=2)
        
        print(f"📁 Сохраняем результаты в: {output_path}")
        print(f"📂 Создана папка масок: {masks_dir}")
        
        # Счётчик для SearchDet масок
        searchdet_mask_count = 0
        
        # Сохраняем только SearchDet маски
        for i, mask in enumerate(masks):
            if i < len(results["annotations"]["defects"]):
                defect_info = results["annotations"]["defects"][i]
                detection_method = defect_info.get("detection_method", "sam2_segmentation")
                category = defect_info.get("category", "defect")
                
                # Сохраняем ТОЛЬКО SearchDet маски
                if detection_method == "searchdet_missing" or category == "missing_element" or category == "defect":
                    # Дополнительная фильтрация по размеру маски
                    mask_area = np.sum(mask > 0)
                    total_image_area = image.shape[0] * image.shape[1]
                    mask_percentage = (mask_area / total_image_area) * 100
                    
                    # Если маска слишком большая, попробуем разделить её на компоненты
                    if mask_percentage > 8.0:  # Уменьшили порог с 15% до 8%
                        split_masks = self._split_large_mask(mask, max_component_percentage=5.0)  # Уменьшили с 10% до 5%
                        if split_masks:
                            print(f"   🔄 Разделили большую маску на {len(split_masks)} компонентов")
                            # Обрабатываем каждый компонент отдельно
                            for j, split_mask in enumerate(split_masks):
                                split_area = np.sum(split_mask > 0)
                                split_percentage = (split_area / total_image_area) * 100
                                
                                if 0.15 <= split_percentage <= 5.0:  # Немного смягчаем: 0.15-5% вместо 0.2-5%
                                    searchdet_mask_count += 1
                                    print(f"   ✅ Дефект-компонент {j+1}: {split_percentage:.1f}% изображения")
                                    self._save_single_mask(split_mask, image, masks_dir, searchdet_mask_count)
                            continue
                        else:
                            print(f"   ⚠️ Пропускаем слишком большую маску: {mask_percentage:.1f}% изображения")
                            continue
                    
                    # Фильтруем слишком большие маски (больше 15% изображения)
                    if mask_percentage > 15.0:  # Уменьшили с 25% до 15%
                        print(f"   ⚠️ Пропускаем слишком большую маску: {mask_percentage:.1f}% изображения")
                        continue
                    
                    # Фильтруем слишком маленькие маски (меньше 0.15% изображения)
                    if mask_percentage < 0.15:  # Уменьшили с 0.2% до 0.15%
                        print(f"   ⚠️ Пропускаем слишком маленькую маску: {mask_percentage:.3f}% изображения")
                        continue
                    
                    # Проверяем компактность маски (соотношение площади к периметру)
                    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        main_contour = max(contours, key=cv2.contourArea)
                        perimeter = cv2.arcLength(main_contour, True)
                        if perimeter > 0:
                            compactness = (4 * np.pi * mask_area) / (perimeter * perimeter)
                            # Фильтруем слишком "разтянутые" маски (compactness < 0.18)
                            if compactness < 0.18:  # Немного смягчаем с 0.2 до 0.18
                                print(f"   ⚠️ Пропускаем слишком разтянутую маску: compactness={compactness:.3f}")
                                continue
                            
                            # Дополнительная проверка: соотношение сторон bounding box
                            x, y, w, h = cv2.boundingRect(main_contour)
                            aspect_ratio = max(w, h) / min(w, h)
                            if aspect_ratio > 4.0:  # Исключаем слишком вытянутые объекты
                                print(f"   ⚠️ Пропускаем слишком вытянутую маску: aspect_ratio={aspect_ratio:.1f}")
                                continue
                    
                    searchdet_mask_count += 1
                    print(f"   ✅ Дефект {searchdet_mask_count}: {mask_percentage:.1f}% изображения")
                    
                    # Сохраняем маску
                    self._save_single_mask(mask, image, masks_dir, searchdet_mask_count)
        
        print(f"   ✅ Сохранено {searchdet_mask_count} SearchDet дефектов в папке masks/")
        
        # Создаём общую визуализацию с ВСЕМИ SearchDet масками
        if searchdet_mask_count > 0:
            combined_visualization = self._create_combined_masks_visualization(image, masks, results)
            combined_path = output_path / f"{image_name}_all_defects_combined.jpg"
            cv2.imwrite(str(combined_path), combined_visualization)
            print(f"   ✅ Общая визуализация всех дефектов: {combined_path.name}")
        else:
            print("   ⚠️ SearchDet дефекты не найдены, общая визуализация не создана")
        
        # Сохраняем оригинальное изображение для сравнения
        original_path = output_path / f"{image_name}_original.jpg"
        cv2.imwrite(str(original_path), image)
        print(f"   ✅ Оригинальное изображение: {original_path.name}")
    
    def _split_large_mask(self, mask, max_component_percentage=5.0):
        """Разделение большой маски на отдельные связанные компоненты"""
        
        # Находим связанные компоненты
        num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
        
        split_masks = []
        total_pixels = mask.shape[0] * mask.shape[1]
        
        for label in range(1, num_labels):  # пропускаем фон (label=0)
            component_mask = (labels == label).astype(np.uint8) * 255
            component_area = np.sum(component_mask > 0)
            component_percentage = (component_area / total_pixels) * 100
            
            # Берём только компоненты разумного размера с более строгими ограничениями
            if 0.3 <= component_percentage <= max_component_percentage:  # Увеличили нижний порог с 0.1 до 0.3
                # Дополнительная проверка компактности для компонентов
                contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    main_contour = max(contours, key=cv2.contourArea)
                    perimeter = cv2.arcLength(main_contour, True)
                    if perimeter > 0:
                        compactness = (4 * np.pi * component_area) / (perimeter * perimeter)
                        if compactness >= 0.25:  # Только достаточно компактные компоненты
                            split_masks.append(component_mask)
        
        return split_masks if split_masks else None
    
    def _save_single_mask(self, mask, image, masks_dir, mask_count):
        """Сохранение одной маски в виде бинарного файла и цветного наложения"""
        
        # Сохраняем бинарную маску (белое на черном)
        binary_mask = np.zeros_like(mask)
        binary_mask[mask > 0] = 255
        
        mask_filename = f"searchdet_defect_{mask_count:02d}.png"
        mask_path = masks_dir / mask_filename
        cv2.imwrite(str(mask_path), binary_mask)
        
        # Также сохраняем цветную маску на оригинальном изображении
        colored_overlay = image.copy()
        red_mask = np.zeros_like(image)
        red_mask[mask > 0] = [0, 0, 255]  # Красный
        cv2.addWeighted(colored_overlay, 0.7, red_mask, 0.3, 0, colored_overlay)
        
        # Добавляем контур
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(colored_overlay, contours, -1, (0, 0, 255), 2)
        
        colored_filename = f"searchdet_defect_{mask_count:02d}_overlay.jpg"
        colored_path = masks_dir / colored_filename
        cv2.imwrite(str(colored_path), colored_overlay)
    
    def _create_combined_masks_visualization(self, image, masks, results):
        """Создание общей визуализации всех SearchDet масок"""
        
        overlay = image.copy()
        red_color = (0, 0, 255)  # Красный для SearchDet
        
        defect_count = 0
        
        for i, mask in enumerate(masks):
            if i < len(results["annotations"]["defects"]):
                defect_info = results["annotations"]["defects"][i]
                detection_method = defect_info.get("detection_method", "sam2_segmentation")
                category = defect_info.get("category", "defect")
                
                # ПОКАЗЫВАЕМ ТОЛЬКО SearchDet маски
                if detection_method == "searchdet_missing" or category == "missing_element" or category == "defect":
                    defect_count += 1
                    
                    # Полупрозрачная красная маска для дефектов
                    red_mask = np.zeros_like(overlay)
                    red_mask[mask > 0] = red_color
                    cv2.addWeighted(overlay, 0.8, red_mask, 0.2, 0, overlay)
                    
                    # Красный контур
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        cv2.drawContours(overlay, contours, -1, red_color, 2)
                        
                        # Добавляем номер дефекта
                        main_contour = max(contours, key=cv2.contourArea)
                        M = cv2.moments(main_contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            
                            # Текст с номером дефекта
                            text = str(defect_count)
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 0.8
                            thickness = 2
                            
                            # Тень
                            cv2.putText(overlay, text, (cx-10, cy+10), font, font_scale, (0, 0, 0), thickness+1)
                            # Основной текст
                            cv2.putText(overlay, text, (cx-8, cy+8), font, font_scale, (255, 255, 255), thickness)
        
        # Добавляем информационную панель
        if defect_count > 0:
            h, w = overlay.shape[:2]
            panel_height = 60
            panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
            
            # Заголовок
            cv2.putText(panel, f"SearchDet: {defect_count} DEFECTS FOUND", 
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(panel, "Red areas = Detected defects and anomalies", 
                       (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 1)
            
            # Объединяем панель с изображением
            overlay = np.vstack([panel, overlay])
        
        return overlay
    
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
        print(f"   🎯 SearchDet дефекты: {len(results['stages']['searchdet_analysis']['result'].get('defect_elements', []))}")
        
        # Подсчитаем только SearchDet маски в визуализации
        searchdet_masks_shown = 0
        for defect in results['annotations']['defects']:
            if (defect.get('detection_method') == 'searchdet_missing' or 
                defect.get('category') == 'missing_element' or
                defect.get('category') == 'defect'):
                searchdet_masks_shown += 1
        
        print(f"   🔴 КРАСНЫХ масок показано: {searchdet_masks_shown} (SearchDet дефекты)")
        print(f"   📝 Всего аннотаций: {len(results['annotations']['defects'])}")
        total_time = sum(stage['duration'] for stage in results['stages'].values())
        print(f"   ⏱️ Общее время: {total_time:.2f} сек")
        print(f"\n🔴 В визуализации показываются ТОЛЬКО КРАСНЫЕ маски от SearchDet (дефекты)!")


if __name__ == "__main__":
    main() 