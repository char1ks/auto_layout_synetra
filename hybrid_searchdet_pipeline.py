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
            
            # Поиск отсутствующих элементов
            missing_elements = self._find_missing_elements(
                example_img, adjusted_queries, similarity_threshold=0.4
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
            min_mask_region_area=100,
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
                
                # Нормализованные координаты
                h, w = image_np.shape[:2]
                bbox_norm = (x_min/w, y_min/h, x_max/w, y_max/h)
                
                missing_elements.append({
                    "type": "missing_element",
                    "bbox": bbox_norm,
                    "area": int(seg.sum()),
                    "confidence": float(1.0 - normalized_similarities[idx][0]),
                    "description": "Potentially missing component detected by SearchDet"
                })
        
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
    
    def _save_hybrid_results(self, image_path, image, masks, results, output_path):
        """Сохранение результатов гибридного анализа"""
        
        image_name = Path(image_path).stem
        
        # JSON с результатами
        json_path = output_path / f"{image_name}_hybrid_analysis.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # Визуализация
        visualization = self._create_hybrid_visualization(image, masks, results)
        cv2.imwrite(str(output_path / f"{image_name}_hybrid_result.jpg"), visualization)
        
        # Отдельные маски
        for i, mask in enumerate(masks):
            mask_path = output_path / f"{image_name}_hybrid_mask_{i+1}.png"
            cv2.imwrite(str(mask_path), mask)
    
    def _create_hybrid_visualization(self, image, masks, results):
        """Создание визуализации с информацией от всех систем"""
        
        vis_image = image.copy()
        
        # Цвета для разных типов детекции
        colors = {
            "sam2_segmentation": (0, 255, 0),      # Зеленый
            "searchdet_missing": (0, 0, 255),     # Красный  
            "llava_coordinates": (255, 0, 0),     # Синий
        }
        
        # Наложение масок
        for i, mask in enumerate(masks):
            defect_info = results["annotations"]["defects"][i] if i < len(results["annotations"]["defects"]) else {}
            detection_method = defect_info.get("detection_method", "sam2_segmentation")
            color = colors.get(detection_method, (255, 255, 255))
            
            # Создание цветной маски
            colored_mask = np.zeros_like(vis_image)
            colored_mask[mask > 0] = color
            
            # Наложение с прозрачностью
            vis_image = cv2.addWeighted(vis_image, 0.7, colored_mask, 0.3, 0)
            
            # Контур
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis_image, contours, -1, color, 2)
        
        # Информация о результатах
        llava_result = results["stages"]["llava_analysis"]["material"]
        searchdet_result = results["stages"]["searchdet_analysis"]["result"]
        
        info_text = f"Material: {llava_result['material']} | Total issues: {len(masks)}"
        missing_text = f"SearchDet missing: {len(searchdet_result.get('missing_elements', []))}"
        
        cv2.putText(vis_image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(vis_image, missing_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Легенда
        cv2.putText(vis_image, "Green: SAM2 | Red: Missing | Blue: LLaVA", (10, vis_image.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return vis_image


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
        print(f"   📝 Финальных аннотаций: {len(results['annotations']['defects'])}")
        total_time = sum(stage['duration'] for stage in results['stages'].values())
        print(f"   ⏱️ Общее время: {total_time:.2f} сек")


if __name__ == "__main__":
    main() 