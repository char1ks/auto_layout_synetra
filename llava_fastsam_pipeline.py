#!/usr/bin/env python3
"""
Автоматическая разметка дефектов: LLaVA → FastSAM → OpenCV
Быстрый pipeline для определения материала и сегментации дефектов
"""

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json
import time
from datetime import datetime
import platform
import os

# GPU Detection
def get_device():
    """Автоматическая детекция лучшего устройства"""
    if torch.cuda.is_available():
        device = 'cuda'
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🔥 GPU найдено: {gpu_name} ({gpu_memory:.1f}GB)")
        return device
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("🍎 MPS (Apple Silicon) найдено")
        return 'mps'
    else:
        print("💻 Используем CPU")
        return 'cpu'

DEVICE = get_device()
print(f"🎯 Устройство: {DEVICE}")

# Platform detection
IS_WINDOWS = platform.system() == 'Windows'
IS_MACOS = platform.system() == 'Darwin'
IS_LINUX = platform.system() == 'Linux'

print(f"🖥️ Операционная система: {platform.system()}")

# LLaVA imports
try:
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
    from llava.conversation import conv_templates, SeparatorStyle
    from PIL import Image
    LLAVA_AVAILABLE = True
except ImportError:
    print("❌ LLaVA не установлена. Устанавливается заглушка...")
    LLAVA_AVAILABLE = False

# FastSAM imports using ultralytics
try:
    from ultralytics import FastSAM
    FASTSAM_AVAILABLE = True
    print("✅ FastSAM через ultralytics найден")
except ImportError as e:
    print(f"❌ FastSAM не найдена: {e}")
    FASTSAM_AVAILABLE = False


class MaterialClassifier:
    """LLaVA для определения типа материала"""
    
    def __init__(self, model_path="liuhaotian/llava-v1.5-7b"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.context_len = None
        
        if LLAVA_AVAILABLE:
            self._load_model()
    
    def _load_model(self):
        """Загрузка модели LLaVA"""
        print("🔄 Загружаем LLaVA модель...")
        try:
            self.model_name = get_model_name_from_path(self.model_path)
            
            # Настройки в зависимости от устройства
            if DEVICE == 'cuda':
                # CUDA (Windows/Linux with NVIDIA GPU)
                self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                    self.model_path, None, self.model_name, 
                    device_map="auto",
                    load_8bit=False, load_4bit=False,
                    torch_dtype=torch.float16  # float16 для GPU экономии памяти
                )
            elif DEVICE == 'mps':
                # Apple Silicon
                self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                    self.model_path, None, self.model_name,
                    device_map="mps",
                    load_8bit=False, load_4bit=False,
                    torch_dtype=torch.float16
                )
            else:
                # CPU fallback
                self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                    self.model_path, None, self.model_name,
                    device_map="cpu",
                    load_8bit=False, load_4bit=False,
                    torch_dtype=torch.float32  # float32 для CPU
                )
                self.model = self.model.float()
            
            print(f"✅ LLaVA модель загружена на {DEVICE}")
            print("✅ LLaVA модель загружена успешно")
        except Exception as e:
            print(f"❌ Ошибка загрузки LLaVA: {e}")
            print("🔄 Пробуем загрузить с упрощенными настройками...")
            try:
                # Второй способ - упрощенная загрузка
                if DEVICE == 'cuda':
                    dtype = torch.float16
                else:
                    dtype = torch.float32
                    
                self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                    self.model_path, None, self.model_name, torch_dtype=dtype
                )
                self.model = self.model.to(DEVICE)
                if DEVICE == 'cpu':
                    self.model = self.model.float()
                print("✅ LLaVA модель загружена с упрощенными настройками")
            except Exception as e2:
                print(f"❌ Повторная ошибка загрузки LLaVA: {e2}")
                self.model = None
    
    def classify_material(self, image_path):
        """Определение типа материала на изображении"""
        if not LLAVA_AVAILABLE or self.model is None:
            # Заглушка
            print("⚠️ LLaVA недоступна, используем заглушку")
            return {
                "material": "metal", 
                "confidence": 0.8,
                "description": "metallic surface with potential defects"
            }
        
        try:
            # Загрузка изображения
            image = Image.open(image_path).convert('RGB')
            
            # Промпт для классификации материала
            prompt = """Look at this image carefully. What type of material is shown? 
            Choose from: metal, wood, plastic, concrete, fabric, glass, ceramic.
            Also describe the surface condition and any visible defects.
            
            Format your answer as:
            Material: [material_type]
            Condition: [surface_description]
            Defects: [visible_issues_or_none]"""
            
            # Подготовка входных данных
            conv = conv_templates["vicuna_v1"].copy()
            conv.append_message(conv.roles[0], f"{DEFAULT_IMAGE_TOKEN}\n{prompt}")
            conv.append_message(conv.roles[1], None)
            prompt_formatted = conv.get_prompt()
            
            # Токенизация с padding
            input_ids = tokenizer_image_token(prompt_formatted, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            
            # Обработка изображения
            image_tensor = process_images([image], self.image_processor, self.model.config)
            
            # Генерация ответа с автоматическим устройством
            input_ids = input_ids.to(DEVICE)
            if hasattr(image_tensor, 'to'):
                image_tensor = image_tensor.to(DEVICE)
            
            # Настройки генерации в зависимости от устройства
            max_tokens = 256 if DEVICE == 'cuda' else 128
            
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=[image.size],
                    do_sample=True,
                    temperature=0.2,
                    max_new_tokens=max_tokens,
                    use_cache=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Декодирование результата
            output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            response = output.split("ASSISTANT:")[-1].strip()
            
            # Парсинг ответа
            material_type = self._parse_material(response)
            
            return {
                "material": material_type,
                "confidence": 0.9,
                "description": response
            }
            
        except Exception as e:
            print(f"❌ Ошибка классификации материала: {e}")
            return {
                "material": "unknown", 
                "confidence": 0.5,
                "description": "failed to analyze"
            }
    
    def _parse_material(self, response):
        """Извлечение типа материала из ответа LLaVA"""
        response_lower = response.lower()
        
        materials = ["metal", "wood", "plastic", "concrete", "fabric", "glass", "ceramic"]
        
        for material in materials:
            if material in response_lower:
                return material
        
        # Попытка найти по ключевым словам
        if any(word in response_lower for word in ["steel", "iron", "aluminum", "copper"]):
            return "metal"
        elif any(word in response_lower for word in ["wooden", "timber", "oak", "pine"]):
            return "wood"
        elif any(word in response_lower for word in ["rubber", "polymer"]):
            return "plastic"
        
        return "unknown"


class DefectSegmenter:
    """FastSAM для быстрой сегментации дефектов"""
    
    def __init__(self, model_path="./models/FastSAM-x.pt"):
        self.model_path = model_path
        self.model = None
        
        if FASTSAM_AVAILABLE:
            self._load_model()
    
    def _load_model(self):
        """Загрузка модели FastSAM"""
        print("🔄 Загружаем FastSAM модель...")
        try:
            # Используем ultralytics для автоматического скачивания
            self.model = FastSAM('FastSAM-x.pt')  # Ultralytics автоматически скачает модель
            print("✅ FastSAM модель загружена успешно")
        except Exception as e:
            print(f"❌ Ошибка загрузки FastSAM: {e}")
            self.model = None
    
    def segment_defects(self, image_path, material_type="unknown"):
        """Сегментация потенциальных дефектов"""
        if not FASTSAM_AVAILABLE or self.model is None:
            # Заглушка
            print("⚠️ FastSAM недоступна, используем простую сегментацию")
            return self._simple_segmentation(image_path)
        
        try:
            # Запуск FastSAM с автоматическим устройством
            # Настройки в зависимости от устройства
            imgsz = 1024 if DEVICE == 'cuda' else 640
            
            results = self.model(
                str(image_path),
                device=DEVICE,
                imgsz=imgsz,
                conf=0.4,
                iou=0.9,
                save=False,
                verbose=False
            )
            
            # Извлечение масок из результатов
            all_masks = []
            if results and len(results) > 0:
                result = results[0]
                
                # Получаем маски если есть
                if hasattr(result, 'masks') and result.masks is not None:
                    masks_data = result.masks.data
                    
                    # Конвертируем в numpy
                    if hasattr(masks_data, 'cpu'):
                        masks_data = masks_data.cpu().numpy()
                    elif hasattr(masks_data, 'numpy'):
                        masks_data = masks_data.numpy()
                    
                    # Разделяем каждую маску
                    for i in range(masks_data.shape[0]):
                        mask = masks_data[i]
                        # Преобразуем в uint8
                        mask = (mask * 255).astype(np.uint8)
                        all_masks.append(mask)
            
            # Фильтрация масок по размеру и форме
            filtered_masks = self._filter_masks(all_masks, material_type)
            
            return {
                "masks": filtered_masks,
                "num_detections": len(filtered_masks),
                "raw_masks": all_masks
            }
            
        except Exception as e:
            print(f"❌ Ошибка сегментации FastSAM: {e}")
            return self._simple_segmentation(image_path)
    
    def _simple_segmentation(self, image_path):
        """Простая сегментация для случая когда FastSAM недоступна"""
        image = cv2.imread(str(image_path))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Простая обработка для поиска контрастных областей
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Поиск контуров
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Создание масок из контуров
        masks = []
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Минимальный размер
                mask = np.zeros_like(gray)
                cv2.fillPoly(mask, [contour], 255)
                masks.append(mask)
        
        return {
            "masks": masks[:10],  # Максимум 10 масок
            "num_detections": len(masks[:10]),
            "raw_masks": masks
        }
    
    def _filter_masks(self, masks, material_type):
        """Фильтрация масок по материалу"""
        if not masks:
            return []
        
        filtered = []
        for mask in masks:
            # Базовая фильтрация по размеру
            if isinstance(mask, np.ndarray):
                area = np.sum(mask > 0)
                if 50 < area < 50000:  # Разумные размеры дефектов
                    filtered.append(mask)
        
        return filtered[:20]  # Максимум 20 лучших масок


class OpenCVPostProcessor:
    """OpenCV постобработка для улучшения результатов"""
    
    def __init__(self):
        self.material_params = {
            "metal": {"min_area": 100, "max_area": 10000, "morph_kernel": 3},
            "wood": {"min_area": 200, "max_area": 15000, "morph_kernel": 5},
            "plastic": {"min_area": 50, "max_area": 8000, "morph_kernel": 2},
            "default": {"min_area": 100, "max_area": 12000, "morph_kernel": 3}
        }
    
    def process_masks(self, image, masks, material_type="default"):
        """Постобработка масок с учетом типа материала"""
        if not masks:
            return []
        
        params = self.material_params.get(material_type, self.material_params["default"])
        processed_masks = []
        
        for mask in masks:
            processed_mask = self._clean_mask(mask, params)
            if processed_mask is not None:
                processed_masks.append(processed_mask)
        
        # Удаление пересекающихся масок
        final_masks = self._remove_overlaps(processed_masks)
        
        return final_masks
    
    def _clean_mask(self, mask, params):
        """Очистка одной маски"""
        try:
            # Конвертация в бинарную маску
            if mask.dtype != np.uint8:
                mask = (mask * 255).astype(np.uint8)
            
            # Морфологические операции
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (params["morph_kernel"], params["morph_kernel"]))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Проверка размера
            area = cv2.countNonZero(mask)
            if params["min_area"] <= area <= params["max_area"]:
                return mask
            
            return None
        except:
            return None
    
    def _remove_overlaps(self, masks, overlap_threshold=0.5):
        """Удаление сильно пересекающихся масок"""
        if len(masks) <= 1:
            return masks
        
        # Вычисление IoU между масками
        final_masks = []
        used_indices = set()
        
        for i, mask1 in enumerate(masks):
            if i in used_indices:
                continue
                
            is_unique = True
            for j, mask2 in enumerate(masks[i+1:], i+1):
                if j in used_indices:
                    continue
                
                iou = self._calculate_iou(mask1, mask2)
                if iou > overlap_threshold:
                    # Оставляем маску с большей площадью
                    area1 = cv2.countNonZero(mask1)
                    area2 = cv2.countNonZero(mask2)
                    if area2 > area1:
                        is_unique = False
                        break
                    else:
                        used_indices.add(j)
            
            if is_unique:
                final_masks.append(mask1)
                used_indices.add(i)
        
        return final_masks
    
    def _calculate_iou(self, mask1, mask2):
        """Вычисление IoU между двумя масками"""
        try:
            intersection = cv2.bitwise_and(mask1, mask2)
            union = cv2.bitwise_or(mask1, mask2)
            
            intersection_area = cv2.countNonZero(intersection)
            union_area = cv2.countNonZero(union)
            
            if union_area == 0:
                return 0
            
            return intersection_area / union_area
        except:
            return 0


class DefectAnalysisPipeline:
    """Основной pipeline для анализа дефектов"""
    
    def __init__(self):
        self.material_classifier = MaterialClassifier()
        self.defect_segmenter = DefectSegmenter()
        self.postprocessor = OpenCVPostProcessor()
    
    def analyze_image(self, image_path, output_dir="./output"):
        """Полный анализ изображения"""
        print(f"\n🔍 Анализируем изображение: {image_path}")
        start_time = time.time()
        
        # Создание директории для результатов
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "image_path": str(image_path),
            "stages": {}
        }
        
        try:
            # Этап 1: Определение материала
            print("🔬 Этап 1: Определение материала...")
            stage1_start = time.time()
            material_result = self.material_classifier.classify_material(image_path)
            stage1_time = time.time() - stage1_start
            
            results["stages"]["material_classification"] = {
                "duration": stage1_time,
                "result": material_result
            }
            print(f"   ✅ Материал: {material_result['material']} (уверенность: {material_result['confidence']:.2f})")
            print(f"   ⏱️ Время: {stage1_time:.2f} сек")
            
            # Этап 2: Сегментация дефектов
            print("🎯 Этап 2: Сегментация дефектов...")
            stage2_start = time.time()
            segmentation_result = self.defect_segmenter.segment_defects(image_path, material_result['material'])
            stage2_time = time.time() - stage2_start
            
            results["stages"]["defect_segmentation"] = {
                "duration": stage2_time,
                "num_raw_detections": segmentation_result['num_detections']
            }
            print(f"   ✅ Найдено сегментов: {segmentation_result['num_detections']}")
            print(f"   ⏱️ Время: {stage2_time:.2f} сек")
            
            # Этап 3: Постобработка
            print("🎨 Этап 3: Постобработка...")
            stage3_start = time.time()
            image = cv2.imread(str(image_path))
            final_masks = self.postprocessor.process_masks(
                image, 
                segmentation_result['masks'], 
                material_result['material']
            )
            stage3_time = time.time() - stage3_start
            
            results["stages"]["postprocessing"] = {
                "duration": stage3_time,
                "num_final_detections": len(final_masks)
            }
            print(f"   ✅ Финальных дефектов: {len(final_masks)}")
            print(f"   ⏱️ Время: {stage3_time:.2f} сек")
            
            # Создание аннотаций
            annotations = self._create_annotations(image, final_masks, material_result)
            results["annotations"] = annotations
            
            # Сохранение результатов
            self._save_results(image_path, image, final_masks, results, output_path)
            
            total_time = time.time() - start_time
            print(f"\n🎉 Анализ завершен за {total_time:.2f} секунд")
            print(f"📁 Результаты сохранены в: {output_path}")
            
            return results
            
        except Exception as e:
            print(f"❌ Ошибка анализа: {e}")
            return None
    
    def _create_annotations(self, image, masks, material_result):
        """Создание аннотаций в формате COCO"""
        annotations = {
            "image_info": {
                "height": image.shape[0],
                "width": image.shape[1],
                "channels": image.shape[2]
            },
            "material": material_result,
            "defects": []
        }
        
        for i, mask in enumerate(masks):
            # Поиск контура
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            
            # Самый большой контур
            main_contour = max(contours, key=cv2.contourArea)
            
            # Bounding box
            x, y, w, h = cv2.boundingRect(main_contour)
            
            # Площадь
            area = cv2.countNonZero(mask)
            
            # Полигон для сегментации
            polygon = main_contour.flatten().tolist()
            
            defect_annotation = {
                "id": i + 1,
                "category": "defect",
                "bbox": [x, y, w, h],
                "area": int(area),
                "segmentation": [polygon],
                "confidence": 0.8  # Базовая уверенность
            }
            
            annotations["defects"].append(defect_annotation)
        
        return annotations
    
    def _save_results(self, image_path, image, masks, results, output_path):
        """Сохранение всех результатов"""
        image_name = Path(image_path).stem
        
        # JSON с результатами
        json_path = output_path / f"{image_name}_analysis.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # Визуализация
        visualization = self._create_visualization(image, masks, results)
        cv2.imwrite(str(output_path / f"{image_name}_result.jpg"), visualization)
        
        # Отдельные маски
        for i, mask in enumerate(masks):
            mask_path = output_path / f"{image_name}_mask_{i+1}.png"
            cv2.imwrite(str(mask_path), mask)
    
    def _create_visualization(self, image, masks, results):
        """Создание визуализации с масками дефектов"""
        vis_image = image.copy()
        
        # Цвета для масок
        colors = [
            (0, 0, 255),    # Красный
            (0, 255, 0),    # Зеленый  
            (255, 0, 0),    # Синий
            (0, 255, 255),  # Желтый
            (255, 0, 255),  # Пурпурный
            (255, 255, 0),  # Голубой
        ]
        
        # Наложение масок
        for i, mask in enumerate(masks):
            color = colors[i % len(colors)]
            
            # Создание цветной маски
            colored_mask = np.zeros_like(vis_image)
            colored_mask[mask > 0] = color
            
            # Наложение с прозрачностью
            vis_image = cv2.addWeighted(vis_image, 0.7, colored_mask, 0.3, 0)
            
            # Контур
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis_image, contours, -1, color, 2)
            
            # Номер дефекта
            if contours:
                M = cv2.moments(contours[0])
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(vis_image, f"D{i+1}", (cx-10, cy+5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Информация о материале
        material_info = results["stages"]["material_classification"]["result"]
        info_text = f"Material: {material_info['material']} | Defects: {len(masks)}"
        cv2.putText(vis_image, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return vis_image


def main():
    parser = argparse.ArgumentParser(description="Автоматическая разметка дефектов: LLaVA + FastSAM + OpenCV")
    parser.add_argument("--image", required=True, help="Путь к изображению для анализа")
    parser.add_argument("--output", default="./output", help="Директория для сохранения результатов")
    
    args = parser.parse_args()
    
    # Проверка файла
    if not Path(args.image).exists():
        print(f"❌ Файл не найден: {args.image}")
        return
    
    # Создание pipeline
    pipeline = DefectAnalysisPipeline()
    
    # Запуск анализа
    results = pipeline.analyze_image(args.image, args.output)
    
    if results:
        print(f"\n📊 Итоговая статистика:")
        print(f"   🔬 Материал: {results['annotations']['material']['material']}")
        print(f"   🎯 Найдено дефектов: {len(results['annotations']['defects'])}")
        print(f"   ⏱️ Общее время: {sum(stage['duration'] for stage in results['stages'].values()):.2f} сек")


if __name__ == "__main__":
    main() 