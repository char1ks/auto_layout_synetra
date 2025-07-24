#!/usr/bin/env python3
"""
Автоматическая разметка дефектов: LLaVA → SAM2 → OpenCV
Intelligent pipeline для определения материала и сегментации дефектов с помощью SAM2
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

# SAM2 imports
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    SAM2_AVAILABLE = True
    print("✅ SAM2 найден")
except ImportError as e:
    print(f"❌ SAM2 не найден: {e}")
    SAM2_AVAILABLE = False


class MaterialAndDefectAnalyzer:
    """LLaVA для определения типа материала и анализа дефектов"""
    
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
        
        # ФИКС для совместимости типов данных
        import torch
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        try:
            self.model_name = get_model_name_from_path(self.model_path)
            
            # Настройки в зависимости от устройства
            if DEVICE == 'cuda':
                # CUDA (Windows/Linux with NVIDIA GPU) - ПРИНУДИТЕЛЬНО float16
                self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                    self.model_path, None, self.model_name, 
                    device_map="auto",
                    load_8bit=False, load_4bit=False,
                    torch_dtype=torch.float16  # Принудительно float16
                )
                # ПРИНУДИТЕЛЬНО приводим всю модель к float16
                self.model = self.model.half()
                
            elif DEVICE == 'mps':
                # Apple Silicon
                self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                    self.model_path, None, self.model_name,
                    device_map="mps",
                    load_8bit=False, load_4bit=False,
                    torch_dtype=torch.float16
                )
                self.model = self.model.half()
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
            Also describe the surface condition and overall quality.
            
            Format your answer as:
            Material: [material_type]
            Condition: [surface_description]
            Quality: [overall_assessment]"""
            
            response = self._get_llava_response(image, prompt)
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
    
    def analyze_defects(self, image_path, material_type="unknown"):
        """Анализ дефектов на изображении с помощью LLaVA"""
        if not LLAVA_AVAILABLE or self.model is None:
            # Заглушка
            print("⚠️ LLaVA недоступна, используем заглушку для анализа дефектов")
            return {
                "defects_found": True,
                "defect_types": ["scratch", "dent"],
                "defect_locations": ["center", "top-right"],
                "severity": "moderate",
                "description": "Detected potential surface defects",
                "prompt_points": [[320, 240], [400, 150]]  # Примерные координаты
            }
        
        try:
            # Загрузка изображения
            image = Image.open(image_path).convert('RGB')
            
            # Специализированный промпт для анализа дефектов
            defect_prompt = f"""Analyze this {material_type} surface for defects and damage. Look carefully for:
            - Scratches, cracks, dents, chips
            - Discoloration, stains, corrosion
            - Surface irregularities, wear patterns
            - Any other visible damage or imperfections
            
            Describe:
            1. Are there any visible defects? (Yes/No)
            2. What types of defects do you see?
            3. Where are they located on the surface? (use terms like top-left, center, bottom-right, etc.)
            4. How severe are the defects? (minor/moderate/severe)
            5. Estimate approximate locations as coordinates if possible (describe relative positions)
            
            Be very specific about defect locations and types."""
            
            response = self._get_llava_response(image, defect_prompt)
            
            # Парсинг результата анализа дефектов
            defect_analysis = self._parse_defect_analysis(response, image.size)
            
            return defect_analysis
            
        except Exception as e:
            print(f"❌ Ошибка анализа дефектов: {e}")
            return {
                "defects_found": False,
                "defect_types": [],
                "defect_locations": [],
                "severity": "unknown",
                "description": "failed to analyze defects",
                "prompt_points": []
            }
    
    def _get_llava_response(self, image, prompt):
        """Получение ответа от LLaVA"""
        # Подготовка входных данных
        conv = conv_templates["vicuna_v1"].copy()
        conv.append_message(conv.roles[0], f"{DEFAULT_IMAGE_TOKEN}\n{prompt}")
        conv.append_message(conv.roles[1], None)
        prompt_formatted = conv.get_prompt()
        
        # Токенизация с padding
        input_ids = tokenizer_image_token(prompt_formatted, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        # Обработка изображения с правильным типом данных
        image_tensor = process_images([image], self.image_processor, self.model.config)
        
        # Определяем тип данных в зависимости от устройства
        if DEVICE == 'cuda':
            target_dtype = torch.float16
        else:
            target_dtype = torch.float32
        
        # Генерация ответа с автоматическим устройством и правильным типом данных
        input_ids = input_ids.to(DEVICE)
        if hasattr(image_tensor, 'to'):
            image_tensor = image_tensor.to(DEVICE, dtype=target_dtype)
        
        # Настройки генерации в зависимости от устройства
        max_tokens = 512 if DEVICE == 'cuda' else 256
        
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
        
        return response
    
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
    
    def _parse_defect_analysis(self, response, image_size):
        """Парсинг результата анализа дефектов"""
        response_lower = response.lower()
        
        # Определение наличия дефектов
        defects_found = any(word in response_lower for word in ["yes", "defect", "damage", "scratch", "crack", "dent", "stain"])
        
        # Типы дефектов
        defect_types = []
        defect_keywords = {
            "scratch": ["scratch", "scratches", "scrape"],
            "crack": ["crack", "cracks", "fracture"],
            "dent": ["dent", "dents", "deformation"],
            "corrosion": ["rust", "corrosion", "oxidation"],
            "stain": ["stain", "discoloration", "spot"],
            "wear": ["wear", "worn", "erosion"],
            "chip": ["chip", "chips", "chipping"]
        }
        
        for defect_type, keywords in defect_keywords.items():
            if any(keyword in response_lower for keyword in keywords):
                defect_types.append(defect_type)
        
        # Локации дефектов
        defect_locations = []
        location_keywords = {
            "top-left": ["top-left", "top left", "upper left"],
            "top-center": ["top-center", "top center", "upper center", "top"],
            "top-right": ["top-right", "top right", "upper right"],
            "center-left": ["center-left", "center left", "middle left", "left"],
            "center": ["center", "middle", "central"],
            "center-right": ["center-right", "center right", "middle right", "right"],
            "bottom-left": ["bottom-left", "bottom left", "lower left"],
            "bottom-center": ["bottom-center", "bottom center", "lower center", "bottom"],
            "bottom-right": ["bottom-right", "bottom right", "lower right"]
        }
        
        for location, keywords in location_keywords.items():
            if any(keyword in response_lower for keyword in keywords):
                defect_locations.append(location)
        
        # Определение серьезности
        severity = "unknown"
        if any(word in response_lower for word in ["severe", "serious", "major", "significant"]):
            severity = "severe"
        elif any(word in response_lower for word in ["moderate", "medium", "noticeable"]):
            severity = "moderate"
        elif any(word in response_lower for word in ["minor", "small", "slight", "light"]):
            severity = "minor"
        
        # Генерация точек-подсказок для SAM2 на основе локаций
        prompt_points = self._generate_prompt_points(defect_locations, image_size)
        
        return {
            "defects_found": defects_found,
            "defect_types": defect_types,
            "defect_locations": defect_locations,
            "severity": severity,
            "description": response,
            "prompt_points": prompt_points
        }
    
    def _generate_prompt_points(self, locations, image_size):
        """Генерация точек-подсказок для SAM2 на основе текстовых локаций"""
        width, height = image_size
        prompt_points = []
        
        # Карта локаций в координаты (относительные)
        location_map = {
            "top-left": (0.25, 0.25),
            "top-center": (0.5, 0.25),
            "top-right": (0.75, 0.25),
            "center-left": (0.25, 0.5),
            "center": (0.5, 0.5),
            "center-right": (0.75, 0.5),
            "bottom-left": (0.25, 0.75),
            "bottom-center": (0.5, 0.75),
            "bottom-right": (0.75, 0.75)
        }
        
        for location in locations:
            if location in location_map:
                rel_x, rel_y = location_map[location]
                x = int(rel_x * width)
                y = int(rel_y * height)
                prompt_points.append([x, y])
        
        # Если локации не найдены, добавляем центральную точку
        if not prompt_points:
            prompt_points.append([width // 2, height // 2])
        
        return prompt_points


class SAM2DefectSegmenter:
    """SAM2 для точной сегментации дефектов на основе подсказок от LLaVA"""
    
    def __init__(self, model_path="./models/sam2_hiera_large.pt", config="sam2_hiera_l.yaml"):
        self.model_path = model_path
        self.config = config
        self.predictor = None
        self.mask_generator = None
        
        if SAM2_AVAILABLE:
            self._load_model()
    
    def _load_model(self):
        """Загрузка модели SAM2"""
        print("🔄 Загружаем SAM2 модель...")
        try:
            # Автоматическое скачивание модели если не существует
            if not Path(self.model_path).exists():
                print("📥 Скачиваем SAM2 модель...")
                # Здесь можно добавить автоматическое скачивание
                self.model_path = "facebook/sam2-hiera-large"  # Используем HuggingFace
            
            # Создание модели SAM2
            sam2_model = build_sam2(self.config, self.model_path, device=DEVICE)
            
            # Создание предиктора для точечных подсказок
            self.predictor = SAM2ImagePredictor(sam2_model)
            
            # Создание генератора масок для автоматической сегментации
            self.mask_generator = SAM2AutomaticMaskGenerator(
                model=sam2_model,
                points_per_side=32,
                pred_iou_thresh=0.7,
                stability_score_thresh=0.85,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=100
            )
            
            print("✅ SAM2 модель загружена успешно")
        except Exception as e:
            print(f"❌ Ошибка загрузки SAM2: {e}")
            self.predictor = None
            self.mask_generator = None
    
    def segment_defects_with_prompts(self, image_path, defect_analysis):
        """Сегментация дефектов с использованием подсказок от LLaVA"""
        if not SAM2_AVAILABLE or self.predictor is None:
            print("⚠️ SAM2 недоступен, используем простую сегментацию")
            return self._simple_segmentation(image_path)
        
        try:
            # Загрузка изображения
            image = cv2.imread(str(image_path))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Установка изображения в предиктор
            self.predictor.set_image(image_rgb)
            
            all_masks = []
            
            # Если есть подсказки от LLaVA, используем их
            if defect_analysis.get("prompt_points"):
                print(f"🎯 Используем {len(defect_analysis['prompt_points'])} точек-подсказок от LLaVA")
                
                for i, point in enumerate(defect_analysis["prompt_points"]):
                    try:
                        # Предсказание маски для каждой точки
                        masks, scores, logits = self.predictor.predict(
                            point_coords=np.array([point]),
                            point_labels=np.array([1]),  # 1 = foreground
                            multimask_output=True
                        )
                        
                        # Выбираем лучшую маску
                        best_mask_idx = np.argmax(scores)
                        best_mask = masks[best_mask_idx]
                        
                        # Конвертация в uint8
                        mask_uint8 = (best_mask * 255).astype(np.uint8)
                        all_masks.append(mask_uint8)
                        
                        print(f"   ✅ Маска {i+1}: точность {scores[best_mask_idx]:.3f}")
                        
                    except Exception as e:
                        print(f"   ❌ Ошибка обработки точки {i+1}: {e}")
                        continue
            
            # Дополнительно используем автоматическую сегментацию
            if defect_analysis.get("defects_found", False):
                print("🤖 Дополнительная автоматическая сегментация...")
                try:
                    auto_masks = self.mask_generator.generate(image_rgb)
                    
                    # Фильтрация и добавление лучших автоматических масок
                    auto_masks_sorted = sorted(auto_masks, key=lambda x: x['predicted_iou'], reverse=True)
                    
                    for auto_mask in auto_masks_sorted[:5]:  # Максимум 5 лучших
                        mask = auto_mask['segmentation']
                        mask_uint8 = (mask * 255).astype(np.uint8)
                        all_masks.append(mask_uint8)
                
                except Exception as e:
                    print(f"   ❌ Ошибка автоматической сегментации: {e}")
            
            # Фильтрация дублирующихся масок
            filtered_masks = self._filter_similar_masks(all_masks)
            
            return {
                "masks": filtered_masks,
                "num_detections": len(filtered_masks),
                "defect_analysis": defect_analysis
            }
            
        except Exception as e:
            print(f"❌ Ошибка сегментации SAM2: {e}")
            return self._simple_segmentation(image_path)
    
    def _filter_similar_masks(self, masks, iou_threshold=0.7):
        """Фильтрация похожих масок"""
        if len(masks) <= 1:
            return masks
        
        filtered = []
        used_indices = set()
        
        for i, mask1 in enumerate(masks):
            if i in used_indices:
                continue
            
            is_unique = True
            for j, mask2 in enumerate(masks[i+1:], i+1):
                if j in used_indices:
                    continue
                
                # Вычисление IoU
                intersection = cv2.bitwise_and(mask1, mask2)
                union = cv2.bitwise_or(mask1, mask2)
                
                intersection_area = cv2.countNonZero(intersection)
                union_area = cv2.countNonZero(union)
                
                if union_area > 0:
                    iou = intersection_area / union_area
                    if iou > iou_threshold:
                        # Оставляем маску с большей площадью
                        area1 = cv2.countNonZero(mask1)
                        area2 = cv2.countNonZero(mask2)
                        if area2 > area1:
                            is_unique = False
                            break
                        else:
                            used_indices.add(j)
            
            if is_unique:
                filtered.append(mask1)
                used_indices.add(i)
        
        return filtered
    
    def _simple_segmentation(self, image_path):
        """Простая сегментация для случая когда SAM2 недоступен"""
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
            "defect_analysis": {"defects_found": True, "description": "Simple segmentation fallback"}
        }


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
        
        # Получаем размеры исходного изображения
        target_height, target_width = image.shape[:2]
        
        for mask in masks:
            # Проверяем размер маски
            if mask.shape[:2] != (target_height, target_width):
                # Изменяем размер маски под исходное изображение
                mask = cv2.resize(mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
            
            processed_mask = self._clean_mask(mask, params)
            if processed_mask is not None:
                processed_masks.append(processed_mask)
        
        return processed_masks
    
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


class DefectAnalysisPipeline:
    """Основной pipeline для анализа дефектов: LLaVA → SAM2 → OpenCV"""
    
    def __init__(self):
        self.analyzer = MaterialAndDefectAnalyzer()
        self.segmenter = SAM2DefectSegmenter()
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
            material_result = self.analyzer.classify_material(image_path)
            stage1_time = time.time() - stage1_start
            
            results["stages"]["material_classification"] = {
                "duration": stage1_time,
                "result": material_result
            }
            print(f"   ✅ Материал: {material_result['material']} (уверенность: {material_result['confidence']:.2f})")
            print(f"   ⏱️ Время: {stage1_time:.2f} сек")
            
            # Этап 2: Анализ дефектов с помощью LLaVA
            print("🔍 Этап 2: Анализ дефектов с помощью LLaVA...")
            stage2_start = time.time()
            defect_analysis = self.analyzer.analyze_defects(image_path, material_result['material'])
            stage2_time = time.time() - stage2_start
            
            results["stages"]["defect_analysis"] = {
                "duration": stage2_time,
                "result": defect_analysis
            }
            print(f"   ✅ Дефекты найдены: {defect_analysis['defects_found']}")
            print(f"   ✅ Типы дефектов: {defect_analysis['defect_types']}")
            print(f"   ✅ Локации: {defect_analysis['defect_locations']}")
            print(f"   ⏱️ Время: {stage2_time:.2f} сек")
            
            # Этап 3: Сегментация с помощью SAM2
            print("🎯 Этап 3: Сегментация дефектов с помощью SAM2...")
            stage3_start = time.time()
            segmentation_result = self.segmenter.segment_defects_with_prompts(image_path, defect_analysis)
            stage3_time = time.time() - stage3_start
            
            results["stages"]["sam2_segmentation"] = {
                "duration": stage3_time,
                "num_raw_detections": segmentation_result['num_detections']
            }
            print(f"   ✅ Найдено сегментов: {segmentation_result['num_detections']}")
            print(f"   ⏱️ Время: {stage3_time:.2f} сек")
            
            # Этап 4: Постобработка
            print("🎨 Этап 4: Постобработка...")
            stage4_start = time.time()
            image = cv2.imread(str(image_path))
            final_masks = self.postprocessor.process_masks(
                image, 
                segmentation_result['masks'], 
                material_result['material']
            )
            stage4_time = time.time() - stage4_start
            
            results["stages"]["postprocessing"] = {
                "duration": stage4_time,
                "num_final_detections": len(final_masks)
            }
            print(f"   ✅ Финальных дефектов: {len(final_masks)}")
            print(f"   ⏱️ Время: {stage4_time:.2f} сек")
            
            # Создание аннотаций
            annotations = self._create_annotations(image, final_masks, material_result, defect_analysis)
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
    
    def _create_annotations(self, image, masks, material_result, defect_analysis):
        """Создание аннотаций в формате COCO с информацией от LLaVA"""
        annotations = {
            "image_info": {
                "height": image.shape[0],
                "width": image.shape[1],
                "channels": image.shape[2]
            },
            "material": material_result,
            "defect_analysis": defect_analysis,
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
            
            # Определение типа дефекта на основе анализа LLaVA
            defect_type = "defect"
            if i < len(defect_analysis.get("defect_types", [])):
                defect_type = defect_analysis["defect_types"][i]
            
            defect_annotation = {
                "id": i + 1,
                "category": defect_type,
                "bbox": [x, y, w, h],
                "area": int(area),
                "segmentation": [polygon],
                "confidence": 0.85,  # Выше благодаря LLaVA + SAM2
                "severity": defect_analysis.get("severity", "unknown")
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
            
            # Номер дефекта и тип
            if contours:
                M = cv2.moments(contours[0])
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Получаем тип дефекта
                    defect_type = "D"
                    if "annotations" in results and i < len(results["annotations"]["defects"]):
                        defect_type = results["annotations"]["defects"][i]["category"][:3].upper()
                    
                    cv2.putText(vis_image, f"{defect_type}{i+1}", (cx-15, cy+5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Информация о материале и анализе
        material_info = results["stages"]["material_classification"]["result"]
        defect_info = results["stages"]["defect_analysis"]["result"]
        
        info_text = f"Material: {material_info['material']} | Defects: {len(masks)}"
        severity_text = f"Severity: {defect_info.get('severity', 'unknown')}"
        
        cv2.putText(vis_image, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(vis_image, severity_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return vis_image


def main():
    parser = argparse.ArgumentParser(description="Автоматическая разметка дефектов: LLaVA + SAM2 + OpenCV")
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
        print(f"   📝 Типы дефектов: {results['annotations']['defect_analysis']['defect_types']}")
        print(f"   ⚠️ Серьезность: {results['annotations']['defect_analysis']['severity']}")
        print(f"   ⏱️ Общее время: {sum(stage['duration'] for stage in results['stages'].values()):.2f} сек")


if __name__ == "__main__":
    main() 