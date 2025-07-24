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
    
    def __init__(self, model_path="liuhaotian/llava-v1.5-13b"):
        self.model_path = model_path
        
        # Доступные модели по уровню детализации
        self.available_models = {
            "detailed": "liuhaotian/llava-v1.5-13b",  # 13B - очень детальный анализ
            "standard": "liuhaotian/llava-v1.5-7b",   # 7B - стандартный
            "latest": "liuhaotian/llava-v1.6-vicuna-13b",  # Новейшая 13B версия
            "onevision": "lmms-lab/llava-onevision-qwen2-7b-ov"  # Специально для детального анализа
        }
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.context_len = None
        
        if LLAVA_AVAILABLE:
            self._load_model()
    
    def switch_model(self, model_type="detailed"):
        """Переключение на другую модель LLaVA для разного уровня детализации"""
        if model_type in self.available_models:
            print(f"🔄 Переключаемся на модель: {model_type}")
            self.model_path = self.available_models[model_type]
            self._load_model()
        else:
            print(f"❌ Неизвестный тип модели: {model_type}")
            print(f"✅ Доступные типы: {list(self.available_models.keys())}")
    
    def get_model_info(self):
        """Информация о текущей модели"""
        return {
            "current_model": self.model_path,
            "available_models": self.available_models,
            "model_loaded": self.model is not None
        }
    
    def _load_model(self):
        """Загрузка модели LLaVA"""
        print("🔄 Загружаем LLaVA модель...")
        
        # ФИКС для совместимости типов данных
        import torch
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        try:
            self.model_name = get_model_name_from_path(self.model_path)
            
            # Определяем размер модели для оптимизации
            is_large_model = "13b" in self.model_path.lower() or "34b" in self.model_path.lower()
            
            print(f"📊 Модель: {self.model_path}")
            print(f"📏 Большая модель: {is_large_model}")
            
            # Настройки в зависимости от устройства и размера модели
            if DEVICE == 'cuda':
                # CUDA - оптимизации для больших моделей
                if is_large_model:
                    print("🔧 Применяем оптимизации для большой модели...")
                    self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                        self.model_path, None, self.model_name, 
                        device_map="auto",
                        load_8bit=True,  # 8bit для экономии памяти на больших моделях
                        load_4bit=False,
                        torch_dtype=torch.float16
                    )
                else:
                    # Стандартная загрузка для 7B модели
                    self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                        self.model_path, None, self.model_name, 
                        device_map="auto",
                        load_8bit=False, load_4bit=False,
                        torch_dtype=torch.float16
                    )
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
        """Двухэтапный анализ дефектов: сначала найти объекты с координатами, потом анализировать дефекты"""
        if not LLAVA_AVAILABLE or self.model is None:
            # Заглушка
            print("⚠️ LLaVA недоступна, используем заглушку для анализа дефектов")
            return {
                "defects_found": True,
                "defect_types": ["scratch", "dent", "missing_part"],
                "defect_locations": ["center", "top-right", "edge"],
                "severity": "moderate",
                "completeness": "incomplete",
                "description": "Detected potential surface defects and structural issues",
                "bounding_boxes": [(0.2, 0.2, 0.8, 0.8)],  # x1, y1, x2, y2 (normalized)
                "prompt_points": [[320, 240], [400, 150], [200, 300]]
            }
        
        try:
            # Загрузка изображения
            image = Image.open(image_path).convert('RGB')
            
            # ЭТАП 1: Найти все объекты с точными координатами
            print("🔍 Этап 1: Поиск объектов и их локализация...")
            object_detection_result = self._detect_objects_with_coordinates(image, material_type)
            
            # ЭТАП 2: Детальный анализ каждого найденного объекта
            print("🔬 Этап 2: Детальный анализ найденных объектов...")
            defect_analysis_result = self._analyze_objects_for_defects(image, object_detection_result, material_type)
            
            # Комбинируем результаты
            combined_result = {
                **defect_analysis_result,
                "bounding_boxes": object_detection_result.get("bounding_boxes", []),
                "objects_found": object_detection_result.get("objects_found", [])
            }
            
            return combined_result
            
        except Exception as e:
            print(f"❌ Ошибка анализа дефектов: {e}")
            return {
                "defects_found": False,
                "defect_types": [],
                "defect_locations": [],
                "severity": "unknown",
                "description": "failed to analyze defects",
                "bounding_boxes": [],
                "prompt_points": []
            }
    
    def _detect_objects_with_coordinates(self, image, material_type):
        """ЭТАП 1: Поиск объектов с точными координатами bounding box"""
        
        # Промпт для поиска объектов с координатами
        detection_prompt = f"""OBJECT DETECTION TASK: Analyze this {material_type} image and identify ALL visible objects/components with their EXACT locations.

        For each object you see, provide:
        1. Object type (wire, cable, connector, screw, pin, contact, etc.)
        2. Bounding box coordinates in format: (x1, y1, x2, y2)
        3. Brief description of the object
        
        COORDINATE FORMAT:
        - Use normalized coordinates (0.0 to 1.0)
        - (x1, y1) = top-left corner
        - (x2, y2) = bottom-right corner
        - Example: wire at top-left would be (0.1, 0.1, 0.4, 0.3)
        
        FOCUS ON FINDING:
        - Individual wires and wire strands
        - Cables and cable bundles
        - Connectors, pins, terminals
        - Screws, bolts, fasteners
        - Electronic components
        - Any damaged or missing areas
        
        OUTPUT FORMAT:
        Object: wire_strand_1, Box: (0.2, 0.3, 0.25, 0.4), Description: thin copper wire
        Object: connector_pin, Box: (0.5, 0.1, 0.55, 0.15), Description: metal contact pin
        Object: missing_area, Box: (0.7, 0.6, 0.8, 0.7), Description: gap where component should be
        
        Be very precise with coordinates. Look for EVERYTHING, including tiny details."""
        
        response = self._get_llava_response(image, detection_prompt)
        
        # Парсинг координат объектов
        return self._parse_object_coordinates(response)
    
    def _analyze_objects_for_defects(self, image, detection_result, material_type):
        """ЭТАП 2: Детальный анализ каждого найденного объекта на предмет дефектов"""
        
        objects_info = "\n".join([f"- {obj['type']}: {obj['description']} at {obj['box']}" 
                                 for obj in detection_result.get("objects_found", [])])
        
        analysis_prompt = f"""DEFECT ANALYSIS: Based on the detected objects below, analyze each one for defects and issues.

        DETECTED OBJECTS:
        {objects_info}
        
        CRITICAL ANALYSIS FOR {material_type.upper()}:
        
        **WIRE & CABLE INSPECTION:**
        - Count individual wire strands in each cable
        - Check if any copper wires are missing from bundles
        - Look for exposed/protruding wire strands
        - Verify insulation integrity
        - Check wire routing and positioning
        
        **CONNECTOR ANALYSIS:**
        - Verify all pins/contacts are present
        - Check for bent or damaged pins
        - Look for corrosion or oxidation
        - Verify proper alignment and seating
        
        **STRUCTURAL INSPECTION:**
        - Identify missing components or fasteners
        - Check for cracks, breaks, or deformation
        - Look for incomplete assemblies
        - Verify proper component orientation
        
        **SURFACE EXAMINATION:**
        - Detect scratches, dents, wear patterns
        - Look for discoloration or staining
        - Check for coating or paint damage
        
        PROVIDE DETAILED ANALYSIS:
        1. Are there visible defects? (Yes/No)
        2. What specific defects do you see for each object?
        3. For wires: Are any strands missing or protruding?
        4. For connectors: Are all pins present and undamaged?
        5. Rate severity: minor/moderate/severe/critical
        6. Is the assembly complete or incomplete?
        7. Describe the overall condition and any missing parts
        
        Focus on identifying subtle issues like individual missing wire strands or slightly bent pins."""
        
        response = self._get_llava_response(image, analysis_prompt)
        
        # Парсинг результата анализа дефектов
        return self._parse_defect_analysis(response, image.size)
    
    def _parse_object_coordinates(self, response):
        """Парсинг координат объектов из ответа LLaVA"""
        import re
        
        objects_found = []
        bounding_boxes = []
        
        # Поиск объектов в формате: Object: name, Box: (x1, y1, x2, y2), Description: desc
        pattern = r'Object:\s*([^,]+),\s*Box:\s*\(([^)]+)\),\s*Description:\s*(.+)'
        matches = re.findall(pattern, response, re.IGNORECASE)
        
        for match in matches:
            object_type = match[0].strip()
            coords_str = match[1].strip()
            description = match[2].strip()
            
            try:
                # Парсинг координат
                coords = [float(x.strip()) for x in coords_str.split(',')]
                if len(coords) == 4:
                    x1, y1, x2, y2 = coords
                    # Валидация координат (должны быть в диапазоне 0-1)
                    if all(0 <= coord <= 1 for coord in coords) and x2 > x1 and y2 > y1:
                        objects_found.append({
                            "type": object_type,
                            "box": (x1, y1, x2, y2),
                            "description": description
                        })
                        bounding_boxes.append((x1, y1, x2, y2))
            except (ValueError, IndexError):
                continue
        
        # Если не найдены объекты в правильном формате, пытаемся найти любые координаты
        if not bounding_boxes:
            coord_pattern = r'\(([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\)'
            coord_matches = re.findall(coord_pattern, response)
            
            for match in coord_matches:
                try:
                    coords = [float(x) for x in match]
                    if all(0 <= coord <= 1 for coord in coords) and coords[2] > coords[0] and coords[3] > coords[1]:
                        bounding_boxes.append(tuple(coords))
                        objects_found.append({
                            "type": "detected_object",
                            "box": tuple(coords),
                            "description": "Object detected from coordinates"
                        })
                except ValueError:
                    continue
        
        return {
            "objects_found": objects_found,
            "bounding_boxes": bounding_boxes,
            "detection_response": response
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
        
        # Настройки генерации в зависимости от устройства и модели
        is_large_model = "13b" in self.model_path.lower() or "34b" in self.model_path.lower()
        
        if DEVICE == 'cuda':
            max_tokens = 1024 if is_large_model else 512  # Больше токенов для детального анализа
        else:
            max_tokens = 512 if is_large_model else 256
        
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
        
        # Определение наличия дефектов/проблем
        defects_found = any(word in response_lower for word in [
            "yes", "defect", "damage", "scratch", "crack", "dent", "stain", 
            "missing", "broken", "incomplete", "separated", "detached", "bent"
        ])
        
        # Расширенные типы дефектов и проблем (включая мелкие детали)
        defect_types = []
        defect_keywords = {
            # Поверхностные дефекты
            "scratch": ["scratch", "scratches", "scrape", "scraping"],
            "crack": ["crack", "cracks", "fracture", "split", "tear"],
            "dent": ["dent", "dents", "deformation", "depression"],
            "corrosion": ["rust", "corrosion", "oxidation", "rusted"],
            "stain": ["stain", "discoloration", "spot", "mark"],
            "wear": ["wear", "worn", "erosion", "abraded"],
            "chip": ["chip", "chips", "chipping", "flaking"],
            
            # Структурные проблемы
            "missing_part": ["missing", "absent", "lost", "hole where", "should be"],
            "broken_off": ["broken off", "detached", "separated", "fell off", "torn off"],
            "bent": ["bent", "twisted", "warped", "deformed", "curved"],
            "incomplete": ["incomplete", "unfinished", "partial", "half"],
            "loose": ["loose", "wobbling", "unstable", "not secure"],
            "gap": ["gap", "space", "opening", "separation"],
            "asymmetry": ["asymmetric", "uneven", "lopsided", "misaligned"],
            
            # Проблемы с проводами и кабелями (НОВОЕ!)
            "wire_missing": ["missing wire", "wire missing", "missing strand", "strand missing", "wire absent"],
            "wire_exposed": ["exposed wire", "wire sticking", "protruding wire", "wire out", "copper showing"],
            "wire_frayed": ["frayed wire", "frayed", "wire broken", "damaged wire", "torn wire"],
            "wire_misrouted": ["misrouted", "wrong position", "incorrect routing", "wire placement"],
            "insulation_damage": ["damaged insulation", "insulation broken", "bare copper", "exposed copper"],
            
            # Проблемы с соединениями и контактами
            "connector_issue": ["missing pin", "pin missing", "connector damage", "contact issue"],
            "solder_defect": ["cold joint", "solder crack", "poor solder", "excess solder"],
            "contact_corrosion": ["corroded contact", "contact corrosion", "oxidized contact"],
            "misalignment": ["misaligned", "not aligned", "crooked", "tilted"],
            
            # Мелкие компоненты
            "tiny_missing": ["tiny", "small missing", "micro component", "fastener missing"],
            "assembly_error": ["not seated", "not inserted", "wrong orientation", "upside down"],
            
            # Проблемы целостности
            "fracture": ["fractured", "broken", "split", "cracked through"],
            "edge_damage": ["edge damage", "torn edge", "damaged edge"],
            "dimensional": ["wrong size", "distorted", "out of shape"]
        }
        
        for defect_type, keywords in defect_keywords.items():
            if any(keyword in response_lower for keyword in keywords):
                defect_types.append(defect_type)
        
        # Расширенные локации дефектов
        defect_locations = []
        location_keywords = {
            "top-left": ["top-left", "top left", "upper left", "corner top-left"],
            "top-center": ["top-center", "top center", "upper center", "top"],
            "top-right": ["top-right", "top right", "upper right", "corner top-right"],
            "center-left": ["center-left", "center left", "middle left", "left"],
            "center": ["center", "middle", "central", "middle area"],
            "center-right": ["center-right", "center right", "middle right", "right"],
            "bottom-left": ["bottom-left", "bottom left", "lower left", "corner bottom-left"],
            "bottom-center": ["bottom-center", "bottom center", "lower center", "bottom"],
            "bottom-right": ["bottom-right", "bottom right", "lower right", "corner bottom-right"],
            "edge": ["edge", "rim", "border", "perimeter"],
            "corner": ["corner", "angle", "joint"],
            "multiple": ["multiple", "several", "various", "throughout"]
        }
        
        for location, keywords in location_keywords.items():
            if any(keyword in response_lower for keyword in keywords):
                defect_locations.append(location)
        
        # Определение серьезности (включая критический уровень)
        severity = "unknown"
        if any(word in response_lower for word in ["critical", "catastrophic", "complete failure", "totally broken"]):
            severity = "critical"
        elif any(word in response_lower for word in ["severe", "serious", "major", "significant", "extensive"]):
            severity = "severe"
        elif any(word in response_lower for word in ["moderate", "medium", "noticeable", "considerable"]):
            severity = "moderate"
        elif any(word in response_lower for word in ["minor", "small", "slight", "light", "superficial"]):
            severity = "minor"
        
        # Определение полноты объекта
        completeness = "unknown"
        if any(word in response_lower for word in ["complete", "intact", "whole", "all parts present"]):
            completeness = "complete"
        elif any(word in response_lower for word in ["incomplete", "missing parts", "partial", "broken off"]):
            completeness = "incomplete"
        
        # Генерация точек-подсказок для SAM2 на основе локаций
        prompt_points = self._generate_prompt_points(defect_locations, image_size)
        
        return {
            "defects_found": defects_found,
            "defect_types": defect_types,
            "defect_locations": defect_locations,
            "severity": severity,
            "completeness": completeness,
            "description": response,
            "prompt_points": prompt_points
        }
    
    def _generate_prompt_points(self, locations, image_size):
        """Генерация точек-подсказок для SAM2 на основе текстовых локаций"""
        width, height = image_size
        prompt_points = []
        
        # Расширенная карта локаций в координаты (относительные)
        location_map = {
            "top-left": (0.25, 0.25),
            "top-center": (0.5, 0.25),
            "top-right": (0.75, 0.25),
            "center-left": (0.25, 0.5),
            "center": (0.5, 0.5),
            "center-right": (0.75, 0.5),
            "bottom-left": (0.25, 0.75),
            "bottom-center": (0.5, 0.75),
            "bottom-right": (0.75, 0.75),
            "edge": (0.5, 0.1),  # Верхний край
            "corner": (0.9, 0.1),  # Правый верхний угол
        }
        
        # Специальная обработка для множественных локаций
        if "multiple" in locations:
            # Добавляем несколько точек для множественных дефектов
            multiple_points = [
                (0.3, 0.3), (0.7, 0.3), (0.3, 0.7), (0.7, 0.7), (0.5, 0.5)
            ]
            for rel_x, rel_y in multiple_points:
                x = int(rel_x * width)
                y = int(rel_y * height)
                prompt_points.append([x, y])
        
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
    
    def _generate_points_from_boxes(self, bounding_boxes, image_size):
        """Генерация точек из bounding boxes для дополнительных промптов"""
        width, height = image_size
        points = []
        
        for bbox in bounding_boxes:
            x1_norm, y1_norm, x2_norm, y2_norm = bbox
            
            # Центральная точка bounding box
            center_x = int((x1_norm + x2_norm) / 2 * width)
            center_y = int((y1_norm + y2_norm) / 2 * height)
            points.append([center_x, center_y])
            
            # Также добавляем несколько точек внутри box для лучшего покрытия
            box_width = (x2_norm - x1_norm) * width
            box_height = (y2_norm - y1_norm) * height
            
            # Добавляем дополнительные точки только для больших объектов
            if box_width > 50 and box_height > 50:
                # Четыре точки в квадрантах
                quad_points = [
                    [int((x1_norm + (x2_norm - x1_norm) * 0.3) * width), 
                     int((y1_norm + (y2_norm - y1_norm) * 0.3) * height)],
                    [int((x1_norm + (x2_norm - x1_norm) * 0.7) * width), 
                     int((y1_norm + (y2_norm - y1_norm) * 0.3) * height)],
                    [int((x1_norm + (x2_norm - x1_norm) * 0.3) * width), 
                     int((y1_norm + (y2_norm - y1_norm) * 0.7) * height)],
                    [int((x1_norm + (x2_norm - x1_norm) * 0.7) * width), 
                     int((y1_norm + (y2_norm - y1_norm) * 0.7) * height)]
                ]
                points.extend(quad_points)
        
        return points


class SAM2DefectSegmenter:
    """SAM2 для точной сегментации дефектов на основе подсказок от LLaVA"""
    
    def __init__(self):
        self.predictor = None
        self.mask_generator = None
        
        if SAM2_AVAILABLE:
            self._load_model()
    
    def _load_model(self):
        """Загрузка модели SAM2"""
        print("🔄 Загружаем SAM2 модель...")
        try:
            import os
            import urllib.request
            
            # Путь для сохранения модели
            models_dir = Path("./models")
            models_dir.mkdir(exist_ok=True)
            
            config_name = "sam2_hiera_l.yaml"
            checkpoint_path = models_dir / "sam2_hiera_large.pt"
            
            # Скачиваем модель если её нет
            if not checkpoint_path.exists():
                print("📥 Скачиваем SAM2 модель (это займет несколько минут)...")
                model_url = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
                
                try:
                    urllib.request.urlretrieve(model_url, checkpoint_path)
                    print(f"✅ SAM2 модель скачана: {checkpoint_path}")
                except Exception as e:
                    print(f"❌ Ошибка скачивания SAM2: {e}")
                    print("🔄 Пробуем использовать torch.hub...")
                    # Fallback: пробуем через torch.hub
                    import torch
                    checkpoint_path = torch.hub.load_state_dict_from_url(
                        "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
                        model_dir=str(models_dir)
                    )
            
            print(f"📥 Загружаем SAM2: {config_name} + {checkpoint_path}")
            
            # Создание модели SAM2
            sam2_model = build_sam2(config_name, str(checkpoint_path), device=DEVICE)
            
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
            print("🔄 Пробуем альтернативный способ загрузки...")
            
            try:
                # Альтернативный способ: используем меньшую модель или упрощенную загрузку
                print("📥 Загружаем упрощенную версию SAM2...")
                sam2_model = build_sam2("sam2_hiera_s.yaml", "sam2_hiera_small.pt", device=DEVICE)
                
                # Создание предиктора для точечных подсказок
                self.predictor = SAM2ImagePredictor(sam2_model)
                
                # Создание генератора масок для автоматической сегментации
                self.mask_generator = SAM2AutomaticMaskGenerator(
                    model=sam2_model,
                    points_per_side=16,  # Меньше точек для упрощения
                    pred_iou_thresh=0.7,
                    stability_score_thresh=0.85,
                    crop_n_layers=1,
                    crop_n_points_downscale_factor=2,
                    min_mask_region_area=100
                )
                
                print("✅ SAM2 упрощенная модель загружена успешно")
            except Exception as e2:
                print(f"❌ Окончательная ошибка загрузки SAM2: {e2}")
                print("⚠️ SAM2 будет недоступен, используется простая сегментация")
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
            
            # ПРИОРИТЕТ 1: Используем bounding boxes от LLaVA (самые точные)
            if defect_analysis.get("bounding_boxes"):
                print(f"📦 Используем {len(defect_analysis['bounding_boxes'])} bounding boxes от LLaVA")
                
                height, width = image_rgb.shape[:2]
                
                for i, bbox in enumerate(defect_analysis["bounding_boxes"]):
                    try:
                        # Конвертация нормализованных координат в пиксели
                        x1_norm, y1_norm, x2_norm, y2_norm = bbox
                        x1 = int(x1_norm * width)
                        y1 = int(y1_norm * height)
                        x2 = int(x2_norm * width)
                        y2 = int(y2_norm * height)
                        
                        # Валидация координат
                        x1, x2 = max(0, min(x1, x2)), min(width-1, max(x1, x2))
                        y1, y2 = max(0, min(y1, y2)), min(height-1, max(y1, y2))
                        
                        if x2 > x1 and y2 > y1:
                            # Предсказание маски с использованием bounding box
                            input_box = np.array([x1, y1, x2, y2])
                            
                            masks, scores, logits = self.predictor.predict(
                                box=input_box[None, :],  # Box prompt
                                multimask_output=False
                            )
                            
                            # Получаем маску
                            mask = masks[0]
                            mask_uint8 = (mask * 255).astype(np.uint8)
                            all_masks.append(mask_uint8)
                            
                            print(f"   ✅ Box {i+1}: точность {scores[0]:.3f}, область ({x1},{y1})-({x2},{y2})")
                        
                    except Exception as e:
                        print(f"   ❌ Ошибка обработки box {i+1}: {e}")
                        continue
            
            # ПРИОРИТЕТ 2: Если нет bounding boxes, используем точки
            elif defect_analysis.get("prompt_points"):
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
                        
                        print(f"   ✅ Точка {i+1}: точность {scores[best_mask_idx]:.3f}")
                        
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
    
    def __init__(self, model_type="detailed"):
        self.analyzer = MaterialAndDefectAnalyzer()
        if model_type != "detailed":
            self.analyzer.switch_model(model_type)
        self.segmenter = SAM2DefectSegmenter()
        self.postprocessor = OpenCVPostProcessor()
    
    def switch_model(self, model_type):
        """Переключение модели LLaVA для разного уровня детализации"""
        self.analyzer.switch_model(model_type)
    
    def get_model_info(self):
        """Информация о текущей модели"""
        return self.analyzer.get_model_info()
    
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
            object_info = None
            
            # Привязка к найденным объектам если они есть
            if defect_analysis.get("objects_found") and i < len(defect_analysis["objects_found"]):
                object_info = defect_analysis["objects_found"][i]
                defect_type = object_info.get("type", "detected_object")
            elif i < len(defect_analysis.get("defect_types", [])):
                defect_type = defect_analysis["defect_types"][i]
            
            defect_annotation = {
                "id": i + 1,
                "category": defect_type,
                "bbox": [x, y, w, h],
                "area": int(area),
                "segmentation": [polygon],
                "confidence": 0.90,  # Выше благодаря LLaVA координатам + SAM2
                "severity": defect_analysis.get("severity", "unknown"),
                "completeness": defect_analysis.get("completeness", "unknown")
            }
            
            # Добавляем информацию об объекте если есть
            if object_info:
                defect_annotation.update({
                    "object_description": object_info.get("description", ""),
                    "llava_bbox": object_info.get("box", None),
                    "detection_method": "llava_coordinates"
                })
            else:
                defect_annotation["detection_method"] = "automatic_segmentation"
            
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
        
        info_text = f"Material: {material_info['material']} | Issues: {len(masks)}"
        severity_text = f"Severity: {defect_info.get('severity', 'unknown')}"
        completeness_text = f"Completeness: {defect_info.get('completeness', 'unknown')}"
        
        # Цвет текста в зависимости от серьезности
        text_color = (255, 255, 255)  # Белый по умолчанию
        if defect_info.get('severity') == 'critical':
            text_color = (0, 0, 255)  # Красный
        elif defect_info.get('severity') == 'severe':
            text_color = (0, 100, 255)  # Оранжевый
        
        cv2.putText(vis_image, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
        cv2.putText(vis_image, severity_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        cv2.putText(vis_image, completeness_text, (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        
        return vis_image


def main():
    parser = argparse.ArgumentParser(description="Автоматическая разметка дефектов: LLaVA + SAM2 + OpenCV")
    parser.add_argument("--image", required=True, help="Путь к изображению для анализа")
    parser.add_argument("--output", default="./output", help="Директория для сохранения результатов")
    parser.add_argument("--model", default="detailed", 
                       choices=["detailed", "standard", "latest", "onevision"],
                       help="Тип модели LLaVA: detailed(13B), standard(7B), latest(13B-v1.6), onevision(7B-специальная)")
    
    args = parser.parse_args()
    
    # Проверка файла
    if not Path(args.image).exists():
        print(f"❌ Файл не найден: {args.image}")
        return
    
    # Создание pipeline с выбранной моделью
    print(f"🤖 Используем модель: {args.model}")
    pipeline = DefectAnalysisPipeline(model_type=args.model)
    
    # Показываем информацию о модели
    model_info = pipeline.get_model_info()
    print(f"📊 Модель загружена: {model_info['current_model']}")
    
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