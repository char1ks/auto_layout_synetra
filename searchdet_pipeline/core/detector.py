#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Автономный SearchDetDetector для модульного пайплайна.
"""

import os
import sys
import cv2
import time
import numpy as np
from pathlib import Path
from PIL import Image

# Добавляем searchdet-main в path для импорта
sys.path.append('./searchdet-main')
try:
    from mask_withsearch import initialize_models as init_searchdet
    SEARCHDET_AVAILABLE = True
except Exception as e:
    print(f"⚠️ SearchDet недоступен: {e}")
    SEARCHDET_AVAILABLE = False

# Импорты модульных компонентов
from .mask_generation import MaskGenerator
from .filtering import MaskFilter  
from .embeddings import EmbeddingExtractor
from .scoring import ScoreCalculator


class SearchDetDetector:
    """Автономный детектор для модульного пайплайна."""
    
    def __init__(self, **kwargs):
        if not SEARCHDET_AVAILABLE:
            raise RuntimeError("SearchDet не найден")

        # Устанавливаем переменную окружения для оптимального feature map
        os.environ['SEARCHDET_FEAT_SHORT_SIDE'] = '384'
        print("🔧 Установлено SEARCHDET_FEAT_SHORT_SIDE=384")

        self.searchdet_resnet, self.searchdet_layer, self.searchdet_transform, self.searchdet_sam = init_searchdet()
        
        # Переопределяем трансформацию
        import torchvision.transforms as transforms
        feat_short_side = int(os.getenv('SEARCHDET_FEAT_SHORT_SIDE', '384'))
        self.searchdet_transform = transforms.Compose([
            transforms.Resize(feat_short_side),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.searchdet_layer = 'layer3'
        print("🔧 Принудительно установлен layer3")
        
        # Сохраняем параметры с дефолтами
        self.mask_backend = kwargs.get('mask_backend', 'fastsam')
        
        # Инициализируем компоненты
        self.mask_generator = MaskGenerator(self)
        self.mask_filter = MaskFilter(self)
        self.embedding_extractor = EmbeddingExtractor(self)
        self.score_calculator = ScoreCalculator(self)

        print("✅ SearchDetDetector инициализирован (автономная модульная версия)")

    def find_present_elements(self, image_path, positive_dir, negative_dir=None):
        """Основная функция поиска объектов."""
        print(f"🔍 Модульный анализ: {image_path}" + "="*60)
        print("🔄 ДЕТАЛЬНАЯ ПОСЛЕДОВАТЕЛЬНОСТЬ ВЫПОЛНЕНИЯ МОДУЛЬНОГО PIPELINE:")
        print("=" * 80)
        print("8️⃣ searchdet_pipeline/core/detector.py → find_present_elements()")
        
        t_total = time.time()
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Не удалось загрузить изображение: {image_path}")
        image_np = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 0) загрузка примеров
        print("9️⃣ Шаг 1: _load_example_images() - загрузка positive/negative примеров")
        pos_imgs = self._load_example_images(positive_dir)
        neg_imgs = self._load_example_images(negative_dir) if negative_dir else []
        if len(pos_imgs) == 0:
            print("   ❌ Нет положительных примеров — прекращаем.")
            return {"found_elements": [], "masks": []}
        print(f"   📁 Positive: {len(pos_imgs)}	📁 Negative: {len(neg_imgs)}")

        # 1) генерация масок
        print("🔟 Шаг 2: MaskGenerator.generate() - генерация масок через SAM/FastSAM")
        masks = self.mask_generator.generate(image_np)
        
        # 2) фильтрация масок
        print("1️⃣1️⃣ Шаг 3-7: MaskFilter.apply_all_filters() - все фильтры масок")
        masks = self.mask_filter.apply_all_filters(masks, image_np)
        
        if not masks:
            print("   ❌ Нет валидных масок после фильтров.")
            return {"found_elements": [], "masks": []}

        # 3) извлечение эмбеддингов
        print("1️⃣2️⃣ Шаг 8: EmbeddingExtractor.extract_mask_embeddings() - эмбеддинги масок")
        mask_vecs, idx_map = self.embedding_extractor.extract_mask_embeddings(image_np, masks)
        if mask_vecs.shape[0] == 0:
            print("   ❌ Не удалось получить эмбеддинги масок.")
            return {"found_elements": [], "masks": []}
        print(f"   📊 Масок с валидными векторами: {mask_vecs.shape[0]}")
        
        print("1️⃣3️⃣ Шаг 9: EmbeddingExtractor.build_queries() - эмбеддинги примеров")
        q_pos, q_neg = self.embedding_extractor.build_queries(pos_imgs, neg_imgs)

        # 4) скоринг и принятие решений
        print("1️⃣4️⃣ Шаг 10: ScoreCalculator.score_and_decide() - скоринг и принятие решений")
        print("🔍 ЭТАП 3: Сопоставление с positive/negative...")
        
        accepted_indices = self.score_calculator.score_and_decide(mask_vecs, q_pos, q_neg)
        
        # 5) формирование результата
        found = []
        result_masks = []
        for i in accepted_indices:
            original_idx = idx_map[i]
            mask_dict = masks[original_idx]
            found.append({
                "mask": mask_dict,
                "confidence": 0.8,  # заглушка
                "bbox": mask_dict.get("bbox", [0, 0, 0, 0])
            })
            result_masks.append(mask_dict)

        print(f"🎯 Принято масок: {len(found)} (после правил и NMS)")
        total_time = time.time() - t_total
        print(f"⏱️ Общее время: {total_time:.2f} сек")
        print("💾 Результаты сохранены в: output")
        
        return {"found_elements": found, "masks": result_masks}

    def _load_example_images(self, dir_path):
        """Загружает изображения из папки."""
        if not dir_path or not Path(dir_path).exists():
            return []
        
        images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            for img_path in Path(dir_path).glob(ext):
                try:
                    img = Image.open(img_path).convert('RGB')
                    images.append(img)
                except Exception as e:
                    print(f"   ⚠️ Не удалось загрузить {img_path}: {e}")
        
        return images