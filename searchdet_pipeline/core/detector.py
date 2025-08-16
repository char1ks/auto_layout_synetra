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
from .step7_result_saving import ResultSaver


class SearchDetDetector:
    """Автономный детектор для модульного пайплайна."""
    
    def __init__(self, **kwargs):
        if not SEARCHDET_AVAILABLE:
            raise RuntimeError("SearchDet не найден")

        # Сохраняем все kwargs для гибкости
        self.params = kwargs

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
        self.mask_backend = self.params.get('mask_backend', 'fastsam')
        
        # Инициализируем компоненты
        self.mask_generator = MaskGenerator(self)
        self.mask_filter = MaskFilter(self, self.params)
        self.embedding_extractor = EmbeddingExtractor(self)
        self.score_calculator = ScoreCalculator(self)
        self.result_saver = ResultSaver()

        print("✅ SearchDetDetector инициализирован (автономная модульная версия)")

    def find_present_elements(self, image_path, positive_dir, negative_dir=None, output_dir="output"):
        """Основная функция поиска объектов."""
        print(f"🔍 Модульный анализ: {image_path}" + "="*60)
        print("🔄 ДЕТАЛЬНАЯ ПОСЛЕДОВАТЕЛЬНОСТЬ ВЫПОЛНЕНИЯ МОДУЛЬНОГО PIPELINE:")
        print("=" * 80)
        print("8️⃣ searchdet_pipeline/core/detector.py → find_present_elements()")
        
        # Детальное отслеживание времени
        timing_info = {}
        t_total = time.time()
        
        # Загрузка изображения
        t_loading = time.time()
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Не удалось загрузить изображение: {image_path}")
        image_np = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        timing_info['image_loading'] = time.time() - t_loading

        # 0) загрузка примеров
        print("9️⃣ Шаг 1: _load_example_images() - загрузка positive/negative примеров")
        t_examples = time.time()
        pos_imgs = self._load_example_images(positive_dir)
        neg_imgs = self._load_example_images(negative_dir) if negative_dir else []
        timing_info['examples_loading'] = time.time() - t_examples
        
        if len(pos_imgs) == 0:
            print("   ❌ Нет положительных примеров — прекращаем.")
            return {"found_elements": [], "masks": []}
        print(f"   📁 Positive: {len(pos_imgs)}	📁 Negative: {len(neg_imgs)}")

        # 1) генерация масок
        print("🔟 Шаг 2: MaskGenerator.generate() - генерация масок через SAM/FastSAM")
        t_masks = time.time()
        masks = self.mask_generator.generate(image_np)
        timing_info['mask_generation'] = time.time() - t_masks
        
        # 2) фильтрация масок
        print("1️⃣1️⃣ Шаг 3-7: MaskFilter.apply_all_filters() - все фильтры масок")
        t_filtering = time.time()
        masks = self.mask_filter.apply_all_filters(masks, image_np)
        timing_info['mask_filtering'] = time.time() - t_filtering
        
        if not masks:
            print("   ❌ Нет валидных масок после фильтров.")
            return {"found_elements": [], "masks": []}

        # 3) извлечение эмбеддингов
        print("1️⃣2️⃣ Шаг 8: EmbeddingExtractor.extract_mask_embeddings() - эмбеддинги масок")
        t_embeddings = time.time()
        mask_vecs, idx_map = self.embedding_extractor.extract_mask_embeddings(image_np, masks)
        if mask_vecs.shape[0] == 0:
            print("   ❌ Не удалось получить эмбеддинги масок.")
            return {"found_elements": [], "masks": []}
        print(f"   📊 Масок с валидными векторами: {mask_vecs.shape[0]}")
        
        print("1️⃣3️⃣ Шаг 9: EmbeddingExtractor.build_queries() - эмбеддинги примеров")
        q_pos, q_neg = self.embedding_extractor.build_queries(pos_imgs, neg_imgs)
        timing_info['embedding_extraction'] = time.time() - t_embeddings

        # 4) скоринг и принятие решений
        print("1️⃣4️⃣ Шаг 10: ScoreCalculator.score_and_decide() - скоринг и принятие решений")
        print("🔍 ЭТАП 3: Сопоставление с positive/negative...")
        t_scoring = time.time()
        accepted_indices = self.score_calculator.score_and_decide(mask_vecs, q_pos, q_neg)
        timing_info['scoring_and_decisions'] = time.time() - t_scoring
        
        # 5) формирование результата
        t_result = time.time()
        found = []
        result_masks = []
        for i in accepted_indices:
            original_idx = idx_map[i]
            mask_dict = masks[original_idx].copy()  # Создаем копию чтобы не изменять оригинал
            
            # Добавляем поля которые ожидает ResultSaver
            confidence = 0.8  # заглушка, можно улучшить в будущем
            mask_dict['confidence'] = confidence
            
            # Убеждаемся что area есть
            if 'area' not in mask_dict and 'segmentation' in mask_dict:
                mask_dict['area'] = int(np.sum(mask_dict['segmentation']))
            
            found.append({
                "mask": mask_dict,
                "confidence": confidence,
                "bbox": mask_dict.get("bbox", [0, 0, 0, 0])
            })
            result_masks.append(mask_dict)
        timing_info['result_formatting'] = time.time() - t_result

        # 6) сохранение результатов
        print("1️⃣5️⃣ Шаг 11: ResultSaver.save_all_results() - сохранение файлов")
        t_saving = time.time()
        image_name = Path(image_path).name
        saved_files = self.result_saver.save_all_results(
            image_np, 
            result_masks, 
            output_dir, 
            image_name,
            pipeline_config={"backend": self.mask_backend}
        )
        timing_info['result_saving'] = time.time() - t_saving

        # Подсчёт общего времени и вывод детальной статистики
        total_time = time.time() - t_total
        timing_info['total_time'] = total_time
        
        print(f"🎯 Принято масок: {len(found)} (после правил и NMS)")
        print(f"⏱️ Общее время: {total_time:.2f} сек")
        print(f"💾 Результаты сохранены в: {output_dir}")
        print(f"📁 Сохранено файлов: {len(saved_files)}")
        
        # Выводим подробную статистику времени
        self._print_timing_statistics(timing_info)
        
        return {
            "found_elements": found, 
            "masks": result_masks,
            "timing_info": timing_info,
            "output_directory": output_dir,
            "saved_files": saved_files
        }

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
    
    def _print_timing_statistics(self, timing_info):
        """Выводит детальную статистику времени выполнения."""
        print("\n" + "="*60)
        print("⏱️ ДЕТАЛЬНАЯ СТАТИСТИКА ВРЕМЕНИ ВЫПОЛНЕНИЯ:")
        print("="*60)
        
        total_time = timing_info['total_time']
        
        # Определяем порядок этапов и их названия
        stages = [
            ('image_loading', '📁 Загрузка изображения'),
            ('examples_loading', '🖼️ Загрузка примеров'),
            ('mask_generation', '🎯 Генерация масок (SAM/FastSAM)'),
            ('mask_filtering', '🔍 Фильтрация масок'),
            ('embedding_extraction', '🧠 Извлечение эмбеддингов'),
            ('scoring_and_decisions', '📊 Скоринг и решения'),
            ('result_formatting', '📋 Формирование результата'),
            ('result_saving', '💾 Сохранение файлов')
        ]
        
        for stage_key, stage_name in stages:
            if stage_key in timing_info:
                stage_time = timing_info[stage_key]
                percentage = (stage_time / total_time * 100) if total_time > 0 else 0
                print(f"{stage_name:<40}: {stage_time:>6.3f}с ({percentage:>5.1f}%)")
        
        print("-" * 60)
        print(f"{'🚀 ОБЩЕЕ ВРЕМЯ':<40}: {total_time:>6.3f}с (100.0%)")
        print("="*60)
        
        # Показываем самые медленные этапы
        stage_times = [(name, timing_info.get(key, 0)) for key, name in stages if key in timing_info]
        stage_times.sort(key=lambda x: x[1], reverse=True)
        
        if len(stage_times) > 1:
            print("\n🐌 САМЫЕ МЕДЛЕННЫЕ ЭТАПЫ:")
            for i, (name, stage_time) in enumerate(stage_times[:3]):
                percentage = (stage_time / total_time * 100) if total_time > 0 else 0
                print(f"   {i+1}. {name}: {stage_time:.3f}с ({percentage:.1f}%)")
        
        # Рекомендации по оптимизации
        if 'mask_generation' in timing_info and timing_info['mask_generation'] > total_time * 0.5:
            print("\n💡 РЕКОМЕНДАЦИИ ПО ОПТИМИЗАЦИИ:")
            print("   • Генерация масок занимает >50% времени")
            print("   • Попробуйте FastSAM вместо SAM-HQ для ускорения")
            print("   • Или уменьшите параметры SAM (points_per_side, imgsz)")
        
        if 'embedding_extraction' in timing_info and timing_info['embedding_extraction'] > total_time * 0.3:
            print("\n💡 РЕКОМЕНДАЦИИ ПО ОПТИМИЗАЦИИ:")
            print("   • Извлечение эмбеддингов занимает >30% времени")
            print("   • Проверьте размер feature map (SEARCHDET_FEAT_SHORT_SIDE)")
            print("   • Убедитесь что используется быстрый метод извлечения")
        
        print()