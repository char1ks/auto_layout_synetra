"""
Основной класс SearchDetDetector - точка входа в пайплайн детекции объектов.
"""

import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image

from ..utils.config import Config
from .step1_data_loading import DataLoader
from .step2_mask_generation import MaskGenerator  
from .step3_mask_filtering import MaskFilter
from .step4_embedding_extraction import EmbeddingExtractor
from .step5_scoring_decisions import ScoringEngine
from .step6_postprocessing import PostProcessor
from .step7_result_saving import ResultSaver


class SearchDetDetector:
    """
    Главный класс для детекции объектов с использованием SearchDet пайплайна.
    
    Объединяет все этапы обработки в единый интерфейс:
    1. Загрузка данных
    2. Генерация масок
    3. Фильтрация масок
    4. Извлечение эмбеддингов
    5. Скоринг и принятие решений
    6. Постобработка
    7. Сохранение результатов
    """
    
    def __init__(self, config: Config):
        """
        Инициализация детектора.
        
        Args:
            config: Конфигурация пайплайна
        """
        self.config = config
        
        # Инициализируем компоненты пайплайна
        self.data_loader = DataLoader()
        self.mask_generator = MaskGenerator(config.mask_generation)
        self.mask_filter = MaskFilter(config.filtering)
        self.scoring_engine = ScoringEngine(config.scoring)
        self.post_processor = PostProcessor(config.post_processing)
        self.result_saver = ResultSaver(config.overlay_alpha)
        
        # Экстрактор эмбеддингов будет инициализирован после загрузки SearchDet моделей
        self.embedding_extractor = None
        self.searchdet_models = None
        
        print("🚀 SearchDetDetector инициализирован")
    
    def initialize_models(self, **model_paths):
        """
        Инициализирует все необходимые модели.
        
        Args:
            **model_paths: Пути к различным моделям (sam_hq_checkpoint, sam2_checkpoint, etc.)
        """
        print("🔧 Инициализация моделей...")
        
        # Инициализируем бэкенд для генерации масок
        backend_config = {
            'checkpoint_path': model_paths.get('sam_hq_checkpoint') or self.config.sam_hq_checkpoint,
            'config_path': model_paths.get('sam2_config') or self.config.sam2_config,
        }
        
        self.mask_generator.initialize_backend(
            self.config.mask_generation.backend,
            **backend_config
        )
        
        # Инициализируем SearchDet модели для эмбеддингов
        try:
            import sys
            sys.path.append('./searchdet-main')
            from mask_withsearch import initialize_models as init_searchdet
            
            self.searchdet_models = init_searchdet()
            self.embedding_extractor = EmbeddingExtractor(self.searchdet_models)
            print("✅ SearchDet модели инициализированы")
            
        except Exception as e:
            print(f"❌ Ошибка инициализации SearchDet: {e}")
            raise RuntimeError("Не удалось инициализировать SearchDet модели")
    
    def find_present_elements(self, image_path: str,
                            positive_dir: Optional[str] = None,
                            negative_dir: Optional[str] = None,
                            output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Основной метод для детекции объектов на изображении.
        
        Args:
            image_path: Путь к целевому изображению
            positive_dir: Директория с положительными примерами
            negative_dir: Директория с отрицательными примерами  
            output_dir: Директория для сохранения результатов
            
        Returns:
            Словарь с результатами детекции
        """
        start_time = time.time()
        
        print("\n" + "="*80)
        print("🔍 ЗАПУСК ПАЙПЛАЙНА ДЕТЕКЦИИ ОБЪЕКТОВ")
        print("="*80)
        print(f"📸 Изображение: {image_path}")
        print(f"➕ Positive примеры: {positive_dir or 'не указано'}")
        print(f"➖ Negative примеры: {negative_dir or 'не указано'}")
        print(f"💾 Выходная директория: {output_dir or 'не указано'}")
        
        try:
            # Этап 1: Загрузка и подготовка данных
            target_image, positive_examples, negative_examples = self.data_loader.prepare_data(
                image_path, positive_dir, negative_dir
            )
            
            # Этап 2: Генерация масок-кандидатов
            candidate_masks = self.mask_generator.generate_masks(target_image)
            
            # Этап 3: Фильтрация масок
            filtered_masks = self.mask_filter.filter_masks(candidate_masks, target_image.shape)
            
            # Этап 4: Извлечение эмбеддингов
            if self.embedding_extractor is None:
                raise RuntimeError("EmbeddingExtractor не инициализирован. Вызовите initialize_models() сначала.")
            
            mask_embeddings, positive_embeddings, negative_embeddings = self.embedding_extractor.extract_embeddings(
                target_image, filtered_masks, positive_examples, negative_examples
            )
            
            # Этап 5: Скоринг и принятие решений
            accept_flags, confidence_scores, positive_scores, negative_scores = self.scoring_engine.score_and_decide(
                mask_embeddings, positive_embeddings, negative_embeddings
            )
            
            # Этап 6: Постобработка
            final_masks = self.post_processor.postprocess(filtered_masks, accept_flags, confidence_scores)
            
            # Этап 7: Сохранение результатов (если указан output_dir)
            saved_files = {}
            if output_dir and self.config.save_all:
                saved_files = self.result_saver.save_all_results(
                    target_image, final_masks, output_dir, image_path, self.config.to_dict()
                )
            
            processing_time = time.time() - start_time
            
            # Формируем результат
            result = {
                'success': True,
                'processing_time': processing_time,
                'detections': final_masks,
                'statistics': {
                    'total_candidate_masks': len(candidate_masks),
                    'filtered_masks': len(filtered_masks), 
                    'accepted_masks': sum(accept_flags) if accept_flags else 0,
                    'final_detections': len(final_masks),
                    'positive_examples': len(positive_examples),
                    'negative_examples': len(negative_examples),
                },
                'saved_files': saved_files,
                'config': self.config.to_dict(),
            }
            
            # Добавляем сводку скоринга
            if confidence_scores:
                result['scoring_summary'] = self.scoring_engine.get_scoring_summary(
                    accept_flags, confidence_scores, positive_scores, negative_scores
                )
            
            # Добавляем сводку детекций
            if final_masks:
                result['detection_summary'] = self.post_processor.create_detection_summary(final_masks)
            
            print("\n" + "="*80)
            print("✅ ПАЙПЛАЙН ЗАВЕРШЁН УСПЕШНО")
            print("="*80)
            print(f"⏱️  Время обработки: {processing_time:.2f} сек")
            print(f"🔍 Найдено объектов: {len(final_masks)}")
            print(f"📊 Кандидатов: {len(candidate_masks)} → Отфильтровано: {len(filtered_masks)} → Принято: {sum(accept_flags) if accept_flags else 0} → Финально: {len(final_masks)}")
            
            if saved_files:
                print(f"💾 Сохранено файлов: {len(saved_files)}")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_result = {
                'success': False,
                'error': str(e),
                'processing_time': processing_time,
                'detections': [],
                'statistics': {},
                'saved_files': {},
                'config': self.config.to_dict(),
            }
            
            print("\n" + "="*80)
            print("❌ ОШИБКА В ПАЙПЛАЙНЕ")
            print("="*80)
            print(f"⚠️  Ошибка: {e}")
            print(f"⏱️  Время до ошибки: {processing_time:.2f} сек")
            
            return error_result
    
    def detect_batch(self, image_paths: List[str],
                    positive_dir: Optional[str] = None,
                    negative_dir: Optional[str] = None,
                    output_base_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Пакетная обработка нескольких изображений.
        
        Args:
            image_paths: Список путей к изображениям
            positive_dir: Директория с положительными примерами
            negative_dir: Директория с отрицательными примерами
            output_base_dir: Базовая директория для сохранения результатов
            
        Returns:
            Список результатов для каждого изображения
        """
        print(f"\n🔄 ПАКЕТНАЯ ОБРАБОТКА {len(image_paths)} ИЗОБРАЖЕНИЙ")
        print("="*80)
        
        results = []
        
        for i, image_path in enumerate(image_paths):
            print(f"\n📸 Обработка {i+1}/{len(image_paths)}: {image_path}")
            
            output_dir = None
            if output_base_dir:
                from pathlib import Path
                image_name = Path(image_path).stem
                output_dir = str(Path(output_base_dir) / f"batch_result_{i:03d}_{image_name}")
            
            result = self.find_present_elements(
                image_path, positive_dir, negative_dir, output_dir
            )
            
            result['batch_index'] = i
            result['image_path'] = image_path
            results.append(result)
        
        # Сводная статистика по всей пачке
        successful_results = [r for r in results if r['success']]
        total_detections = sum(len(r['detections']) for r in successful_results)
        total_time = sum(r['processing_time'] for r in results)
        
        print(f"\n📊 ИТОГИ ПАКЕТНОЙ ОБРАБОТКИ:")
        print(f"   • Успешно обработано: {len(successful_results)}/{len(image_paths)}")
        print(f"   • Общее время: {total_time:.2f} сек")
        print(f"   • Среднее время на изображение: {total_time/len(image_paths):.2f} сек") 
        print(f"   • Всего найдено объектов: {total_detections}")
        
        return results
    
    def get_config(self) -> Config:
        """Возвращает текущую конфигурацию."""
        return self.config
    
    def update_config(self, **kwargs):
        """
        Обновляет конфигурацию.
        
        Args:
            **kwargs: Параметры для обновления
        """
        # Здесь можно добавить логику обновления конфигурации
        print("⚠️ Обновление конфигурации во время выполнения пока не поддерживается")
    
    def validate_setup(self) -> Dict[str, bool]:
        """
        Проверяет корректность настройки всех компонентов.
        
        Returns:
            Словарь с результатами проверки каждого компонента
        """
        validation_results = {
            'data_loader': self.data_loader is not None,
            'mask_generator': self.mask_generator is not None,
            'mask_filter': self.mask_filter is not None,
            'scoring_engine': self.scoring_engine is not None,
            'post_processor': self.post_processor is not None,
            'result_saver': self.result_saver is not None,
            'embedding_extractor': self.embedding_extractor is not None,
            'searchdet_models': self.searchdet_models is not None,
        }
        
        all_valid = all(validation_results.values())
        
        print(f"🔧 Валидация компонентов: {'✅ Все компоненты готовы' if all_valid else '❌ Некоторые компоненты не готовы'}")
        
        for component, status in validation_results.items():
            status_icon = "✅" if status else "❌"
            print(f"   {status_icon} {component}")
        
        return validation_results
