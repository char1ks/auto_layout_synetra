"""
Этап 4: ИЗВЛЕЧЕНИЕ ЭМБЕДДИНГОВ (ВЕКТОРНЫХ ПРЕДСТАВЛЕНИЙ)

Основная задача: Преобразовать маски-кандидаты и изображения-примеры в векторы 
в общем пространстве признаков для их последующего сравнения.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image

try:
    import sys
    sys.path.append('./searchdet-main')
    from mask_withsearch import (
        get_vector,
        adjust_embedding,
        extract_features_from_masks,
    )
    SEARCHDET_AVAILABLE = True
except Exception as e:
    print(f"⚠️ SearchDet недоступен: {e}")
    SEARCHDET_AVAILABLE = False


class EmbeddingExtractor:
    """Класс для извлечения эмбеддингов из масок и изображений."""
    
    def __init__(self, searchdet_models=None):
        """
        Инициализация экстрактора эмбеддингов.
        
        Args:
            searchdet_models: Инициализированные модели SearchDet
        """
        if not SEARCHDET_AVAILABLE:
            raise RuntimeError("SearchDet не доступен для извлечения эмбеддингов")
        
        self.searchdet_models = searchdet_models
    
    def extract_embeddings(self, image: np.ndarray, masks: List[Dict[str, Any]],
                          positive_examples: List[Image.Image], 
                          negative_examples: List[Image.Image]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Извлекает эмбеддинги для масок и примеров.
        
        Args:
            image: Целевое изображение
            masks: Список масок-кандидатов
            positive_examples: Положительные примеры
            negative_examples: Отрицательные примеры
            
        Returns:
            Кортеж (mask_embeddings, positive_embeddings, negative_embeddings)
        """
        print("\n🔄 ЭТАП 4: ИЗВЛЕЧЕНИЕ ЭМБЕДДИНГОВ")
        print("=" * 60)
        
        if not masks:
            print("   ⚠️ Нет масок для извлечения эмбеддингов")
            return np.array([]), np.array([]), np.array([])
        
        # 4.1. Извлечение эмбеддингов для масок-кандидатов
        mask_embeddings = self._extract_mask_embeddings(image, masks)
        
        # 4.2. Извлечение эмбеддингов для примеров
        positive_embeddings, negative_embeddings = self._build_queries(
            positive_examples, negative_examples
        )
        
        print(f"   📊 Результат извлечения эмбеддингов:")
        print(f"     • Маски: {mask_embeddings.shape}")
        print(f"     • Positive примеры: {positive_embeddings.shape}")
        print(f"     • Negative примеры: {negative_embeddings.shape}")
        
        return mask_embeddings, positive_embeddings, negative_embeddings
    
    def _extract_mask_embeddings(self, image: np.ndarray, 
                                masks: List[Dict[str, Any]]) -> np.ndarray:
        """
        4.1. Извлечение эмбеддингов для масок-кандидатов.
        
        Args:
            image: Целевое изображение в формате RGB
            masks: Список словарей с масками
            
        Returns:
            Массив эмбеддингов масок [N_masks, embedding_dim]
        """
        print(f"   🔍 Извлечение эмбеддингов для {len(masks)} масок...")
        
        if not masks:
            return np.array([])
        
        try:
            # Подготавливаем маски для SearchDet
            mask_list = []
            for mask in masks:
                # SearchDet ожидает маски в формате boolean
                segmentation = mask['segmentation'].astype(bool)
                mask_list.append(segmentation)
            
            # Используем функцию extract_features_from_masks из SearchDet
            # Эта функция:
            # 1. Пропускает изображение через ResNet
            # 2. Извлекает feature map с указанного слоя 
            # 3. Применяет Masked Pooling для каждой маски
            # 4. Выполняет Global Average Pooling
            # 5. Применяет L2 normalization
            mask_embeddings = extract_features_from_masks(
                image, mask_list, self.searchdet_models
            )
            
            print(f"     ✅ Эмбеддинги масок извлечены: {mask_embeddings.shape}")
            return mask_embeddings
            
        except Exception as e:
            print(f"     ❌ Ошибка извлечения эмбеддингов масок: {e}")
            return np.array([])
    
    def _build_queries(self, positive_examples: List[Image.Image], 
                      negative_examples: List[Image.Image]) -> Tuple[np.ndarray, np.ndarray]:
        """
        4.2. Построение запросов из positive/negative примеров.
        
        Args:
            positive_examples: Список positive изображений
            negative_examples: Список negative изображений
            
        Returns:
            Кортеж (positive_embeddings, negative_embeddings)
        """
        print(f"   🎯 Построение запросов из примеров...")
        print(f"     • Positive примеры: {len(positive_examples)}")
        print(f"     • Negative примеры: {len(negative_examples)}")
        
        # Извлекаем эмбеддинги для positive примеров
        positive_embeddings = self._extract_example_embeddings(
            positive_examples, "positive"
        )
        
        # Извлекаем эмбеддинги для negative примеров
        negative_embeddings = self._extract_example_embeddings(
            negative_examples, "negative"
        )
        
        # Если есть negative примеры, корректируем positive эмбеддинги
        if len(negative_embeddings) > 0 and len(positive_embeddings) > 0:
            positive_embeddings = self._adjust_positive_embeddings(
                positive_embeddings, negative_embeddings
            )
        
        return positive_embeddings, negative_embeddings
    
    def _extract_example_embeddings(self, examples: List[Image.Image], 
                                   example_type: str) -> np.ndarray:
        """
        Извлекает эмбеддинги для списка изображений-примеров.
        
        Args:
            examples: Список изображений PIL
            example_type: Тип примеров ("positive" или "negative")
            
        Returns:
            Массив эмбеддингов [N_examples, embedding_dim]
        """
        if not examples:
            return np.array([])
        
        embeddings = []
        
        for i, example in enumerate(examples):
            try:
                # Конвертируем PIL Image в numpy для SearchDet
                example_np = np.array(example)
                
                # Используем get_vector из SearchDet для извлечения эмбеддинга
                # Эта функция:
                # 1. Пропускает изображение через ResNet
                # 2. Применяет Global Average Pooling по всей feature map
                # 3. Применяет L2 normalization
                embedding = get_vector(example_np, self.searchdet_models)
                embeddings.append(embedding)
                
            except Exception as e:
                print(f"     ❌ Ошибка извлечения эмбеддинга для {example_type} примера {i}: {e}")
                continue
        
        if embeddings:
            embeddings_array = np.array(embeddings)
            print(f"     ✅ {example_type.capitalize()} эмбеддинги: {embeddings_array.shape}")
            return embeddings_array
        else:
            return np.array([])
    
    def _adjust_positive_embeddings(self, positive_embeddings: np.ndarray, 
                                   negative_embeddings: np.ndarray) -> np.ndarray:
        """
        Корректирует positive эмбеддинги с учетом negative примеров.
        
        Args:
            positive_embeddings: Массив positive эмбеддингов
            negative_embeddings: Массив negative эмбеддингов
            
        Returns:
            Скорректированные positive эмбеддинги
        """
        print(f"     🔧 Корректировка positive эмбеддингов с учетом {len(negative_embeddings)} negative примеров...")
        
        try:
            adjusted_embeddings = []
            
            for pos_emb in positive_embeddings:
                # Используем adjust_embedding из SearchDet
                # Эта функция:
                # 1. Вычисляет центроид negative примеров
                # 2. "Сдвигает" positive вектор от negative центроида
                # 3. Применяет L2 normalization к результату
                adjusted_emb = adjust_embedding(
                    pos_emb, negative_embeddings, self.searchdet_models
                )
                adjusted_embeddings.append(adjusted_emb)
            
            adjusted_array = np.array(adjusted_embeddings)
            print(f"     ✅ Positive эмбеддинги скорректированы: {adjusted_array.shape}")
            return adjusted_array
            
        except Exception as e:
            print(f"     ❌ Ошибка корректировки positive эмбеддингов: {e}")
            # Возвращаем исходные эмбеддинги в случае ошибки
            return positive_embeddings
    
    @staticmethod
    def validate_embeddings(embeddings: np.ndarray, name: str) -> bool:
        """
        Проверяет корректность извлеченных эмбеддингов.
        
        Args:
            embeddings: Массив эмбеддингов для проверки
            name: Название эмбеддингов для логирования
            
        Returns:
            True если эмбеддинги корректны
        """
        if embeddings.size == 0:
            print(f"     ⚠️ {name}: пустой массив эмбеддингов")
            return False
        
        if len(embeddings.shape) != 2:
            print(f"     ❌ {name}: неправильная размерность {embeddings.shape}")
            return False
        
        if np.any(np.isnan(embeddings)):
            print(f"     ❌ {name}: содержит NaN значения")
            return False
        
        if np.any(np.isinf(embeddings)):
            print(f"     ❌ {name}: содержит бесконечные значения")
            return False
        
        # Проверяем L2 нормализацию (векторы должны иметь единичную длину)
        norms = np.linalg.norm(embeddings, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-6):
            print(f"     ⚠️ {name}: векторы не нормализованы (нормы: {norms.min():.6f}-{norms.max():.6f})")
        
        print(f"     ✅ {name}: валидация пройдена")
        return True
    
    def get_embedding_stats(self, embeddings: np.ndarray, name: str) -> Dict[str, Any]:
        """
        Возвращает статистики эмбеддингов.
        
        Args:
            embeddings: Массив эмбеддингов
            name: Название для логирования
            
        Returns:
            Словарь со статистиками
        """
        if embeddings.size == 0:
            return {"name": name, "count": 0}
        
        stats = {
            "name": name,
            "count": len(embeddings),
            "dimensions": embeddings.shape[1] if len(embeddings.shape) > 1 else 0,
            "mean_norm": np.linalg.norm(embeddings, axis=1).mean() if len(embeddings.shape) > 1 else 0,
            "std_norm": np.linalg.norm(embeddings, axis=1).std() if len(embeddings.shape) > 1 else 0,
            "min_value": embeddings.min(),
            "max_value": embeddings.max(),
            "mean_value": embeddings.mean(),
            "std_value": embeddings.std(),
        }
        
        return stats
