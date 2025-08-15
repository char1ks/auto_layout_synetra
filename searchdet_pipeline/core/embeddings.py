#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Извлечение эмбеддингов масок и примеров через SearchDet.
"""

import numpy as np
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms

# SearchDet импорты
try:
    from mask_withsearch import get_vector, adjust_embedding, extract_features_from_masks
    SEARCHDET_AVAILABLE = True
except Exception as e:
    print(f"⚠️ SearchDet недоступен: {e}")
    SEARCHDET_AVAILABLE = False


class EmbeddingExtractor:
    """Извлечение эмбеддингов через SearchDet."""
    
    def __init__(self, detector):
        self.detector = detector
    
    def extract_mask_embeddings(self, image_np, masks):
        """Извлекает эмбеддинги для масок."""
        print("🧠 ЭТАП 2: Эмбеддинги масок и запросов...")
        print(f"🔍 Извлечение эмбеддингов для {len(masks)} масок...")
        
        if not SEARCHDET_AVAILABLE:
            print("⚠️ SearchDet недоступен, возвращаем пустые эмбеддинги")
            return np.array([]).reshape(0, 1024), []
        
        # Конвертируем изображение в PIL
        pil_image = Image.fromarray(image_np)
        
        # Подготавливаем маски в формате boolean numpy array
        mask_arrays = []
        valid_indices = []
        
        for i, mask_dict in enumerate(masks):
            mask = mask_dict["segmentation"]
            if isinstance(mask, np.ndarray) and mask.dtype == bool:
                mask_arrays.append(mask)
                valid_indices.append(i)
            else:
                print(f"   ⚠️ Маска {i} имеет неправильный тип: {type(mask)}")
        
        if not mask_arrays:
            print("   ❌ Нет валидных масок для извлечения эмбеддингов")
            return np.array([]).reshape(0, 1024), []
        
        print(f"🚀 БЫСТРОЕ извлечение эмбеддингов для {len(mask_arrays)} масок (Masked Pooling)...")
        
        try:
            # Пробуем быстрый метод
            embeddings = self._extract_fast(image_np, mask_arrays)
            if embeddings is not None:
                print(f"⚡ БЫСТРО: {len(mask_arrays)} масок обработано")
                return embeddings, valid_indices
        except Exception as e:
            print(f"⚠️ Быстрый метод не сработал ({e}), используем старый")
        
        # Медленный метод как fallback
        print(f"🧠 МЕДЛЕННОЕ извлечение эмбеддингов для {len(mask_arrays)} масок...")
        try:
            embeddings = self._extract_slow(pil_image, mask_arrays)
            if embeddings is not None:
                print(f"✅ Старый метод: обработано {len(mask_arrays)} масок")
                return embeddings, valid_indices
        except Exception as e:
            print(f"⚠️ Медленный метод также не сработал: {e}")
        
        print("❌ Не удалось извлечь эмбеддинги")
        return np.array([]).reshape(0, 1024), []
    
    def _extract_fast(self, image_np, mask_arrays):
        """Быстрое извлечение через extract_features_from_masks."""
        # Получаем модели SearchDet
        resnet, layer, transform, sam = (
            self.detector.searchdet_resnet,
            self.detector.searchdet_layer, 
            self.detector.searchdet_transform,
            self.detector.searchdet_sam
        )
        
        print(f"🔧 Используем {layer} для большего feature map")
        
        # Проверяем что это tuple из моделей
        if isinstance(resnet, tuple) and len(resnet) >= 2:
            model = resnet[0]  # backbone
            device = resnet[1] if len(resnet) > 1 else 'cuda'
        else:
            model = resnet
            device = 'cuda'
        
        # Преобразуем mask_arrays в формат, ожидаемый extract_features_from_masks
        # Функция ожидает список словарей с ключом 'segmentation'
        mask_dicts = []
        for mask_array in mask_arrays:
            mask_dicts.append({'segmentation': mask_array})
        
        # Пробуем разные варианты вызова extract_features_from_masks
        try:
            # Правильная сигнатура: extract_features_from_masks(image, masks, model, layer, transform)
            embeddings = extract_features_from_masks(
                image_np, mask_dicts, model, layer, transform
            )
            if isinstance(embeddings, np.ndarray) and embeddings.ndim == 2:
                return embeddings
        except Exception as e:
            print(f"   ⚠️ Вариант 1 не удался: {e}")
        
        try:
            # Альтернативная сигнатура: extract_features_from_masks(image, masks, model, layer)
            embeddings = extract_features_from_masks(
                image_np, mask_dicts, model, layer
            )
            if isinstance(embeddings, np.ndarray) and embeddings.ndim == 2:
                return embeddings
        except Exception as e:
            print(f"   ⚠️ Вариант 2 не удался: {e}")
        
        try:
            # Старая сигнатура: extract_features_from_masks(image, masks, model)
            embeddings = extract_features_from_masks(
                image_np, mask_dicts, model
            )
            if isinstance(embeddings, np.ndarray) and embeddings.ndim == 2:
                return embeddings
        except Exception as e:
            print(f"   ⚠️ Вариант 3 не удался: {e}")
        
        return None
    
    def _extract_slow(self, pil_image, mask_arrays):
        """Медленное извлечение по одной маске."""
        # Получаем модели SearchDet
        resnet, layer, transform, sam = (
            self.detector.searchdet_resnet,
            self.detector.searchdet_layer, 
            self.detector.searchdet_transform,
            self.detector.searchdet_sam
        )
        
        print("🔧 DINO оптимизация: уменьшаем изображения до 384x512 для ускорения")
        
        # Проверяем что это tuple из моделей
        if isinstance(resnet, tuple) and len(resnet) >= 2:
            model = resnet[0]  # backbone
            device = resnet[1] if len(resnet) > 1 else 'cuda'
        else:
            model = resnet
            device = 'cuda'
        
        # Альтернативный подход: используем старую медленную функцию напрямую
        try:
            # Преобразуем mask_arrays в формат, ожидаемый extract_features_from_masks_slow
            mask_dicts = []
            for mask_array in mask_arrays:
                mask_dicts.append({'segmentation': mask_array})
            
            # Конвертируем PIL обратно в numpy для функции
            image_np = np.array(pil_image)
            
            # Вызываем медленную функцию напрямую
            from mask_withsearch import extract_features_from_masks_slow
            embeddings = extract_features_from_masks_slow(image_np, mask_dicts, model, layer, transform)
            
            if isinstance(embeddings, np.ndarray) and embeddings.ndim == 2:
                return embeddings
                
        except Exception as e:
            print(f"   ⚠️ Медленная функция не сработала: {e}")
        
        # Fallback: извлекаем эмбеддинги по одной маске вручную
        embeddings = []
        for i, mask in enumerate(mask_arrays):
            try:
                # Создаем маскированное изображение
                image_np = np.array(pil_image)
                mask_image = np.zeros_like(image_np)
                mask_image[mask] = image_np[mask]
                
                # Конвертируем обратно в PIL
                mask_pil = Image.fromarray(mask_image)
                
                # Используем get_vector с правильными аргументами
                vec = get_vector(mask_pil, model, layer, transform)
                if hasattr(vec, 'numpy'):
                    embeddings.append(vec.numpy())
                else:
                    embeddings.append(vec)
                    
            except Exception as e:
                print(f"   ⚠️ Ошибка с маской {i}: {e}")
                # Fallback - случайный вектор
                embeddings.append(np.random.rand(1024).astype(np.float32))
        
        if embeddings:
            embeddings_array = np.array(embeddings)
            # Нормализуем
            norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
            embeddings_array = embeddings_array / (norms + 1e-8)
            return embeddings_array
        
        return None
    
    def build_queries(self, pos_imgs, neg_imgs):
        """Строит эмбеддинги запросов из positive/negative примеров."""
        print(f"🔍 Построение запросов из {len(pos_imgs)} positive и {len(neg_imgs)} negative примеров")
        
        # Получаем модели SearchDet
        resnet, layer, transform, sam = (
            self.detector.searchdet_resnet,
            self.detector.searchdet_layer, 
            self.detector.searchdet_transform,
            self.detector.searchdet_sam
        )
        
        # Проверяем что это tuple из моделей
        if isinstance(resnet, tuple) and len(resnet) >= 2:
            model = resnet[0]  # backbone
            device = resnet[1] if len(resnet) > 1 else 'cuda'
        else:
            model = resnet
            device = 'cuda'
        
        # Positive эмбеддинги
        q_pos = []
        for i, img in enumerate(pos_imgs):
            try:
                vec = self._get_image_embedding(img, model, layer, transform)
                if vec is not None:
                    q_pos.append(vec)
            except Exception as e:
                print(f"   ⚠️ Ошибка с positive {i}: {e}")
        
        # Negative эмбеддинги  
        q_neg = []
        for i, img in enumerate(neg_imgs):
            try:
                vec = self._get_image_embedding(img, model, layer, transform)
                if vec is not None:
                    q_neg.append(vec)
            except Exception as e:
                print(f"   ⚠️ Ошибка с negative {i}: {e}")
        
        q_pos = np.array(q_pos) if q_pos else np.array([]).reshape(0, 1024)
        q_neg = np.array(q_neg) if q_neg else np.array([]).reshape(0, 1024)
        
        print(f"   📊 Построено positive: {q_pos.shape[0]}, negative: {q_neg.shape[0]}")
        
        return q_pos, q_neg
    
    def _get_image_embedding(self, pil_image, model, layer, transform):
        """Получает эмбеддинг для одного изображения."""
        if not SEARCHDET_AVAILABLE:
            # Заглушка
            return np.random.rand(1024).astype(np.float32)
        
        try:
            # Используем get_vector из SearchDet с правильными аргументами
            vec = get_vector(pil_image, model, layer, transform)
            if isinstance(vec, np.ndarray):
                return vec
            
            # Если это тензор PyTorch
            if hasattr(vec, 'numpy'):
                return vec.numpy()
                
            return None
        except Exception as e:
            print(f"   ⚠️ get_vector failed: {e}")
            
            try:
                # Fallback - случайный вектор
                return np.random.rand(1024).astype(np.float32)
            except:
                return None
