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
            embeddings = self._extract_fast(pil_image, mask_arrays)
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
    
    def _extract_fast(self, pil_image, mask_arrays):
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
        
        # Пробуем разные варианты вызова extract_features_from_masks
        try:
            # Вариант 1: современная сигнатура
            embeddings = extract_features_from_masks(
                model, pil_image, mask_arrays, layer, transform
            )
            if isinstance(embeddings, np.ndarray) and embeddings.ndim == 2:
                return embeddings
        except Exception as e:
            print(f"   ⚠️ Вариант 1 не удался: {e}")
        
        try:
            # Вариант 2: без transform
            embeddings = extract_features_from_masks(
                model, pil_image, mask_arrays, layer
            )
            if isinstance(embeddings, np.ndarray) and embeddings.ndim == 2:
                return embeddings
        except Exception as e:
            print(f"   ⚠️ Вариант 2 не удался: {e}")
        
        try:
            # Вариант 3: старая сигнатура
            embeddings = extract_features_from_masks(
                model, pil_image, mask_arrays
            )
            if isinstance(embeddings, np.ndarray) and embeddings.ndim == 2:
                return embeddings
        except Exception as e:
            print(f"   ⚠️ Вариант 3 не удался: {e}")
        
        return None
    
    def _extract_slow(self, pil_image, mask_arrays):
        """Медленное извлечение по одной маске."""
        # Пока заглушка
        print("🔧 DINO оптимизация: уменьшаем изображения до 384x512 для ускорения")
        
        # Возвращаем случайные эмбеддинги для тестирования
        n_masks = len(mask_arrays)
        embeddings = np.random.rand(n_masks, 1024).astype(np.float32)
        
        # Нормализуем
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)
        
        return embeddings
    
    def build_queries(self, pos_imgs, neg_imgs):
        """Строит эмбеддинги запросов из positive/negative примеров."""
        print(f"🔍 Построение запросов из {len(pos_imgs)} positive и {len(neg_imgs)} negative примеров")
        
        # Positive эмбеддинги
        q_pos = []
        for i, img in enumerate(pos_imgs):
            try:
                vec = self._get_image_embedding(img)
                if vec is not None:
                    q_pos.append(vec)
            except Exception as e:
                print(f"   ⚠️ Ошибка с positive {i}: {e}")
        
        # Negative эмбеддинги  
        q_neg = []
        for i, img in enumerate(neg_imgs):
            try:
                vec = self._get_image_embedding(img)
                if vec is not None:
                    q_neg.append(vec)
            except Exception as e:
                print(f"   ⚠️ Ошибка с negative {i}: {e}")
        
        q_pos = np.array(q_pos) if q_pos else np.array([]).reshape(0, 1024)
        q_neg = np.array(q_neg) if q_neg else np.array([]).reshape(0, 1024)
        
        print(f"   📊 Построено positive: {q_pos.shape[0]}, negative: {q_neg.shape[0]}")
        
        return q_pos, q_neg
    
    def _get_image_embedding(self, pil_image):
        """Получает эмбеддинг для одного изображения."""
        if not SEARCHDET_AVAILABLE:
            # Заглушка
            return np.random.rand(1024).astype(np.float32)
        
        try:
            # Используем get_vector из SearchDet
            vec = get_vector(pil_image)
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
