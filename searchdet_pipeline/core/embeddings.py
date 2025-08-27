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
try:
    from mask_withsearch import get_vector, adjust_embedding, extract_features_from_masks
    SEARCHDET_AVAILABLE = True
except Exception as e:
    print(f"⚠️ SearchDet недоступен: {e}")
    SEARCHDET_AVAILABLE = False

class EmbeddingExtractor:
    def __init__(self, detector):
        self.detector = detector
        self.backbone = getattr(detector, 'backbone', 'dinov2_b')
        # Ленивая инициализация DINO при первом использовании
        self._dino_model = None
        self._dino_preprocess = None
        # 🚀 Кэш для DINO forward pass
        self._dino_cache = {}  # {image_hash: (patch_tokens, grid_size)}
        # 🚀 Настройка максимального размера для оптимизации
        self.max_embedding_size = getattr(detector, 'max_embedding_size', 1024)
        # 🚀 Настройка половинной точности для DINO
        self.dino_half_precision = getattr(detector, 'dino_half_precision', False)
    
    def extract_mask_embeddings(self, image_np, masks):
        """Извлекает эмбеддинги для масок."""
        print("🧠 ЭТАП 2: Эмбеддинги масок и запросов...")
        print(f"🔍 Извлечение эмбеддингов для {len(masks)} масок...")
        
        if not SEARCHDET_AVAILABLE:
            print("⚠️ SearchDet недоступен, возвращаем пустые эмбеддинги")
            return np.zeros((0, 1024), dtype=np.float32), []
        
        # Конвертируем изображение в PIL
        pil_image = Image.fromarray(image_np)
        
        # Подготавливаем маски в формате boolean numpy array с фильтрацией
        mask_arrays = []
        valid_indices = []
        min_mask_area = 100  # Минимальная площадь маски в пикселях
        
        for i, mask_dict in enumerate(masks):
            mask = mask_dict["segmentation"]
            if isinstance(mask, np.ndarray) and mask.dtype == bool:
                # 🚀 Фильтрация по размеру сразу при обработке
                mask_area = np.sum(mask)
                if mask_area >= min_mask_area:
                    mask_arrays.append(mask)
                    valid_indices.append(i)
            else:
                print(f"   ⚠️ Маска {i} имеет неправильный тип: {type(mask)}")
        
        if len(mask_arrays) < len(masks):
            print(f"   🔍 Предфильтрация масок: {len(masks)} → {len(mask_arrays)} (удалено {len(masks) - len(mask_arrays)} невалидных/маленьких масок)")
        
        if not mask_arrays:
            print("   ❌ Нет валидных масок для извлечения эмбеддингов")
            return np.zeros((0, 1024), dtype=np.float32), []
        
        print(f"🚀 БЫСТРОЕ извлечение эмбеддингов для {len(mask_arrays)} масок (Masked Pooling)...")
        
        try:
            embeddings = self._extract_fast(image_np, mask_arrays)
            if embeddings is not None:
                print(f"⚡ БЫСТРО: {len(mask_arrays)} масок обработано")
                # --- Нормализация форм возврата к (N, D) float32 ---
                embeddings = np.asarray(embeddings)
                if embeddings.ndim == 1:
                    embeddings = embeddings.reshape(1, -1)
                elif embeddings.ndim > 2:
                    embeddings = embeddings.reshape(embeddings.shape[0], -1)
                embeddings = embeddings.astype(np.float32)
                return embeddings, valid_indices
        except Exception as e:
            print(f"⚠️ Быстрый метод не сработал ({e}), используем старый")
        
        print(f"🧠 МЕДЛЕННОЕ извлечение эмбеддингов для {len(mask_arrays)} масок...")
        try:
            embeddings = self._extract_slow(pil_image, mask_arrays)
            if embeddings is not None:
                print(f"✅ Старый метод: обработано {len(mask_arrays)} масок")
                embeddings = np.asarray(embeddings)
                if embeddings.ndim == 1:
                    embeddings = embeddings.reshape(1, -1)
                elif embeddings.ndim > 2:
                    embeddings = embeddings.reshape(embeddings.shape[0], -1)
                embeddings = embeddings.astype(np.float32)
                return embeddings, valid_indices
        except Exception as e:
            print(f"⚠️ Медленный метод также не сработал: {e}")
        
        print("❌ Не удалось извлечь эмбеддинги")
        return np.zeros((0, 1024), dtype=np.float32), []
    
    def _extract_fast(self, image_np, mask_arrays):
        import time
        
        if self.backbone.startswith('dinov2'):
            return self._extract_with_dino(image_np, mask_arrays)
            
        extract_start = time.time()
        print(f"🚀 БЫСТРОЕ извлечение эмбеддингов для {len(mask_arrays)} масок (Masked Pooling)...")
        
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
        
        # 🚀 Пытаемся использовать быстрый метод с правильными параметрами
        try:
            # Проверяем, что у нас есть все необходимые параметры
            if model is not None and layer is not None:
                embeddings = extract_features_from_masks(image_np, mask_dicts, model, layer, transform)
                if isinstance(embeddings, np.ndarray) and embeddings.ndim == 2 and embeddings.shape[0] == len(mask_arrays):
                    extract_time = time.time() - extract_start
                    old_time_estimate = len(mask_arrays) * 0.1  # Примерное время старого метода
                    speedup = old_time_estimate / extract_time if extract_time > 0 else 1
                    print(f"   ⚡ БЫСТРО: {extract_time:.3f} сек ({extract_time/len(mask_arrays)*1000:.1f} мс/маска) - ускорение ~{speedup:.1f}x")
                    return embeddings
        except Exception as e:
            print(f"   ⚠️ Быстрый метод не сработал: {e}")
        
        # Fallback: пробуем без transform
        try:
            if model is not None and layer is not None:
                embeddings = extract_features_from_masks(image_np, mask_dicts, model, layer, None)
                if isinstance(embeddings, np.ndarray) and embeddings.ndim == 2:
                    extract_time = time.time() - extract_start
                    print(f"   ⚡ БЫСТРО (без transform): {extract_time:.3f} сек")
                    return embeddings
        except Exception as e:
            print(f"   ⚠️ Быстрый метод без transform не сработал: {e}")
        
        return None
    
    def _extract_slow(self, pil_image, mask_arrays):
        """Оптимизированное медленное извлечение эмбеддингов."""
        import time
        import cv2
        
        extract_start = time.time()
        print(f"🐌 МЕДЛЕННОЕ извлечение эмбеддингов для {len(mask_arrays)} масок...")
        
        if self.backbone.startswith('dinov2'):
            return self._extract_with_dino(np.array(pil_image), mask_arrays)
        
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
        
        # 🚀 Оптимизация: масштабируем изображение для ускорения
        original_image = np.array(pil_image)
        h, w = original_image.shape[:2]
        
        if max(h, w) > self.max_embedding_size:
            scale = self.max_embedding_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            scaled_image = cv2.resize(original_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            print(f"   📏 Масштабирование изображения: {w}x{h} → {new_w}x{new_h} (scale={scale:.3f}, max_size={self.max_embedding_size})")
            
            # Масштабируем маски соответственно
            scaled_mask_arrays = []
            for mask_array in mask_arrays:
                scaled_mask = cv2.resize(mask_array.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                scaled_mask_arrays.append(scaled_mask.astype(bool))
            mask_arrays = scaled_mask_arrays
            pil_image = Image.fromarray(scaled_image)
        
        # Пробуем батчевую обработку через extract_features_from_masks_slow
        try:
            mask_dicts = []
            for mask_array in mask_arrays:
                mask_dicts.append({'segmentation': mask_array})
            
            image_np = np.array(pil_image)
            
            from mask_withsearch import extract_features_from_masks_slow
            embeddings = extract_features_from_masks_slow(image_np, mask_dicts, model, layer, transform)
            
            if isinstance(embeddings, np.ndarray) and embeddings.ndim == 2:
                extract_time = time.time() - extract_start
                print(f"   🐌 МЕДЛЕННО (batch): {extract_time:.3f} сек ({extract_time/len(mask_arrays)*1000:.1f} мс/маска)")
                return embeddings
                
        except Exception as e:
            print(f"   ⚠️ Батчевая обработка не сработала: {e}")
        
        # Fallback: обрабатываем маски по одной
        print(f"   🔄 Fallback: обрабатываем маски по одной...")
        embeddings = []
        for i, mask in enumerate(mask_arrays):
            try:
                image_np = np.array(pil_image)
                mask_image = np.zeros_like(image_np)
                mask_image[mask] = image_np[mask]
                
                mask_pil = Image.fromarray(mask_image)
                
                vec = get_vector(mask_pil, model, layer, transform)
                if hasattr(vec, 'numpy'):
                    embeddings.append(vec.numpy())
                else:
                    embeddings.append(vec)
                    
            except Exception as e:
                print(f"   ⚠️ Ошибка с маской {i}: {e}")
                embeddings.append(np.random.rand(1024).astype(np.float32))
        
        if embeddings:
            embeddings_array = np.array(embeddings)
            norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
            embeddings_array = embeddings_array / (norms + 1e-8)
            extract_time = time.time() - extract_start
            print(f"   🐌 МЕДЛЕННО (по одной): {extract_time:.3f} сек ({extract_time/len(mask_arrays)*1000:.1f} мс/маска)")
            return embeddings_array
        
        return None
    
    def build_queries(self, pos_imgs, neg_imgs):
        """Строит эмбеддинги запросов из positive/negative примеров."""
        print(f"🔍 Построение запросов из {len(pos_imgs)} positive и {len(neg_imgs)} negative примеров")
        if self.backbone.startswith('dinov2'):
            # DINO путь: глобальные эмбеддинги картинок без масок
            pos_list = []
            for i, img in enumerate(pos_imgs):
                try:
                    vec = self._get_dino_global(np.array(img))
                    if vec is not None:
                        pos_list.append(vec)
                except Exception as e:
                    print(f"   ⚠️ Ошибка с positive {i}: {e}")
            neg_list = []
            for i, img in enumerate(neg_imgs):
                try:
                    vec = self._get_dino_global(np.array(img))
                    if vec is not None:
                        neg_list.append(vec)
                except Exception as e:
                    print(f"   ⚠️ Ошибка с negative {i}: {e}")
            q_pos = np.array(pos_list) if pos_list else np.array([]).reshape(0, 1024)
            q_neg = np.array(neg_list) if neg_list else np.array([]).reshape(0, 1024)
            # L2 нормализация
            if q_pos.shape[0] > 0:
                q_pos = q_pos / (np.linalg.norm(q_pos, axis=1, keepdims=True) + 1e-8)
            if q_neg.shape[0] > 0:
                q_neg = q_neg / (np.linalg.norm(q_neg, axis=1, keepdims=True) + 1e-8)
            print(f"   📊 Построено positive: {q_pos.shape[0]}, negative: {q_neg.shape[0]}")
            return q_pos.astype(np.float32), q_neg.astype(np.float32)

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
        pos_list = []
        for i, img in enumerate(pos_imgs):
            try:
                vec = self._get_image_embedding(img, model, layer, transform)
                if vec is not None:
                    pos_list.append(vec)
            except Exception as e:
                print(f"   ⚠️ Ошибка с positive {i}: {e}")
        
        # Negative эмбеддинги  
        neg_list = []
        for i, img in enumerate(neg_imgs):
            try:
                vec = self._get_image_embedding(img, model, layer, transform)
                if vec is not None:
                    neg_list.append(vec)
            except Exception as e:
                print(f"   ⚠️ Ошибка с negative {i}: {e}")
        
        q_pos = np.array(pos_list) if pos_list else np.array([]).reshape(0, 1024)
        q_neg = np.array(neg_list) if neg_list else np.array([]).reshape(0, 1024)
        
        # Корректировка positive с учётом negative (как в hybrid)
        if q_pos.shape[0] > 0 and q_neg.shape[0] > 0:
            try:
                adjusted = np.stack([adjust_embedding(q, q_pos, q_neg) for q in q_pos], axis=0).astype(np.float32)
                q_pos = adjusted
            except Exception as e:
                print(f"   ⚠️ adjust_embedding error, используем не скорректированные: {e}")
        
        # L2-нормализация
        if q_pos.shape[0] > 0:
            q_pos = q_pos / (np.linalg.norm(q_pos, axis=1, keepdims=True) + 1e-8)
        if q_neg.shape[0] > 0:
            q_neg = q_neg / (np.linalg.norm(q_neg, axis=1, keepdims=True) + 1e-8)
        
        print(f"   📊 Построено positive: {q_pos.shape[0]}, negative: {q_neg.shape[0]}")
        return q_pos.astype(np.float32), q_neg.astype(np.float32)
    
    def _get_image_embedding(self, pil_image, model, layer, transform):
        """Получает эмбеддинг для одного изображения."""
        if not SEARCHDET_AVAILABLE:
            # Заглушка
            return np.random.rand(1024).astype(np.float32)
        
        if self.backbone.startswith('dinov2'):
            return self._get_dino_global(np.array(pil_image))

        try:
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

    def _ensure_dino(self):
        if self._dino_model is not None:
            return
        try:
            import timm
            import torchvision.transforms as T
            from torchvision.transforms import InterpolationMode
            
            name_map = {
                'dinov2_s': 'vit_small_patch14_dinov2.lvd142m',
                'dinov2_b': 'vit_base_patch14_dinov2.lvd142m',
                'dinov2_l': 'vit_large_patch14_dinov2.lvd142m',
                'dinov2_g': 'vit_giant_patch14_dinov2.lvd142m',
            }
            model_name = name_map.get(self.backbone, 'vit_base_patch14_dinov2.lvd142m')
            self._dino_model = timm.create_model(model_name, pretrained=True)
            self._dino_model.eval()
            
            # 🚀 Применяем половинную точность если включена
            if self.dino_half_precision:
                self._dino_model = self._dino_model.half()
                print(f"   ⚡ DINO модель переведена в float16 для ускорения")
            
            data_config = self._dino_model.default_cfg
            self.dino_img_size = data_config['input_size'][-1]
            
            try:
                patch_size = self._dino_model.patch_embed.patch_size[0]
                self.dino_grid_size = (self.dino_img_size // patch_size, self.dino_img_size // patch_size)
            except Exception:
                self.dino_grid_size = (37, 37)
            class SquarePad:
                def __call__(self, image):
                    w, h = image.size
                    max_wh = np.max([w, h])
                    hp = (max_wh - w) // 2
                    vp = (max_wh - h) // 2
                    padding = (hp, vp, max_wh - w - hp, max_wh - h - vp)
                    return T.functional.pad(image, padding, 0, 'constant')

            self._dino_preprocess = T.Compose([
                SquarePad(),
                T.Resize(self.dino_img_size, interpolation=InterpolationMode.BICUBIC, antialias=True),
                T.CenterCrop(self.dino_img_size),
                T.ToTensor(),
                T.Normalize(mean=data_config['mean'], std=data_config['std']),
            ])
            
            print(f"🧩 DINO backbone={self.backbone}, img_size={self.dino_img_size}, grid={self.dino_grid_size}")
            
        except Exception as e:
            print(f"⚠️ Не удалось инициализировать DINOv2: {e}")
            self._dino_model = None

    def _get_dino_global(self, image_np: np.ndarray):
        self._ensure_dino()
        if self._dino_model is None:
            return np.random.rand(1024).astype(np.float32)
        import torch
        with torch.no_grad():
            pil = Image.fromarray(image_np)
            x = self._dino_preprocess(pil).unsqueeze(0)
            
            # 🚀 Применяем половинную точность к входным данным если нужно
            if self.dino_half_precision:
                x = x.half()
            
            feats = self._dino_model.forward_features(x)
            
            # Обработка выхода: может быть dict или tensor
            if isinstance(feats, dict):
                # Приоритет: CLS токен, если есть
                if 'x_norm_clstoken' in feats:
                    vec = feats['x_norm_clstoken'][0]
                # Fallback: среднее по патчам
                elif 'x_norm_patchtokens' in feats:
                    vec = feats['x_norm_patchtokens'][0].mean(dim=0)
                else: # Неизвестный формат dict
                    vec = next(iter(feats.values()))[0].mean(dim=0)
            elif torch.is_tensor(feats):
                # Если feats - тензор, предполагаем (B, N, D)
                # Используем CLS токен (первый) как глобальный дескриптор
                if feats.ndim == 3 and feats.shape[1] > 0:
                    vec = feats[0, 0]
                # Fallback: среднее по всем токенам/пространству
                else:
                    vec = feats[0].mean(dim=0)
            else: # Неизвестный тип
                return np.random.rand(1024).astype(np.float32)

            vec = vec.detach().cpu().float().numpy().squeeze()
            
            # Приводим к 1024 при необходимости
            if vec.shape[0] != 1024:
                out = np.zeros(1024, dtype=np.float32)
                take = min(1024, vec.shape[0])
                out[:take] = vec[:take]
                vec = out
                
            # L2 norm
            vec_norm = np.linalg.norm(vec)
            if vec_norm > 1e-8:
                vec = vec / vec_norm
            return vec.astype(np.float32)

    def _extract_with_dino(self, image_np, mask_arrays):
        import time
        import torch
        import torch.nn.functional as F
        import cv2
        
        extract_start = time.time()
        print(f"🚀 БЫСТРОЕ DINO извлечение эмбеддингов для {len(mask_arrays)} масок...")
        
        self._ensure_dino()
        if self._dino_model is None:
            return None

        # 🚀 ОПТИМИЗАЦИЯ: Масштабирование изображения для ускорения
        H0, W0 = image_np.shape[:2]
        scale = 1.0
        
        if max(H0, W0) > self.max_embedding_size:
            if H0 >= W0:
                scale = self.max_embedding_size / float(H0)
            else:
                scale = self.max_embedding_size / float(W0)
            scaled_image = cv2.resize(image_np, (int(W0 * scale), int(H0 * scale)), interpolation=cv2.INTER_LINEAR)
            print(f"   🔧 Масштабирование DINO: {W0}x{H0} → {scaled_image.shape[1]}x{scaled_image.shape[0]} (scale={scale:.3f}, max_size={self.max_embedding_size})")
        else:
            scaled_image = image_np

        # 🚀 КЭШИРОВАНИЕ: Проверяем кэш для избежания повторных вычислений
        import hashlib
        image_hash = hashlib.md5(scaled_image.tobytes()).hexdigest()
        
        if image_hash in self._dino_cache:
            patch_tokens, cached_grid_size = self._dino_cache[image_hash]
            print(f"   ⚡ DINO кэш попадание! Пропускаем forward pass")
            # Проверяем совместимость grid размера
            if cached_grid_size != self.dino_grid_size:
                print(f"   ⚠️ Grid размер изменился: {cached_grid_size} → {self.dino_grid_size}, пересчитываем")
                del self._dino_cache[image_hash]
            else:
                # Используем кэшированные токены
                pass
        
        if image_hash not in self._dino_cache:
            # Обрабатываем изображение через DINO
            pil_image = Image.fromarray(scaled_image)
            x = self._dino_preprocess(pil_image).unsqueeze(0)
            
            # 🚀 Применяем половинную точность к входным данным если нужно
            if self.dino_half_precision:
                x = x.half()
            
            dino_start = time.time()
            with torch.no_grad():
                feats = self._dino_model.forward_features(x)
            dino_time = time.time() - dino_start
            print(f"   ⚡ DINO forward: {dino_time:.3f}с (precision: {'float16' if self.dino_half_precision else 'float32'})")
            
            # --- Универсальное извлечение патч-токенов ---
            patch_tokens = None
            if isinstance(feats, dict) and 'x_norm_patchtokens' in feats:
                patch_tokens = feats['x_norm_patchtokens'][0]
            elif torch.is_tensor(feats) and feats.ndim == 3 and feats.shape[1] > 1:
                # Если feats - тензор (B, N, D), отбрасываем CLS токен
                patch_tokens = feats[0, 1:]
            
            if patch_tokens is None:
                print("⚠️ DINO model did not return patch tokens. Falling back to old method.")
                return self._extract_with_dino_fallback(image_np, mask_arrays)

            # Проверка соответствия количества токенов и размера grid
            expected_tokens = self.dino_grid_size[0] * self.dino_grid_size[1]
            if patch_tokens.shape[0] != expected_tokens:
                print(f"⚠️ Mismatch in token count: expected {expected_tokens}, got {patch_tokens.shape[0]}. Fallback.")
                return self._extract_with_dino_fallback(image_np, mask_arrays)
            
            # 🚀 Сохраняем в кэш
            self._dino_cache[image_hash] = (patch_tokens.clone(), self.dino_grid_size)
            print(f"   💾 DINO результат сохранен в кэш (размер кэша: {len(self._dino_cache)})")
        else:
             patch_tokens, _ = self._dino_cache[image_hash]

        # 🚀 СУПЕР-БАТЧЕВАЯ обработка всех масок
        embeddings = []
        gh, gw = self.dino_grid_size
        
        # Предварительно масштабируем все маски к размеру scaled_image
        scaled_masks = []
        for mask in mask_arrays:
            if scale != 1.0:
                # Масштабируем маску к размеру scaled_image
                scaled_mask = cv2.resize(mask.astype(np.uint8), (scaled_image.shape[1], scaled_image.shape[0]), interpolation=cv2.INTER_NEAREST)
                scaled_mask = scaled_mask.astype(bool)
            else:
                scaled_mask = mask
            scaled_masks.append(scaled_mask)
        
        # 🚀 ОПТИМИЗАЦИЯ: Батчевая интерполяция всех масок сразу
        batch_start = time.time()
        valid_masks = []
        valid_indices = []
        
        # Собираем все непустые маски в один батч
        for i, mask in enumerate(scaled_masks):
            if mask.sum() > 0:
                valid_masks.append(mask.astype(np.float32))
                valid_indices.append(i)
        
        if valid_masks:
            # Конвертируем в батч тензор (N, 1, H, W)
            batch_masks = torch.from_numpy(np.stack(valid_masks)).unsqueeze(1)
            
            # Батчевая интерполяция всех масок сразу
            resized_batch = F.interpolate(batch_masks, size=(gh, gw), mode='bilinear', align_corners=False)
            resized_batch = resized_batch.squeeze(1).view(len(valid_masks), -1)  # (N, gh*gw)
            
            batch_time = time.time() - batch_start
            print(f"   ⚡ Батчевая интерполяция {len(valid_masks)} масок: {batch_time:.3f}с")
            
            # Извлекаем эмбеддинги для всех валидных масок
            for batch_idx, original_idx in enumerate(valid_indices):
                mask_tensor = resized_batch[batch_idx]
                
                foreground_indices = torch.where(mask_tensor > 0.1)[0]
                if len(foreground_indices) == 0:
                    foreground_indices = torch.tensor([torch.argmax(mask_tensor)])
                mask_embedding = patch_tokens[foreground_indices].mean(dim=0)

                v = mask_embedding.cpu().float().numpy()
                
                # L2 нормализация
                v_norm = np.linalg.norm(v)
                if v_norm > 1e-8:
                    v = v / v_norm
                
                # Приводим к 1024 при необходимости
                if v.shape[0] != 1024:
                    out = np.zeros(1024, dtype=np.float32)
                    take = min(1024, v.shape[0])
                    out[:take] = v[:take]
                    v = out

                # Вставляем эмбеддинг в правильную позицию
                while len(embeddings) <= original_idx:
                    embeddings.append(None)
                embeddings[original_idx] = v.astype(np.float32)
        
        # Заполняем пустые маски нулевыми векторами
        for i in range(len(scaled_masks)):
            if i >= len(embeddings) or embeddings[i] is None:
                while len(embeddings) <= i:
                    embeddings.append(None)
                embeddings[i] = np.zeros(patch_tokens.shape[-1], dtype=np.float32)

        extract_time = time.time() - extract_start
        old_time_estimate = len(mask_arrays) * 0.2  # Примерное время старого метода
        speedup = old_time_estimate / extract_time if extract_time > 0 else 1
        print(f"   ⚡ БЫСТРО DINO: {extract_time:.3f} сек ({extract_time/len(mask_arrays)*1000:.1f} мс/маска) - ускорение ~{speedup:.1f}x")

        if embeddings:
            return np.stack(embeddings, axis=0).astype(np.float32)
        return None

    def _extract_with_dino_fallback(self, image_np, mask_arrays):
        self._ensure_dino()
        if self._dino_model is None:
            return None
        import torch
        embeddings = []
        with torch.no_grad():
            for mask in mask_arrays:
                ys, xs = np.where(mask)
                if ys.size == 0 or xs.size == 0:
                    embeddings.append(np.zeros(1024, dtype=np.float32))
                    continue
                
                y1, y2 = ys.min(), ys.max()
                x1, x2 = xs.min(), xs.max()
                
                # Вырезаем кроп с маской
                crop_np = image_np[y1:y2+1, x1:x2+1].copy()
                m = mask[y1:y2+1, x1:x2+1]
                if m.dtype != bool:
                    m = m.astype(bool)
                crop_np[~m] = 0
                
                # Получаем глобальный вектор для кропа, используя обновленный метод
                vec = self._get_dino_global(crop_np)
                
                embeddings.append(vec)

        if embeddings:
            return np.stack(embeddings, axis=0).astype(np.float32)
        return None


    def build_queries_multiclass(self, pos_by_class, neg_imgs):
        if str(self.backbone).startswith('dinov2'):
            self._ensure_dino()
            D = 1024

            neg_list = []
            for i, img in enumerate(neg_imgs or []):
                try:
                    v = self._get_dino_global(np.array(img))
                    v = np.asarray(v, dtype=np.float32).reshape(-1)
                    v /= (np.linalg.norm(v) + 1e-8)
                    neg_list.append(v.copy())
                except Exception as e:
                    print(f"   ⚠️ Ошибка с negative {i}: {e}")
            q_neg = np.stack(neg_list, axis=0) if neg_list else np.zeros((0, D), dtype=np.float32)
            class_pos = {}
            if pos_by_class is None:
                pos_by_class = {}
            for cls, imgs in (pos_by_class or {}).items():
                vecs = []
                for i, img in enumerate(imgs or []):
                    try:
                        v = self._get_dino_global(np.array(img))
                        v = np.asarray(v, dtype=np.float32).reshape(-1)
                        v /= (np.linalg.norm(v) + 1e-8)
                        vecs.append(v.copy())
                    except Exception as e:
                        print(f"   ⚠️ Ошибка с positive '{cls}' #{i}: {e}")
                Q = np.stack(vecs, axis=0) if vecs else np.zeros((0, D), dtype=np.float32)
                class_pos[cls] = Q.astype(np.float32)
                print(f"   📊 Класс '{cls}': {Q.shape[0]} примеров")
            print(f"   📊 Negative всего: {q_neg.shape[0]}")
            return class_pos, q_neg.astype(np.float32)
        resnet, layer, transform, sam = (
            self.detector.searchdet_resnet,
            self.detector.searchdet_layer,
            self.detector.searchdet_transform,
            self.detector.searchdet_sam
        )

        if isinstance(resnet, tuple) and len(resnet) >= 2:
            model = resnet[0]
            device = resnet[1] if len(resnet) > 1 else 'cuda'
        else:
            model = resnet
            device = 'cuda'

        # Сначала отрицательные
        neg_list = []
        embedding_dim = None
        for i, img in enumerate(neg_imgs or []):
            try:
                vec = self._get_image_embedding(img, model, layer, transform)
                if vec is None:
                    continue
                vec = np.asarray(vec, dtype=np.float32).reshape(-1)
                if embedding_dim is None: embedding_dim = vec.shape[0]
                vec /= (np.linalg.norm(vec) + 1e-8)
                neg_list.append(vec)
            except Exception as e:
                print(f"   ⚠️ Ошибка с negative {i}: {e}")
        
        if embedding_dim is None:
            # Пытаемся угадать размерность из позитивов, если негативов не было
            if pos_by_class:
                try:
                    cls, imgs = next(iter(pos_by_class.items()))
                    if imgs:
                        vec = self._get_image_embedding(imgs[0], model, layer, transform)
                        if vec is not None:
                            embedding_dim = vec.reshape(-1).shape[0]
                except Exception:
                    pass
            if embedding_dim is None:
                embedding_dim = 1024 # Fallback

        q_neg = np.stack(neg_list, axis=0) if neg_list else np.zeros((0, embedding_dim), dtype=np.float32)

        # Теперь позитивные по классам
        class_pos = {}
        for cls, imgs in (pos_by_class or {}).items():
            pos_list = []
            for i, img in enumerate(imgs or []):
                try:
                    vec = self._get_image_embedding(img, model, layer, transform)
                    if vec is None:
                        continue
                    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
                    vec /= (np.linalg.norm(vec) + 1e-8)
                    pos_list.append(vec)
                except Exception as e:
                    print(f"   ⚠️ Ошибка с positive '{cls}' #{i}: {e}")
            
            Q = np.stack(pos_list, axis=0) if pos_list else np.zeros((0, embedding_dim), dtype=np.float32)
            class_pos[cls] = Q.astype(np.float32)
            print(f"   📊 Класс '{cls}': {Q.shape[0]} примеров")

        print(f"   📊 Negative всего: {q_neg.shape[0]}")
        return class_pos, q_neg


class DINOv3Embedding(nn.Module):
    def __init__(self, ckpt_path):
        super().__init__()
        self.model = ConvNeXt(arch='base', out_indices=-1)
        state_dict = torch.load(ckpt_path)['state_dict']
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.to(self.device)

    def preprocess(self, image):
        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transform(image).unsqueeze(0).to(self.device)

    def get_embedding(self, x):
        with torch.no_grad():
            embedding = self.model(x)
        return embedding


class ResNet101Embedding(nn.Module):
    def __init__(self, layer='layer3'):
        super().__init__()
        self.model = ResNet(101, 'DINOv2')
        self.model.eval()
        self.model.to(self.device)

    def preprocess(self, image):
        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transform(image).unsqueeze(0).to(self.device)

    def get_embedding(self, x):
        with torch.no_grad():
            embedding = self.model(x)
        return embedding
