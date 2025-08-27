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
import os
try:
    from mask_withsearch import get_vector, adjust_embedding, extract_features_from_masks
    SEARCHDET_AVAILABLE = True
except Exception as e:
    print(f"⚠️ SearchDet недоступен: {e}")
    SEARCHDET_AVAILABLE = False

import torch.nn as nn
import time


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
        # DINOv3 ConvNeXt-B поля
        # Используем предзагруженную модель из детектора если доступна
        self._dinov3_model = getattr(detector, 'dinov3_model', None)
        self._dinov3_preprocess = None
        self.dinov3_checkpoint_path = getattr(detector, 'dinov3_checkpoint_path', None)
        
        # Устройство для DINOv3 (CPU/GPU)
        self._dinov3_device = getattr(detector, 'dinov3_device', None)
    
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
        import os
        # === DINOv3 ConvNeXt-B: специальная ветка ===
        if self.backbone.startswith('dinov3'):
            return self._extract_with_dinov3_convnext(image_np, mask_arrays)
        
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
        
        if self.backbone.startswith('dinov3'):
            return self._extract_with_dinov3_convnext(np.array(pil_image), mask_arrays)
        
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
        
        if self.backbone.startswith('dinov3'):
            return self._get_dinov3_global(np.array(pil_image))
        
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

    def _extract_with_dino(self, image_np, valid_masks):
        import torch.nn.functional as F
        
        self._ensure_dino()
        if self._dino_model is None:
            return np.zeros((0, 1024), dtype=np.float32)

        image_pil = Image.fromarray(image_np)
        image_hash = hash(image_pil.tobytes())

        # === Кэширование ===
        if image_hash in self._dino_cache:
            patch_tokens, grid_size = self._dino_cache[image_hash]
            gh, gw = grid_size
        else:
            with torch.no_grad():
                # 🚀 Используем кастомный forward_features
                features_dict = self._dino_model.forward_features(self._dino_preprocess(image_pil).unsqueeze(0).to(self._dino_device))
                patch_tokens = features_dict['x_norm_patchtokens'].squeeze(0) # (T, D)
                
                # Нормализуем патч-токены перед кэшированием
                if isinstance(patch_tokens, torch.Tensor):
                    patch_tokens = F.normalize(patch_tokens.float(), p=2, dim=-1)
                else:
                    patch_tokens = torch.from_numpy(patch_tokens).float()
                    patch_tokens = F.normalize(patch_tokens, p=2, dim=-1)

                gh, gw = features_dict['hw']
                self.dino_grid_size = (gh, gw)
                self._dino_cache[image_hash] = (patch_tokens.clone(), self.dino_grid_size)

        # === Батчевая обработка масок ===
        mask_tensors = [torch.from_numpy(m.astype(np.float32)).unsqueeze(0).unsqueeze(0) for m in valid_masks]
        batch_masks = torch.cat(mask_tensors, dim=0).to(self._dino_device)
        
        # БЫЛО: 'bilinear', что размывает маску. СТАЛО: 'nearest'
        resized_batch = F.interpolate(batch_masks, size=(gh, gw), mode='nearest')
        mask_bool = (resized_batch.squeeze(1) > 0.5) # [N, gh, gw]

        embeddings = []
        for batch_idx in range(len(valid_masks)):
            
            foreground_indices = torch.where(mask_bool[batch_idx].flatten())[0]
            if len(foreground_indices) == 0:
                # Защита на случай тонких масок — берём максимум по вероятности пикселя
                flat = resized_batch.squeeze(1)[batch_idx].flatten()
                foreground_indices = torch.tensor([int(torch.argmax(flat))], device=flat.device)

            if len(foreground_indices) > 0:
                mask_patch_tokens = patch_tokens[foreground_indices]
                embedding = mask_patch_tokens.mean(dim=0)
            else:
                # Fallback: если маска пуста, используем CLS токен (глобальный)
                embedding = patch_tokens.mean(dim=0) # Усредняем все патчи как fallback

            # Финальная L2 нормализация
            embedding = F.normalize(embedding.float(), p=2, dim=0)
            embeddings.append(embedding.cpu().numpy())

        return np.array(embeddings).astype(np.float32)

    # Modify negative embeddings to use central region instead of global average
    def _get_dinov3_central(self, image_np):
        """Получает центральный эмбеддинг для изображения."""
        if self._dinov3_model is None:
            raise ValueError("DINOv3 модель не инициализирована")
    
        # Преобразование изображения в PIL перед обработкой
        if isinstance(image_np, np.ndarray):
            image_pil = Image.fromarray(image_np)
        else:
            raise TypeError(f"Unexpected type {type(image_np)}")
    
        # Преобразование изображения
        x = self._dinov3_preprocess(image_pil).unsqueeze(0).to(self._dinov3_device)
    
        # Получение фичей
        feats = self._dinov3_model.forward_features(x)
    
        # Центральный регион
        h, w = feats.shape[-2:]
        center_h, center_w = h // 2, w // 2
        central_feats = feats[:, :, center_h - 1:center_h + 2, center_w - 1:center_w + 2]
    
        # Агрегация
        central_embedding = central_feats.mean(dim=[-2, -1])
        return central_embedding.py()

    # Update build_queries_multiclass to use central embedding for negatives
    def build_queries_multiclass(self, pos_by_class, neg_imgs):
        if str(self.backbone).startswith('dinov3'):
            self._ensure_dinov3_convnext()
            D = 1024

            # NEGATIVE
            neg_list = []
            for img in (neg_imgs or []):
                v = self._get_dinov3_central(np.array(img))  # Use central embedding
                v = v.astype(np.float32); v /= (np.linalg.norm(v)+1e-8)
                neg_list.append(v)
            q_neg = np.stack(neg_list, axis=0) if neg_list else np.zeros((0,D), np.float32)

            # POSITIVE
            class_pos = {}
            for cls, imgs in (pos_by_class or {}).items():
                vecs = []
                for img in (imgs or []):
                    v = self._get_dinov3_global(np.array(img))
                    v = v.astype(np.float32); v /= (np.linalg.norm(v)+1e-8)
                    vecs.append(v)
                Q = np.stack(vecs, axis=0) if vecs else np.zeros((0,D), np.float32)
                class_pos[cls] = Q

            # --- Фильтр негативов, похожих на позитивы ---
            pos_stack = []
            for cls2, Q2 in (class_pos or {}).items():
                if Q2 is not None and Q2.size:
                    pos_stack.append(Q2.astype(np.float32))
            if len(pos_stack):
                import os
                P = np.vstack(pos_stack)
                P /= (np.linalg.norm(P, axis=1, keepdims=True) + 1e-8)
                if q_neg.shape[0] > 0:
                    qn = q_neg.astype(np.float32)
                    qn /= (np.linalg.norm(qn, axis=1, keepdims=True) + 1e-8)
                    sims_np = qn @ P.T
                    thr = float(os.getenv('SEARCHDET_NEG_FILTER_THR', '0.60'))
                    keep = (np.max(sims_np, axis=1) <= thr)
                    dropped = int((~keep).sum())
                    if dropped:
                        print(f"   🧹 Убрано {dropped} негативов (слишком похожи на позитивы; thr={thr:.2f})")
                    q_neg = q_neg[keep]

            return class_pos, q_neg
        
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

        # Сначала отрицательные - используем МНОЖЕСТВЕННЫЕ СЛУЧАЙНЫЕ ОБЛАСТИ вместо глобального эмбеддинга
        neg_list = []
        embedding_dim = None
        for i, img in enumerate(neg_imgs or []):
            try:
                # Генерируем несколько случайных областей из negative изображения
                neg_regions = self._extract_random_regions_from_image(img, num_regions=5)
                for region_vec in neg_regions:
                    if region_vec is None:
                        continue
                    vec = np.asarray(region_vec, dtype=np.float32).reshape(-1)
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

        # Теперь позитивные по классам - используем ЦЕНТРАЛЬНЫЕ ОБЛАСТИ вместо глобального эмбеддинга
        class_pos = {}
        for cls, imgs in (pos_by_class or {}).items():
            pos_list = []
            for i, img in enumerate(imgs or []):
                try:
                    # Для positive примеров используем центральную область (предполагаем что объект в центре)
                    pos_regions = self._extract_central_region_from_image(img)
                    for region_vec in pos_regions:
                        if region_vec is None:
                            continue
                        vec = np.asarray(region_vec, dtype=np.float32).reshape(-1)
                        vec /= (np.linalg.norm(vec) + 1e-8)
                        pos_list.append(vec)
                except Exception as e:
                    print(f"   ⚠️ Ошибка с positive '{cls}' #{i}: {e}")
            
            Q = np.stack(pos_list, axis=0) if pos_list else np.zeros((0, embedding_dim), dtype=np.float32)
            class_pos[cls] = Q.astype(np.float32)
            print(f"   📊 Класс '{cls}': {Q.shape[0]} примеров")

        print(f"   📊 Negative всего: {q_neg.shape[0]}")
        return class_pos, q_neg

    def _extract_random_regions_from_image(self, pil_image, num_regions=5):
        """Извлекает эмбеддинги из случайных областей изображения для более справедливого сравнения с масками."""
        import random
        import numpy as np
        
        # Конвертируем PIL в numpy
        image_np = np.array(pil_image)
        h, w = image_np.shape[:2]
        
        # Размер случайной области (примерно как средняя маска)
        region_size = min(h, w) // 4  # Четверть от минимального размера
        region_size = max(32, region_size)  # Минимум 32 пикселя
        
        regions = []
        for _ in range(num_regions):
            # Случайные координаты для области
            x = random.randint(0, max(0, w - region_size))
            y = random.randint(0, max(0, h - region_size))
            
            # Создаем маску для этой области
            mask = np.zeros((h, w), dtype=bool)
            mask[y:y+region_size, x:x+region_size] = True
            
            try:
                # Извлекаем эмбеддинг для этой области как для обычной маски
                if self.backbone.startswith('dinov3'):
                    region_emb = self._extract_with_dinov3_convnext(image_np, [mask])
                elif self.backbone.startswith('dinov2'):
                    region_emb = self._extract_with_dino(image_np, [mask])
                else:
                    # Для ResNet используем старый метод с маской
                    region_emb = self._extract_slow(pil_image, [mask])
                
                if region_emb is not None and len(region_emb) > 0:
                    regions.append(region_emb[0])  # Берем первый (и единственный) эмбеддинг
            except Exception as e:
                print(f"   ⚠️ Ошибка извлечения региона: {e}")
                continue
        
        return regions

    def _extract_central_region_from_image(self, pil_image):
        """Извлекает эмбеддинг из центральной области изображения для positive примеров."""
        import numpy as np
        
        # Конвертируем PIL в numpy
        image_np = np.array(pil_image)
        h, w = image_np.shape[:2]
        
        # Размер центральной области (60% от изображения)
        region_h = int(h * 0.6)
        region_w = int(w * 0.6)
        
        # Центрируем область
        start_y = (h - region_h) // 2
        start_x = (w - region_w) // 2
        
        # Создаем маску для центральной области
        mask = np.zeros((h, w), dtype=bool)
        mask[start_y:start_y+region_h, start_x:start_x+region_w] = True
        
        regions = []
        try:
            # Извлекаем эмбеддинг для центральной области
            if self.backbone.startswith('dinov3'):
                region_emb = self._extract_with_dinov3_convnext(image_np, [mask])
            elif self.backbone.startswith('dinov2'):
                region_emb = self._extract_with_dino(image_np, [mask])
            else:
                # Для ResNet используем старый метод с маской
                region_emb = self._extract_slow(pil_image, [mask])
            
            if region_emb is not None and len(region_emb) > 0:
                regions.append(region_emb[0])  # Берем первый (и единственный) эмбеддинг
        except Exception as e:
            print(f"   ⚠️ Ошибка извлечения центральной области: {e}")
            # Fallback к глобальному эмбеддингу если не получилось
            try:
                resnet, layer, transform, sam = (
                    self.detector.searchdet_resnet,
                    self.detector.searchdet_layer,
                    self.detector.searchdet_transform,
                    self.detector.searchdet_sam
                )
                if isinstance(resnet, tuple) and len(resnet) >= 2:
                    model = resnet[0]
                else:
                    model = resnet
                
                vec = self._get_image_embedding(pil_image, model, layer, transform)
                if vec is not None:
                    regions.append(vec)
            except Exception:
                pass
        
        return regions


    # =========================
    # DINOv3 ConvNeXt-B support
    # =========================
    def _ensure_dinov3_convnext(self):
        import os, torch, timm
        from torchvision import transforms as T
        from torchvision.transforms import InterpolationMode

        # Если модель уже есть (например, предзагружена детектором), убедимся что выставлен device и препроцессинг
        if self._dinov3_model is not None:
            if self._dinov3_device is None:
                try:
                    self._dinov3_device = next(self._dinov3_model.parameters()).device
                except StopIteration:
                    self._dinov3_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if self._dinov3_preprocess is None:
                img_size = int(os.getenv('SEARCHDET_FEAT_SHORT_SIDE', '224'))
                self._dinov3_preprocess = T.Compose([
                    T.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
                    T.ToTensor(),
                    T.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
                ])
            return

        if not self.dinov3_ckpt:
            raise FileNotFoundError("Укажи --dinov3-ckpt путь к .pth")

        # ConvNeXt-B без классификатора -> model(x) даёт pooled features
        self._dinov3_model = timm.create_model('convnext_base', pretrained=False, num_classes=0)
        sd = torch.load(self.dinov3_ckpt, map_location='cpu')
        if isinstance(sd, dict) and 'model' in sd: sd = sd['model']
        self._dinov3_model.load_state_dict(sd, strict=False)
        self._dinov3_model.eval()

        # Определяем устройство и переносим модель
        self._dinov3_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._dinov3_model.to(self._dinov3_device)

        img_size = int(os.getenv('SEARCHDET_FEAT_SHORT_SIDE', '224'))
        self._dinov3_preprocess = T.Compose([
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ])

    def _get_dinov3_global(self, image_np):
        import torch
        from PIL import Image as PILImage
        self._ensure_dinov3_convnext()
        x = self._dinov3_preprocess(PILImage.fromarray(image_np)).unsqueeze(0).to(self._dinov3_device)
        with torch.no_grad():
            feats = self._dinov3_model.forward_features(x)
            if isinstance(feats, dict):
                feats = feats.get('x', None) or feats.get('features', None)
            if feats.ndim == 3:
                feats = feats.unsqueeze(0)
            assert feats.ndim == 4, "Ожидали [B,C,Hf,Wf] от forward_features"
            fmap = feats[0].detach().cpu().float().numpy().transpose(1, 2, 0)  # (Hf, Wf, C)
            v = fmap.reshape(-1, fmap.shape[-1]).mean(axis=0).astype(np.float32)  # spatial mean
        v /= (np.linalg.norm(v) + 1e-8)
        if v.shape[0] != 1024:
            out = np.zeros(1024, dtype=np.float32)
            out[:min(1024, v.shape[0])] = v[:min(1024, v.shape[0])]
            v = out
        return v.astype(np.float32)


    def _extract_with_dinov3_convnext(self, image_np, mask_arrays):
        import torch, cv2, numpy as np
        from PIL import Image as PILImage

        self._ensure_dinov3_convnext()
        if self._dinov3_model is None:
            return np.zeros((0, 1024), dtype=np.float32)

        # 1) Один прогон полного изображения -> spatial feature map
        x_full = self._dinov3_preprocess(PILImage.fromarray(image_np)).unsqueeze(0).to(self._dinov3_device)
        with torch.no_grad():
            feats = self._dinov3_model.forward_features(x_full)
            # feats: [1, C, Hf, Wf] для ConvNeXt; иногда timm вернёт dict -> приведи к тензору
            if isinstance(feats, dict):
                # timm convnext обычно возвращает тензор; на всякий случай:
                feats = feats.get('x', None) or feats.get('features', None)
            if feats.ndim == 3:  # [C,Hf,Wf]
                feats = feats.unsqueeze(0)
            assert feats.ndim == 4, "Ожидали [B,C,Hf,Wf] от forward_features"

        feats_cpu = feats.detach().cpu()
        B, C, Hf, Wf = feats_cpu.shape
        fmap = feats_cpu[0].float().numpy().transpose(1,2,0)  # (Hf,Wf,C)

        embs = []
        for mask in mask_arrays:
            # 2) Маску ресайзим к размеру фич-карты и усредняем ТОЛЬКО по активным позициям
            m = cv2.resize(mask.astype(np.uint8), (Wf, Hf), interpolation=cv2.INTER_NEAREST).astype(bool)
            if not m.any():
                # Маска пуста после ресайза. Найдем центр масс исходной маски
                # и возьмем вектор из этой точки на карте признаков.
                if np.any(mask):
                    moments = cv2.moments(mask.astype(np.uint8))
                    if moments["m00"] > 0:
                        orig_x = int(moments["m10"] / moments["m00"])
                        orig_y = int(moments["m01"] / moments["m00"])
                    else:
                        # Fallback for very weird masks where moments are zero.
                        orig_y_arr, orig_x_arr = np.where(mask)
                        orig_y, orig_x = int(np.mean(orig_y_arr)), int(np.mean(orig_x_arr))

                    fy = int(orig_y / mask.shape[0] * Hf)
                    fx = int(orig_x / mask.shape[1] * Wf)
                    fy = np.clip(fy, 0, Hf - 1)
                    fx = np.clip(fx, 0, Wf - 1)
                    v = fmap[fy, fx].copy().astype(np.float32)
                else:
                    # Если и исходная маска пуста, то глобальное усреднение
                    v = fmap.reshape(-1, C).mean(axis=0).astype(np.float32)
            else:
                v = fmap[m].mean(axis=0).astype(np.float32)
            # 3) Нормализация L2
            v /= (np.linalg.norm(v) + 1e-8)
            # ConvNeXt-B -> 1024-D
            if v.shape[0] != 1024:
                out = np.zeros(1024, dtype=np.float32)
                out[:min(1024, v.shape[0])] = v[:min(1024, v.shape[0])]
                v = out
            embs.append(v)
        
        # Добавляем отладочный принт для проверки дисперсии
        if embs:
            embs_array = np.stack(embs, axis=0)
            print("DEBUG mask_emb std:", np.std(embs_array, axis=0)[:8], "||", np.std(embs_array))
            return embs_array
        return np.zeros((0,1024), dtype=np.float32)
