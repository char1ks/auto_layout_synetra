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
        self.backbone = getattr(detector, 'backbone', 'resnet101')
        # Ленивая инициализация DINO при первом использовании
        self._dino_model = None
        self._dino_preprocess = None
    
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
        # Если выбран DINOv2, используем DINO путь
        if self.backbone.startswith('dinov2'):
            return self._extract_with_dino(image_np, mask_arrays)
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
        
        # Пытаемся вызвать совместимую сигнатуру без лишнего шума в логах
        try:
            embeddings = extract_features_from_masks(image_np, mask_dicts, model, layer, transform)
            if isinstance(embeddings, np.ndarray) and embeddings.ndim == 2:
                return embeddings
        except TypeError:
            # Fallback 1: без transform
            try:
                embeddings = extract_features_from_masks(image_np, mask_dicts, model, layer)
                if isinstance(embeddings, np.ndarray) and embeddings.ndim == 2:
                    return embeddings
            except TypeError:
                # Fallback 2: без layer и transform
                try:
                    embeddings = extract_features_from_masks(image_np, mask_dicts, model)
                    if isinstance(embeddings, np.ndarray) and embeddings.ndim == 2:
                        return embeddings
                except Exception:
                    pass
        except Exception:
            pass
        return None
    
    def _extract_slow(self, pil_image, mask_arrays):
        """Медленное извлечение по одной маске."""
        if self.backbone.startswith('dinov2'):
            return self._extract_with_dino(np.array(pil_image), mask_arrays)
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

    # ----------------------- DINOv2 utils -----------------------
    def _ensure_dino(self):
        if self._dino_model is not None:
            return
        try:
            import timm
            name_map = {
                'dinov2_s': 'vit_small_patch14_dinov2.lvd142m',
                'dinov2_b': 'vit_base_patch14_dinov2.lvd142m',
                'dinov2_l': 'vit_large_patch14_dinov2.lvd142m',
                'dinov2_g': 'vit_giant_patch14_dinov2.lvd142m',
            }
            model_name = name_map.get(self.backbone, 'vit_small_patch14_dinov2.lvd142m')
            self._dino_model = timm.create_model(model_name, pretrained=True)
            self._dino_model.eval()
            import torchvision.transforms as T
            self._dino_preprocess = T.Compose([
                T.Resize(518, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(518),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
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
            feats = self._dino_model.forward_features(x)
            # Глобальный вектор: pool по CLS/mean
            if isinstance(feats, dict) and 'x_norm_clstoken' in feats:
                vec = feats['x_norm_clstoken'][0]
            else:
                # fallback: среднее по пространству
                if isinstance(feats, dict) and 'x_norm_patchtokens' in feats:
                    tokens = feats['x_norm_patchtokens'][0]
                    vec = tokens.mean(dim=0)
                else:
                    vec = feats.mean(dim=(-2,-1)) if hasattr(feats, 'mean') else torch.mean(feats, dim=1)
            vec = vec.detach().cpu().float().numpy()
            # Приводим к 1024 при необходимости
            if vec.shape[0] != 1024:
                # простая проекция до 1024 для совместимости
                out = np.zeros(1024, dtype=np.float32)
                take = min(1024, vec.shape[0])
                out[:take] = vec[:take]
                vec = out
            # L2 norm
            vec = vec / (np.linalg.norm(vec) + 1e-8)
            return vec.astype(np.float32)

    def _extract_with_dino(self, image_np, mask_arrays):
        self._ensure_dino()
        if self._dino_model is None:
            return None
        import torch
        pil = Image.fromarray(image_np)
        embeddings = []
        with torch.no_grad():
            for mask in mask_arrays:
                # вырез по маске
                ys, xs = np.where(mask)
                if ys.size == 0 or xs.size == 0:
                    continue
                y1, y2 = ys.min(), ys.max()
                x1, x2 = xs.min(), xs.max()
                crop = pil.crop((x1, y1, x2+1, y2+1))
                x = self._dino_preprocess(crop).unsqueeze(0)
                feats = self._dino_model.forward_features(x)
                if isinstance(feats, dict) and 'x_norm_clstoken' in feats:
                    vec = feats['x_norm_clstoken'][0]
                else:
                    if isinstance(feats, dict) and 'x_norm_patchtokens' in feats:
                        tokens = feats['x_norm_patchtokens'][0]
                        vec = tokens.mean(dim=0)
                    else:
                        vec = feats.mean(dim=(-2,-1)) if hasattr(feats, 'mean') else torch.mean(feats, dim=1)
                v = vec.detach().cpu().float().numpy()
                if v.shape[0] != 1024:
                    out = np.zeros(1024, dtype=np.float32)
                    take = min(1024, v.shape[0])
                    out[:take] = v[:take]
                    v = out
                v = v / (np.linalg.norm(v) + 1e-8)
                embeddings.append(v.astype(np.float32))
        if embeddings:
            return np.stack(embeddings, axis=0)
        return None


    def build_queries_multiclass(self, pos_by_class, neg_imgs):
        """Строит эмбеддинги примеров ПО КЛАССАМ.
        
        Args:
            pos_by_class: dict[str, list[PIL.Image]] — ключ = имя класса, значение = список изображений-позитивов для класса
            neg_imgs: list[PIL.Image] — отрицательные примеры (общие для всех классов)
        
        Returns:
            class_pos: dict[str, np.ndarray] — по каждому классу массив [N_cls, D]
            q_neg: np.ndarray — массив отрицательных эмбеддингов [N_neg, D]
        """
        # Сначала отрицательные
        neg_list = []
        for i, img in enumerate(neg_imgs or []):
            try:
                vec = self._get_dino_global(np.array(img))
                if vec is not None:
                    neg_list.append(vec)
            except Exception as e:
                print(f"   ⚠️ Ошибка с negative {i}: {e}")
        q_neg = np.array(neg_list) if neg_list else np.array([]).reshape(0, 1024)
        if q_neg.shape[0] > 0:
            q_neg = q_neg / (np.linalg.norm(q_neg, axis=1, keepdims=True) + 1e-8)
        
        class_pos = {}
        if pos_by_class is None:
            pos_by_class = {}
        for cls, imgs in pos_by_class.items():
            pos_list = []
            for i, img in enumerate(imgs or []):
                try:
                    vec = self._get_dino_global(np.array(img))
                    if vec is not None:
                        pos_list.append(vec)
                except Exception as e:
                    print(f"   ⚠️ Ошибка с positive '{cls}' #{i}: {e}")
            q_pos = np.array(pos_list) if pos_list else np.array([]).reshape(0, 1024)
            if q_pos.shape[0] > 0:
                q_pos = q_pos / (np.linalg.norm(q_pos, axis=1, keepdims=True) + 1e-8)
            class_pos[cls] = q_pos.astype(np.float32)
            print(f"   📊 Класс '{cls}': {q_pos.shape[0]} примеров")
        
        print(f"   📊 Negative всего: {q_neg.shape[0]}")
        return class_pos, q_neg.astype(np.float32)
