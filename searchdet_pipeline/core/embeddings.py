#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Извлечение эмбеддингов масок и примеров для пайплайна.
- DINOv3: через общий энкодер (encode_mask / encode_image)
- DINOv2 / ResNet путь оставлен как fallback (если требуется старыми режимами)
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from PIL import Image
import cv2

# SearchDet (если есть)
try:
    from mask_withsearch import get_vector, adjust_embedding, extract_features_from_masks
    SEARCHDET_AVAILABLE = True
except Exception:
    SEARCHDET_AVAILABLE = False

# === Новый энкодер DINOv3 ===
try:
    from .dinov3_encoder import DinoV3Encoder
    DINOV3_OK = True
except Exception as e:
    print(f"⚠️ DINOv3 encoder unavailable: {e}")
    DINOV3_OK = False


class EmbeddingExtractor:
    """
    detector должен нести:
      - backbone: str (напр. dinov3_vitb16 / dinov3_convnext_base / dinov2_b / resnet101)
      - dinov3_ckpt: Optional[str]
      - device: 'cuda'|'cpu' (опционально)
      - vit_pooling: 'cls'|'mean' (опционально)
      - max_embedding_size: int (масштаб для fallback путей)
    """

    def __init__(self, detector):
        self.detector = detector
        self.backbone = getattr(detector, "backbone", "dinov2_b")
        self.device = getattr(detector, "device", "cuda")
        self.max_embedding_size = getattr(detector, "max_embedding_size", 1024)

        # --- добавлено: DINOv3 encoder (единая точка входа, как в standalone скрипте) ---
        self._d3 = None
        if str(self.backbone).startswith("dinov3"):
            from .dinov3_encoder import DinoV3Encoder
            self._d3 = DinoV3Encoder(
                backbone = self.backbone.replace("dinov3_", ""),  # convnext_base / vitb16 / ...
                ckpt     = getattr(detector, "dinov3_ckpt", None),
                device   = getattr(detector, "encoder_device", "cuda"),
                use_half = bool(getattr(detector, "dino_half_precision", False)),
                loader   = getattr(detector, "loader", "timm"),
                repo_dir = getattr(detector, "repo_dir", None),
            )
            print(f"🧩 Using DINOv3 encoder: {self.backbone} @ {self._d3.device}, half={self._d3.use_half}")

        # Кэш формата: {image_id: np.ndarray} — по желанию можно расширить
        self.cache = {}
        
        # Параметры для генерации query из масок
        self.pos_as_query_masks = bool(getattr(detector, 'pos_as_query_masks', True))
        self.min_mask_area = int(getattr(detector, 'min_mask_area', 100))

    # =========================
    # Маски
    # =========================
    def extract_mask_embeddings(self, image_np: np.ndarray, masks: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[int]]:
        """
        Возвращает:
          - эмбеддинги (N, D) float32
          - индексы валидных масок (исходные индексы)
        """
        print("🧠 ЭТАП 2: Эмбеддинги масок и запросов...")
        valid_vecs: List[np.ndarray] = []
        valid_idx: List[int] = []

        # Короткое замыкание для DINOv3 - предотвращаем fallback на SearchDet
        if self._d3 is not None:
            embs = []
            for i, m in enumerate(masks):
                seg = m.get("segmentation", None)
                if isinstance(seg, np.ndarray) and seg.dtype == bool and seg.sum() > 0:
                    try:
                        # сделаем masked-image: фон занулим, чтобы был «объект»
                        masked = image_np.copy()
                        masked[~seg] = 0
                        vec = self._d3.encode_image(Image.fromarray(masked))  # вернёт L2-нормированный вектор
                        embs.append(vec.astype(np.float32))
                        valid_idx.append(i)
                    except Exception as e:
                        print(f"   ⚠️ mask {i} failed: {e}")
            if embs:
                return np.vstack(embs), valid_idx
            else:
                return np.zeros((0, self._d3.model.num_features), dtype=np.float32), valid_idx

        # === Fallback: старый SearchDet-режим (если нужен для dinov2/resnet) ===
        if not SEARCHDET_AVAILABLE:
            print("⚠️ SearchDet backend is not available → empty embeddings")
            return np.zeros((0, 1024), dtype=np.float32), []

        mask_dicts = []
        keep = []
        for i, m in enumerate(masks):
            seg = m.get("segmentation", None)
            if isinstance(seg, np.ndarray) and seg.dtype == bool and seg.sum() > 0:
                mask_dicts.append({"segmentation": seg})
                keep.append(i)
        if not mask_dicts:
            return np.zeros((0, 1024), dtype=np.float32), []

        resnet, layer, transform = (
            getattr(self.detector, "searchdet_resnet", None),
            getattr(self.detector, "searchdet_layer", None),
            getattr(self.detector, "searchdet_transform", None),
        )
        try:
            vecs = extract_features_from_masks(image_np, mask_dicts, resnet, layer, transform)
            vecs = np.asarray(vecs, dtype=np.float32)
            # L2
            vecs /= (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8)
            return vecs, keep
        except Exception as e:
            print(f"⚠️ extract_features_from_masks failed: {e}")
            return np.zeros((0, 1024), dtype=np.float32), []
    def _extract_with_dinov3_convnext(self, image_np, mask_arrays):
        """ConvNeXt+DINOv3: эмбеддинги масок через bbox-crop + тот же preprocess, что и для positive."""
        import time, torch, cv2
        from PIL import Image
        t0 = time.time()

        self._ensure_dinov3_convnext()
        if self._dinov3_model is None:
            return None

        H0, W0 = image_np.shape[:2]
        scale = 1.0
        if max(H0, W0) > self.max_embedding_size:
            scale = float(self.max_embedding_size) / float(max(H0, W0))
            newW, newH = int(W0 * scale), int(H0 * scale)
            image_np = cv2.resize(image_np, (newW, newH), interpolation=cv2.INTER_LINEAR)

        embeddings = []
        ctx = float(getattr(self.detector, "mask_context", 0.08))  # 8% контекст по умолчанию
        use_half = bool(getattr(self.detector, "dino_half_precision", False))
        device = next(self._dinov3_model.parameters()).device
        model_dtype = next(self._dinov3_model.parameters()).dtype  # float32 или float16

        with torch.no_grad():
            for i, m in enumerate(mask_arrays):
                # ресайз маски под scale (если был)
                if scale != 1.0:
                    m = cv2.resize(m.astype(np.uint8), (image_np.shape[1], image_np.shape[0]),
                                interpolation=cv2.INTER_NEAREST).astype(bool)

                ys, xs = np.where(m)
                if ys.size == 0:
                    embeddings.append(np.zeros(1024, dtype=np.float32))
                    continue

                y1, y2 = ys.min(), ys.max()
                x1, x2 = xs.min(), xs.max()

                # контекст вокруг bbox
                h, w = image_np.shape[:2]
                pad_y = int((y2 - y1 + 1) * ctx)
                pad_x = int((x2 - x1 + 1) * ctx)
                y1 = max(0, y1 - pad_y); y2 = min(h - 1, y2 + pad_y)
                x1 = max(0, x1 - pad_x); x2 = min(w - 1, x2 + pad_x)

                crop = image_np[y1:y2+1, x1:x2+1].copy()

                pil = Image.fromarray(crop)
                x = self._dinov3_preprocess(pil).unsqueeze(0)  # (1,3,224,224)
                x = x.to(device)
                # приводим dtype входа к dtype модели (важно для half)
                if model_dtype == torch.float16:
                    x = x.half()
                else:
                    x = x.float()

                z = self._dinov3_model(x)           # (1, D), num_classes=0 => pre-logits
                v = z[0].detach().cpu().float().numpy()

                # L2-нормализация и приведение к D=1024 (если вдруг другое D)
                v = v.astype(np.float32)
                n = np.linalg.norm(v) + 1e-8
                v = v / n
                if v.shape[0] != 1024:
                    out = np.zeros(1024, dtype=np.float32)
                    take = min(1024, v.shape[0])
                    out[:take] = v[:take]
                    v = out
                embeddings.append(v)

        print(f"   ⚡ DINOv3 ConvNeXt-B (bbox-crop): {time.time()-t0:.3f}s, N={len(embeddings)}")
        return np.stack(embeddings, axis=0).astype(np.float32)

    # =========================
    # Примеры (positive/negative)
    # =========================
    def build_queries(self, pos_imgs: List[Image.Image], neg_imgs: List[Image.Image]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Возвращает (q_pos, q_neg) — матрицы эмбеддингов примеров.
        Совпадает по нормировке с масками (L2) и со скриптом dinov3_similarity.
        """
        print(f"🔍 Построение запросов из {len(pos_imgs)} positive и {len(neg_imgs)} negative примеров")
        
        # Короткое замыкание для DINOv3
        if self._d3 is not None and self.pos_as_query_masks:
            # 1) q_pos: все валидные маски из positive
            pos_vecs = []
            for i, pil in enumerate(pos_imgs or []):
                img_np = np.array(pil)
                masks = self._gen_masks_for_image(img_np)
                for md in masks:
                    m = md.get('segmentation', None)
                    if isinstance(m, np.ndarray) and m.dtype == bool and m.sum() >= self.min_mask_area:
                        pos_vecs.append(self._encode_mask_d3(img_np, m))
            q_pos = np.stack(pos_vecs, 0) if pos_vecs else np.zeros((0, 1024), np.float32)

            # 2) q_neg: по желанию — аналогично (или пусто, если не используешь)
            neg_vecs = []
            for i, pil in enumerate(neg_imgs or []):
                img_np = np.array(pil)
                masks = self._gen_masks_for_image(img_np)
                for md in masks:
                    m = md.get('segmentation', None)
                    if isinstance(m, np.ndarray) and m.dtype == bool and m.sum() >= self.min_mask_area:
                        neg_vecs.append(self._encode_mask_d3(img_np, m))
            q_neg = np.stack(neg_vecs, 0) if neg_vecs else np.zeros((0, 1024), np.float32)

            print(f"   📊 Построено positive-масок: {q_pos.shape[0]}, negative-масок: {q_neg.shape[0]}")
            return q_pos.astype(np.float32), q_neg.astype(np.float32)
        
        elif self._d3 is not None:
            def _vec(img):
                v = self._d3.encode_image(img)
                n = np.linalg.norm(v); v = v / (n + 1e-8)
                if v.shape[0] != 1024:
                    out = np.zeros(1024, np.float32); out[:min(1024, v.shape[0])] = v[:min(1024, v.shape[0])]
                    v = out
                return v.astype(np.float32)

            pos = np.stack([_vec(im) for im in pos_imgs], 0) if pos_imgs else np.zeros((0,1024), np.float32)
            neg = np.stack([_vec(im) for im in neg_imgs], 0) if neg_imgs else np.zeros((0,1024), np.float32)
            print(f"   📊 Построено positive: {pos.shape[0]}, negative: {neg.shape[0]}")
            return pos, neg

        # Fallback: старый путь (если нужен)
        if not SEARCHDET_AVAILABLE:
            return np.zeros((0, 1024), dtype=np.float32), np.zeros((0, 1024), dtype=np.float32)

        pos_vecs, neg_vecs = [], []
        resnet, layer, transform = (
            getattr(self.detector, "searchdet_resnet", None),
            getattr(self.detector, "searchdet_layer", None),
            getattr(self.detector, "searchdet_transform", None),
        )
        def _one(img: Image.Image) -> Optional[np.ndarray]:
            try:
                v = get_vector(img, resnet, layer, transform)
                v = np.asarray(v, dtype=np.float32).reshape(-1)
                v /= (np.linalg.norm(v) + 1e-8)
                return v
            except Exception:
                return None

        for im in (pos_imgs or []):
            v = _one(im)
            if v is not None:
                pos_vecs.append(v)
        for im in (neg_imgs or []):
            v = _one(im)
            if v is not None:
                neg_vecs.append(v)

        D = pos_vecs[0].shape[0] if pos_vecs else (neg_vecs[0].shape[0] if neg_vecs else 1024)
        Qp = np.stack(pos_vecs, axis=0) if pos_vecs else np.zeros((0, D), dtype=np.float32)
        Qn = np.stack(neg_vecs, axis=0) if neg_vecs else np.zeros((0, D), dtype=np.float32)
        return Qp, Qn

    # мультикласс: {cls: [PIL,...]}, neg:[PIL,...]
    def build_queries_multiclass(self, pos_by_class: Dict[str, List[Image.Image]], neg_imgs: List[Image.Image]):
        # Короткое замыкание для DINOv3
        if self._d3 is not None:
            def _vec(img):
                v = self._d3.encode_image(img)
                n = np.linalg.norm(v); v = v / (n + 1e-8)
                if v.shape[0] != 1024:
                    out = np.zeros(1024, np.float32); out[:min(1024, v.shape[0])] = v[:min(1024, v.shape[0])]
                    v = out
                return v.astype(np.float32)

            # neg
            neg = np.stack([_vec(im) for im in neg_imgs], 0) if neg_imgs else np.zeros((0,1024), np.float32)
            
            # pos by class
            class_pos = {}
            for cls, imgs in (pos_by_class or {}).items():
                Q = np.stack([_vec(im) for im in imgs], 0) if imgs else np.zeros((0,1024), np.float32)
                class_pos[cls] = Q

            print(f"   📊 Negative всего: {neg.shape[0]}")
            for c, Q in class_pos.items():
                print(f"   📊 Класс '{c}': {Q.shape[0]} примеров")
            return class_pos, neg

        # Fallback старого пути:
        class_pos = {}
        for cls, imgs in (pos_by_class or {}).items():
            vecs = []
            for im in (imgs or []):
                try:
                    v = get_vector(im, getattr(self.detector, "searchdet_resnet", None),
                                   getattr(self.detector, "searchdet_layer", None),
                                   getattr(self.detector, "searchdet_transform", None))
                    v = np.asarray(v, dtype=np.float32).reshape(-1)
                    v /= (np.linalg.norm(v) + 1e-8)
                    vecs.append(v)
                except Exception:
                    pass
            D = vecs[0].shape[0] if vecs else 1024
            class_pos[cls] = (np.stack(vecs, axis=0) if vecs else np.zeros((0, D), dtype=np.float32))
        neg_vecs = []
        for im in (neg_imgs or []):
            try:
                v = get_vector(im, getattr(self.detector, "searchdet_resnet", None),
                               getattr(self.detector, "searchdet_layer", None),
                               getattr(self.detector, "searchdet_transform", None))
                v = np.asarray(v, dtype=np.float32).reshape(-1)
                v /= (np.linalg.norm(v) + 1e-8)
                neg_vecs.append(v)
            except Exception:
                pass
        Dn = neg_vecs[0].shape[0] if neg_vecs else 1024
        Qn = (np.stack(neg_vecs, axis=0) if neg_vecs else np.zeros((0, Dn), dtype=np.float32))
        return class_pos, Qn

    # =========================
    # Вспомогательные методы
    # =========================
    
    def _encode_mask_d3(self, image_np: np.ndarray, mask_bool: np.ndarray) -> np.ndarray:
        """Зануляем фон и считаем эмбеддинг через DINOv3."""
        masked = image_np.copy()
        masked[~mask_bool] = 0
        v = self._d3.encode_image(masked)  # внутри уже правильный device/dtype
        # L2 для устойчивости
        n = np.linalg.norm(v)
        if n > 1e-8:
            v = v / n
        return v.astype(np.float32)
    
    def _gen_masks_for_image(self, image_np: np.ndarray):
        """Вызываем тот же генератор масок, что и в пайплайне, с твоими порогами."""
        if hasattr(self.detector, "generate_masks"):
            return self.detector.generate_masks(image_np)  # должен вернуть [{'segmentation': bool np.ndarray, ...}, ...]
        # если нет — грубый fallback: одна «вся картинка»
        h,w = image_np.shape[:2]
        return [{'segmentation': np.ones((h,w), dtype=bool)}]
