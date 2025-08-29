# embeddings.py

import numpy as np
from PIL import Image
import cv2
import torch

class EmbeddingExtractor:
    def __init__(self, detector):
        self.detector = detector
        # ничего больше здесь не нужно — всё возьмём из detector.dinov3_encoder

    # ---------- НАДЁЖНОЕ КОДИРОВАНИЕ ROI МАСКИ ЧЕРЕЗ DINOv3 ----------
    def _encode_mask_roi_with_dinov3(self, image_np: np.ndarray, mask_bool: np.ndarray) -> np.ndarray:
        """
        1) берём bbox маски,
        2) вырезаем crop,
        3) зануляем фон (по маске),
        4) подаём в DINOv3-энкодер,
        5) возвращаем L2-нормированный вектор (float32).
        """
        enc = getattr(self.detector, "dinov3_encoder", None)
        if enc is None:
            # safety: пустой вектор (но лучше всегда иметь энкодер)
            return np.zeros(1024, dtype=np.float32)

        ys, xs = np.where(mask_bool)
        if ys.size == 0 or xs.size == 0:
            return np.zeros(enc.output_dim, dtype=np.float32)  # fallback

        y1, y2 = int(ys.min()), int(ys.max())
        x1, x2 = int(xs.min()), int(xs.max())

        crop = image_np[y1:y2+1, x1:x2+1].copy()
        m    = mask_bool[y1:y2+1, x1:x2+1]
        # зануляем фон
        if m.dtype != bool:
            m = m.astype(bool)
        crop[~m] = 0

        pil = Image.fromarray(crop)
        vec = enc.encode(pil)  # уже float32 + L2 norm внутри энкодера
        return vec

    # ---------- МАСКИ: ВСЕГДА ЧЕРЕЗ DINOv3 ----------
    def extract_mask_embeddings(self, image_np, masks):
        """
        Возвращает:
          - np.ndarray (N, D) float32, L2-нормированные эмбеддинги масок,
          - список индексов валидных масок (как раньше).
        """
        print("🧠 ЭТАП 2: Эмбеддинги масок и запросов...")
        if not isinstance(masks, (list, tuple)) or len(masks) == 0:
            print("   ❌ Нет масок на входе")
            return np.zeros((0, 1024), dtype=np.float32), []

        enc = getattr(self.detector, "dinov3_encoder", None)
        if enc is None:
            print("   ❌ DINOv3-энкодер не инициализирован в detector")
            return np.zeros((0, 1024), dtype=np.float32), []

        vecs = []
        valid_ids = []
        for i, md in enumerate(masks):
            m = md.get("segmentation", None)
            if m is None or not isinstance(m, np.ndarray):
                continue
            try:
                v = self._encode_mask_roi_with_dinov3(image_np, m.astype(bool))
                if v is not None and v.size > 0:
                    vecs.append(v.astype(np.float32, copy=False))
                    valid_ids.append(i)
            except Exception as e:
                print(f"   ⚠️ Маска {i}: ошибка кодирования через DINOv3: {e}")

        if not vecs:
            print("   ❌ Не удалось получить эмбеддинги масок.")
            return np.zeros((0, 1024), dtype=np.float32), []

        X = np.stack(vecs, axis=0).astype(np.float32, copy=False)
        # sanity: векторы уже нормированы в энкодере
        print(f"   📊 Масок с валидными векторами: {X.shape[0]}")
        return X, valid_ids

    # ---------- ПРИМЕРЫ: ВСЕГДА ЧЕРЕЗ DINOv3 ----------
    def build_queries_multiclass(self, pos_by_class, neg_imgs):
        """
        Строит словарь: {класс: np.ndarray (Kc, D)} и q_neg: (M, D)
        Все векторы — L2-нормированные float32.
        """
        enc = getattr(self.detector, "dinov3_encoder", None)
        if enc is None:
            print("   ❌ DINOv3-энкодер не инициализирован в detector")
            # пустые, чтобы скоринг корректно отработал
            return {}, np.zeros((0, 1024), dtype=np.float32)

        class_pos = {}
        total_pos = 0

        for cls_name, img_list in (pos_by_class or {}).items():
            cls_vecs = []
            for i, pil_img in enumerate(img_list or []):
                try:
                    v = enc.encode(pil_img)  # float32, L2
                    if v is not None and v.size > 0:
                        cls_vecs.append(v.astype(np.float32, copy=False))
                except Exception as e:
                    print(f"   ⚠️ Класс '{cls_name}', пример {i}: {e}")
            if cls_vecs:
                Q = np.stack(cls_vecs, axis=0).astype(np.float32, copy=False)
            else:
                Q = np.zeros((0, enc.output_dim), dtype=np.float32)
            class_pos[cls_name] = Q
            total_pos += Q.shape[0]
            print(f"   📊 Класс '{cls_name}': {Q.shape[0]} примеров")

        # отрицательные опциональны; если их нет — возвращаем (0, D), но не используем их в вычитаниях
        neg_vecs = []
        for i, pil_img in enumerate(neg_imgs or []):
            try:
                v = enc.encode(pil_img)
                if v is not None and v.size > 0:
                    neg_vecs.append(v.astype(np.float32, copy=False))
            except Exception as e:
                print(f"   ⚠️ Negative пример {i}: {e}")
        q_neg = np.stack(neg_vecs, axis=0).astype(np.float32, copy=False) if neg_vecs else \
                np.zeros((0, enc.output_dim), dtype=np.float32)

        print(f"   📊 Negative всего: {q_neg.shape[0]}")
        return class_pos, q_neg
