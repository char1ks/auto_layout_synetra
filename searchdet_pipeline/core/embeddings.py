# embeddings.py

import os
import glob
import numpy as np
from PIL import Image
import cv2
import torch

from .dinov3_encoder import DinoV3Encoder


def _iter_images(root):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp", "*.tif", "*.tiff")
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(root, e)))
    return sorted(paths)

def extract_features_from_masks(encoder, masked_crops):
    """
    masked_crops: List[PIL.Image] — вырезки под маски
    Возвращает: np.ndarray (N,D)
    """
    feats = []
    # Получаем эталонный dtype/device из модели
    p = next(encoder.model.parameters())
    dev, dt = p.device, p.dtype

    for im in masked_crops:
        x = encoder.tf(im).unsqueeze(0)                  # cpu float32
        x = x.to(device=dev, dtype=dt, non_blocking=True)
        with torch.no_grad():
            f = encoder.model.forward_features(x)
            # тот же способ, что и в encoder.encode
            if hasattr(encoder.model, "forward_head"):
                out = encoder.model.forward_head(f, pre_logits=True)
            else:
                out = f
            if out.ndim == 2:
                v = out[0]
            elif out.ndim == 3:
                if encoder.pooling == "mean" and out.shape[1] > 1:
                    v = out[0, 1:].mean(dim=0)
                else:
                    v = out[0, 0]
            elif out.ndim == 4:
                v = out.mean(dim=(2, 3))[0]
            else:
                v = out.flatten(1)[0]
            v = torch.nn.functional.normalize(v.float(), dim=0)
            feats.append(v.cpu().numpy().astype(np.float32))

    if not feats:
        return np.zeros((0, 1), dtype=np.float32)
    return np.stack(feats, axis=0)  # (N,D)


class EmbeddingExtractor:
    def __init__(self, detector):
        self.detector = detector
        self.encoder = DinoV3Encoder(device=detector.device)

    # ---------- НАДЁЖНОЕ КОДИРОВАНИЕ ROI МАСКИ ЧЕРЕЗ DINOv3 ----------
    def _encode_mask_roi_with_dinov3(self, image_np: np.ndarray, mask_bool: np.ndarray) -> np.ndarray:
        """
        1) берём bbox маски,
        2) вырезаем crop,
        3) зануляем фон (по маске),
        4) подаём в DINOv3-энкодер,
        5) возвращаем L2-нормированный вектор (float32).
        """
        if self.encoder is None:
            # safety: пустой вектор (но лучше всегда иметь энкодер)
            return np.zeros(1024, dtype=np.float32)

        ys, xs = np.where(mask_bool)
        if ys.size == 0 or xs.size == 0:
            return np.zeros(self.encoder.model.embed_dim, dtype=np.float32)  # fallback

        y1, y2 = int(ys.min()), int(ys.max())
        x1, x2 = int(xs.min()), int(xs.max())

        crop = image_np[y1:y2+1, x1:x2+1].copy()
        m    = mask_bool[y1:y2+1, x1:x2+1]
        # зануляем фон
        if m.dtype != bool:
            m = m.astype(bool)
        crop[~m] = 0

        pil = Image.fromarray(crop)
        vec = self.encoder.encode(pil)  # уже float32 + L2 norm внутри энкодера
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

        if self.encoder is None:
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
    def build_queries_multiclass(self, pos_input: dict | str,
                              negative_dir: str | None,
                              pos_as_query_masks: bool,
                              mask_crops_by_class: dict[str, list] | None = None,
                              max_per_class: int = 64) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """
        Возвращает:
          q_pos: dict[class_name] -> (K_i, D) — НОРМИРОВАННЫЕ векторы примеров
          q_neg: (K_neg, D) или shape (0, D), если нет негативов

        Если negative_dir отсутствует/пуст, будет собран онлайн-негатив из фоновых масок
        (передай его позже в score_multiclass через параметр `online_negatives`).
        """
        q_pos: dict[str, np.ndarray] = {}
        q_neg_list: list[np.ndarray] = []

        if self.encoder is None:
            print("   ❌ DINOv3-энкодер не инициализирован в detector")
            # пустые, чтобы скоринг корректно отработал
            return {}, np.zeros((0, 1024), dtype=np.float32)

        # POS: могут быть как dict {class:[paths]} или путь к папке с подпапками-классами
        pos_class_iterator = None
        is_dir_mode = False
        if isinstance(pos_input, str) and os.path.isdir(pos_input):
            pos_class_iterator = sorted(os.listdir(pos_input))
            is_dir_mode = True
        elif isinstance(pos_input, dict):
            pos_class_iterator = sorted(pos_input.keys())
            is_dir_mode = False

        if pos_class_iterator:
            for cls in pos_class_iterator:
                image_paths = []
                if is_dir_mode:
                    cls_dir = os.path.join(pos_input, cls)
                    if not os.path.isdir(cls_dir):
                        continue
                    image_paths = _iter_images(cls_dir)
                else: # dict
                    image_paths = pos_input[cls]

                vecs: list[np.ndarray] = []
                if pos_as_query_masks and mask_crops_by_class and cls in mask_crops_by_class:
                    # Используем маски из positive в качестве запросов
                    for im in mask_crops_by_class[cls][:max_per_class]:
                        vecs.append(self.encoder.encode(im))
                else:
                    # Берём целые изображения из класса
                    for item in image_paths[:max_per_class]:
                        try:
                            if isinstance(item, Image.Image):
                                im = item.convert("RGB")
                            else:
                                im = Image.open(item).convert("RGB")
                            vecs.append(self.encoder.encode(im))
                        except Exception as e:
                            print(f"   ⚠️ Ошибка обработки positive-примера {item}: {e}")
                            continue

                if vecs:
                    arr = np.stack(vecs, 0)
                    # Приведение к единичным (на всякий)
                    arr = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8)
                    q_pos[cls] = arr

        # NEG: реальные негативы
        if negative_dir and os.path.isdir(negative_dir):
            for p in _iter_images(negative_dir)[:256]:
                try:
                    im = Image.open(p).convert("RGB")
                    v = self.encoder.encode(im)
                    q_neg_list.append(v / (np.linalg.norm(v) + 1e-8))
                except Exception:
                    continue

        q_neg = np.stack(q_neg_list, 0) if q_neg_list else np.zeros((0, 1), dtype=np.float32)
        return q_pos, q_neg
