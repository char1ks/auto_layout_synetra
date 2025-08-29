# embeddings.py

import os
import glob
import numpy as np
from PIL import Image
import cv2
import torch

from .dinov3_encoder import DinoV3Encoder

def _to_pil_any(x):
    """Надёжно приводим всё к PIL.Image."""
    if isinstance(x, Image.Image):
        return x
    if isinstance(x, str):
        # путь к файлу
        return Image.open(x).convert("RGB")
    if isinstance(x, np.ndarray):
        if x.ndim == 2:
            x = np.stack([x, x, x], axis=-1)
        if x.dtype != np.uint8:
            x = np.clip(x, 0, 255).astype(np.uint8)
        return Image.fromarray(x, mode="RGB")
    if torch.is_tensor(x):
        t = x.detach().cpu()
        if t.ndim == 3 and t.shape[0] in (1,3):  # [C,H,W] -> [H,W,C]
            t = t.permute(1,2,0).contiguous()
        t = t.numpy()
        return _to_pil_any(t)
    # неизвестный формат
    raise TypeError(f"Cannot convert type {type(x)} to PIL.Image")


def _l2n(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)


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
        self.encoder = DinoV3Encoder(
            backbone_name=getattr(detector, 'dinov3_backbone', 'vitb16'),
            device=detector.device,
            ckpt_path=getattr(detector, 'dinov3_ckpt', None)
        )

    def _safe_stack(self, arrs, axis=0):
        if not arrs:
            return np.zeros((0, 768), dtype=np.float32)  # дефолт D=768, подменится ниже
        return np.stack(arrs, axis=axis)

    def _filter_bad(self, mat: np.ndarray, cls_name: str = None, kind: str = ""):
        """Удаляет строки с NaN/Inf/нулевой нормой."""
        if mat.size == 0:
            return mat
        bad = ~np.isfinite(mat).all(axis=1)
        norms = np.linalg.norm(mat, axis=1)
        bad |= (norms < 1e-6)
        if bad.any():
            print(f"   ⚠️ Dropped {bad.sum()} bad {kind} embeddings"
                  + (f" in class '{cls_name}'" if cls_name else ""))
        return mat[~bad]

    def _encode_pil_list(self, pil_list):
        """Кодируем список PIL-изображений → (N,D) без NaN."""
        out = []
        for i, im in enumerate(pil_list):
            try:
                v = self.encoder.encode(im)  # (D,)
                if not np.isfinite(v).all():
                    v = np.zeros_like(v, dtype=np.float32)
                out.append(v.astype(np.float32))
            except Exception as e:
                print(f"   ⚠️ Pos/Neg encode failed ({i}): {e}")
        if not out:
            return np.zeros((0, 768), dtype=np.float32)
        M = self._safe_stack(out, axis=0)
        M = self._filter_bad(M, kind="example")
        return M

    # ---------- НАДЁЖНОЕ КОДИРОВАНИЕ ROI МАСКИ ЧЕРЕЗ DINOv3 ----------
    def _encode_mask_roi_with_dinov3(self, image_np: np.ndarray, mask_bool: np.ndarray, pad: int = 8) -> np.ndarray:
        """
        image_np: HxWx3 (uint8), mask_bool: HxW (bool)
        Вырезает bbox(mask) с небольшим полем, фон заливает средним цветом, кодирует через encoder.encode(PIL).
        Возвращает np.float32 вектор признаков (D,) без NaN.
        """
        assert image_np.ndim == 3 and image_np.shape[2] == 3, "image_np must be HxWx3"
        H, W, _ = image_np.shape
        mb = mask_bool.astype(bool)
        
        print(f"     🔍 ROI ДИАГНОСТИКА: Площадь маски = {mb.sum()} пикселей из {mb.size}")
        
        if mb.sum() == 0:
            print(f"     ⚠️ ROI: Пустая маска, возвращаем нулевой вектор")
            return np.zeros((getattr(self.encoder, "feat_dim", 768),), dtype=np.float32)

        ys, xs = np.where(mb)
        y0, y1 = max(0, ys.min()-pad), min(H, ys.max()+1+pad)
        x0, x1 = max(0, xs.min()-pad), min(W, xs.max()+1+pad)
        
        print(f"     🔍 ROI: Bounding box = ({y0}, {x0}) -> ({y1}, {x1})")

        crop = image_np[y0:y1, x0:x1].copy()
        m_crop = mb[y0:y1, x0:x1]
        
        print(f"     🔍 ROI: Размер кропа = {crop.shape}")

        # заполняем фон средним цветом из ROI (уменьшает «засвет»)
        mean_color = crop[m_crop].mean(axis=0) if m_crop.any() else np.array([128,128,128], dtype=np.float32)
        bg = np.tile(mean_color.reshape(1,1,3), (crop.shape[0], crop.shape[1], 1))
        crop = np.where(m_crop[...,None], crop, bg).astype(np.uint8)
        
        print(f"     🔍 ROI: Средний цвет фона = {mean_color}")

        try:
            pil = Image.fromarray(crop, mode="RGB")
            v = self.encoder.encode(pil)
            
            print(f"     🔍 ROI: Исходный вектор - finite: {np.isfinite(v).all()}, норма: {np.linalg.norm(v):.2e}")
            
            if not np.isfinite(v).all():
                print(f"     ⚠️ ROI: Обнаружены NaN/Inf, применяем nan_to_num")
                v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                
            n = np.linalg.norm(v)
            if not np.isfinite(n) or n < 1e-6:
                print(f"     ⚠️ ROI: Нулевая норма ({n:.2e}), обнуляем вектор")
                v = np.zeros_like(v, dtype=np.float32)
            else:
                print(f"     ✅ ROI: Валидный вектор с нормой {n:.2e}")
                
        except Exception as e:
            print(f"     ❌ ROI: Ошибка кодирования - {e}")
            import traceback
            traceback.print_exc()
            v = np.zeros((getattr(self.encoder, "feat_dim", 768),), dtype=np.float32)
        return v.astype(np.float32)

    # ---------- МАСКИ: ВСЕГДА ЧЕРЕЗ DINOv3 ----------
    def extract_mask_embeddings(self, image_pil, masks):
        """
        Возвращает матрицу (M,D). Поддерживает словари SAM/FastSAM и «кропы».
        Любые NaN → обнуляем, пустые/некорректные вектора → отбрасываем.
        """
        vecs = []
        image_np = np.array(image_pil)

        for i, m in enumerate(masks):
            print(f"   🔍 Обрабатываем маску {i+1}/{len(masks)}, тип = {type(m)}")
            try:
                # 1) Стандарт SAM/FastSAM: {'segmentation': np.ndarray[bool], ...}
                if isinstance(m, dict) and 'segmentation' in m:
                    seg = m['segmentation']
                    if isinstance(seg, np.ndarray):
                        print(f"   🔍 Маска {i+1}: SAM формат, segmentation shape={seg.shape}, dtype={seg.dtype}")
                        v = self._encode_mask_roi_with_dinov3(image_np, seg.astype(bool), pad=8)
                        print(f"   🔍 Маска {i+1}: Получен вектор с нормой {np.linalg.norm(v):.2e}")
                        vecs.append(v.astype(np.float32))
                    else:
                        print(f"   ⚠️ Mask {i+1}: 'segmentation' is not numpy array (type={type(seg)}), skip")
                        continue

                # 2) Объект с .crop(image) -> любой тип → конвертируем в PIL
                elif hasattr(m, "crop"):
                    print(f"   🔍 Маска {i+1}: Объект с .crop() методом")
                    crop_any = m.crop(image_pil)
                    crop_pil = _to_pil_any(crop_any)
                    print(f"   🔍 Маска {i+1}: Crop размер = {crop_pil.size}")
                    v = self.encoder.encode(crop_pil)
                    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                    print(f"   🔍 Маска {i+1}: Получен вектор с нормой {np.linalg.norm(v):.2e}")
                    vecs.append(v)

                # 3) Прямые заготовки: str/ndarray/tensor/PIL
                elif isinstance(m, (str, np.ndarray, torch.Tensor, Image.Image)):
                    print(f"   🔍 Маска {i+1}: Прямая заготовка, тип = {type(m)}")
                    crop_pil = _to_pil_any(m)
                    print(f"   🔍 Маска {i+1}: Конвертирован в PIL размер = {crop_pil.size}")
                    v = self.encoder.encode(crop_pil)
                    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                    print(f"   🔍 Маска {i+1}: Получен вектор с нормой {np.linalg.norm(v):.2e}")
                    vecs.append(v)

                # 4) Кортежи (img, mask) — используем только картинку
                elif isinstance(m, (tuple, list)) and len(m) == 2:
                    print(f"   🔍 Маска {i+1}: Кортеж/список из 2 элементов")
                    crop_pil = _to_pil_any(m[0])
                    print(f"   🔍 Маска {i+1}: Первый элемент конвертирован в PIL размер = {crop_pil.size}")
                    v = self.encoder.encode(crop_pil)
                    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                    print(f"   🔍 Маска {i+1}: Получен вектор с нормой {np.linalg.norm(v):.2e}")
                    vecs.append(v)

                # 5) Словари с "image"/"crop"
                elif isinstance(m, dict):
                    print(f"   🔍 Маска {i+1}: Словарь с ключами {list(m.keys())}")
                    crop_any = m.get("image", m.get("crop", None))
                    if crop_any is None:
                        print(f"   ⚠️ Mask {i+1}: dict has no 'image' or 'crop', skip")
                        continue
                    crop_pil = _to_pil_any(crop_any)
                    print(f"   🔍 Маска {i+1}: Значение конвертировано в PIL размер = {crop_pil.size}")
                    v = self.encoder.encode(crop_pil)
                    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                    print(f"   🔍 Маска {i+1}: Получен вектор с нормой {np.linalg.norm(v):.2e}")
                    vecs.append(v)

                else:
                    print(f"   ⚠️ Mask {i+1}: unknown format {type(m)}, skip")
                    continue

            except Exception as e:
                print(f"   ❌ Mask {i+1} embedding failed: {e}")
                import traceback
                print(f"   📍 Stack trace для маски {i+1}:")
                traceback.print_exc()

        if not vecs:
            D = getattr(self.encoder, "feat_dim", 768)
            return np.zeros((0, D), dtype=np.float32)

        V = np.stack(vecs, axis=0)  # (M,D)
        # фильтр мусора: finite + ненулевая норма
        finite = np.isfinite(V).all(axis=1)
        norms = np.linalg.norm(V, axis=1)
        good = finite & (norms >= 1e-6)
        
        if not good.all():
            dropped_count = (~good).sum()
            print(f"   ⚠️ Dropped {dropped_count} bad mask embeddings")
            print(f"   📍 ДЕТАЛИ ОТБРАСЫВАНИЯ:")
            print(f"     - Не finite: {(~finite).sum()} масок")
            print(f"     - Нулевая норма (<1e-6): {(norms < 1e-6).sum()} масок")
            print(f"     - Нормы векторов: {norms}")
            
            # Показываем какие именно маски отброшены
            for i, is_good in enumerate(good):
                if not is_good:
                    print(f"     - Маска {i+1}: finite={finite[i]}, норма={norms[i]:.2e}")
                    
        return V[good].astype(np.float32)

    # ---------- ПРИМЕРЫ: ВСЕГДА ЧЕРЕЗ DINOv3 ----------
    def build_queries_multiclass(self, pos_dict, neg_list, pos_as_query_masks=False):
        """
        pos_dict: {class_name: [PIL, PIL, ...]}
        neg_list: [PIL, PIL, ...]

        return:
            q_pos: {class_name: np.ndarray (K,D)}  — без NaN/нулей
            q_neg: np.ndarray (N,D)                — без NaN/нулей
        """
        q_pos = {}
        for cls, pil_list in pos_dict.items():
            M = self._encode_pil_list(pil_list)    # (K,D)
            M = self._filter_bad(M, cls_name=cls, kind="q_pos")
            if M.shape[0] > 0:
                q_pos[cls] = M
            else:
                print(f"   ⚠️ Класс '{cls}' пуст после фильтрации — пропущен")

        q_neg = self._encode_pil_list(neg_list) if neg_list else np.zeros((0, 768), dtype=np.float32)
        q_neg = self._filter_bad(q_neg, kind="q_neg")

        # Если все классы выпали — вернём пустые, чтобы скоринг корректно отработал
        if not q_pos:
            print("   ❌ Нет валидных классов с эмбеддингами после фильтрации.")
        return q_pos, q_neg


def build_queries_multiclass(
    pos_by_class: dict[str, list[np.ndarray] | np.ndarray],
    neg_list: list[np.ndarray] | np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """
    Собирает эмбеддинги запросов для мультикласса.
    Возвращает:
      q_pos: dict {class_name -> np.ndarray [N_c, D]} (все строки L2-нормированы)
      q_neg: np.ndarray [M, D] (может быть (0, D)), L2-нормирован
    """
    q_pos: dict[str, np.ndarray] = {}
    D: int | None = None

    # --- позитивы по классам
    for cls, vecs in (pos_by_class or {}).items():
        if vecs is None:
            arr = np.zeros((0, 0), dtype=np.float32)
        else:
            if isinstance(vecs, np.ndarray):
                arr = vecs
            else:
                arr = np.array(vecs, dtype=np.float32)
        if arr.size == 0:
            q_pos[cls] = np.zeros((0, 0), dtype=np.float32)
            continue
        if arr.ndim == 1:
            arr = arr[None, :]
        arr = arr.astype(np.float32, copy=False)
        arr = _l2n(arr, axis=1)
        q_pos[cls] = arr
        if D is None:
            D = arr.shape[1]

    # --- негативы
    if neg_list is None:
        q_neg = np.zeros((0, D if D is not None else 0), dtype=np.float32)
    else:
        if isinstance(neg_list, np.ndarray):
            neg_arr = neg_list
        else:
            neg_arr = np.array(neg_list, dtype=np.float32)
        if neg_arr.size == 0:
            q_neg = np.zeros((0, D if D is not None else 0), dtype=np.float32)
        else:
            if neg_arr.ndim == 1:
                neg_arr = neg_arr[None, :]
            neg_arr = neg_arr.astype(np.float32, copy=False)
            neg_arr = _l2n(neg_arr, axis=1)
            q_neg = neg_arr

    # отладка как в логах
    try:
        total_neg = int(q_neg.shape[0])
    except Exception:
        total_neg = 0
    print(f"   DEBUG: build_queries_multiclass returning: q_pos type={type(q_pos)}, "
          f"q_neg type={type(q_neg)}, q_neg shape={getattr(q_neg, 'shape', None)}")
    return q_pos, q_neg
