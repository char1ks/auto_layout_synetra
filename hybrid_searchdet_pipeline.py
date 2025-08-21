#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SearchDet-only pipeline (без LLaVA):
 - SAM-HQ → автогенерация масок (с опциональным даунскейлом для ускорения)
 - Эмбеддинги масок ↔ эмбеддинги positive/negative
 - Гео-фильтр: только по площади (min/max) + НОВОЕ: жёсткий запрет масок, касающихся краёв кадра
 - Правила приёма: pos/neg + консенсус, NMS
 - Фоллбэк-визуализация (если ничего не принято) — рисуем top-K кандидатов по score
 - Сохранение визуализаций и аннотаций, включая total_mask (обнуляем края, если требуется)
"""

import os
import sys
import cv2
import json
import time
import argparse
import numpy as np
import subprocess
import shlex
from pathlib import Path
from PIL import Image
from datetime import datetime
import urllib.request

# --- SearchDet helpers -------------------------------------------------
sys.path.append('./searchdet-main')
try:
    from mask_withsearch import (
        initialize_models as init_searchdet,
        get_vector,
        adjust_embedding,
        extract_features_from_masks,
    )
    SEARCHDET_AVAILABLE = True
except Exception as e:
    print(f"⚠️ SearchDet недоступен: {e}")
    SEARCHDET_AVAILABLE = False


# ============================== FastSAM Utils =============================

def _pip_install(requirement: str) -> bool:
    """Возвращает True, если установка прошла успешно."""
    try:
        cmd = f"{sys.executable} -m pip install -U {requirement}"
        print(f"   ⬇️ pip: {cmd}")
        res = subprocess.run(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False, text=True)
        print(res.stdout[-1000:])  # последние строки лога
        return res.returncode == 0
    except Exception as e:
        print(f"   ⚠️ pip ошибка: {e}")
        return False


def _ensure_ultralytics(auto_install: bool = True):
    """Гарантирует, что импорт ultralytics возможен. При необходимости — ставит пакет."""
    try:
        from ultralytics import FastSAM  # noqa: F401
        return True
    except Exception as e:
        print(f"   ℹ️ ultralytics не найден: {e}")
        if not auto_install:
            return False
        ok = _pip_install("ultralytics>=8.1.0")
        if not ok:
            print("   ❌ Не удалось установить ultralytics.")
            return False
        # повторим импорт
        try:
            from ultralytics import FastSAM  # noqa: F401
            return True
        except Exception as e2:
            print(f"   ❌ Импорт ultralytics всё ещё не работает: {e2}")
            return False


def _load_fastsam_model(model_path: str | None):
    """
    Возвращает объект FastSAM. Если файлы весов не найдены:
    1) пробуем 'FastSAM-x.pt' (автодокачка у ultralytics),
    2) затем 'FastSAM-s.pt'.
    """
    # гарантируем наличие пакета
    if not _ensure_ultralytics(auto_install=True):
        raise RuntimeError("Не удалось подготовить ultralytics/FastSAM.")

    try:
        from ultralytics import FastSAM
    except Exception:
        # старый путь в некоторых сборках
        from ultralytics.models.fastsam import FastSAM

    # 0) создадим папку models/
    Path("models").mkdir(parents=True, exist_ok=True)

    # 1) если указан путь и файл существует — грузим по нему
    if model_path and os.path.exists(model_path):
        print(f"   ✅ FastSAM веса найдены локально: {model_path}")
        return FastSAM(model_path)

    # 2) иначе даём шанс автозагрузке ultralytics по имени файла
    for name in [model_path, "FastSAM-x.pt", "FastSAM-s.pt"]:
        if not name:
            continue
        try:
            print(f"   ⬇️ Пытаемся загрузить FastSAM по имени: {name} (ultralytics auto-download)")
            return FastSAM(name)  # ultralytics сам скачает весa в кэш
        except Exception as e:
            print(f"   ⚠️ Не удалось: {name} → {e}")

    # 3) Если тут — всё плохо. Подсказываем прямой путь.
    raise FileNotFoundError(
        "Веса FastSAM не найдены и не удалось автоскачать. "
        "Скачай вручную 'FastSAM-x.pt' или 'FastSAM-s.pt' и укажи --fastsam-model путь."
    )


# =============================== Core ==================================
class SearchDetDetector:
    def __init__(
        self,
        # приём решений (смягчённые дефолты)
        min_confidence: float = 0.60,
        margin: float = -0.10,
        ratio: float = 0.80,
        neg_cap: float = 0.95,
        consensus_k: int = 3,
        consensus_thr: float = 0.60,
        topk: int = 3,
        max_masks: int = 100,

        # SAM & предфильтры (УСКОРЕНИЕ: параметры из test.py)
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.90,
        points_per_side: int = 64,
        points_per_side_multi: list[int] | None = None,  # например [64, 80, 96]
        points_per_batch: int = 128,
        box_nms_thresh: float = 0.5,
        crop_nms_thresh: float = 0.5,
        crop_n_layers: int = 0,  # УСКОРЕНИЕ: отключили crop layers (как в test.py)
        crop_n_points_downscale_factor: int = 3,  # УСКОРЕНИЕ: увеличили downscale (как в test.py)
        sam_min_region: int = 0,      # 0 = авто по min_area_frac, иначе абсолютное число пикселей
        sam_long_side: int | None = 1800,
        sam_max_after_merge: int = 0, # 0 = без лимита на выходе SAM

        # площадь (геометрический фильтр)
        min_area: int = 800,
        min_area_frac: float = 0.03,
        max_area_frac: float = 0.90,

        # НОВОЕ: фильтр краёв
        ban_border_masks: bool = True,   # по умолчанию — запрещаем любые маски, касающиеся рамки
        border_width: int = 2,           # ширина рамки, px
        border_clip_small: bool = False, # если True, мелкие касания не удаляем, а срезаем по краю
        border_clip_max_frac: float = 0.02,  # максимум доли кадра для "клипа" вместо удаления

        # NMS
        nms_iou: float = 0.60,
        containment_iou_threshold: float = 0.95, # IoU порог для вложенных масок
        perfect_rectangle_iou_threshold: float = 0.99, # IoU порог для идеальных прямоугольников

        # backbone layer
        layer_override: str | None = "layer2",

        # debug
        debug_topk_pre: int = 10,

        # выбор бэкенда генерации масок
        mask_backend: str = "sam-hq",          # 'sam-hq' или 'fastsam'

        # FastSAM параметры
        fastsam_model: str = "models/FastSAM-x.pt",
        fastsam_imgsz: int = 1024,
        fastsam_conf: float = 0.4,
        fastsam_iou: float = 0.9,
        fastsam_retina: bool = True,
        fastsam_device: str | None = None,

        # SAM2 параметры
        sam_model: str = "sam-hq/pretrained_checkpoint/sam_hq_vit_l.pth",
        sam_encoder: str = "vit_l",
        sam2_weights: str = None,
    ):
        if not SEARCHDET_AVAILABLE:
            raise RuntimeError("SearchDet не найден. Убедись, что searchdet-main в проекте и mask_withsearch импортируется.")

        # Устанавливаем переменную окружения для оптимального feature map
        import os
        os.environ['SEARCHDET_FEAT_SHORT_SIDE'] = '384'  # Оптимальный размер для баланса скорости и качества
        print("🔧 Установлено SEARCHDET_FEAT_SHORT_SIDE=384 для оптимального feature map")

        self.searchdet_resnet, self.searchdet_layer, self.searchdet_transform, self.searchdet_sam = init_searchdet()
        
        # Переопределяем трансформацию для правильного размера
        import torchvision.transforms as transforms
        feat_short_side = int(os.getenv('SEARCHDET_FEAT_SHORT_SIDE', '384'))
        self.searchdet_transform = transforms.Compose([
            transforms.Resize(feat_short_side),  # используем переменную окружения, БЕЗ CenterCrop
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        print(f"🔧 Переопределена трансформация с Resize({feat_short_side}) без CenterCrop")
        
        # Принудительно устанавливаем layer3 для оптимального feature map
        if hasattr(self.searchdet_resnet, 'layer3'):
            self.searchdet_layer = 'layer3'
            print("🔧 Принудительно установлен layer3 для лучшего feature map")
        elif layer_override:
            self.searchdet_layer = layer_override

        # thresholds
        self.min_confidence = float(min_confidence)
        self.margin = float(margin)
        self.ratio = float(ratio)
        self.neg_cap = float(neg_cap)
        self.consensus_k = int(consensus_k)
        self.consensus_thr = float(consensus_thr)
        self.topk = int(topk)
        self.max_masks = int(max_masks)

        # sam
        self.pred_iou_thresh = float(pred_iou_thresh)
        self.stability_score_thresh = float(stability_score_thresh)
        self.points_per_side = int(points_per_side)
        self.points_per_side_multi = points_per_side_multi
        self.points_per_batch = int(points_per_batch)
        self.box_nms_thresh = float(box_nms_thresh)
        self.crop_nms_thresh = float(crop_nms_thresh)
        self.crop_n_layers = int(crop_n_layers)
        self.crop_n_points_downscale_factor = int(crop_n_points_downscale_factor)
        self.sam_min_region = int(sam_min_region)
        self.sam_long_side = int(sam_long_side) if sam_long_side else None
        self.sam_max_after_merge = int(sam_max_after_merge)

        # area
        self.min_area = int(min_area)
        self.min_area_frac = float(min_area_frac)
        self.max_area_frac = float(max_area_frac)

        # border
        self.ban_border_masks = bool(ban_border_masks)
        self.border_width = int(border_width)
        self.border_clip_small = bool(border_clip_small)
        self.border_clip_max_frac = float(border_clip_max_frac)

        # nms
        self.nms_iou = float(nms_iou)
        self.containment_iou_threshold = float(containment_iou_threshold)
        self.perfect_rectangle_iou_threshold = float(perfect_rectangle_iou_threshold)

        # debug
        self.debug_topk_pre = int(debug_topk_pre)
        self._debug_pre_top = []

        # --- backend выбора генератора масок ---
        self.mask_backend = (mask_backend or "sam-hq").lower()
        self.sam_model_path = sam_model
        self.sam_encoder = sam_encoder
        self.sam2_weights = sam2_weights
        self.fastsam_model = fastsam_model
        self.fastsam_imgsz = fastsam_imgsz
        self.fastsam_conf = fastsam_conf
        self.fastsam_iou = fastsam_iou
        self.fastsam_retina = fastsam_retina
        self.fastsam_device = fastsam_device

        if self.mask_backend == "fastsam":
            # FastSAM (уже реализовано)
            try:
                import torch  # noqa: F401
                if self.fastsam_device is None:
                    import torch as _torch
                    self.fastsam_device = "cuda" if _torch.cuda.is_available() else "cpu"
            except Exception:
                self.fastsam_device = "cpu"
            self.fastsam = _load_fastsam_model(self.fastsam_model)  # FIX: было self.fastsam_model_path
            print(f"   ✅ FastSAM готов (device={self.fastsam_device})")
        elif self.mask_backend == "sam2":
            # SAM2: поддержка разных энкодеров и весов
            try:
                from sam2 import Sam2AutomaticMaskGenerator, sam_model_registry
            except ImportError:
                raise RuntimeError("Для SAM2 требуется пакет 'sam2'. Установите его вручную.")
            encoder = self.sam_encoder or "vit_l"
            weights = self.sam2_weights or self.sam_model_path
            if not os.path.exists(weights):
                Path(os.path.dirname(weights)).mkdir(parents=True, exist_ok=True)
                ok = _download_sam2_weights(encoder, weights)
                if not ok:
                    raise RuntimeError(f"Не удалось получить веса SAM2 для {encoder}. Укажите путь вручную.")
            print(f"   🎯 Загружаем SAM2: encoder={encoder}, weights={weights}")
            self.searchdet_sam = sam_model_registry[encoder](checkpoint=weights)
            self.sam2_mask_generator = Sam2AutomaticMaskGenerator(self.searchdet_sam)
        else:
            # SAM-HQ (по умолчанию)
            try:
                from segment_anything_hq import sam_model_registry as sam_hq_registry
            except ImportError:
                raise RuntimeError("Для SAM-HQ требуется пакет 'segment_anything_hq'. Установите его вручную.")
            encoder = self.sam_encoder or "vit_l"
            weights = self.sam_model_path
            print(f"   🎯 Загружаем SAM-HQ: encoder={encoder}, weights={weights}")
            self.searchdet_sam = sam_hq_registry[encoder](checkpoint=weights)
            self.sam2_mask_generator = None

        self.fastsam = None
        self.fastsam_device = fastsam_device
        if self.mask_backend == "fastsam":
            try:
                import torch  # noqa: F401
                if self.fastsam_device is None:
                    import torch as _torch
                    self.fastsam_device = "cuda" if _torch.cuda.is_available() else "cpu"
            except Exception:
                self.fastsam_device = "cpu"

            # ← главное отличие: автоматическая установка и загрузка
            self.fastsam = _load_fastsam_model(self.fastsam_model)  # FIX: было self.fastsam_model_path
            print(f"   ✅ FastSAM готов (device={self.fastsam_device})")

    # --------------------------- I/O helpers --------------------------- #
    def _load_example_images(self, directory):
        imgs = []
        if not directory or not os.path.exists(directory):
            return imgs
        for name in sorted(os.listdir(directory)):
            if name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
                p = os.path.join(directory, name)
                try:
                    imgs.append(Image.open(p).convert("RGB"))
                except Exception as e:
                    print(f"   ⚠️ Не удалось загрузить {name}: {e}")
        return imgs

    # --------------------------- SAM masks ---------------------------- #
    def _generate_sam_masks(self, image_np):
        # Если выбран FastSAM — используем его
        if getattr(self, "mask_backend", "sam-hq") == "fastsam":
            return self._generate_fastsam_masks(image_np)
        # Если выбран SAM2 — используем его генератор
        if getattr(self, "mask_backend", "sam-hq") == "sam2":
            print(f"🚀 ЭТАП 1: SAM2 автогенерация масок ...")
            t0 = time.time()
            masks = self.sam2_mask_generator.generate(image_np)
            print(f"   🔍 SAM2 всего сгенерировал {len(masks)} масок за {time.time()-t0:.2f} сек")
            return masks
        # иначе — стандартный SAM-HQ путь ниже
        from segment_anything_hq import SamAutomaticMaskGenerator
        original_h, original_w = image_np.shape[:2]
        run_img = image_np
        scale = 1.0

        if self.sam_long_side and max(original_h, original_w) > self.sam_long_side:
            if original_h >= original_w:
                scale = self.sam_long_side / float(original_h)
            else:
                scale = self.sam_long_side / float(original_w)
            run_img = cv2.resize(image_np, (int(original_w * scale), int(original_h * scale)), interpolation=cv2.INTER_LINEAR)

        h, w = run_img.shape[:2]
        min_region = self.sam_min_region if self.sam_min_region > 0 else max(200, int(self.min_area_frac * (h * w)))

        pps_list = sorted(set(self.points_per_side_multi if self.points_per_side_multi else [self.points_per_side]))
        print(f"🚀 ЭТАП 1: SAM автогенерация масок на {w}x{h} (scale={scale:.3f}), сетки={pps_list} ...")

        all_masks = []
        t0 = time.time()
        for pps in pps_list:
            mg = SamAutomaticMaskGenerator(
                model=self.searchdet_sam,
                points_per_side=int(pps),
                points_per_batch=self.points_per_batch,
                pred_iou_thresh=self.pred_iou_thresh,
                stability_score_thresh=self.stability_score_thresh,
                min_mask_region_area=min_region,
                box_nms_thresh=self.box_nms_thresh,
                crop_nms_thresh=self.crop_nms_thresh,
                crop_n_layers=self.crop_n_layers,
                crop_n_points_downscale_factor=self.crop_n_points_downscale_factor,
            )
            masks = mg.generate(run_img)
            print(f"   • Сетка {pps}: {len(masks)} масок")
            all_masks.extend(masks)
        print(f"   🔍 SAM всего сгенерировал {len(all_masks)} масок за {time.time()-t0:.2f} сек (до рескейла)")

        # если был даунскейл — вернём маски к исходному размеру
        if scale != 1.0:
            scaled = []
            for m in all_masks:
                seg = m['segmentation'].astype(np.uint8)
                seg = cv2.resize(seg, (original_w, original_h), interpolation=cv2.INTER_NEAREST) > 0
                rows = np.any(seg, axis=1); cols = np.any(seg, axis=0)
                if rows.any() and cols.any():
                    yidx = np.where(rows)[0]; xidx = np.where(cols)[0]
                    y1, y2 = yidx[0], yidx[-1]; x1, x2 = xidx[0], xidx[-1]
                    bbox = [x1, y1, x2-x1, y2-y1]
                else:
                    bbox = [0, 0, 0, 0]
                m2 = dict(m)
                m2['segmentation'] = seg
                m2['bbox'] = bbox
                scaled.append(m2)
            all_masks = scaled

        if self.sam_max_after_merge and self.sam_max_after_merge > 0:
            all_masks = sorted(all_masks, key=lambda m: int(np.sum(m['segmentation'] > 0)), reverse=True)[: self.sam_max_after_merge]

        return all_masks

    def _generate_fastsam_masks(self, image_np):
        """Генерация масок через FastSAM с тем же даунскейлом по long_side."""
        assert self.fastsam is not None, "FastSAM не инициализирован"

        import cv2
        import time
        H0, W0 = image_np.shape[:2]
        run_img = image_np
        scale = 1.0

        if self.sam_long_side and max(H0, W0) > self.sam_long_side:
            if H0 >= W0:
                scale = self.sam_long_side / float(H0)
            else:
                scale = self.sam_long_side / float(W0)
            run_img = cv2.resize(image_np, (int(W0 * scale), int(H0 * scale)), interpolation=cv2.INTER_LINEAR)

        h, w = run_img.shape[:2]
        print(f"🚀 ЭТАП 1: FastSAM автогенерация масок на {w}x{h} (scale={scale:.3f}) ...")

        # Прогон FastSAM (everything)
        fastsam_start = time.time()
        results = self.fastsam(
            source=run_img,
            imgsz=self.fastsam_imgsz,
            conf=self.fastsam_conf,
            iou=self.fastsam_iou,
            device=self.fastsam_device,
            retina_masks=self.fastsam_retina,
            verbose=False,
        )
        fastsam_time = time.time() - fastsam_start

        if not results:
            print("   ⚠️ FastSAM вернул пустой результат")
            return []

        r0 = results[0]
        if getattr(r0, "masks", None) is None or getattr(r0.masks, "data", None) is None:
            print("   ⚠️ FastSAM: нет масок в результате")
            return []

        import numpy as np
        masks_t = r0.masks.data  # [N, Hm, Wm] (torch)
        try:
            masks_np = masks_t.cpu().numpy().astype(np.uint8)  # 0/1
        except Exception:
            masks_np = np.array(masks_t).astype(np.uint8)

        out = []
        for seg in masks_np:
            seg_u8 = (seg > 0).astype(np.uint8)

            # скейлим к исходному размеру
            if scale != 1.0:
                seg_u8 = cv2.resize(seg_u8, (W0, H0), interpolation=cv2.INTER_NEAREST)
            H, W = seg_u8.shape[:2]

            ys, xs = np.where(seg_u8 > 0)
            if ys.size and xs.size:
                x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
                bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
            else:
                bbox = [0, 0, 0, 0]

            out.append({
                "segmentation": (seg_u8 > 0),
                "bbox": bbox,
                "area": int(seg_u8.sum()),
                "stability_score": 1.0,   # FastSAM не даёт этот скор — ставим фиктивно
                "predicted_iou": 1.0,     # то же
                "crop_box": [0, 0, W, H],
            })

        print(f"   🔍 FastSAM сгенерировал {len(out)} масок за {fastsam_time:.3f} сек")
        return out

    def _filter_nested_masks(self, masks):
        """Фильтрует вложенные маски, оставляя только внешние.
        Если маска почти полностью содержится в другой, меньшая удаляется.
        """
        if not masks or len(masks) <= 1:
            return masks

        # Сортируем маски по площади по убыванию, чтобы сначала обрабатывать большие маски
        sorted_masks = sorted(masks, key=lambda m: m['area'], reverse=True)

        keep_masks_indices = set(range(len(sorted_masks)))

        for i, m_outer in enumerate(sorted_masks):
            if i not in keep_masks_indices: # Если внешняя маска уже удалена
                continue

            seg_outer = m_outer['segmentation']

            for j, m_inner in enumerate(sorted_masks):
                if i == j or j not in keep_masks_indices: # Пропускаем ту же маску или уже удаленные
                    continue

                seg_inner = m_inner['segmentation']
                are_inner = m_inner['area']

                # Проверяем, содержится ли m_inner в m_outer
                if self._is_contained(seg_inner, seg_outer, are_inner, self.containment_iou_threshold):
                    keep_masks_indices.discard(j) # Удаляем меньшую маску

        final_masks = [sorted_masks[i] for i in sorted(list(keep_masks_indices))]
        print(f"   🔗 Фильтр вложенных масок ({self.containment_iou_threshold:.2f} IoU): {len(masks)} → {len(final_masks)}")
        return final_masks

    def _merge_overlapping_masks(self, masks, iou_threshold=0.7):
        if not masks:
            return []
        merged, used = [], set()
        for i, m1 in enumerate(masks):
            if i in used: continue
            seg1 = m1['segmentation']
            group = [m1]; idxs = {i}
            for j, m2 in enumerate(masks[i+1:], i+1):
                if j in used: continue
                seg2 = m2['segmentation']
                inter = np.logical_and(seg1, seg2).sum()
                union = np.logical_or(seg1, seg2).sum()
                if union > 0 and inter / union > iou_threshold:
                    group.append(m2); idxs.add(j)
            if len(group) > 1:
                seg = group[0]['segmentation'].copy()
                for g in group[1:]:
                    seg = np.logical_or(seg, g['segmentation'])
                rows = np.any(seg, axis=1); cols = np.any(seg, axis=0)
                if rows.any() and cols.any():
                    yidx = np.where(rows)[0]; xidx = np.where(cols)[0]
                    y1, y2 = yidx[0], yidx[-1]; x1, x2 = xidx[0], xidx[-1]
                    bbox = [x1, y1, x2-x1, y2-y1]
                else:
                    bbox = [0,0,0,0]
                merged.append({
                    'segmentation': seg,
                    'area': int(seg.sum()),
                    'bbox': bbox,
                    'stability_score': max(g.get('stability_score',0) for g in group),
                    'predicted_iou': max(g.get('predicted_iou',0) for g in group),
                    'crop_box': group[0].get('crop_box', [0,0,seg.shape[1], seg.shape[0]])
                })
            else:
                merged.append(m1)
            used.update(idxs)
        print(f"   🔗 Слияние масок: {len(masks)} → {len(merged)}")
        return merged

    # ----------------------- border handling -------------------------- #
    def _drop_or_clip_border_masks(self, masks, image_np):
        if not masks or not self.ban_border_masks:
            return masks
        H, W = image_np.shape[:2]
        bw = max(1, min(self.border_width, H // 2, W // 2))

        # Готовим булеву маску рамки шириной bw
        border_ring = np.zeros((H, W), dtype=bool)
        border_ring[:bw, :] = True
        border_ring[-bw:, :] = True
        border_ring[:, :bw] = True
        border_ring[:, -bw:] = True

        keep = []
        dropped = clipped = 0
        total_pix = float(H * W)

        for m in masks:
            seg = m['segmentation'].astype(bool)
            if not seg.any():
                continue
            touches = bool(np.any(seg & border_ring))
            if not touches:
                keep.append(m)
                continue

            area = int(seg.sum())
            area_frac = area / (total_pix + 1e-9)

            if self.border_clip_small and area_frac <= self.border_clip_max_frac:
                # Аккуратно срезаем пиксели на рамке
                new_seg = seg.copy()
                new_seg[border_ring] = False
                if new_seg.any():
                    rows = np.any(new_seg, axis=1); cols = np.any(new_seg, axis=0)
                    yidx = np.where(rows)[0]; xidx = np.where(cols)[0]
                    y1, y2 = int(yidx[0]), int(yidx[-1]); x1, x2 = int(xidx[0]), int(xidx[-1])
                    m2 = dict(m)
                    m2['segmentation'] = new_seg
                    m2['bbox'] = [x1, y1, x2-x1, y2-y1]
                    m2['area'] = int(new_seg.sum())
                    keep.append(m2)
                    clipped += 1
                else:
                    dropped += 1
            else:
                dropped += 1
        print(f"   🧱 Фильтр краёв (bw={bw}px): {len(masks)} → {len(keep)} (dropped={dropped}, clipped={clipped})")
        return keep

    # ----------------------- area-only filter ------------------ #
    def _filter_by_area_only(self, masks, image_np):
        if not masks:
            return []
        H, W = image_np.shape[:2]
        total = float(H * W)
        keep = []
        dropped_small = dropped_big = 0
        for m in masks:
            seg = m['segmentation']
            if seg is None:
                continue
            a = int(np.sum(seg > 0))
            frac = a / (total + 1e-9)
            if frac < self.min_area_frac:
                dropped_small += 1
                continue
            if frac > self.max_area_frac:
                dropped_big += 1
                continue
            keep.append(m)
        print(f"   📏 Фильтр размера ({self.min_area_frac*100:.1f}% - {self.max_area_frac*100:.1f}%): {len(masks)} → {len(keep)} (small={dropped_small}, big={dropped_big})")
        return keep

    # --------------------------- Embeddings --------------------------- #
    def _extract_mask_embeddings(self, image_np, masks):
        if not masks:
            return np.zeros((0, 1), dtype=np.float32), []
        print(f"   🔧 Извлечение эмбеддингов: layer={self.searchdet_layer}, transform={type(self.searchdet_transform)}")
        vecs = extract_features_from_masks(
            image_np, masks, self.searchdet_resnet, self.searchdet_layer, self.searchdet_transform
        )
        vecs = np.asarray(vecs, dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs[None, :]
        vecs = np.nan_to_num(vecs, nan=0.0, posinf=1.0, neginf=-1.0)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        ok = (norms.squeeze(-1) > 1e-8) & np.isfinite(norms.squeeze(-1))
        vecs = vecs[ok]
        idx_map = np.nonzero(ok)[0].tolist()
        if vecs.size == 0:
            return np.zeros((0,1), dtype=np.float32), []
        vecs = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-8)
        print(f"   🔍 Embeddings shape: {vecs.shape}, range: [{vecs.min():.6f}, {vecs.max():.6f}]")
        return vecs.astype(np.float32), idx_map

    def _build_queries(self, pos_imgs, neg_imgs):
        if len(pos_imgs) == 0:
            return np.zeros((0,1), dtype=np.float32), np.zeros((0,1), dtype=np.float32)

        pos_emb = np.stack(
            [get_vector(img, self.searchdet_resnet, self.searchdet_layer, self.searchdet_transform).numpy()
             for img in pos_imgs], axis=0
        ).astype(np.float32)

        if len(neg_imgs) > 0:
            neg_emb = np.stack(
                [get_vector(img, self.searchdet_resnet, self.searchdet_layer, self.searchdet_transform).numpy()
                 for img in neg_imgs], axis=0
            ).astype(np.float32)
        else:
            neg_emb = np.zeros((0, pos_emb.shape[1]), dtype=np.float32)

        if neg_emb.shape[0] == 0:
            queries = pos_emb
        else:
            queries = np.stack([adjust_embedding(q, pos_emb, neg_emb) for q in pos_emb], axis=0).astype(np.float32)

        queries = queries / np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-8)
        if neg_emb.shape[0] > 0:
            neg_emb = neg_emb / np.maximum(np.linalg.norm(neg_emb, axis=1, keepdims=True), 1e-8)

        return queries.astype(np.float32), neg_emb.astype(np.float32)

    # --------------------------- Scoring ------------------------------- #
    @staticmethod
    def _cosine_matrix(A, B):
        if A.size == 0 or B.size == 0:
            return np.zeros((A.shape[0], B.shape[0]), dtype=np.float32)
        return (A @ B.T).astype(np.float32)

    def _aggregate_positive(self, sims_pos):
        if sims_pos.size == 0:
            return np.zeros((sims_pos.shape[0],), dtype=np.float32)
        maxv = sims_pos.max(axis=1)
        k = min(self.topk, sims_pos.shape[1]) if sims_pos.shape[1] > 0 else 1
        topk_mean = (np.partition(sims_pos, -k, axis=1)[:, -k:].mean(axis=1)) if k > 1 else maxv
        agg = 0.7 * maxv + 0.3 * topk_mean
        agg = (agg + 1.0) * 0.5  # → [0..1]
        return np.nan_to_num(agg, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # --------------------------- NMS ---------------------------------- #
    @staticmethod
    def _iou_xyxy(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1 + 1), max(0, iy2 - iy1 + 1)
        inter = iw * ih
        aarea = (ax2 - ax1 + 1) * (ay2 - ay1 + 1)
        barea = (bx2 - bx1 + 1) * (by2 - by1 + 1)
        union = aarea + barea - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _calculate_mask_bbox_iou(mask_segmentation, mask_bbox) -> float:
        """
        Вычисляет IoU между маской и ее ограничивающим прямоугольником.
        """
        if not mask_segmentation.any():
            return 0.0

        # Создаем маску из bbox
        x, y, w, h = mask_bbox
        bbox_mask = np.zeros_like(mask_segmentation, dtype=bool)
        bbox_mask[y:y+h, x:x+w] = True

        intersection = np.logical_and(mask_segmentation, bbox_mask).sum()
        union = np.logical_or(mask_segmentation, bbox_mask).sum()

        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _is_contained(seg_inner, seg_outer, are_inner, containment_iou_threshold) -> bool:
        """
        Проверяет, содержится ли seg_inner (меньшая маска) внутри seg_outer (большей маски)
        по заданному IoU порогу.
        """
        if not seg_inner.any() or not seg_outer.any():
            return False

        intersection = np.logical_and(seg_inner, seg_outer).sum()
        
        if are_inner == 0: # Avoid division by zero
            return False

        # Containment IoU: intersection over area of the inner mask
        containment_iou = intersection / are_inner
        return containment_iou >= containment_iou_threshold

    def _filter_perfect_rectangles(self, masks):
        """
        Фильтрует маски, которые слишком похожи на идеальный прямоугольник.
        """
        if not masks:
            return []

        filtered_masks = []
        initial_count = len(masks)
        for i, m in enumerate(masks):
            seg = m['segmentation']
            bbox = m['bbox']

            # Вычисляем IoU маски с ее ограничивающим прямоугольником
            bbox_iou = self._calculate_mask_bbox_iou(seg, bbox)

            # Если IoU очень высокое, считаем маску прямоугольной и отбрасываем
            if bbox_iou >= self.perfect_rectangle_iou_threshold:
                # print(f"   🔗 DEBUG: Dropping mask {i} (area {m['area']}) - bbox_iou {bbox_iou:.4f} >= {self.perfect_rectangle_iou_threshold}")
                continue
            
            filtered_masks.append(m)
        
        print(f"   🔗 Фильтр прямоугольных масок ({self.perfect_rectangle_iou_threshold:.2f} IoU): {initial_count} → {len(filtered_masks)}")
        return filtered_masks

    def _filter_nested_masks(self, masks):
        """Фильтрует вложенные маски, оставляя только внешние.
        Если маска почти полностью содержится в другой, меньшая удаляется.
        """
        if not masks or len(masks) <= 1:
            return masks

        # Сортируем маски по площади по убыванию, чтобы сначала обрабатывать большие маски
        sorted_masks = sorted(masks, key=lambda m: m['area'], reverse=True)

        keep_masks_indices = set(range(len(sorted_masks)))

        for i, m_outer in enumerate(sorted_masks):
            if i not in keep_masks_indices: # Если внешняя маска уже удалена
                continue

            seg_outer = m_outer['segmentation']

            for j, m_inner in enumerate(sorted_masks):
                if i == j or j not in keep_masks_indices: # Пропускаем ту же маску или уже удаленные
                    continue

                seg_inner = m_inner['segmentation']
                are_inner = m_inner['area']

                # Проверяем, содержится ли m_inner в m_outer
                if self._is_contained(seg_inner, seg_outer, are_inner, self.containment_iou_threshold):
                    keep_masks_indices.discard(j) # Удаляем меньшую маску

        final_masks = [sorted_masks[i] for i in sorted(list(keep_masks_indices))]
        print(f"   🔗 Фильтр вложенных масок ({self.containment_iou_threshold:.2f} IoU): {len(masks)} → {len(final_masks)}")
        return final_masks

    def _merge_overlapping_masks(self, masks, iou_threshold=0.7):
        if not masks:
            return []
        merged, used = [], set()
        for i, m1 in enumerate(masks):
            if i in used: continue
            seg1 = m1['segmentation']
            group = [m1]; idxs = {i}
            for j, m2 in enumerate(masks[i+1:], i+1):
                if j in used: continue
                seg2 = m2['segmentation']
                inter = np.logical_and(seg1, seg2).sum()
                union = np.logical_or(seg1, seg2).sum()
                if union > 0 and inter / union > iou_threshold:
                    group.append(m2); idxs.add(j)
            if len(group) > 1:
                seg = group[0]['segmentation'].copy()
                for g in group[1:]:
                    seg = np.logical_or(seg, g['segmentation'])
                rows = np.any(seg, axis=1); cols = np.any(seg, axis=0)
                if rows.any() and cols.any():
                    yidx = np.where(rows)[0]; xidx = np.where(cols)[0]
                    y1, y2 = yidx[0], yidx[-1]; x1, x2 = xidx[0], xidx[-1]
                    bbox = [x1, y1, x2-x1, y2-y1]
                else:
                    bbox = [0,0,0,0]
                merged.append({
                    'segmentation': seg,
                    'area': int(seg.sum()),
                    'bbox': bbox,
                    'stability_score': max(g.get('stability_score',0) for g in group),
                    'predicted_iou': max(g.get('predicted_iou',0) for g in group),
                    'crop_box': group[0].get('crop_box', [0,0,seg.shape[1], seg.shape[0]])
                })
            else:
                merged.append(m1)
            used.update(idxs)
        print(f"   🔗 Слияние масок: {len(masks)} → {len(merged)}")
        return merged

    # ----------------------- border handling -------------------------- #
    def _drop_or_clip_border_masks(self, masks, image_np):
        if not masks or not self.ban_border_masks:
            return masks
        H, W = image_np.shape[:2]
        bw = max(1, min(self.border_width, H // 2, W // 2))

        # Готовим булеву маску рамки шириной bw
        border_ring = np.zeros((H, W), dtype=bool)
        border_ring[:bw, :] = True
        border_ring[-bw:, :] = True
        border_ring[:, :bw] = True
        border_ring[:, -bw:] = True

        keep = []
        dropped = clipped = 0
        total_pix = float(H * W)

        for m in masks:
            seg = m['segmentation'].astype(bool)
            if not seg.any():
                continue
            touches = bool(np.any(seg & border_ring))
            if not touches:
                keep.append(m)
                continue

            area = int(seg.sum())
            area_frac = area / (total_pix + 1e-9)

            if self.border_clip_small and area_frac <= self.border_clip_max_frac:
                # Аккуратно срезаем пиксели на рамке
                new_seg = seg.copy()
                new_seg[border_ring] = False
                if new_seg.any():
                    rows = np.any(new_seg, axis=1); cols = np.any(new_seg, axis=0)
                    yidx = np.where(rows)[0]; xidx = np.where(cols)[0]
                    y1, y2 = int(yidx[0]), int(yidx[-1]); x1, x2 = int(xidx[0]), int(xidx[-1])
                    m2 = dict(m)
                    m2['segmentation'] = new_seg
                    m2['bbox'] = [x1, y1, x2-x1, y2-y1]
                    m2['area'] = int(new_seg.sum())
                    keep.append(m2)
                    clipped += 1
                else:
                    dropped += 1
            else:
                dropped += 1
        print(f"   🧱 Фильтр краёв (bw={bw}px): {len(masks)} → {len(keep)} (dropped={dropped}, clipped={clipped})")
        return keep

    # ----------------------- area-only filter ------------------ #
    def _filter_by_area_only(self, masks, image_np):
        if not masks:
            return []
        H, W = image_np.shape[:2]
        total = float(H * W)
        keep = []
        dropped_small = dropped_big = 0
        for m in masks:
            seg = m['segmentation']
            if seg is None:
                continue
            a = int(np.sum(seg > 0))
            frac = a / (total + 1e-9)
            if frac < self.min_area_frac:
                dropped_small += 1
                continue
            if frac > self.max_area_frac:
                dropped_big += 1
                continue
            keep.append(m)
        print(f"   📏 Фильтр размера ({self.min_area_frac*100:.1f}% - {self.max_area_frac*100:.1f}%): {len(masks)} → {len(keep)} (small={dropped_small}, big={dropped_big})")
        return keep

    # --------------------------- Embeddings --------------------------- #
    def _extract_mask_embeddings(self, image_np, masks):
        if not masks:
            return np.zeros((0, 1), dtype=np.float32), []
        print(f"   🔧 Извлечение эмбеддингов: layer={self.searchdet_layer}, transform={type(self.searchdet_transform)}")
        vecs = extract_features_from_masks(
            image_np, masks, self.searchdet_resnet, self.searchdet_layer, self.searchdet_transform
        )
        vecs = np.asarray(vecs, dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs[None, :]
        vecs = np.nan_to_num(vecs, nan=0.0, posinf=1.0, neginf=-1.0)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        ok = (norms.squeeze(-1) > 1e-8) & np.isfinite(norms.squeeze(-1))
        vecs = vecs[ok]
        idx_map = np.nonzero(ok)[0].tolist()
        if vecs.size == 0:
            return np.zeros((0,1), dtype=np.float32), []
        vecs = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-8)
        print(f"   🔍 Embeddings shape: {vecs.shape}, range: [{vecs.min():.6f}, {vecs.max():.6f}]")
        return vecs.astype(np.float32), idx_map

    def _build_queries(self, pos_imgs, neg_imgs):
        if len(pos_imgs) == 0:
            return np.zeros((0,1), dtype=np.float32), np.zeros((0,1), dtype=np.float32)

        pos_emb = np.stack(
            [get_vector(img, self.searchdet_resnet, self.searchdet_layer, self.searchdet_transform).numpy()
             for img in pos_imgs], axis=0
        ).astype(np.float32)

        if len(neg_imgs) > 0:
            neg_emb = np.stack(
                [get_vector(img, self.searchdet_resnet, self.searchdet_layer, self.searchdet_transform).numpy()
                 for img in neg_imgs], axis=0
            ).astype(np.float32)
        else:
            neg_emb = np.zeros((0, pos_emb.shape[1]), dtype=np.float32)

        if neg_emb.shape[0] == 0:
            queries = pos_emb
        else:
            queries = np.stack([adjust_embedding(q, pos_emb, neg_emb) for q in pos_emb], axis=0).astype(np.float32)

        queries = queries / np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-8)
        if neg_emb.shape[0] > 0:
            neg_emb = neg_emb / np.maximum(np.linalg.norm(neg_emb, axis=1, keepdims=True), 1e-8)

        return queries.astype(np.float32), neg_emb.astype(np.float32)

    # --------------------------- Scoring ------------------------------- #
    @staticmethod
    def _cosine_matrix(A, B):
        if A.size == 0 or B.size == 0:
            return np.zeros((A.shape[0], B.shape[0]), dtype=np.float32)
        return (A @ B.T).astype(np.float32)

    def _aggregate_positive(self, sims_pos):
        if sims_pos.size == 0:
            return np.zeros((sims_pos.shape[0],), dtype=np.float32)
        maxv = sims_pos.max(axis=1)
        k = min(self.topk, sims_pos.shape[1]) if sims_pos.shape[1] > 0 else 1
        topk_mean = (np.partition(sims_pos, -k, axis=1)[:, -k:].mean(axis=1)) if k > 1 else maxv
        agg = 0.7 * maxv + 0.3 * topk_mean
        agg = (agg + 1.0) * 0.5  # → [0..1]
        return np.nan_to_num(agg, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # --------------------------- NMS ---------------------------------- #
    @staticmethod
    def _iou_xyxy(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1 + 1), max(0, iy2 - iy1 + 1)
        inter = iw * ih
        aarea = (ax2 - ax1 + 1) * (ay2 - ay1 + 1)
        barea = (bx2 - bx1 + 1) * (by2 - by1 + 1)
        union = aarea + barea - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _calculate_mask_bbox_iou(mask_segmentation, mask_bbox) -> float:
        """
        Вычисляет IoU между маской и ее ограничивающим прямоугольником.
        """
        if not mask_segmentation.any():
            return 0.0

        # Создаем маску из bbox
        x, y, w, h = mask_bbox
        bbox_mask = np.zeros_like(mask_segmentation, dtype=bool)
        bbox_mask[y:y+h, x:x+w] = True

        intersection = np.logical_and(mask_segmentation, bbox_mask).sum()
        union = np.logical_or(mask_segmentation, bbox_mask).sum()

        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _is_contained(seg_inner, seg_outer, are_inner, containment_iou_threshold) -> bool:
        """
        Проверяет, содержится ли seg_inner (меньшая маска) внутри seg_outer (большей маски)
        по заданному IoU порогу.
        """
        if not seg_inner.any() or not seg_outer.any():
            return False

        intersection = np.logical_and(seg_inner, seg_outer).sum()
        
        if are_inner == 0: # Avoid division by zero
            return False

        # Containment IoU: intersection over area of the inner mask
        containment_iou = intersection / are_inner
        return containment_iou >= containment_iou_threshold

    def _nms(self, elements):
        if len(elements) <= 1:
            return elements
        order = sorted(range(len(elements)), key=lambda i: elements[i]['confidence'], reverse=True)
        keep, taken = [], set()
        for i in order:
            if i in taken: continue
            keep.append(elements[i])
            bi = elements[i]['bbox_xyxy']
            for j in order:
                if j == i or j in taken: continue
                bj = elements[j]['bbox_xyxy']
                if self._iou_xyxy(bi, bj) >= self.nms_iou:
                    taken.add(j)
        return keep

    # --------------------------- Main search -------------------------- #
    def find_present_elements(self, image_path, positive_dir, negative_dir=None):
        print(f"🔍 SearchDet-only анализ: {image_path}" + "="*60)
        print("🔄 ДЕТАЛЬНАЯ ПОСЛЕДОВАТЕЛЬНОСТЬ ВЫПОЛНЕНИЯ HYBRID PIPELINE:")
        print("=" * 80)
        print("7️⃣ hybrid_searchdet_pipeline.py → find_present_elements()")
        t_total = time.time()
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Не удалось загрузить изображение: {image_path}")
        image_np = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 0) примеры
        print("8️⃣ Шаг 1: _load_example_images() - загрузка positive/negative примеров")
        pos_imgs = self._load_example_images(positive_dir)
        neg_imgs = self._load_example_images(negative_dir) if negative_dir else []
        if len(pos_imgs) == 0:
            print("   ❌ Нет положительных примеров — прекращаем.")
            return {"found_elements": [], "masks": []}
        print(f"   📁 Positive: {len(pos_imgs)}	📁 Negative: {len(neg_imgs)}")

        # 1) SAM маски
        print("9️⃣ Шаг 2: _generate_sam_masks() - генерация масок через SAM/FastSAM")
        masks = self._generate_sam_masks(image_np)
        print("🔟 Шаг 3: _filter_perfect_rectangles() - фильтр идеальных прямоугольников")
        masks = self._filter_perfect_rectangles(masks)
        print("1️⃣1️⃣ Шаг 4: _drop_or_clip_border_masks() - обработка границ")
        masks = self._drop_or_clip_border_masks(masks, image_np)
        print("1️⃣2️⃣ Шаг 5: _filter_nested_masks() - фильтр вложенных масок")
        masks = self._filter_nested_masks(masks)
        print("1️⃣3️⃣ Шаг 6: _merge_overlapping_masks() - объединение перекрывающихся масок")
        masks = self._merge_overlapping_masks(masks, iou_threshold=0.7)
        print("1️⃣4️⃣ Шаг 7: _filter_by_area_only() - фильтр по площади")
        # Фильтр по площади включен: учитывает --min-area-frac и --max-area-frac
        masks = self._filter_by_area_only(masks, image_np)
        if not masks:
            print("   ❌ Нет валидных масок после фильтров.")
            return {"found_elements": [], "masks": []}

        # 2) эмбеддинги
        print("1️⃣5️⃣ Шаг 8: _extract_mask_embeddings() - извлечение эмбеддингов масок")
        print("🧠 ЭТАП 2: Эмбеддинги масок и запросов...")
        mask_vecs, idx_map = self._extract_mask_embeddings(image_np, masks)
        if mask_vecs.shape[0] == 0:
            print("   ❌ Не удалось получить эмбеддинги масок.")
            return {"found_elements": [], "masks": []}
        print(f"   📊 Масок с валидными векторами: {mask_vecs.shape[0]}")
        print("1️⃣6️⃣ Шаг 9: _build_queries() - построение эмбеддингов примеров")
        q_pos, q_neg = self._build_queries(pos_imgs, neg_imgs)

        # 3) скоринг
        print("🔍 ЭТАП 3: Сопоставление с positive/negative...")
        sims_pos_raw = self._cosine_matrix(mask_vecs, q_pos)  # [-1..1]
        pos_score = self._aggregate_positive(sims_pos_raw)    # [0..1]
        if q_neg.shape[0] > 0:
            sims_neg = self._cosine_matrix(mask_vecs, q_neg)
            neg_max = ((sims_neg.max(axis=1) + 1.0) * 0.5).astype(np.float32)  # [0..1]
        else:
            neg_max = np.zeros_like(pos_score)

        # консенсус по позитивам
        consensus_cnt = (((sims_pos_raw + 1.0) * 0.5) >= self.consensus_thr).sum(axis=1)

        # правила приёма
        eps = 1e-6
        accept = (pos_score >= self.min_confidence) & \
                 ((pos_score - neg_max) >= self.margin) & \
                 ((pos_score / (neg_max + eps)) >= self.ratio) & \
                 (consensus_cnt >= max(1, self.consensus_k)) & \
                 (neg_max <= self.neg_cap)

        # pre-debug: запомним топ-K кандидатов по pos_score
        pre_order = np.argsort(pos_score)[::-1][:self.debug_topk_pre]
        H, W = image_np.shape[:2]
        self._debug_pre_top = []
        for idx in pre_order:
            m = masks[idx_map[idx]]
            seg = m['segmentation'].astype(bool)
            ys, xs = np.where(seg)
            if ys.size == 0 or xs.size == 0:
                continue
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
            self._debug_pre_top.append({
                'mask': seg,
                'bbox_xyxy': [int(x1), int(y1), int(x2), int(y2)],
                'bbox_xyxy_norm': [x1/W, y1/H, x2/W, y2/H],
                'confidence': float(pos_score[idx]),
                'area': int(seg.sum()),
                'source': 'pre-top',
            })

        # 4) формирование результатов
        candidates = []
        for local_idx, ok in enumerate(accept):
            if not ok:
                continue
            m = masks[idx_map[local_idx]]
            seg = m['segmentation'].astype(bool)
            ys, xs = np.where(seg)
            if ys.size == 0 or xs.size == 0:
                continue
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
            candidates.append({
                'mask': seg,
                'bbox_xyxy': [int(x1), int(y1), int(x2), int(y2)],
                'bbox_xyxy_norm': [x1/W, y1/H, x2/W, y2/H],
                'confidence': float(pos_score[local_idx]),
                'area': int(seg.sum()),
                'source': 'auto',
            })

        # сортировка и ограничение по количеству
        candidates.sort(key=lambda e: e['confidence'], reverse=True)
        if self.max_masks > 0:
            candidates = candidates[:self.max_masks]

        # NMS по боксам
        found = self._nms(candidates)
        print(f"   🎯 Принято масок: {len(found)} (после правил и NMS)")
        print(f"   ⏱️ Общее время: {time.time() - t_total:.2f} сек")
        return {"found_elements": found, "masks": masks}

    # --------------------------- Annotations -------------------------- #
    def build_annotations(self, image_bgr, found):
        ann = {
            "image_info": {
                "height": int(image_bgr.shape[0]),
                "width": int(image_bgr.shape[1]),
                "channels": int(image_bgr.shape[2]),
            },
            "searchdet": {"found": len(found)},
            "defects": []
        }
        for i, e in enumerate(found, 1):
            seg = (e['mask'] * 255).astype(np.uint8)
            contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            poly = c.flatten().tolist()
            ann["defects"].append({
                "id": i,
                "category": "found_detail",
                "bbox": [int(x), int(y), int(w), int(h)],
                "area": int((seg > 0).sum()),
                "segmentation": [poly],
                "confidence": float(e.get('confidence', 0.0)),
                "detection_method": "searchdet_found"
            })
        return ann

    # --------------------------- Visualization & Save ----------------- #
    def _overlay_masks(self, image_bgr, to_draw, draw_boxes=True, title=None):
        out = image_bgr.copy()
        if title:
            cv2.putText(out, title, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        for i, e in enumerate(to_draw, 1):
            seg = e['mask']
            mask_u8 = (seg * 255).astype(np.uint8)
            # Используем красный цвет для всех масок
            color = (0, 0, 255)  # Красный цвет в BGR
            colored = np.zeros_like(out)
            colored[seg] = color
            cv2.addWeighted(out, 1 - 0.3, colored, 0.3, 0, out)
            cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                cv2.drawContours(out, cnts, -1, color, 2)
                if draw_boxes:
                    x1, y1, x2, y2 = e['bbox_xyxy']
                    cv2.rectangle(out, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(out, f"#{i} {e['confidence']:.2f}", (x1, max(0,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        return out

    def _contours_only(self, image_bgr, to_draw):
        out = image_bgr.copy()
        for i, e in enumerate(to_draw, 1):
            mask_u8 = (e['mask'] * 255).astype(np.uint8)
            color = (0, 0, 255) if e.get('source') != 'pre-top' else (255, 0, 0)
            cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                cv2.drawContours(out, cnts, -1, color, 3)
                x1, y1, x2, y2 = e['bbox_xyxy']
                cv2.putText(out, f"#{i} {e['confidence']:.2f}", (x1, max(0,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        return out

    def _composite_mask(self, found, shape_hw):
        h, w = shape_hw
        comp = np.zeros((h, w), dtype=np.uint8)
        for i, e in enumerate(found, 1):
            comp[e['mask']] = i
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(1, len(found) + 1):
            rgb[comp == i] = [(i * 50) % 255, (i * 80 + 100) % 255, (i * 120 + 150) % 255]
        return rgb

    def save_all(self, image_path, output_dir, found, masks, annotations):
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        name = Path(image_path).stem
        img_bgr = cv2.imread(str(image_path))

        to_draw = found if len(found) > 0 else getattr(self, "_debug_pre_top", [])

        overlay = self._overlay_masks(img_bgr, to_draw, draw_boxes=True, title=("FOUND" if len(found)>0 else "TOP-K CANDIDATES"))
        contours = self._contours_only(img_bgr, to_draw)
        composite = self._composite_mask(found, img_bgr.shape[:2]) if found else np.zeros_like(img_bgr)

        cv2.imwrite(str(out_dir / f"{name}_overlay.jpg"), overlay)
        cv2.imwrite(str(out_dir / f"{name}_contours.jpg"), contours)
        cv2.imwrite(str(out_dir / f"{name}_composite.png"), composite)

        # отдельные маски
        mdir = out_dir / "masks"
        mdir.mkdir(exist_ok=True)
        for i, e in enumerate(to_draw, 1):
            cv2.imwrite(str(mdir / f"{name}_mask_{i:02d}.png"), (e['mask'] * 255).astype(np.uint8))

        # total binary — белые маски на чёрном фоне (всегда создаём)
        total = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
        src_for_total = found if len(found) > 0 else to_draw
        for e in src_for_total:
            total[e['mask']] = 255

        # если всё ещё пусто — подстрахуемся: склеим всё из папки masks/
        if int(total.sum()) == 0 and mdir.exists():
            for fname in sorted(os.listdir(mdir)):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
                    mm = cv2.imread(str(mdir / fname), cv2.IMREAD_GRAYSCALE)
                    if mm is None:
                        continue
                    if mm.shape[:2] != img_bgr.shape[:2]:
                        mm = cv2.resize(mm, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
                    total[mm > 127] = 255

        # НОВОЕ: обнуляем рамку и в total, если включён запрет
        if self.ban_border_masks:
            bw = max(1, min(self.border_width, total.shape[0] // 2, total.shape[1] // 2))
            total[:bw, :] = 0
            total[-bw:, :] = 0
            total[:, :bw] = 0
            total[:, -bw:] = 0

        # сохранить во всех ожидаемых форматах имён
        cv2.imwrite(str(out_dir / "total_mask.png"), total)
        cv2.imwrite(str(out_dir / f"total_mask_{name}.png"), total)
        cv2.imwrite(str(out_dir / f"{name}_total_mask.png"), total)

        # json
        payload = {
            "timestamp": datetime.now().isoformat(),
            "image_path": str(image_path),
            "params": {
                "min_confidence": self.min_confidence,
                "margin": self.margin,
                "ratio": self.ratio,
                "neg_cap": self.neg_cap,
                "consensus_k": self.consensus_k,
                "consensus_thr": self.consensus_thr,
                "topk": self.topk,
                "max_masks": self.max_masks,
                "pred_iou_thresh": self.pred_iou_thresh,
                "stability_score_thresh": self.stability_score_thresh,
                "points_per_side": self.points_per_side,
                "points_per_side_multi": self.points_per_side_multi,
                "points_per_batch": self.points_per_batch,
                "box_nms_thresh": self.box_nms_thresh,
                "crop_nms_thresh": self.crop_nms_thresh,
                "crop_n_layers": self.crop_n_layers,
                "crop_n_points_downscale_factor": self.crop_n_points_downscale_factor,
                "sam_min_region": self.sam_min_region,
                "sam_long_side": self.sam_long_side,
                "sam_max_after_merge": self.sam_max_after_merge,
                "min_area": self.min_area,
                "min_area_frac": self.min_area_frac,
                "max_area_frac": self.max_area_frac,
                "ban_border_masks": self.ban_border_masks,
                "border_width": self.border_width,
                "border_clip_small": self.border_clip_small,
                "border_clip_max_frac": self.border_clip_max_frac,
                "nms_iou": self.nms_iou,
                "containment_iou_threshold": self.containment_iou_threshold,
                "perfect_rectangle_iou_threshold": self.perfect_rectangle_iou_threshold,
                "layer": self.searchdet_layer,
                "mask_backend": getattr(self, "mask_backend", "sam-hq"),
            },
            "annotations": annotations,
        }
        with open(out_dir / f"{name}_searchdet_results.json", 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"   💾 Результаты сохранены в: {out_dir}")


# ================================ CLI =================================

def _parse_pps_multi(s: str | None):
    if not s:
        return None
    try:
        vals = [int(x.strip()) for x in s.split(',') if x.strip()]
        return [v for v in vals if v > 0]
    except Exception:
        return None

def _download_sam2_weights(encoder: str, dest_path: str):
    url = SAM2_URLS.get(encoder)
    if not url:
        print(f"   ❌ Неизвестный энкодер для SAM2: {encoder}. Укажите путь к весам вручную.")
        return False
    print(f"   ⬇️ Скачиваем веса SAM2 для {encoder} из {url} ...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        print(f"   ✅ Веса SAM2 скачаны: {dest_path}")
        return True
    except Exception as e:
        print(f"   ❌ Не удалось скачать веса SAM2: {e}")
        return False

def main():
    p = argparse.ArgumentParser(description="SearchDet-only: поиск присутствующих элементов по папке positive (анти-FP фильтры)")
    p.add_argument("--image", required=True, help="Путь к изображению")
    p.add_argument("--positive", required=True, help="Папка с примерами искомого объекта (кропы без фона)")
    p.add_argument("--negative", default=None, help="Папка с фоном/анти-примерами (опционально)")
    p.add_argument("--output", default="./output", help="Папка для сохранения результатов")

    # NMS
    p.add_argument("--nms-iou", type=float, default=0.60, help="IoU-порог для NMS по боксам")
    p.add_argument("--containment-iou-threshold", type=float, default=0.95, help="IoU-порог для фильтрации вложенных масок")
    p.add_argument("--perfect-rectangle-iou-threshold", type=float, default=0.99, help="IoU-порог для фильтрации идеальных прямоугольных масок")

    # пороги приёма
    p.add_argument("--min-confidence", type=float, default=0.60)
    p.add_argument("--margin", type=float, default=-0.10)
    p.add_argument("--ratio", type=float, default=0.80)
    p.add_argument("--neg-cap", type=float, default=0.95)
    p.add_argument("--consensus-k", type=int, default=3)
    p.add_argument("--consensus-thr", type=float, default=0.60)
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--max-masks", type=int, default=100)

    # площадь
    p.add_argument("--min-area", type=int, default=800)
    p.add_argument("--min-area-frac", type=float, default=0.03)
    p.add_argument("--max-area-frac", type=float, default=0.60,
                   help="Удалить маску, если её площадь > доли кадра (по умолчанию 0.60 = 60%)")

    # SAM
    p.add_argument("--pred-iou", type=float, default=0.88)
    p.add_argument("--stability", type=float, default=0.90)
    p.add_argument("--pps", type=int, default=64, help="Точки на сторону для одной сетки SAM")
    p.add_argument("--pps-multi", type=str, default=None,
                   help="Несколько сеток SAM через запятую, напр. '48,64,80' — сильно увеличит число масок")
    p.add_argument("--ppb", type=int, default=128, help="points_per_batch для SAM")
    p.add_argument("--box-nms", type=float, default=0.5)
    p.add_argument("--crop-nms", type=float, default=0.5)
    p.add_argument("--crop-layers", type=int, default=0, help="Количество crop layers (0 = отключено для ускорения)")
    p.add_argument("--crop-pts-downscale", type=int, default=3, help="Downscale factor для crop points (3 = быстрее)")
    p.add_argument("--sam-min-region", type=int, default=0,
                   help="Абсолютный минимум площади маски в пикселях (0 = авто)")
    p.add_argument("--sam-long-side", type=int, default=1800)
    p.add_argument("--sam-max-after-merge", type=int, default=0,
                   help="Ограничение количества масок сразу после SAM (0 = без лимита)")

    # выбор бэкенда масок
    p.add_argument("--mask-backend", type=str, default="sam-hq",
                   choices=["sam-hq", "fastsam", "sam2"],
                   help="Генератор масок: 'sam-hq' (по умолчанию), 'fastsam' или 'sam2'")
    p.add_argument("--sam-model", type=str, default="sam-hq/pretrained_checkpoint/sam_hq_vit_l.pth", help="Путь к весам SAM-HQ или SAM2")
    p.add_argument("--sam-encoder", type=str, default="vit_l", help="Энкодер для SAM2: vit_b, vit_l, vit_h и т.д.")
    p.add_argument("--sam2-weights", type=str, default=None, help="Путь к весам SAM2 (если отличается от --sam-model)")

    # FastSAM
    p.add_argument("--fastsam-model", type=str, default="models/FastSAM-x.pt")
    p.add_argument("--fastsam-imgsz", type=int, default=1024)
    p.add_argument("--fastsam-conf", type=float, default=0.4)
    p.add_argument("--fastsam-iou", type=float, default=0.9)
    p.add_argument("--fastsam-retina", action="store_true", default=True)
    p.add_argument("--no-fastsam-retina", dest="fastsam_retina", action="store_false")
    p.add_argument("--fastsam-device", type=str, default=None,
                   help="cuda / cpu (по умолчанию авто)")

    # Фильтр краёв
    border_group = p.add_mutually_exclusive_group()
    border_group.add_argument("--ban-border-masks", dest="ban_border_masks", action="store_true",
                              help="Удалять любые маски, касающиеся рамки изображения (по умолчанию)")
    border_group.add_argument("--no-ban-border-masks", dest="ban_border_masks", action="store_false",
                              help="Разрешить маски, касающиеся рамки")
    p.set_defaults(ban_border_masks=True)
    p.add_argument("--border-width", type=int, default=2, help="Толщина запрещённой рамки, px")
    p.add_argument("--border-clip-small", action="store_true",
                   help="Не удалять маленькие маски на рамке, а срезать край")
    p.add_argument("--border-clip-max-frac", type=float, default=0.02,
                   help="Макс. доля кадра для клипа вместо удаления (по умолчанию 0.02 = 2%)")

    # backbone
    p.add_argument("--layer", type=str, default="layer2")

    # debug
    p.add_argument("--debug-topk-pre", type=int, default=10, help="Сколько кандидатов рисовать, если ничего не принято")

    # GT
    p.add_argument("--ground-truth", default=None, help="Путь к бинарной GT-маске (опционально)")

    args = p.parse_args()

    det = SearchDetDetector(
        min_confidence=args.min_confidence,
        margin=args.margin,
        ratio=args.ratio,
        neg_cap=args.neg_cap,
        consensus_k=args.consensus_k,
        consensus_thr=args.consensus_thr,
        topk=args.topk,
        max_masks=args.max_masks,
        pred_iou_thresh=args.pred_iou,
        stability_score_thresh=args.stability,
        points_per_side=args.pps,
        points_per_side_multi=_parse_pps_multi(args.pps_multi),
        points_per_batch=args.ppb,
        box_nms_thresh=args.box_nms,
        crop_nms_thresh=args.crop_nms,
        crop_n_layers=args.crop_layers,
        crop_n_points_downscale_factor=args.crop_pts_downscale,
        sam_min_region=args.sam_min_region,
        sam_long_side=args.sam_long_side,
        sam_max_after_merge=args.sam_max_after_merge,
        min_area=args.min_area,
        min_area_frac=args.min_area_frac,
        max_area_frac=args.max_area_frac,
        ban_border_masks=args.ban_border_masks,
        border_width=args.border_width,
        border_clip_small=args.border_clip_small,
        border_clip_max_frac=args.border_clip_max_frac,
        nms_iou=args.nms_iou,
        containment_iou_threshold=args.containment_iou_threshold,
        perfect_rectangle_iou_threshold=args.perfect_rectangle_iou_threshold,
        layer_override=args.layer,
        debug_topk_pre=args.debug_topk_pre,
        mask_backend=args.mask_backend,
        sam_model=args.sam_model,
        sam_encoder=args.sam_encoder,
        sam2_weights=args.sam2_weights,
        fastsam_model=args.fastsam_model,
        fastsam_imgsz=args.fastsam_imgsz,
        fastsam_conf=args.fastsam_conf,
        fastsam_iou=args.fastsam_iou,
        fastsam_retina=args.fastsam_retina,
        fastsam_device=args.fastsam_device,
    )

    res = det.find_present_elements(args.image, args.positive, args.negative)
    img_bgr = cv2.imread(str(args.image))
    ann = det.build_annotations(img_bgr, res['found_elements'])
    det.save_all(args.image, args.output, res['found_elements'], res['masks'], ann)

    # GT сравнение (опционально)
    if args.ground_truth:
        gt = cv2.imread(args.ground_truth, cv2.IMREAD_GRAYSCALE)
        if gt is None:
            print(f"   ⚠️ GT не найден: {args.ground_truth}")
        else:
            gt = (cv2.resize(gt, (img_bgr.shape[1], img_bgr.shape[0])) > 127).astype(np.uint8)
            pred = np.zeros_like(gt)
            for e in res['found_elements']:
                pred[e['mask']] = 1
            inter = np.logical_and(gt, pred).sum()
            union = np.logical_or(gt, pred).sum()
            gt_area = gt.sum(); pr_area = pred.sum()
            iou = inter/union if union>0 else 0.0
            dice = (2*inter)/(gt_area+pr_area) if (gt_area+pr_area)>0 else 0.0
            prec = inter/pr_area if pr_area>0 else 0.0
            rec = inter/gt_area if gt_area>0 else 0.0
            print("📊 GT сравнение:")
            print(f"   IoU: {iou*100:.2f}%  |  Dice: {dice*100:.2f}%  |  Precision: {prec*100:.2f}%  |  Recall: {rec*100:.2f}%")


if __name__ == "__main__":
    main()



  """
        ПОДРОБНОЕ ОПИСАНИЕ ПАЙПЛАЙНА `find_present_elements()`

        Основная задача этого пайплайна — найти на целевом изображении объекты, похожие на предоставленные "положительные" примеры, и отфильтровать всё остальное, используя "отрицательные" примеры и различные эвристики.

        ПАЙПЛАЙН ВЫПОЛНЕНИЯ:

        1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
           ЦЕЛЬ: Загрузить все необходимые изображения (целевое, позитивные/негативные примеры) и подготовить их к обработке.
           ├─ 1.1. Загрузка целевого изображения
           │   ├── `cv2.imread(image_path)`: Изображение читается с диска в виде NumPy массива. OpenCV по умолчанию использует цветовое пространство BGR (Blue-Green-Red).
           │   └── `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`: Цветовое пространство конвертируется из BGR в RGB. Это стандарт для большинства других библиотек, включая PIL и PyTorch, и обеспечивает корректное восприятие цветов моделью.
           └─ 1.2. Загрузка примеров (_load_example_images)
               ├── Вызывается для `positive_dir` и `negative_dir`.
               ├── `os.listdir(directory)`: Получает список всех файлов в указанной директории.
               ├── `name.lower().endswith(...)`: Проверяется расширение каждого файла. Загружаются только файлы с расширениями .png, .jpg, .jpeg, .bmp, .webp.
               └── `PIL.Image.open(path).convert("RGB")`: Каждый файл-пример открывается с помощью Pillow и приводится к формату RGB для консистентности.
               └── Результат: два списка изображений в формате `PIL.Image`.

        2. ГЕНЕРАЦИЯ МАСОК-КАНДИДАТОВ (_generate_sam_masks)
           ЦЕЛЬ: Разбить исходное изображение на множество потенциальных объектов (масок) с помощью одной из моделей семейства Segment Anything (SAM).
           ├─ 2.1. Выбор и инициализация бэкенда
           │   ├── В зависимости от параметра `mask_backend` выбирается одна из трёх моделей:
           │   │   ├── `"sam-hq"` (по умолчанию): Segment Anything Model High Quality. Обеспечивает высокое качество масок, хорошо прорабатывает детали и границы.
           │   │   ├── `"sam2"`: Segment Anything Model 2. Новая версия модели с улучшенной производительностью и качеством.
           │   │   └── `"fastsam"`: Быстрая аппроксимация SAM, основанная на YOLOv8. Значительно быстрее, но может давать менее точные маски.
           │   └── Модели инициализируются при создании объекта `SearchDetDetector`, загружая веса с диска.
           │
           ├─ 2.2. Опциональный даунскейл изображения (для SAM-HQ/SAM2/FastSAM)
           │   ├── Проверяется, превышает ли длинная сторона изображения порог `sam_long_side`.
           │   ├── Если да, вычисляется коэффициент масштабирования `scale = sam_long_side / max(height, width)`.
           │   ├── `cv2.resize(image, ..., interpolation=cv2.INTER_LINEAR)`: Изображение пропорционально уменьшается. Используется линейная интерполяция, так как это хороший компромисс между скоростью и качеством для фотореалистичных изображений.
           │   └── ЦЕЛЬ: Ускорить работу SAM, которая очень чувствительна к размеру входного изображения, и снизить потребление видеопамяти.
           │
           ├─ 2.3. Процесс генерации масок (детали по бэкендам)
           │   ├─ A) SAM-HQ (`SamAutomaticMaskGenerator.generate`)
           │   │   ├── Создается объект `SamAutomaticMaskGenerator` с множеством параметров, управляющих процессом:
           │   │   │   ├── `points_per_side`: Плотность сетки точек, размещаемой на изображении. Чем больше значение, тем больше мелких деталей может быть найдено, но тем дольше работает алгоритм. Возможно использование нескольких сеток (`points_per_side_multi`).
           │   │   │   ├── `pred_iou_thresh`: Порог IoU для предсказания качества маски.
           │   │   │   ├── `stability_score_thresh`: Порог "стабильности" маски при изменении порога бинаризации. Отсеивает "шумные" маски.
           │   │   │   ├── `min_mask_region_area`: Минимальная площадь маски в пикселях. Отсеивает слишком мелкие объекты на раннем этапе.
           │   │   │   └── `crop_n_layers`, `crop_nms_thresh`: Параметры, управляющие многоуровневой обработкой кропов изображения для нахождения объектов разного масштаба.
           │   │   └── `mg.generate(run_img)`: Запускается основной процесс, который генерирует маски для каждой точки сетки, фильтрует и объединяет их.
           │   │
           │   ├─ B) SAM2 (`Sam2AutomaticMaskGenerator.generate`)
           │   │   ├── Использует более новый и оптимизированный генератор масок. Процесс концептуально схож с SAM-HQ, но реализован в библиотеке `sam2`.
           │   │   └── `self.sam2_mask_generator.generate(image_np)`: Вызов API проще, параметры инкапсулированы внутри генератора.
           │   │
           │   └─ C) FastSAM (`_generate_fastsam_masks`)
           │       ├── `self.fastsam(...)`: Модель FastSAM применяется к изображению.
           │       ├── `results[0].masks.data`: Результат — это тензор PyTorch формы `[N, H, W]`, где N — количество найденных масок.
           │       └── Конвертация в стандартный формат: Каждая маска из тензора преобразуется в словарь, аналогичный выходу SAM-HQ/SAM2, с полями `segmentation`, `bbox`, `area`. Поля `stability_score` и `predicted_iou` заполняются значениями по умолчанию (1.0).
           │
           └─ 2.4. Рескейл масок к исходному размеру
               ├── Если на шаге 2.2 изображение было уменьшено (`scale != 1.0`), маски нужно вернуть к исходному разрешению.
               ├── `cv2.resize(mask, ..., interpolation=cv2.INTER_NEAREST)`: Каждая бинарная маска увеличивается обратно. Используется интерполяция по ближайшему соседу (`INTER_NEAREST`), чтобы избежать появления полутонов (значения между 0 и 1), сохранив четкие границы маски.
               └── `bbox`, `area`: После рескейла маски, её bounding box и площадь пересчитываются заново.
               └── Результат: Список словарей, где каждый словарь представляет одну маску-кандидата и её свойства (`segmentation`, `bbox`, `area`, ...).

        3. ПОСЛЕДОВАТЕЛЬНАЯ ФИЛЬТРАЦИЯ МАСОК
           ЦЕЛЬ: Уменьшить количество масок-кандидатов, удалив заведомо ложные или избыточные. Фильтры применяются один за другим.
           ├─ 3.1. _filter_perfect_rectangles() — Удаление идеальных прямоугольников
           │   ├── `_calculate_mask_bbox_iou()`: Для каждой маски вычисляется её IoU (Intersection over Union) с её же bounding box.
           │   ├── `perfect_rectangle_iou_threshold`: Если IoU превышает этот порог (например, 0.99), это означает, что форма маски почти идеально совпадает с прямоугольником.
           │   └── ЗАЧЕМ? Часто это соответствует рамкам, плашкам, баннерам или ошибкам сегментации, а не реальным объектам сложной формы. Такие маски удаляются.
           │
           ├─ 3.2. _drop_or_clip_border_masks() — Удаление или обрезка масок на краях
           │   ├── `border_ring`: Создается бинарная маска-кольцо толщиной `border_width` пикселей по периметру изображения.
           │   ├── `np.any(seg & border_ring)`: Проверяется, пересекается ли маска объекта с этим "кольцом".
           │   ├── `ban_border_masks=True`: Если да, маска по умолчанию удаляется. ЗАЧЕМ? Часто фон или части других объектов "прилипают" к краям кадра, и этот фильтр помогает от них избавиться.
           │   └── `border_clip_small=True`: Альтернативное поведение. Если маска касается края, но она маленькая (площадь < `border_clip_max_frac`), то она не удаляется целиком, а лишь обрезается по границе кадра.
           │
           ├─ 3.3. _filter_nested_masks() — Удаление вложенных масок
           │   ├── Сортировка масок по площади по убыванию (от больших к меньшим).
           │   ├── Для каждой пары масок (большой `m_outer` и меньшей `m_inner`) вычисляется "Containment IoU".
           │   ├── Containment IoU = `intersection_area / small_mask_area`. Он показывает, какая доля маленькой маски находится внутри большой.
           │   ├── `containment_iou_threshold`: Если Containment IoU превышает порог (например, 0.9), меньшая маска `m_inner` считается полностью вложенной и удаляется.
           │   └── ЗАЧЕМ? SAM часто генерирует несколько масок для одного объекта: одну для всего объекта, и несколько для его частей. Этот фильтр оставляет только самую большую, внешнюю маску.
           │
           ├─ 3.4. _merge_overlapping_masks() — Объединение сильно перекрывающихся масок
           │   ├── Поиск групп масок, у которых попарный IoU (обычный) превышает `iou_threshold`.
           │   ├── `np.logical_or(mask1, mask2, ...)`: Все маски в найденной группе объединяются в одну большую маску с помощью логического "ИЛИ".
           │   ├── Пересчитываются `bbox` и `area` для новой, объединенной маски.
           │   └── ЗАЧЕМ? Иногда SAM разбивает один объект на несколько смежных, сильно перекрывающихся частей. Этот шаг "склеивает" их обратно в единый объект.
           │
           └─ 3.5. _filter_by_area_only() — Фильтр по относительной площади
               ├── Для каждой маски вычисляется её площадь как доля от общей площади кадра.
               ├── `min_area_frac`, `max_area_frac`: Маска удаляется, если её относительная площадь меньше `min_area_frac` или больше `max_area_frac`.
               └── ЗАЧЕМ? Это базовый, но эффективный способ отсеять слишком мелкий "мусор" или слишком большие маски, захватившие весь фон.

        4. ИЗВЛЕЧЕНИЕ ЭМБЕДДИНГОВ (ВЕКТОРНЫХ ПРЕДСТАВЛЕНИЙ)
           ЦЕЛЬ: Преобразовать маски-кандидаты и изображения-примеры в векторы в общем пространстве признаков для их последующего сравнения.
           ├─ 4.1. _extract_mask_embeddings() — Эмбеддинги для масок-кандидатов
           │   ├── `extract_features_from_masks_fast()`: Вызывается функция из библиотеки SearchDet.
           │   │   ├── Внутри неё: изображение и маски проходят через нейросеть-эмбеддер (ResNet).
           │   │   ├── Извлекается карта признаков (feature map) с указанного слоя (`layer2`, `layer3` или `layer4`).
           │   │   ├── Каждая маска сжимается до размера карты признаков.
           │   │   ├── Masked Pooling: признаки из карты признаков берутся только в тех местах, где маска имеет значение "1".
           │   │   ├── Global Average Pooling: признаки усредняются по всей области маски, чтобы получить один вектор.
           │   └── `L2 Normalization`: Полученные векторы нормализуются (приводятся к единичной длине). Это необходимо для того, чтобы в дальнейшем использовать скалярное произведение как меру косинусного сходства.
           │   └── Результат: `mask_vecs` — NumPy массив формы `[N_valid_masks, embedding_dim]`.
           │
           └─ 4.2. _build_queries() — Эмбеддинги для positive/negative примеров
               ├── `get_vector()`: Для каждого изображения-примера (positive и negative) извлекается вектор признаков. В отличие от масок, здесь используется Global Average Pooling по всей feature map изображения, так как примеры предполагаются "чистыми" (содержат только целевой объект или фон).
               ├── `adjust_embedding()`: Если предоставлены negative-примеры, эмбеддинги positive-примеров корректируются.
               │   ├── Вычисляется центроид (средний вектор) всех negative-примеров.
               │   ├── Каждый positive-вектор "сдвигается" в направлении от этого "негативного" центроида.
               │   └── ЗАЧЕМ? Это делает positive-запросы более уникальными и менее похожими на фон, повышая качество распознавания.
               └── `L2 Normalization`: Итоговые векторы запросов (`queries`) и негативные эмбеддинги (`neg_emb`) также L2-нормализуются.

        5. СКОРИНГ И ПРИНЯТИЕ РЕШЕНИЙ
           ЦЕЛЬ: Для каждой маски-кандидата вычислить оценку (confidence) и принять решение, является ли она искомым объектом.
           ├─ 5.1. _cosine_matrix() — Вычисление сходства
           │   ├── `A @ B.T`: Выполняется матричное умножение между эмбеддингами масок и эмбеддингами запросов. Так как векторы L2-нормализованы, результат этой операции эквивалентен косинусному сходству между векторами.
           │   ├── `sims_pos`: Матрица сходства масок с positive-примерами.
           │   └── `sims_neg`: Матрица сходства масок с negative-примерами.
           │
           ├─ 5.2. _aggregate_positive() — Агрегация positive-скоров
           │   ├── Для каждой маски (строки в `sims_pos`) вычисляется единый positive-скор.
           │   ├── `max_sim`: Максимальное сходство с одним из positive-примеров.
           │   ├── `topk_mean`: Среднее из `k` лучших сходств.
           │   ├── `0.7 * max_sim + 0.3 * topk_mean`: Финальный скор — это взвешенная комбинация. Это делает оценку более робастной, чем простое взятие максимума.
           │   └── `(score + 1.0) * 0.5`: Результат переводится из диапазона `[-1, 1]` в диапазон `[0, 1]`, где 1 — максимальное сходство.
           │
           ├─ 5.3. Проверка consensus (согласия)
           │   ├── `consensus_cnt`: Для каждой маски подсчитывается, сколько positive-примеров дали сходство выше порога `consensus_thr`.
           │   └── ЗАЧЕМ? Чтобы считать маску "принятой", она должна быть похожа не на один случайный positive-пример, а на некоторое минимальное их количество (`consensus_k`).
           │
           └─ 5.4. Применение финальных правил приема
               └── Маска принимается (`accept=True`) только если выполняются ВСЕ условия:
                   ├── `confidence >= min_confidence`: Её итоговый positive-скор выше минимального порога.
                   ├── `positive_score - negative_score >= margin`: Разница между positive и negative скорами должна быть достаточной.
                   ├── `positive_score / (negative_score + eps) >= ratio`: Отношение positive к negative скору должно быть выше порога.
                   └── `negative_score <= neg_cap`: Сходство с негативными примерами не должно быть слишком высоким.

        6. ПОСТОБРАБОТКА
           ЦЕЛЬ: Финальная очистка принятых масок.
           ├─ 6.1. Сортировка и ограничение
           │   ├── Кандидаты сортируются по `confidence` по убыванию.
           │   └── `max_masks`: Оставляется только `top-N` лучших кандидатов.
           │
           └─ 6.2. _nms() — Non-Maximum Suppression (подавление немаксимумов)
               ├── Алгоритм проходит по отсортированному списку кандидатов.
               ├── Самый уверенный кандидат (`confidence`) всегда остается.
               ├── Все остальные кандидаты, у которых `bbox` сильно перекрывается с ним (IoU > `nms_iou_threshold`), удаляются.
               ├── Процесс повторяется для следующего по уверенности кандидата из оставшихся.
               └── ЗАЧЕМ? Убирает дублирующиеся детекции одного и того же объекта.

        7. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ (save_all)
           ЦЕЛЬ: Сохранить все артефакты анализа в удобном для человека и машины виде.
           ├─ 7.1. Визуализации
           │   ├── `_overlay_masks`: Исходное изображение с полупрозрачными наложенными масками, их контурами и `bbox`.
           │   ├── `_contours_only`: Изображение только с контурами найденных объектов.
           │   └── `_composite_mask`: "Семантическая" маска, где каждый найденный объект раскрашен в свой уникальный цвет.
           ├─ 7.2. Маски
           │   ├── Каждая найденная маска сохраняется как отдельный .png файл.
           │   └── `total_mask.png`: Единая бинарная маска, являющаяся объединением всех найденных масок.
           └─ 7.3. Аннотации (JSON)
               ├── `build_annotations`: Формирует детальный JSON-отчет.
               └── Содержит информацию об изображении, параметры запуска, и для каждого найденного объекта: `bbox`, площадь, полигон сегментации (`segmentation`) и `confidence`.

        ПОТОК ДАННЫХ:
        Изображение → [SAM/FastSAM/SAM2] → Маски-кандидаты → [Фильтры] → Очищенные маски → [ResNet-эмбеддер] → Векторы масок ↔ Векторы примеров → [Скоринг] → Оценки confidence → [Правила/NMS] → Финальные объекты → [Визуализация/JSON] → Результаты
        """