#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Генерация масок через различные бэкенды (SAM-HQ, FastSAM, SAM2).
"""

import os
import sys
import subprocess
import shlex
import numpy as np
from pathlib import Path
import urllib.request


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
    """Гарантирует, что импорт ultralytics возможен."""
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
        try:
            from ultralytics import FastSAM  # noqa: F401
            return True
        except Exception as e2:
            print(f"   ❌ Импорт ultralytics всё ещё не работает: {e2}")
            return False


def _load_fastsam_model(model_path: str | None):
    """Возвращает объект FastSAM."""
    if not _ensure_ultralytics(auto_install=True):
        raise RuntimeError("Не удалось подготовить ultralytics/FastSAM.")

    try:
        from ultralytics import FastSAM
    except Exception:
        from ultralytics.models.fastsam import FastSAM

    # Создадим папку models/
    Path("models").mkdir(parents=True, exist_ok=True)

    # Если указан путь и файл существует — грузим по нему
    if model_path and os.path.exists(model_path):
        print(f"   ✅ FastSAM веса найдены локально: {model_path}")
        return FastSAM(model_path)

    # Иначе пробуем автозагрузку
    for name in [model_path, "FastSAM-x.pt", "FastSAM-s.pt"]:
        if not name:
            continue
        try:
            print(f"   ⬇️ Автодокачка FastSAM: {name}")
            return FastSAM(name)
        except Exception as e:
            print(f"   ⚠️ Не удалось: {name} → {e}")

    raise FileNotFoundError("Веса FastSAM не найдены и не удалось автоскачать.")


class MaskGenerator:
    """Генератор масок через различные бэкенды."""
    
    def __init__(self, detector):
        self.detector = detector
        self._fastsam_model = None
        self._sam_generator = None
        
    def generate(self, image_np):
        """Генерирует маски для изображения."""
        print(f"🚀 ЭТАП 1: {self.detector.mask_backend.upper()} автогенерация масок")
        
        if self.detector.mask_backend == "fastsam":
            return self._generate_fastsam_masks(image_np)
        elif self.detector.mask_backend == "sam-hq":
            return self._generate_sam_masks(image_np)
        else:
            raise ValueError(f"Неподдерживаемый бэкенд: {self.detector.mask_backend}")
    
    def _generate_fastsam_masks(self, image_np):
        """Генерация масок через FastSAM."""
        import torch
        
        # Инициализируем модель если нужно
        if self._fastsam_model is None:
            print("⬇️ Автодокачка FastSAM...")
            self._fastsam_model = _load_fastsam_model(getattr(self.detector, 'fastsam_model', None))
            device = getattr(self.detector, 'fastsam_device', None) or ('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"✅ FastSAM готов (device={device})")
        
        # Конвертируем numpy в PIL для FastSAM
        from PIL import Image
        pil_img = Image.fromarray(image_np)
        
        # Параметры FastSAM
        imgsz = getattr(self.detector, 'fastsam_imgsz', 1024)
        conf = getattr(self.detector, 'fastsam_conf', 0.4)
        iou = getattr(self.detector, 'fastsam_iou', 0.9)
        retina = getattr(self.detector, 'fastsam_retina', True)
        
        print(f"   на {image_np.shape[1]}x{image_np.shape[0]} (imgsz={imgsz})")
        
        import time
        t_start = time.time()
        
        # Запускаем FastSAM
        results = self._fastsam_model(
            pil_img,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            retina_masks=retina,
            verbose=False
        )
        
        t_gen = time.time() - t_start
        
        # Извлекаем маски
        if not results or len(results) == 0:
            print("   ❌ FastSAM не вернул результатов")
            return []
        
        result = results[0]
        if result.masks is None:
            print("   ❌ FastSAM не сгенерировал масок")
            return []
        
        # Конвертируем в формат mask_dict
        masks = []
        mask_tensor = result.masks.data  # [N, H, W]
        
        for i in range(mask_tensor.shape[0]):
            mask_np = mask_tensor[i].cpu().numpy().astype(bool)
            
            # Вычисляем bbox
            y_indices, x_indices = np.where(mask_np)
            if len(y_indices) == 0:
                continue
                
            x1, y1 = x_indices.min(), y_indices.min()
            x2, y2 = x_indices.max(), y_indices.max()
            
            mask_dict = {
                "segmentation": mask_np,
                "bbox": [x1, y1, x2, y2],
                "area": mask_np.sum(),
                "predicted_iou": 0.9,  # заглушка
                "stability_score": 0.9,  # заглушка
            }
            masks.append(mask_dict)
        
        print(f"🔍 FastSAM сгенерировал {len(masks)} масок за {t_gen:.3f} сек")
        print(f"🎯 Сгенерировано {len(masks)} масок-кандидатов")
        
        return masks
    
    def _generate_sam_masks(self, image_np):
        """Генерация масок через SAM-HQ."""
        import time
        
        # Инициализируем модель если нужно
        if self._sam_generator is None:
            print("🎯 Инициализация SAM-HQ...")
            self._sam_generator = self._init_sam_hq()
        
        print(f"   на {image_np.shape[1]}x{image_np.shape[0]}")
        
        t_start = time.time()
        
        # Генерируем маски
        masks = self._sam_generator.generate(image_np)
        
        t_gen = time.time() - t_start
        
        print(f"🔍 SAM-HQ сгенерировал {len(masks)} масок за {t_gen:.3f} сек")
        print(f"🎯 Сгенерировано {len(masks)} масок-кандидатов")
        
        return masks
    
    def _init_sam_hq(self):
        """Инициализация SAM-HQ модели."""
        try:
            # Импорты SAM-HQ
            from segment_anything_hq import sam_model_registry, SamAutomaticMaskGenerator
        except ImportError:
            try:
                from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
                print("   ⚠️ Используем обычную SAM вместо SAM-HQ")
            except ImportError:
                raise RuntimeError("Не найден ни SAM-HQ, ни обычная SAM. Установите segment-anything-hq или segment-anything")
        
        # Ищем checkpoint
        sam_checkpoint = self._find_sam_checkpoint()
        
        if not sam_checkpoint:
            raise RuntimeError("Не найден checkpoint SAM-HQ. Запустите с --backend fastsam или установите SAM-HQ")
        
        print(f"   📦 Загружаем SAM из: {sam_checkpoint}")
        
        # Загружаем модель
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_type = "vit_l"  # или "vit_h", "vit_b"
        
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        
        # Создаем генератор масок
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=32,
            points_per_batch=64,
            pred_iou_thresh=0.8,
            stability_score_thresh=0.9,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,
            box_nms_thresh=0.5,
            crop_nms_thresh=0.5,
            crop_overlap_ratio=512 / 1500,
        )
        
        print(f"   ✅ SAM-HQ готов (device={device})")
        return mask_generator
    
    def _find_sam_checkpoint(self):
        """Ищет checkpoint SAM-HQ."""
        import os
        from pathlib import Path
        
        # Возможные пути к checkpoint
        possible_paths = [
            "sam-hq/pretrained_checkpoint/sam_hq_vit_l.pth",
            "models/sam_hq_vit_l.pth", 
            "sam_hq_vit_l.pth",
            # Также пути для обычной SAM
            "models/sam_vit_l_0b3195.pth",
            "sam_vit_l_0b3195.pth"
        ]
        
        for path in possible_paths:
            if os.path.exists(path) and os.path.getsize(path) > 100_000_000:  # Больше 100MB
                return path
        
        # Попробуем автоскачивание SAM-HQ
        return self._download_sam_hq()
    
    def _download_sam_hq(self):
        """Автоскачивание SAM-HQ модели."""
        import urllib.request
        from pathlib import Path
        
        sam_dir = Path("sam-hq/pretrained_checkpoint")
        sam_checkpoint = sam_dir / "sam_hq_vit_l.pth"
        
        if sam_checkpoint.exists() and sam_checkpoint.stat().st_size > 100_000_000:
            return str(sam_checkpoint)
        
        print("   📥 Скачиваем SAM-HQ модель...")
        sam_dir.mkdir(parents=True, exist_ok=True)
        
        url = "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_l.pth"
        
        try:
            urllib.request.urlretrieve(url, sam_checkpoint)
            if sam_checkpoint.stat().st_size > 100_000_000:
                print("   ✅ SAM-HQ скачана успешно")
                return str(sam_checkpoint)
            else:
                print("   ❌ Файл скачался неполностью")
                sam_checkpoint.unlink()  # Удаляем поврежденный файл
        except Exception as e:
            print(f"   ❌ Ошибка скачивания: {e}")
        
        return None
