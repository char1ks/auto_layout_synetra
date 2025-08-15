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
        """Генерация масок через SAM-HQ (заглушка)."""
        print("⚠️ SAM-HQ генерация масок пока не реализована в модульной версии")
        return []
