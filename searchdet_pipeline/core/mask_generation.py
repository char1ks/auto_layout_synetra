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
        import torch, cv2, time
        
        # Инициализируем модель если нужно
        if self._fastsam_model is None:
            print("⬇️ Автодокачка FastSAM...")
            self._fastsam_model = _load_fastsam_model(getattr(self.detector, 'fastsam_model', None))
            self._fastsam_device = getattr(self.detector, 'fastsam_device', None) or ('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"✅ FastSAM готов (device={self._fastsam_device})")
        
        H0, W0 = image_np.shape[:2]
        run_img = image_np
        scale = 1.0
        sam_long_side = getattr(self.detector, 'sam_long_side', None)
        if sam_long_side and max(H0, W0) > sam_long_side:
            if H0 >= W0:
                scale = sam_long_side / float(H0)
            else:
                scale = sam_long_side / float(W0)
            run_img = cv2.resize(image_np, (int(W0 * scale), int(H0 * scale)), interpolation=cv2.INTER_LINEAR)
        h, w = run_img.shape[:2]
        
        # Параметры FastSAM
        imgsz = getattr(self.detector, 'fastsam_imgsz', 1024)
        conf = getattr(self.detector, 'fastsam_conf', 0.4)
        iou = getattr(self.detector, 'fastsam_iou', 0.9)
        retina = getattr(self.detector, 'fastsam_retina', True)
        
        print(f"   на {w}x{h} (scale={scale:.3f}, imgsz={imgsz})")
        t_start = time.time()
        
        # Запускаем FastSAM
        results = self._fastsam_model(
            source=run_img,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=self._fastsam_device,
            retina_masks=retina,
            verbose=False
        )
        t_gen = time.time() - t_start
        
        if not results:
            print("   ⚠️ FastSAM вернул пустой результат")
            return []
        r0 = results[0]
        if getattr(r0, 'masks', None) is None or getattr(r0.masks, 'data', None) is None:
            print("   ⚠️ FastSAM: нет масок в результате")
            return []
        
        mask_tensor = r0.masks.data
        try:
            masks_np = mask_tensor.cpu().numpy().astype(np.uint8)
        except Exception:
            masks_np = np.array(mask_tensor).astype(np.uint8)
        
        out = []
        for seg in masks_np:
            seg_u8 = (seg > 0).astype(np.uint8)
            # Возвращаем к исходному размеру при необходимости
            if scale != 1.0:
                seg_u8 = cv2.resize(seg_u8, (W0, H0), interpolation=cv2.INTER_NEAREST)
            H, W = seg_u8.shape[:2]
            ys, xs = np.where(seg_u8 > 0)
            if ys.size and xs.size:
                x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
                bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]  # XYWH
            else:
                bbox = [0, 0, 0, 0]
            out.append({
                'segmentation': (seg_u8 > 0),
                'bbox': bbox,
                'area': int((seg_u8 > 0).sum()),
                'stability_score': 1.0,
                'predicted_iou': 1.0,
                'crop_box': [0, 0, W, H],
            })
        print(f"🔍 FastSAM сгенерировал {len(out)} масок за {t_gen:.3f} сек")
        print(f"🎯 Сгенерировано {len(out)} масок-кандидатов")
        return out
    
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
        # Сначала проверяем явно указанный путь
        sam_checkpoint = getattr(self.detector, 'sam_model', None)
        if not sam_checkpoint:
            sam_checkpoint = self._find_sam_checkpoint()
        
        if not sam_checkpoint:
            raise RuntimeError("Не найден checkpoint SAM-HQ. Запустите с --backend fastsam или установите SAM-HQ")
        
        print(f"   📦 Загружаем SAM из: {sam_checkpoint}")
        
        # Загружаем модель
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Позволяем выбрать энкодер через detector.sam_encoder (vit_b/vit_l/vit_h)
        model_type = getattr(self.detector, 'sam_encoder', 'vit_l') or 'vit_l'
        
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
        
        # Определяем модель из параметров детектора
        model_type = getattr(self.detector, 'sam_encoder', 'vit_l')
        
        # Возможные пути к checkpoint для указанной модели
        possible_paths = [
            f"sam-hq/pretrained_checkpoint/sam_hq_{model_type}.pth",
            f"models/sam_hq_{model_type}.pth", 
            f"sam_hq_{model_type}.pth",
        ]
        
        # Если не указан конкретный тип, проверяем все варианты
        if not hasattr(self.detector, 'sam_encoder'):
            possible_paths.extend([
                "sam-hq/pretrained_checkpoint/sam_hq_vit_l.pth",
                "models/sam_hq_vit_l.pth", 
                "sam_hq_vit_l.pth",
                "sam-hq/pretrained_checkpoint/sam_hq_vit_h.pth",
                "models/sam_hq_vit_h.pth", 
                "sam_hq_vit_h.pth",
                "sam-hq/pretrained_checkpoint/sam_hq_vit_b.pth",
                "models/sam_hq_vit_b.pth", 
                "sam_hq_vit_b.pth",
                # Также пути для обычной SAM
                "models/sam_vit_l_0b3195.pth",
                "sam_vit_l_0b3195.pth"
            ])
        
        for path in possible_paths:
            if os.path.exists(path) and os.path.getsize(path) > 100_000_000:  # Больше 100MB
                print(f"✅ SAM-HQ модель уже существует: {path}")
                return path
        
        # Попробуем автоскачивание SAM-HQ
        return self._download_sam_hq()
    
    def _download_sam_hq(self):
        """Автоскачивание SAM-HQ модели."""
        import urllib.request
        from pathlib import Path
        
        # Определяем модель из параметров детектора
        model_type = getattr(self.detector, 'sam_encoder', 'vit_l')
        
        sam_dir = Path("sam-hq/pretrained_checkpoint")
        sam_checkpoint = sam_dir / f"sam_hq_{model_type}.pth"
        
        if sam_checkpoint.exists() and sam_checkpoint.stat().st_size > 100_000_000:
            return str(sam_checkpoint)
        
        print(f"   📥 SAM-HQ модель не найдена, скачиваем {model_type}...")
        sam_dir.mkdir(parents=True, exist_ok=True)
        
        # URL для разных моделей
        urls = {
            'vit_b': "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_b.pth",
            'vit_l': "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_l.pth", 
            'vit_h': "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_h.pth"
        }
        
        if model_type not in urls:
            print(f"   ❌ Неизвестный тип модели: {model_type}")
            return None
            
        url = urls[model_type]
        print(f"🔄 Скачиваем SAM-HQ модель из {url}")
        
        # Размеры файлов для проверки
        expected_sizes = {'vit_b': 375_000_000, 'vit_l': 1_200_000_000, 'vit_h': 2_400_000_000}
        print(f"⚠️ Это займет несколько минут (~{expected_sizes[model_type]/1_000_000_000:.1f}GB)...")
        
        try:
            def progress_hook(block_num, block_size, total_size):
                if total_size > 0:
                    percent = min(100.0, (block_num * block_size / total_size) * 100)
                    print(f"\r📥 Прогресс: {percent:.1f}%", end='', flush=True)
            
            urllib.request.urlretrieve(url, sam_checkpoint, reporthook=progress_hook)
            print()  # Новая строка после прогресса
            
            if sam_checkpoint.stat().st_size > expected_sizes[model_type] * 0.8:  # 80% от ожидаемого размера
                print(f"   ✅ SAM-HQ модель скачана: {sam_checkpoint}")
                return str(sam_checkpoint)
            else:
                print("   ❌ Файл скачался неполностью")
                sam_checkpoint.unlink()  # Удаляем поврежденный файл
        except Exception as e:
            print(f"   ❌ Ошибка скачивания: {e}")
        
        return None
