#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import subprocess
import shlex
import numpy as np
from pathlib import Path
import urllib.request


def _pip_install(requirement: str) -> bool:
    try:
        cmd = f"{sys.executable} -m pip install -U {requirement}"
        print(f"   ⬇️ pip: {cmd}")
        res = subprocess.run(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False, text=True)
        print(res.stdout[-1000:])
        return res.returncode == 0
    except Exception as e:
        print(f"   ⚠️ pip ошибка: {e}")
        return False


def _ensure_ultralytics(auto_install: bool = True):
    try:
        from ultralytics import FastSAM
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
            from ultralytics import FastSAM
            return True
        except Exception as e2:
            print(f"   ❌ Импорт ultralytics всё ещё не работает: {e2}")
            return False


def _load_fastsam_model(model_path: str | None):
    if not _ensure_ultralytics(auto_install=True):
        raise RuntimeError("Не удалось подготовить ultralytics/FastSAM.")

    try:
        from ultralytics import FastSAM
    except Exception:
        from ultralytics.models.fastsam import FastSAM

    Path("models").mkdir(parents=True, exist_ok=True)

    if model_path and os.path.exists(model_path):
        print(f"   ✅ FastSAM веса найдены локально: {model_path}")
        return FastSAM(model_path)

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
    def __init__(self, detector):
        self.detector = detector
        self._fastsam_model = None
        self._sam_generator = None
        
        # Предзагрузка моделей при инициализации
        self._preload_models()
    
    def _preload_models(self):
        """Предзагружает модели при инициализации для избежания задержек во время инференса"""
        print(f"🔄 Предзагрузка {self.detector.mask_backend.upper()} модели...")
        
        if self.detector.mask_backend == "fastsam":
            self._preload_fastsam()
        elif self.detector.mask_backend == "sam-hq":
            self._preload_sam_hq()
        else:
            print(f"⚠️ Неизвестный бэкенд: {self.detector.mask_backend}")
    
    def _preload_fastsam(self):
        """Предзагружает FastSAM модель"""
        import torch
        print("⬇️ Загрузка FastSAM...")
        self._fastsam_model = _load_fastsam_model(getattr(self.detector, 'fastsam_model', None))
        self._fastsam_device = getattr(self.detector, 'fastsam_device', None) or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"✅ FastSAM готов (device={self._fastsam_device})")
    
    def _preload_sam_hq(self):
        """Предзагружает SAM-HQ модель"""
        print("🎯 Инициализация SAM-HQ...")
        self._sam_generator = self._init_sam_hq()
        print("✅ SAM-HQ готов")
    
    def _cleanup_gpu_memory(self):
        """Очищает GPU память для предотвращения накопления"""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # Дополнительная синхронизация для полной очистки
                torch.cuda.synchronize()
        except Exception as e:
            # Игнорируем ошибки очистки памяти, чтобы не прерывать основной процесс
            pass
        
    def generate(self, image_np):
        print(f"🚀 ЭТАП 1: {self.detector.mask_backend.upper()} автогенерация масок")
        
        if self.detector.mask_backend == "fastsam":
            return self._generate_fastsam_masks(image_np)
        elif self.detector.mask_backend == "sam-hq":
            return self._generate_sam_masks(image_np)
        else:
            raise ValueError(f"Неподдерживаемый бэкенд: {self.detector.mask_backend}")
    
    def _generate_fastsam_masks(self, image_np):
        import torch, cv2, time
        
        # Модель уже предзагружена в __init__
        if self._fastsam_model is None:
            raise RuntimeError("FastSAM модель не была предзагружена")
        
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
        
        imgsz = getattr(self.detector, 'fastsam_imgsz', 1024)
        conf = getattr(self.detector, 'fastsam_conf', 0.4)
        iou = getattr(self.detector, 'fastsam_iou', 0.9)
        retina = getattr(self.detector, 'fastsam_retina', True)
        
        print(f"   на {w}x{h} (scale={scale:.3f}, imgsz={imgsz})")
        t_start = time.time()
        
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
                'segmentation': (seg_u8 > 0),
                'bbox': bbox,
                'area': int((seg_u8 > 0).sum()),
                'stability_score': 1.0,
                'predicted_iou': 1.0,
                'crop_box': [0, 0, W, H],
            })
        print(f"🔍 FastSAM сгенерировал {len(out)} масок за {t_gen:.3f} сек")
        print(f"🎯 Сгенерировано {len(out)} масок-кандидатов")
        
        # Очистка GPU памяти после генерации
        self._cleanup_gpu_memory()
        
        return out
    
    def _generate_sam_masks(self, image_np):
        import time
        import cv2
        import numpy as np
        
        # Модель уже предзагружена в __init__
        if self._sam_generator is None:
            raise RuntimeError("SAM-HQ генератор не был предзагружен")
        
        # Добавляем масштабирование изображения для ускорения SAM-HQ
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
            print(f"   🔧 Масштабирование: {W0}x{H0} → {run_img.shape[1]}x{run_img.shape[0]} (scale={scale:.3f})")
        
        print(f"   на {run_img.shape[1]}x{run_img.shape[0]}")
        
        t_start = time.time()
        
        masks = self._sam_generator.generate(run_img)
        
        # Если изображение было масштабировано, нужно восстановить маски к оригинальному размеру
        if scale != 1.0:
            print(f"   🔄 Восстановление масок к оригинальному размеру {W0}x{H0}...")
            for mask in masks:
                # Масштабируем сегментацию обратно
                seg = mask['segmentation']
                seg_resized = cv2.resize(seg.astype(np.uint8), (W0, H0), interpolation=cv2.INTER_NEAREST)
                mask['segmentation'] = seg_resized.astype(bool)
                
                # Пересчитываем bbox и area для оригинального размера
                ys, xs = np.where(seg_resized)
                if ys.size and xs.size:
                    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
                    mask['bbox'] = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
                    mask['area'] = int(seg_resized.sum())
                else:
                    mask['bbox'] = [0, 0, 0, 0]
                    mask['area'] = 0
                
                # Обновляем crop_box
                mask['crop_box'] = [0, 0, W0, H0]
        
        t_gen = time.time() - t_start
        
        print(f"🔍 SAM-HQ сгенерировал {len(masks)} масок за {t_gen:.3f} сек")
        print(f"🎯 Сгенерировано {len(masks)} масок-кандидатов")
        
        # Очистка GPU памяти после генерации
        self._cleanup_gpu_memory()
        
        return masks
    
    def _init_sam_hq(self):
        try:
            from segment_anything_hq import sam_model_registry, SamAutomaticMaskGenerator
        except ImportError:
            try:
                from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
                print("   ⚠️ Используем обычную SAM вместо SAM-HQ")
            except ImportError:
                raise RuntimeError("Не найден ни SAM-HQ, ни обычная SAM. Установите segment-anything-hq или segment-anything")
        
        sam_checkpoint = getattr(self.detector, 'sam_model', None)
        if not sam_checkpoint:
            sam_checkpoint = self._find_sam_checkpoint()
        
        if not sam_checkpoint:
            raise RuntimeError("Не найден checkpoint SAM-HQ. Запустите с --backend fastsam или установите SAM-HQ")
        
        print(f"   📦 Загружаем SAM из: {sam_checkpoint}")
        
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_type = getattr(self.detector, 'sam_encoder', 'vit_l') or 'vit_l'
        
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        
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
        import os
        from pathlib import Path
        
        model_type = getattr(self.detector, 'sam_encoder', 'vit_l')
        
        possible_paths = [
            f"sam-hq/pretrained_checkpoint/sam_hq_{model_type}.pth",
            f"models/sam_hq_{model_type}.pth", 
            f"sam_hq_{model_type}.pth",
        ]
        
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
                "models/sam_vit_l_0b3195.pth",
                "sam_vit_l_0b3195.pth"
            ])
        
        for path in possible_paths:
            if os.path.exists(path) and os.path.getsize(path) > 100_000_000:
                print(f"✅ SAM-HQ модель уже существует: {path}")
                return path
        
        return self._download_sam_hq()
    
    def _download_sam_hq(self):
        import urllib.request
        from pathlib import Path
        
        model_type = getattr(self.detector, 'sam_encoder', 'vit_l')
        
        sam_dir = Path("sam-hq/pretrained_checkpoint")
        sam_checkpoint = sam_dir / f"sam_hq_{model_type}.pth"
        
        if sam_checkpoint.exists() and sam_checkpoint.stat().st_size > 100_000_000:
            return str(sam_checkpoint)
        
        print(f"   📥 SAM-HQ модель не найдена, скачиваем {model_type}...")
        sam_dir.mkdir(parents=True, exist_ok=True)
        
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
        
        expected_sizes = {'vit_b': 375_000_000, 'vit_l': 1_200_000_000, 'vit_h': 2_400_000_000}
        print(f"⚠️ Это займет несколько минут (~{expected_sizes[model_type]/1_000_000_000:.1f}GB)...")
        
        try:
            def progress_hook(block_num, block_size, total_size):
                if total_size > 0:
                    percent = min(100.0, (block_num * block_size / total_size) * 100)
                    print(f"\r📥 Прогресс: {percent:.1f}%", end='', flush=True)
            
            urllib.request.urlretrieve(url, sam_checkpoint, reporthook=progress_hook)
            print()
            
            if sam_checkpoint.stat().st_size > expected_sizes[model_type] * 0.8:
                print(f"   ✅ SAM-HQ модель скачана: {sam_checkpoint}")
                return str(sam_checkpoint)
            else:
                print("   ❌ Файл скачался неполностью")
                sam_checkpoint.unlink()
        except Exception as e:
            print(f"   ❌ Ошибка скачивания: {e}")
        
        return None
