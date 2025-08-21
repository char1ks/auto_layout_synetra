"""
Вспомогательные функции для SearchDet Pipeline.
"""

import sys
import cv2
import subprocess
import shlex
import urllib.request
import numpy as np
from pathlib import Path
from typing import Tuple, List, Any, Optional


# URL для скачивания весов SAM2
SAM2_URLS = {
    "vit_h": "https://huggingface.co/ybelkada/segment-anything-2-hiera-large/resolve/main/sam2_hiera_l.pt",
    "vit_l": "https://huggingface.co/ybelkada/segment-anything-2-hiera-large/resolve/main/sam2_hiera_l.pt", 
    "vit_b": "https://huggingface.co/ybelkada/segment-anything-2-hiera-large/resolve/main/sam2_hiera_l.pt",
}


def pip_install(requirement: str) -> bool:
    """
    Устанавливает пакет через pip.
    
    Args:
        requirement: Название пакета для установки
        
    Returns:
        True если установка прошла успешно
    """
    try:
        cmd = f"{sys.executable} -m pip install -U {requirement}"
        print(f"   ⬇️ pip: {cmd}")
        res = subprocess.run(
            shlex.split(cmd), 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            check=False, 
            text=True
        )
        print(res.stdout[-1000:])  # последние строки лога
        return res.returncode == 0
    except Exception as e:
        print(f"   ⚠️ pip ошибка: {e}")
        return False


def ensure_ultralytics(auto_install: bool = True) -> bool:
    """
    Гарантирует доступность пакета ultralytics.
    
    Args:
        auto_install: Автоматически устанавливать пакет если не найден
        
    Returns:
        True если пакет доступен
    """
    try:
        from ultralytics import FastSAM  # noqa: F401
        return True
    except Exception as e:
        print(f"   ℹ️ ultralytics не найден: {e}")
        if not auto_install:
            return False
        ok = pip_install("ultralytics>=8.1.0")
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


def iou_xyxy(bbox_a: Tuple[int, int, int, int], bbox_b: Tuple[int, int, int, int]) -> float:
    """
    Вычисляет IoU для двух bounding box в формате (x1, y1, x2, y2).
    
    Args:
        bbox_a: Первый bbox
        bbox_b: Второй bbox
        
    Returns:
        Значение IoU от 0 до 1
    """
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b
    
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    
    iw, ih = max(0, ix2 - ix1 + 1), max(0, iy2 - iy1 + 1)
    inter = iw * ih
    
    aarea = (ax2 - ax1 + 1) * (ay2 - ay1 + 1)
    barea = (bx2 - bx1 + 1) * (by2 - by1 + 1)
    union = aarea + barea - inter
    
    return inter / union if union > 0 else 0.0


def calculate_mask_bbox_iou(mask_segmentation: np.ndarray, mask_bbox: Tuple[int, int, int, int]) -> float:
    """
    Вычисляет IoU между маской и ее ограничивающим прямоугольником.
    
    Args:
        mask_segmentation: Бинарная маска
        mask_bbox: Bounding box в формате (x, y, w, h)
        
    Returns:
        Значение IoU от 0 до 1
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


def is_contained(seg_inner: np.ndarray, seg_outer: np.ndarray, 
                area_inner: int, containment_iou_threshold: float) -> bool:
    """
    Проверяет, содержится ли одна маска внутри другой.
    
    Args:
        seg_inner: Меньшая маска
        seg_outer: Большая маска  
        area_inner: Площадь меньшей маски
        containment_iou_threshold: Порог IoU для определения вложенности
        
    Returns:
        True если маска вложена
    """
    if not seg_inner.any() or not seg_outer.any():
        return False

    intersection = np.logical_and(seg_inner, seg_outer).sum()
    
    if area_inner == 0:  # Избегаем деления на ноль
        return False

    # Containment IoU: пересечение / площадь внутренней маски
    containment_iou = intersection / area_inner
    return containment_iou >= containment_iou_threshold


def parse_points_per_side_multi(s: Optional[str]) -> Optional[List[int]]:
    """
    Парсит строку с множественными значениями points_per_side.
    
    Args:
        s: Строка вида "32,64,128"
        
    Returns:
        Список значений или None
    """
    if not s:
        return None
    try:
        vals = [int(x.strip()) for x in s.split(',') if x.strip()]
        return [v for v in vals if v > 0]
    except Exception:
        return None


def download_sam2_weights(encoder: str, dest_path: str) -> bool:
    """
    Скачивает веса для SAM2.
    
    Args:
        encoder: Тип энкодера (vit_h, vit_l, vit_b)
        dest_path: Путь для сохранения весов
        
    Returns:
        True если скачивание успешно
    """
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


def load_image(image_path: str) -> np.ndarray:
    """
    Загружает изображение и конвертирует в RGB.
    
    Args:
        image_path: Путь к изображению
        
    Returns:
        Изображение в формате RGB
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_json(data: Any, filepath: str) -> None:
    """
    Сохраняет данные в JSON файл с поддержкой numpy типов.
    
    Args:
        data: Данные для сохранения
        filepath: Путь к файлу
    """
    import json
    
    class NumpyEncoder(json.JSONEncoder):
        """Кастомный JSON encoder для numpy типов."""
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            return super().default(obj)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)


def create_output_dir(base_path: str, image_name: str) -> Path:
    """
    Создает директорию для выходных файлов.
    
    Args:
        base_path: Базовый путь
        image_name: Имя изображения
        
    Returns:
        Путь к созданной директории
    """
    output_dir = Path(base_path) / f"results_{Path(image_name).stem}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
