#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dinov3_similarity.py
Сравнение изображения-запроса с набором изображений по эмбеддингам DINOv3.

Поддержка:
- ViT: vitb16 (vit_base_patch16_224), vitl16 (vit_large_patch16_224), vith14 (vit_huge_patch14_224)
  Пулинг: CLS (по умолчанию) или mean по патчам.
- ConvNeXt: convnext_base / small / tiny / large (GAP-пулинг).

Особенности:
- Корректные transforms из timm (resolve_model_data_config + create_transform).
- Умная загрузка state_dict (отбрасывание голов, префикса module., strict=False).
- Вектор признаков берём из forward_head(..., pre_logits=True) → (D,).
- Исправлен сбор целевых путей: файлы в target-dir теперь находятся и без --recursive.
"""

import os
import sys
import csv
import glob
import argparse
from typing import List, Tuple, Optional, Dict

import numpy as np
from PIL import Image, ImageOps

import torch
import torch.nn.functional as F

import timm
from timm.data import resolve_model_data_config, create_transform


# -------------------------
# Архитектуры по сокращениям
# -------------------------
VIT_BACKBONES: Dict[str, str] = {
    "vitb16": "vit_base_patch16_224",
    "vitl16": "vit_large_patch16_224",
    "vith14": "vit_huge_patch14_224",
}

CONVN_BACKBONES: Dict[str, str] = {
    "convnext_base":  "convnext_base",
    "convnext_small": "convnext_small",
    "convnext_tiny":  "convnext_tiny",
    "convnext_large": "convnext_large",
}

ALL_BACKBONES = sorted(list(VIT_BACKBONES.keys()) + list(CONVN_BACKBONES.keys()))


# -------------------------
# Утилиты
# -------------------------
def load_pil(path: str) -> Optional[Image.Image]:
    try:
        img = Image.open(path).convert("RGB")
        img = ImageOps.exif_transpose(img)  # корректируем ориентацию по EXIF
        return img
    except Exception as e:
        print(f"   ⚠️ Не удалось открыть '{path}': {e}", file=sys.stderr)
        return None


def iter_image_paths(target_dir: Optional[str], target_globs: List[str], recursive: bool) -> List[str]:
    """Собирает список путей к изображениям из папки и/или шаблонов."""
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp", "*.tif", "*.tiff")
    paths: List[str] = []

    if target_dir:
        # Если не рекурсивно: ищем файлы прямо в директории.
        # Если рекурсивно: **/ext
        if recursive:
            for ext in exts:
                pat = os.path.join(target_dir, "**", ext)
                paths.extend(glob.glob(pat, recursive=True))
        else:
            for ext in exts:
                pat = os.path.join(target_dir, ext)
                paths.extend(glob.glob(pat, recursive=False))

    for g in target_globs:
        paths.extend(glob.glob(g, recursive=True))

    # уникализируем и сортируем
    paths = sorted(list(dict.fromkeys(paths)))
    return paths


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def _infer_vit_arch_from_state_dict(sd: dict) -> Optional[str]:
    """
    Грубый автодетект ViT-архитектуры по размерности эмбеддинга и размеру патча.
    Возвращает timm-имя архитектуры или None.
    """
    dim = None
    patch = None
    if "cls_token" in sd:
        dim = sd["cls_token"].shape[-1]
    elif "pos_embed" in sd:
        dim = sd["pos_embed"].shape[-1]
    if "patch_embed.proj.weight" in sd:
        w = sd["patch_embed.proj.weight"]
        dim = w.shape[0]
        patch = w.shape[-1]

    if dim == 768 and (patch in (None, 16)):
        return "vit_base_patch16_224"
    if dim == 1024 and (patch in (None, 16)):
        return "vit_large_patch16_224"
    if dim == 1280 and (patch in (None, 14)):
        return "vit_huge_patch14_224"
    return None


# -------------------------
# Энкодер DINOv3
# -------------------------
class DinoV3Encoder:
    """
    timm-модель (num_classes=0), загрузка DINOv3 pretrain, корректные трансформы.
    Для ViT используем global_pool="token", для ConvNeXt — "avg".
    """

    def __init__(
        self,
        backbone: str,
        ckpt: Optional[str],
        device: str = "cuda",
        half: bool = False,
        pooling: str = "cls",  # для ViT: 'cls' или 'mean'
    ):
        self.device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
        self.half = bool(half)
        self.pooling = pooling.lower().strip()
        self.is_vit = backbone in VIT_BACKBONES

        if backbone in VIT_BACKBONES:
            self.img_arch = VIT_BACKBONES[backbone]
        elif backbone in CONVN_BACKBONES:
            self.img_arch = CONVN_BACKBONES[backbone]
        else:
            raise ValueError(f"Неизвестный backbone '{backbone}'. Допустимые: {ALL_BACKBONES}")

        # Создаём timm-модель без головы; пул — зависит от архитектуры
        gp = "token" if self.is_vit else "avg"
        self.model = timm.create_model(self.img_arch, pretrained=False, num_classes=0, global_pool=gp)
        self.model.eval().to(self.device)
        if self.half:
            self.model.half()

        # Корректные трансформы под конкретную модель
        self.data_cfg = resolve_model_data_config(self.model)
        self.tf = create_transform(**self.data_cfg, is_training=False)

        # Загружаем чекпойнт, если указан
        if ckpt:
            self._smart_load(ckpt)

    def _smart_load(self, path: str):
        if not os.path.isfile(path):
            print(f"⚠️ Чекпойнт не найден: {path}", file=sys.stderr)
            return

        print(f"🔧 Loading checkpoint: {path}")
        sd = torch.load(path, map_location="cpu")
        if isinstance(sd, dict):
            sd = sd.get("state_dict", sd.get("model", sd))

        # чистим ключи (сбрасываем головы и training-обвязку)
        cleaned = {}
        drop_prefixes = [
            "head", "fc", "classifier", "pre_logits",
            "seg_head", "decode_head", "aux_head",
            "mask_head", "detr_head", "m2f_head", "linear_head",
            "teacher", "student",
        ]
        for k, v in sd.items():
            if k.startswith("module."):
                k = k[7:]
            if any(k.startswith(p) for p in drop_prefixes):
                continue
            cleaned[k] = v

        # Если это ViT и арх не совпала — попробуем угадать и пересоздать
        if self.is_vit:
            inferred = _infer_vit_arch_from_state_dict(cleaned)
            if inferred and inferred != self.img_arch:
                print(f"ℹ️ Auto-detect: ckpt выглядит как '{inferred}', пересоздаю модель…")
                self.img_arch = inferred
                gp = "token"  # для ViT
                self.model = timm.create_model(self.img_arch, pretrained=False, num_classes=0, global_pool=gp)
                self.model.eval().to(self.device)
                if self.half:
                    self.model.half()
                self.data_cfg = resolve_model_data_config(self.model)
                self.tf = create_transform(**self.data_cfg, is_training=False)

        missing, unexpected = self.model.load_state_dict(cleaned, strict=False)
        if unexpected:
            print(f"   ℹ️ Пропущены ключи (unexpected): {len(unexpected)}")
        if missing:
            print(f"   ℹ️ Недостающие ключи (missing): {len(missing)}")
        print("   ✅ Веса загружены (strict=False)")

    @torch.no_grad()
    def _prep(self, img: Image.Image) -> torch.Tensor:
        x = self.tf(img).unsqueeze(0)
        if self.half:
            x = x.half()
        return x.to(self.device)

    @torch.no_grad()
    def encode(self, img: Image.Image) -> np.ndarray:
        """
        Возвращает L2-нормированный вектор признаков (D,).
        """
        x = self._prep(img)
        feats = self.model.forward_features(x)

        vec = None
        # Унифицированный путь: "pre-logits"
        if hasattr(self.model, "forward_head"):
            out = self.model.forward_head(feats, pre_logits=True)
            if out.ndim == 2:            # (B, D)
                vec = out[0]
            elif out.ndim == 3:          # (B, T, D) для некоторых ViT-конфигов
                if self.pooling == "mean":
                    if out.shape[1] > 1:
                        vec = out[0, 1:].mean(dim=0)
                    else:
                        vec = out[0, 0]
                else:
                    vec = out[0, 0]

        # Фолбэк на всякий случай
        if vec is None:
            t = feats["x"] if isinstance(feats, dict) and "x" in feats else feats
            if t.ndim == 3:              # (B, T, D) ViT
                if self.pooling == "mean":
                    vec = (t[0, 1:].mean(dim=0) if t.shape[1] > 1 else t[0, 0])
                else:
                    vec = t[0, 0]
            elif t.ndim == 4:            # (B, C, H, W) ConvNeXt
                vec = t.mean(dim=(2, 3))[0]
            else:                        # (B, D) или плоское
                vec = t.flatten(1)[0]

        vec = F.normalize(vec.float(), dim=0).cpu().numpy().astype(np.float32)
        return vec


# -------------------------
# Основной скрипт
# -------------------------
def main():
    p = argparse.ArgumentParser(
        description="DINOv3 image similarity (query vs targets).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--backbone", required=True, choices=ALL_BACKBONES,
                   help="Архитектура энкодера.")
    p.add_argument("--ckpt", required=True,
                   help="Путь к весам DINOv3 (pretrain).")
    p.add_argument("--query", required=True,
                   help="Путь к эталонному изображению (query).")
    p.add_argument("--target-dir", default=None,
                   help="Папка с изображениями для сравнения.")
    p.add_argument("--targets", nargs="*", default=[],
                   help="Список glob-шаблонов файлов для сравнения (можно вместо target-dir).")
    p.add_argument("--recursive", action="store_true",
                   help="Рекурсивный поиск в target-dir.")
    p.add_argument("--pooling", default="cls", choices=["cls", "mean"],
                   help="Пулинг для ViT: CLS или среднее по патчам.")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                   help="Устройство.")
    p.add_argument("--half", action="store_true",
                   help="FP16 (если поддерживается).")
    p.add_argument("--topk", type=int, default=15,
                   help="Сколько ближайших вывести.")
    p.add_argument("--save-csv", default=None,
                   help="Путь для сохранения CSV (путь/файл.csv).")

    args = p.parse_args()

    # Инициализация энкодера
    enc = DinoV3Encoder(
        backbone=args.backbone,
        ckpt=args.ckpt,
        device=args.device,
        half=args.half,
        pooling=args.pooling,
    )

    # Загружаем query
    query_img = load_pil(args.query)
    if query_img is None:
        sys.exit(1)
    q_vec = enc.encode(query_img)

    # Собираем цели
    target_paths = iter_image_paths(args.target_dir, args.targets, args.recursive)
    if not target_paths:
        print("⚠️ Не найдено ни одного целевого изображения.")
        return

    # Сравнение
    rows: List[Tuple[str, float]] = []
    vectors: List[np.ndarray] = []
    for pth in target_paths:
        img = load_pil(pth)
        if img is None:
            continue
        try:
            v = enc.encode(img)
            sim = cosine(q_vec, v)
            rows.append((pth, sim))
            vectors.append(v)
        except Exception as e:
            print(f"   ⚠️ Ошибка с {pth}: {e}", file=sys.stderr)

    # Санити-чек на «коллапс» эмбеддингов
    if len(vectors) >= 4:
        sample = vectors[:min(16, len(vectors))]
        sims = []
        for i in range(len(sample) - 1):
            for j in range(i + 1, len(sample)):
                sims.append(float(np.dot(sample[i], sample[j])))
        if sims and np.mean(sims) > 0.85:
            print("⚠️ Внимание: эмбеддинги целей почти одинаковые (mean cos ~{:.3f}). "
                  "Проверьте соответствие веса↔архитектура и корректность трансформов."
                  .format(np.mean(sims)))

    # Сортировка и вывод top-k
    rows.sort(key=lambda x: x[1], reverse=True)
    k = min(args.topk, len(rows))
    print("\n=== TOP SIMILAR ===")
    for i in range(k):
        print(f"  {i+1:>2}.  {rows[i][1]:.4f}   {rows[i][0]}")

    # Сохранение CSV
    if args.save_csv:
        try:
            with open(args.save_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["rank", "similarity", "path"])
                for i, (pth, sim) in enumerate(rows, start=1):
                    w.writerow([i, f"{sim:.6f}", pth])
            print(f"\n💾 CSV: {args.save_csv}")
        except Exception as e:
            print(f"⚠️ Не удалось сохранить CSV: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
