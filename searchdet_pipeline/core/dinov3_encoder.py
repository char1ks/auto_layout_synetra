#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Единый энкодер DINOv3 для пайплайна:
- ViT: vitb16 / vitl16 / vith14 (global_pool="token", pooling='cls' или 'mean')
- ConvNeXt: convnext_{tiny,small,base,large} (global_pool="avg")

Методы:
- encode_image(PIL.Image) -> (D,)
- encode_mask(image_np, mask_bool) -> (D,)  # маска бинарная на исходном изображении
"""

from __future__ import annotations
import os
from typing import Dict, Optional

import numpy as np
from PIL import Image, ImageOps

import torch
import torch.nn.functional as F
import timm
from timm.data import resolve_model_data_config, create_transform


VIT_BACKBONES: Dict[str, str] = {
    "dinov3_vitb16": "vit_base_patch16_224",
    "dinov3_vitl16": "vit_large_patch16_224",
    "dinov3_vith14": "vit_huge_patch14_224",
    # короткие синонимы
    "vitb16": "vit_base_patch16_224",
    "vitl16": "vit_large_patch16_224",
    "vith14": "vit_huge_patch14_224",
}
CONVN_BACKBONES: Dict[str, str] = {
    "dinov3_convnext_tiny":  "convnext_tiny",
    "dinov3_convnext_small": "convnext_small",
    "dinov3_convnext_base":  "convnext_base",
    "dinov3_convnext_large": "convnext_large",
    # короткие
    "convnext_tiny":  "convnext_tiny",
    "convnext_small": "convnext_small",
    "convnext_base":  "convnext_base",
    "convnext_large": "convnext_large",
}
ALL_BACKBONES = set(VIT_BACKBONES.keys()) | set(CONVN_BACKBONES.keys())


def _load_pil(path: str) -> Optional[Image.Image]:
    try:
        im = Image.open(path).convert("RGB")
        return ImageOps.exif_transpose(im)
    except Exception:
        return None


def _infer_vit_arch_from_state_dict(sd: dict) -> Optional[str]:
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


class DinoV3Encoder:
    def __init__(
        self,
        backbone: str,
        ckpt: Optional[str],
        device: str = "cuda",
        half: bool = False,
        vit_pooling: str = "cls",  # 'cls' | 'mean'
        use_half: bool = False,
        loader: str = "timm",  # "timm" | "hub"
        repo_dir: Optional[str] = None,
    ):
        if backbone not in ALL_BACKBONES:
            raise ValueError(f"Unknown DINOv3 backbone '{backbone}'")
        self.backbone = backbone
        self.is_vit = backbone in VIT_BACKBONES
        self.arch = VIT_BACKBONES.get(backbone, CONVN_BACKBONES.get(backbone))
        self.device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
        self.half = bool(half or use_half)  # Поддержка обоих параметров
        self.use_half = self.half  # Унификация
        self.vit_pooling = vit_pooling.lower().strip()
        self.loader = loader
        self.repo_dir = repo_dir

        if self.loader == "hub":
            self._build_from_hub(backbone, ckpt)
        else:
            self._build_from_timm(backbone, ckpt)

    def _build_from_timm(self, backbone: str, ckpt: Optional[str]):
        """Создание модели через timm (текущий способ)."""
        gp = "token" if self.is_vit else "avg"
        self.model = timm.create_model(self.arch, pretrained=False, num_classes=0, global_pool=gp)
        self.model.eval().to(self.device)
        if self.half and self.device.type == "cuda":
            self.model.half()
        else:
            self.half = False  # страховка: на CPU только float32

        self.data_cfg = resolve_model_data_config(self.model)
        self.tf = create_transform(**self.data_cfg, is_training=False)

        if ckpt:
            self._smart_load(ckpt)

    def _build_from_hub(self, backbone: str, ckpt: str):
        """Создание модели через torch.hub (официальный способ)."""
        if self.repo_dir is None:
            raise ValueError("Для loader=hub нужно указать repo_dir (путь к локальному клону DINOv3).")

        # Карта имён hub-функций
        HUB_NAME = None
        if   backbone in ("vitb16", "vit_base_patch16_224"):     HUB_NAME = "dinov3_vitb16"
        elif backbone in ("vitl16", "vit_large_patch16_224"):    HUB_NAME = "dinov3_vitl16"
        elif backbone in ("vith14", "vit_huge_patch14_224"):     HUB_NAME = "dinov3_vith16plus"  # ближ. вариант
        elif backbone in ("convnext_tiny",):                     HUB_NAME = "dinov3_convnext_tiny"
        elif backbone in ("convnext_small",):                    HUB_NAME = "dinov3_convnext_small"
        elif backbone in ("convnext_base", "dinov3_convnext_base"): HUB_NAME = "dinov3_convnext_base"
        elif backbone in ("convnext_large",):                    HUB_NAME = "dinov3_convnext_large"
        else:
            raise ValueError(f"Неизвестный backbone для hub: {backbone}")

        print(f"🧩 Загрузка через torch.hub: {HUB_NAME}")
        self.model = torch.hub.load(self.repo_dir, HUB_NAME, source="local", weights=ckpt)
        self.model.eval().to(self.device)
        if self.half:
            self.model.half()

        # Попробуем взять data_config у модели; если нет — дефолты ImageNet
        try:
            self.data_cfg = resolve_model_data_config(self.model)
            self.tf = create_transform(**self.data_cfg, is_training=False)
        except Exception:
            from torchvision import transforms
            self.tf = transforms.Compose([
                transforms.Resize(224, interpolation=3),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
            ])

        # Определим, ViT это или ConvNeXt, чтобы выбрать pooling по умолчанию
        name = self.model.__class__.__name__.lower()
        self.is_vit = ("vit" in name)
        print(f"   ⚙️ HUB модель: {self.model.__class__.__name__} (is_vit={self.is_vit})")

    def _smart_load(self, path: str):
        if not os.path.isfile(path):
            print(f"⚠️ DINOv3 ckpt not found: {path}")
            return
        print(f"🔧 Loading DINOv3 ckpt: {path}")
        sd = torch.load(path, map_location="cpu")
        if isinstance(sd, dict):
            sd = sd.get("state_dict", sd.get("model", sd))

        # drop heads & training wrappers
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

        if self.is_vit:
            inferred = _infer_vit_arch_from_state_dict(cleaned)
            if inferred and inferred != self.arch:
                print(f"ℹ️ Auto-detect arch: {inferred} (rebuild)")
                gp = "token"
                self.arch = inferred
                self.model = timm.create_model(self.arch, pretrained=False, num_classes=0, global_pool=gp)
                self.model.eval().to(self.device)
                if self.half and self.device.type == "cuda":
                    self.model.half()
                else:
                    self.half = False  # страховка: на CPU только float32
                self.data_cfg = resolve_model_data_config(self.model)
                self.tf = create_transform(**self.data_cfg, is_training=False)

        missing, unexpected = self.model.load_state_dict(cleaned, strict=False)
        if unexpected:
            print(f"   ℹ️ unexpected keys: {len(unexpected)}")
        if missing:
            print(f"   ℹ️ missing keys   : {len(missing)}")
        print("   ✅ weights loaded (strict=False)")

    @torch.no_grad()
    def _prep(self, img: Image.Image) -> torch.Tensor:
        x = self.tf(img).unsqueeze(0)
        if self.half:
            x = x.half()
        return x.to(self.device)

    @torch.no_grad()
    def encode_image(self, img: Image.Image) -> np.ndarray:
        x = self._prep(img)
        feats = self.model.forward_features(x)

        vec = None
        if hasattr(self.model, "forward_head"):
            out = self.model.forward_head(feats, pre_logits=True)
            if out.ndim == 2:
                vec = out[0]
            elif out.ndim == 3:
                if self.vit_pooling == "mean" and out.shape[1] > 1:
                    vec = out[0, 1:].mean(dim=0)
                else:
                    vec = out[0, 0]
        if vec is None:
            t = feats["x"] if isinstance(feats, dict) and "x" in feats else feats
            if t.ndim == 3:  # ViT
                if self.vit_pooling == "mean" and t.shape[1] > 1:
                    vec = t[0, 1:].mean(dim=0)
                else:
                    vec = t[0, 0]
            elif t.ndim == 4:  # ConvNeXt
                vec = t.mean(dim=(2, 3))[0]
            else:
                vec = t.flatten(1)[0]

        vec = F.normalize(vec.float(), dim=0).cpu().numpy().astype(np.float32)
        return vec

    @torch.no_grad()
    def encode_mask(self, image_np: np.ndarray, mask_bool: np.ndarray) -> np.ndarray:
        """Простая и надёжная маскировка: зануляем фон и кодируем как целое изображение."""
        if mask_bool.dtype != np.bool_:
            mask_bool = mask_bool.astype(bool)
        if mask_bool.sum() == 0:
            # пустая маска → нулевой вектор
            return np.zeros(self.model.num_features, dtype=np.float32)

        # делаем masked image
        img_masked = image_np.copy()
        img_masked[~mask_bool] = 0
        pil = Image.fromarray(img_masked).convert("RGB")
        
        x = self.tf(pil).unsqueeze(0)
        x = x.to(self.device)
        if self.half:
            x = x.half()
        
        feats = self.model.forward_features(x)

        vec = None
        if hasattr(self.model, "forward_head"):
            out = self.model.forward_head(feats, pre_logits=True)
            if out.ndim == 2:
                vec = out[0]
            elif out.ndim == 3:
                if self.vit_pooling == "mean" and out.shape[1] > 1:
                    vec = out[0, 1:].mean(dim=0)
                else:
                    vec = out[0, 0]
        if vec is None:
            t = feats["x"] if isinstance(feats, dict) and "x" in feats else feats
            if t.ndim == 3:  # ViT
                if self.vit_pooling == "mean" and t.shape[1] > 1:
                    vec = t[0, 1:].mean(dim=0)
                else:
                    vec = t[0, 0]
            elif t.ndim == 4:  # ConvNeXt
                vec = t.mean(dim=(2, 3))[0]
            else:
                vec = t.flatten(1)[0]

        vec = F.normalize(vec.float(), dim=0).cpu().numpy().astype(np.float32)
        return vec
