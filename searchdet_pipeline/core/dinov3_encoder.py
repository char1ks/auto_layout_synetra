import sys
from pathlib import Path
import torch
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image
import numpy as np
from typing import Any, Dict, Optional

# Рекомендуется: включить TF32 на CUDA (стабильнее, чем fp16)
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass

def _to_pil_any(x: object) -> Image.Image:
    """Преобразует путь/ndarray/tensor/PIL -> PIL RGB."""
    if isinstance(x, Image.Image):
        return x.convert("RGB")
    if isinstance(x, str):
        return Image.open(x).convert("RGB")
    if isinstance(x, np.ndarray):
        arr = x
        if arr.ndim == 2:
            arr = np.stack([arr]*3, axis=-1)
        if arr.dtype != np.uint8:
            # нормализация в 0..255
            if arr.max() <= 1.0:
                arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")
    if isinstance(x, torch.Tensor):
        t = x.detach().cpu()
        if t.ndim == 2:
            t = t.unsqueeze(-1).repeat(1, 1, 3)
        if t.ndim == 3:
            # CHW -> HWC, 0..1 -> 0..255
            if t.shape[0] in (1, 3):
                t = (t.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
            else:
                t = (t.clamp(0, 1) * 255).byte().numpy()
        return Image.fromarray(t, mode="RGB")
    raise TypeError(f"Unsupported image type: {type(x)}")

# Импортируем модуль бэкендов DINOv3 и выбираем архитектуру по имени
try:
    from dinov3.hub import backbones as dino_backbones
except ImportError:
    # Если не найден, добавим корень проекта и внутреннюю папку dinov3 в sys.path
    project_root = Path(__file__).resolve().parent.parent.parent
    dinov3_repo_path = project_root
    if str(dinov3_repo_path) not in sys.path:
        sys.path.insert(0, str(dinov3_repo_path))
    inner_dinov3_path = project_root / 'dinov3'
    if str(inner_dinov3_path) not in sys.path:
        sys.path.insert(0, str(inner_dinov3_path))
    from dinov3.hub import backbones as dino_backbones


class DinoV3Encoder:
    def __init__(self, backbone_name='vitb16', device='cpu', ckpt_path=None):
        self.device = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
        # Форсим fp32: любые half-режимы часто дают NaN на ViT
        self.half = False
        self.pooling = "cls"
        self.is_vit = True

        # Нормализуем имя бэкенда и выберем конструктор
        name = (backbone_name or 'vitb16').lower()
        # Разрешим часто используемые алиасы
        aliases = {
            'vits16': 'dinov3_vits16',
            'vits16plus': 'dinov3_vits16plus',
            'vitb16': 'dinov3_vitb16',
            'vitl16': 'dinov3_vitl16',
            'vitl16plus': 'dinov3_vitl16plus',
            'vith16plus': 'dinov3_vith16plus',
            'vit7b16': 'dinov3_vit7b16',
            'convnext_tiny': 'dinov3_convnext_tiny',
            'convnext_small': 'dinov3_convnext_small',
            'convnext_base': 'dinov3_convnext_base',
            'convnext_large': 'dinov3_convnext_large',
        }
        # Если имя уже в полном формате, оставим как есть
        if name in aliases:
            fn_name = aliases[name]
        elif name.startswith('dinov3_'):
            fn_name = name
        else:
            # Попробуем вытащить из имени паттерны
            if 'vit7b16' in name:
                fn_name = 'dinov3_vit7b16'
            elif 'vitb16' in name:
                fn_name = 'dinov3_vitb16'
            elif 'vits16plus' in name:
                fn_name = 'dinov3_vits16plus'
            elif 'vits16' in name:
                fn_name = 'dinov3_vits16'
            elif 'vitl16plus' in name:
                fn_name = 'dinov3_vitl16plus'
            elif 'vitl16' in name:
                fn_name = 'dinov3_vitl16'
            elif 'vith16plus' in name:
                fn_name = 'dinov3_vith16plus'
            elif 'convnext_large' in name:
                fn_name = 'dinov3_convnext_large'
            elif 'convnext_base' in name:
                fn_name = 'dinov3_convnext_base'
            elif 'convnext_small' in name:
                fn_name = 'dinov3_convnext_small'
            elif 'convnext_tiny' in name:
                fn_name = 'dinov3_convnext_tiny'
            else:
                fn_name = 'dinov3_vitb16'
        self.is_vit = not fn_name.startswith('dinov3_convnext')

        if not hasattr(dino_backbones, fn_name):
            print(f"⚠️ DINOv3: Неизвестный backbone '{backbone_name}', используем vitb16")
            fn_name = 'dinov3_vitb16'
        backbone_fn = getattr(dino_backbones, fn_name)
        print(f"🔧 DINOv3: используем архитектуру {fn_name}")

        # Создаем модель и загружаем веса
        if ckpt_path:
            print(f"🔧 DINOv3: загружаем checkpoint из {ckpt_path}")
            self.model = backbone_fn(pretrained=False).to(self.device)
            ckpt = torch.load(ckpt_path, map_location=self.device)
            
            # Попробуем извлечь state_dict из разных распространенных ключей
            if isinstance(ckpt, dict):
                if 'model' in ckpt and isinstance(ckpt['model'], dict):
                    state_dict = ckpt['model']
                    print(f"   📦 Извлекли state_dict из ключа 'model'")
                elif 'state_dict' in ckpt and isinstance(ckpt['state_dict'], dict):
                    state_dict = ckpt['state_dict']
                    print(f"   📦 Извлекли state_dict из ключа 'state_dict'")
                elif 'backbone' in ckpt and isinstance(ckpt['backbone'], dict):
                    state_dict = ckpt['backbone']
                    print(f"   📦 Извлекли state_dict из ключа 'backbone'")
                else:
                    # Возможно это уже state_dict
                    state_dict = {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}
                    print(f"   📦 Используем checkpoint как state_dict напрямую")
            else:
                state_dict = ckpt
                print(f"   📦 Checkpoint не является словарем, используем как есть")

            # Анализируем совместимость ключей перед загрузкой
            model_keys = set(self.model.state_dict().keys())
            ckpt_keys = set(state_dict.keys())
            matching_keys = model_keys & ckpt_keys
            
            print(f"   🔍 Анализ совместимости: модель имеет {len(model_keys)} ключей, checkpoint - {len(ckpt_keys)}")
            print(f"   🔍 Совпадающих ключей: {len(matching_keys)} из {len(model_keys)}")
            
            if len(matching_keys) < len(model_keys) * 0.1:  # Менее 10% совпадений
                print(f"   ❌ КРИТИЧЕСКОЕ ПРЕДУПРЕЖДЕНИЕ: Очень мало совпадающих ключей ({len(matching_keys)}/{len(model_keys)})!")
                print(f"   ❌ Возможно, это checkpoint не для backbone DINOv3, а для другой модели (например, DETR head)")
                print(f"   ❌ Рекомендуется использовать правильный backbone checkpoint или убрать --dinov3-ckpt")
                
                # Показываем примеры ключей для диагностики
                print(f"   🔍 Примеры ключей модели: {list(model_keys)[:5]}")
                print(f"   🔍 Примеры ключей checkpoint: {list(ckpt_keys)[:5]}")

            # Попробуем загрузить с ослабленным строгим режимом и выведем статистику совпадений
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            try:
                n_total = sum(p.numel() for p in self.model.state_dict().values())
                n_loaded = sum(state_dict[k].numel() for k in self.model.state_dict().keys() if k in state_dict)
                load_percentage = (n_loaded / n_total) * 100 if n_total > 0 else 0
                print(f"   📦 DINOv3 ckpt: загружено параметров ~{n_loaded}/{n_total} ({load_percentage:.1f}%)")
                print(f"   📦 Отсутствующих ключей: {len(missing)}, лишних ключей: {len(unexpected)}")
                
                if len(missing) > 0:
                    print(f"   ⚠️ Отсутствуют ключи (первые 5): {list(missing)[:5]}")
                if len(unexpected) > 0:
                    print(f"   ⚠️ Лишние ключи (первые 5): {list(unexpected)[:5]}")
                    
                if load_percentage < 50:
                    print(f"   ❌ ВНИМАНИЕ: Загружено менее 50% параметров! Модель может работать некорректно.")
                    print(f"   ❌ Рекомендуется проверить совместимость checkpoint с выбранной архитектурой {fn_name}")
                    
            except Exception as e:
                print(f"   ⚠️ Ошибка при подсчете статистики загрузки: {e}")
        else:
            print(f"🔧 DINOv3: используем предобученные веса из hub")
            self.model = backbone_fn(pretrained=True).to(self.device)

        # ВАЖНО: держим веса в float32
        self.model.eval().float()

        self.transform = T.Compose([
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _compute_dtype(self):
        """Какой dtype реально использовать в вычислениях."""
        if self.device.type == "cuda" and hasattr(self, 'half') and self.half:
            return torch.float16
        return torch.float32

    @staticmethod
    def _pick_feature(feats: Any, pooling: str = "cls") -> torch.Tensor:
        """
        Возвращает тензор [B,D] из разных форматов, которые выдают ViT/ConvNeXt в DINOv3.
        Порядок приоритета:
          - dict['x_norm_clstoken'] (готовый cls)
          - dict['x_prenorm'][:,0]  (cls до головы)
          - dict['x_norm_patchtokens'].mean(dim=1) (mean по патчам)
          - dict['x'] (на всякий случай)
          - просто тензор (если не dict)
        """
        if isinstance(feats, dict):
            if 'x_norm_clstoken' in feats:
                z = feats['x_norm_clstoken']           # [B,D]
            elif 'x_prenorm' in feats:
                z = feats['x_prenorm']
                if z.ndim == 3:                        # [B,T,D]
                    z = z[:, 0, :]                     # cls
            elif 'x_norm_patchtokens' in feats:
                z = feats['x_norm_patchtokens']        # [B,T,D]
                if z.ndim == 3:
                    z = z[:, 1:, :].mean(dim=1) if z.shape[1] > 1 else z[:, 0, :]
            elif 'x' in feats:
                z = feats['x']
                if z.ndim == 4:                        # ConvNeXt [B,C,H,W]
                    z = z.mean(dim=(2,3))
                elif z.ndim == 3:                      # [B,T,D]
                    z = z[:, 0, :] if pooling == "cls" else z[:, 1:, :].mean(dim=1)
                elif z.ndim == 2:
                    pass
                else:
                    z = z.flatten(1)
            else:
                # непонятный dict -> сваливаемся в mean по всем тензорам
                vs = []
                for v in feats.values():
                    if torch.is_tensor(v):
                        vv = v
                        if vv.ndim == 4: vv = vv.mean(dim=(2,3))
                        if vv.ndim == 3: vv = vv.mean(dim=1)
                        if vv.ndim == 2: vs.append(vv)
                z = torch.stack(vs, dim=0).mean(dim=0) if vs else None
        else:
            z = feats

        if not torch.is_tensor(z):
            raise RuntimeError("DINOv3: не удалось извлечь вектор признаков из forward_features()")

        if z.ndim == 4:  # [B,C,H,W]
            z = z.mean(dim=(2,3))
        elif z.ndim == 3:  # [B,T,D]
            z = z[:, 0, :] if pooling == "cls" else z[:, 1:, :].mean(dim=1)
        elif z.ndim == 1:
            z = z[None, :]
        return z  # [B,D]

    # безопасный препроцесс
    @torch.no_grad()
    def _prep(self, img_pil):
        x = self.transform(img_pil).unsqueeze(0)         # [1,3,H,W]
        x = x.to(self.device, dtype=torch.float32)
        return x

    @torch.no_grad()
    def _prep_tensor(self, img_pil: Image.Image) -> torch.Tensor:
        """PIL -> (1,3,H,W) на правильном device и с dtype модели."""
        print(f"   🔍 ДИАГНОСТИКА _prep_tensor: PIL размер = {img_pil.size}")
        
        x = self.transform(img_pil).unsqueeze(0)            # float32 CPU
        
        # Диагностика после трансформации
        x_has_nan = torch.isnan(x).any()
        x_has_inf = torch.isinf(x).any()
        x_min, x_max = x.min().item(), x.max().item()
        print(f"   🔍 ДИАГНОСТИКА _prep_tensor: После transform {x.shape}, NaN={x_has_nan}, Inf={x_has_inf}, min={x_min:.4f}, max={x_max:.4f}")
        
        x = x.to(self.device)
        wanted_dtype = next(self.model.parameters()).dtype
        if x.dtype != wanted_dtype:
            x = x.to(dtype=wanted_dtype)             # выравниваем precision с моделью
        return x

    @torch.no_grad()
    def _forward_safe(self, x: torch.Tensor) -> torch.Tensor:
        """Единый путь извлечения признаков -> L2-норм (B,D), float32 CPU."""
        # Диагностика входного тензора
        x_has_nan = torch.isnan(x).any()
        x_has_inf = torch.isinf(x).any()
        x_min, x_max = x.min().item(), x.max().item()
        print(f"   🔍 ДИАГНОСТИКА _forward_safe: Входной тензор {x.shape}, NaN={x_has_nan}, Inf={x_has_inf}, min={x_min:.4f}, max={x_max:.4f}")
        
        if x_has_nan or x_has_inf:
            print(f"   ❌ ДИАГНОСТИКА: Входной тензор содержит NaN/Inf, заменяем на нули")
            x = torch.zeros_like(x)
        
        feats = self.model.forward_features(x)
        
        # Диагностика выхода модели
        print(f"   🔍 ДИАГНОСТИКА: model.forward_features вернул тип = {type(feats)}")
        if isinstance(feats, dict):
            print(f"   🔍 ДИАГНОСТИКА: Ключи в feats = {list(feats.keys())}")
            for k, v in feats.items():
                if isinstance(v, torch.Tensor):
                    v_has_nan = torch.isnan(v).any()
                    v_has_inf = torch.isinf(v).any()
                    print(f"   🔍 ДИАГНОСТИКА: feats['{k}'] shape={v.shape}, NaN={v_has_nan}, Inf={v_has_inf}")
        elif isinstance(feats, torch.Tensor):
            feats_has_nan = torch.isnan(feats).any()
            feats_has_inf = torch.isinf(feats).any()
            print(f"   🔍 ДИАГНОСТИКА: feats tensor shape={feats.shape}, NaN={feats_has_nan}, Inf={feats_has_inf}")
        
        if hasattr(self.model, "forward_head"):
            z = self.model.forward_head(feats, pre_logits=True)
        else:
            t = feats["x"] if isinstance(feats, dict) and "x" in feats else feats
            # Безопасная проверка типа перед обращением к ndim
            if isinstance(t, dict):
                # Если t все еще словарь, попробуем извлечь тензор
                if "x_norm_clstoken" in t:
                    z = t["x_norm_clstoken"]
                elif "x_norm_patchtokens" in t:
                    z = t["x_norm_patchtokens"][:, 0]  # Берем первый токен
                else:
                    # Берем первое значение из словаря, которое является тензором
                    tensor_values = [v for v in t.values() if isinstance(v, torch.Tensor)]
                    if tensor_values:
                        z = tensor_values[0]
                        if z.ndim == 3:
                            z = z[:, 0]  # Берем CLS токен
                        elif z.ndim > 2:
                            z = z.flatten(1)
                    else:
                        # Fallback: создаем нулевой тензор
                        z = torch.zeros((x.shape[0], 768), device=x.device, dtype=x.dtype)
            elif hasattr(t, 'ndim'):
                if t.ndim == 4:      # (B,C,H,W) ConvNeXt
                    z = t.mean(dim=(2, 3))
                elif t.ndim == 3:    # (B,T,D) ViT
                    z = t[:, 0]
                else:                # (B,D)
                    z = t.flatten(1)
            else:
                # Fallback для неизвестных типов
                z = torch.zeros((x.shape[0], 768), device=x.device, dtype=x.dtype)

        # Диагностика z после извлечения
        z_has_nan = torch.isnan(z).any()
        z_has_inf = torch.isinf(z).any()
        print(f"   🔍 ДИАГНОСТИКА: z после извлечения shape={z.shape}, NaN={z_has_nan}, Inf={z_has_inf}")
        
        z = z.float()
        # запомним размерность один раз
        if not hasattr(self, "feat_dim"):
            self.feat_dim = int(z.shape[-1])

        # Диагностика до нормализации
        z_norm_before = torch.norm(z, dim=-1)
        print(f"   🔍 ДИАГНОСТИКА DINOv3: Норма до нормализации = {z_norm_before.item():.8f}")
        
        z = F.normalize(z, dim=-1, eps=1e-6)
        
        # Диагностика после нормализации
        z_norm_after = torch.norm(z, dim=-1)
        print(f"   🔍 ДИАГНОСТИКА DINOv3: Норма после нормализации = {z_norm_after.item():.8f}")
        
        return z.cpu()

    @torch.no_grad()
    def _forward_features_safe(self, x: torch.Tensor) -> torch.Tensor:
        """
        Жёсткий fp32-путь + санация NaN. При NaN на GPU — повторяем на CPU.
        Дополнительно: если норма признака близка к нулю при CLS-пулинге —
        пробуем альтернативный MEAN-пулинг по патчам и выбираем наиболее информативный.
        Возвращает [D] (для одного изображения).
        """
        # Первый прогон на текущем девайсе в float32 без autocast
        self.model.float().eval()
        x = x.to(self.device, dtype=torch.float32)

        feats = self.model.forward_features(x)

        # Извлекаем признак CLS и альтернативный MEAN без повторного forward
        z_cls = self._pick_feature(feats, pooling="cls")  # [1,D]
        z_mean = self._pick_feature(feats, pooling="mean")  # [1,D]

        # Проверим на NaN/Inf
        need_cpu_retry = (not torch.isfinite(z_cls).all()) or (not torch.isfinite(z_mean).all())
        if need_cpu_retry:
            # повтор на CPU → float32
            cpu = torch.device("cpu")
            self.model = self.model.to(cpu, dtype=torch.float32).eval()
            x_cpu = x.to(cpu, dtype=torch.float32)
            feats = self.model.forward_features(x_cpu)
            z_cls = self._pick_feature(feats, pooling="cls")
            z_mean = self._pick_feature(feats, pooling="mean")

        # финальная санация
        z_cls = torch.nan_to_num(z_cls, nan=0.0, posinf=0.0, neginf=0.0).float()
        z_mean = torch.nan_to_num(z_mean, nan=0.0, posinf=0.0, neginf=0.0).float()

        # Посчитаем нормы до нормализации и выведем диагностику
        n_cls = torch.norm(z_cls, dim=-1).item()
        n_mean = torch.norm(z_mean, dim=-1).item()
        print(f"   🔍 DINOv3: нормы до нормализации — CLS={n_cls:.6f}, MEAN={n_mean:.6f}")

        # Выбираем тот, у которого норма выше и не является около-нулевой
        eps = 1e-6
        if (n_cls < eps) and (n_mean >= eps):
            print("   ⚠️ DINOv3: CLS-норма ~0, используем MEAN-пулинг")
            z = z_mean
        elif (n_mean < eps) and (n_cls >= eps):
            z = z_cls
        else:
            # Если оба валидны — берем с большей нормой; если оба ~0 — возьмём CLS (дальше обнулится после проверки)
            z = z_cls if n_cls >= n_mean else z_mean

        # L2-норма (если z ~ 0, останется нулевым вектором — это обработаем выше по стеку)
        z = torch.nn.functional.normalize(z, dim=-1)
        return z[0]  # [D]

    @torch.no_grad()
    def encode(self, img_pil) -> np.ndarray:
        x = self._prep(img_pil)
        z = self._forward_features_safe(x)               # [D], float32
        v = z.detach().cpu().numpy().astype(np.float32)

        # второй уровень проверки
        if not np.isfinite(v).all():
            v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        # если норма вдруг ~0 (редко) — вернём нулевой вектор (лучше, чем NaN)
        n = np.linalg.norm(v)
        if not np.isfinite(n) or n < 1e-6:
            v = np.zeros_like(v, dtype=np.float32)
        return v
