

"""
Автономный SearchDetDetector для модульного пайплайна.
"""

import os
import sys
import cv2
import time
import numpy as np
from pathlib import Path
from PIL import Image
sys.path.append('./searchdet-main')
try:
    from mask_withsearch import initialize_models as init_searchdet
    SEARCHDET_AVAILABLE = True
except Exception as e:
    print(f"⚠️ SearchDet недоступен: {e}")
    SEARCHDET_AVAILABLE = False
from .mask_generation import MaskGenerator
from .filtering import MaskFilter  
from .embeddings import EmbeddingExtractor
from .scoring import ScoreCalculator
from .step7_result_saving import ResultSaver
from .embeddings import ResNet101Embedding, DINOv3Embedding
from .sam_predictor import SAMPredictor
from .utils import get_image_size, get_feature_map_size, upsample_feature_map


class SearchDetDetector:
    def __init__(self, **kwargs):
        if not SEARCHDET_AVAILABLE:
            raise RuntimeError("SearchDet не найден")
        self.params = kwargs
        self.sam_encoder = self.params.get('sam_encoder', 'vit_l')
        self.sam_model = self.params.get('sam_model', None)
        self.backbone = self.params.get('backbone', 'dinov2_b')
        self.dinov3_ckpt = self.params.get('dinov3_ckpt', None)
        if not self.backbone.startswith('dinov2'):
            feat_short = str(self.params.get('feat_short_side', 384))
            os.environ['SEARCHDET_FEAT_SHORT_SIDE'] = feat_short
            print(f"🔧 Установлено SEARCHDET_FEAT_SHORT_SIDE={feat_short}")
        else:
            print(f"🔧 DINOv2 бэкенд: используется собственный размер модели")
        print(f"🔧 Выбран SAM энкодер: {self.sam_encoder}")
        self.searchdet_resnet, self.searchdet_layer, self.searchdet_transform, self.searchdet_sam = init_searchdet()
        if not self.backbone.startswith('dinov2'):
            import torchvision.transforms as transforms
            feat_short_side = int(os.getenv('SEARCHDET_FEAT_SHORT_SIDE', '384'))
            self.searchdet_transform = transforms.Compose([
                transforms.Resize(feat_short_side),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            print(f"🔧 DINOv2: трансформация будет определена в EmbeddingExtractor")  
        self.searchdet_layer = self.params.get('layer', 'layer3')
        print(f"🔧 Используется слой для эмбеддингов: {self.searchdet_layer}")
        self.mask_backend = self.params.get('mask_backend', 'fastsam')
        self.consensus_k = int(self.params.get('consensus_k', 3))
        self.consensus_thr = float(self.params.get('consensus_thr', 0.60))
        self.nms_iou = float(self.params.get('nms_iou', 0.60))
        # 🚀 Параметры оптимизации эмбеддингов
        self.max_embedding_size = self.params.get('max_embedding_size', 1024)
        self.dino_half_precision = self.params.get('dino_half_precision', False)

        # 🚀 Параметры генерации масок
        default_sam_long = min(1800, self.max_embedding_size)
        self.sam_long_side = int(self.params.get('sam_long_side', default_sam_long))
        self.fastsam_imgsz = int(self.params.get('fastsam_imgsz', 1024))
        self.fastsam_conf = float(self.params.get('fastsam_conf', 0.4))
        self.fastsam_iou = float(self.params.get('fastsam_iou', 0.9))
        self.fastsam_retina = bool(self.params.get('fastsam_retina', True))
        self.ban_border_masks = bool(self.params.get('border_ban', True))
        self.border_width = int(self.params.get('border_width', 2))
        # 🚀 Предзагрузка DINOv3 при инициализации (если используется)
        if self.backbone.startswith('dinov3') and self.dinov3_ckpt:
            print(f"🔧 Предзагрузка DINOv3 ConvNeXt-B: {self.dinov3_ckpt}")
            self._preload_dinov3()
        
        self.mask_generator = MaskGenerator(self)
        self.mask_filter = MaskFilter(self, self.params)
        self.embedding_extractor = EmbeddingExtractor(self)
        self.score_calculator = ScoreCalculator(self, self.params)
        self.result_saver = ResultSaver()
        print("✅ SearchDetDetector инициализирован (автономная модульная версия)")
    
    def _preload_dinov3(self):
        """Предзагрузка DINOv3 ConvNeXt-B модели при инициализации детектора."""
        try:
            import torch
            import timm
            import torchvision.transforms as T
            from torchvision.transforms import InterpolationMode
            
            print("🔄 Загрузка DINOv3 ConvNeXt-B модели...")
            
            # Создаем модель ConvNeXt-B без классификатора (эмбеддинги)
            self.dinov3_model = timm.create_model('convnext_base', pretrained=False, num_classes=0)
            
            # Загружаем веса DINOv3
            if self.dinov3_ckpt and os.path.exists(self.dinov3_ckpt):
                print(f"🔧 Загружаем DINOv3 веса из: {self.dinov3_ckpt}")
                state_dict = torch.load(self.dinov3_ckpt, map_location='cpu')
                # Обрабатываем разные форматы checkpoint
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                elif 'model' in state_dict:
                    state_dict = state_dict['model']
                self.dinov3_model.load_state_dict(state_dict, strict=False)
            else:
                print(f"⚠️ DINOv3 checkpoint не найден: {self.dinov3_ckpt}, используем случайные веса")
            
            # Создаем препроцессор (стандартный ImageNet)
            self.dinov3_preprocess = T.Compose([
                T.Resize(256, interpolation=InterpolationMode.BICUBIC),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
            # Переводим в eval режим и на GPU если доступно
            self.dinov3_model.eval()
            if torch.cuda.is_available():
                self.dinov3_model = self.dinov3_model.cuda()
                print("🚀 DINOv3 модель загружена на GPU")
            else:
                print("💻 DINOv3 модель загружена на CPU")
                
            # Применяем половинную точность если включено
            if self.dino_half_precision and torch.cuda.is_available():
                self.dinov3_model = self.dinov3_model.half()
                print("⚡ DINOv3 переведена в половинную точность")
                
            print("✅ DINOv3 ConvNeXt-B успешно предзагружена")
            
        except Exception as e:
            print(f"❌ Ошибка предзагрузки DINOv3: {e}")
            # Устанавливаем None чтобы использовать ленивую загрузку
            self.dinov3_model = None
            self.dinov3_preprocess = None
    
    def find_present_elements(self, image_path, positive_dir, negative_dir=None, output_dir="output"):
        """Основная функция поиска объектов."""
        print(f"🔍 Модульный анализ: {image_path}" + "="*60)
        print("🔄 ДЕТАЛЬНАЯ ПОСЛЕДОВАТЕЛЬНОСТЬ ВЫПОЛНЕНИЯ МОДУЛЬНОГО PIPELINE:")
        print("=" * 80)
        print("8️⃣ searchdet_pipeline/core/detector.py → find_present_elements()")
        timing_info = {}
        t_total = time.time()
        t_loading = time.time()
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Не удалось загрузить изображение: {image_path}")
        image_np = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        timing_info['image_loading'] = time.time() - t_loading
        print("9️⃣ Шаг 1: _load_example_images() - загрузка positive/negative примеров")
        t_examples = time.time()   
        pos_by_class = self._load_positive_by_class(positive_dir)
        if len(pos_by_class) == 0:
            print("   ❌ Нет положительных примеров — прекращаем.")
            return {"found_elements": [], "masks": []}
        total_pos = sum(len(v) for v in pos_by_class.values())
        neg_imgs = self._load_example_images(negative_dir) if negative_dir else []
        timing_info['examples_loading'] = time.time() - t_examples
        print(f"   📁 Positive: {total_pos} в {len(pos_by_class)} классах	📁 Negative: {len(neg_imgs)}")
        print("🔟 Шаг 2: MaskGenerator.generate() - генерация масок через SAM/FastSAM")
        t_masks = time.time()
        masks = self.mask_generator.generate(image_np)
        timing_info['mask_generation'] = time.time() - t_masks
        print("1️⃣1️⃣ Шаг 3-7: MaskFilter.apply_all_filters() - все фильтры масок")
        t_filtering = time.time()
        masks = self.mask_filter.apply_all_filters(masks, image_np)
        timing_info['mask_filtering'] = time.time() - t_filtering
        if not masks:
            print("   ❌ Нет валидных масок после фильтров.")
            return {"found_elements": [], "masks": []}
        print("1️⃣2️⃣ Шаг 8: EmbeddingExtractor.extract_mask_embeddings() - эмбеддинги масок")
        t_embeddings = time.time()
        mask_vecs, idx_map = self.embedding_extractor.extract_mask_embeddings(image_np, masks)
        if mask_vecs.shape[0] == 0:
            print("   ❌ Не удалось получить эмбеддинги масок.")
            return {"found_elements": [], "masks": []}
        print(f"   📊 Масок с валидными векторами: {mask_vecs.shape[0]}")
        print("1️⃣3️⃣ Шаг 9: EmbeddingExtractor.build_queries_multiclass() - эмбеддинги примеров по классам")
        class_pos, q_neg = self.embedding_extractor.build_queries_multiclass(pos_by_class, neg_imgs)
        timing_info['embedding_extraction'] = time.time() - t_embeddings
        print("1️⃣4️⃣ Шаг 10: ScoreCalculator.score_multiclass() - скоринг и принятие решений")
        print("🔍 ЭТАП 3: Сопоставление с positive/negative по классам...")
        t_scoring = time.time()
        decisions = self.score_calculator.score_multiclass(mask_vecs, class_pos, q_neg)
        timing_info['scoring_and_decisions'] = time.time() - t_scoring
        t_result = time.time()
        found = []
        result_masks = []
        candidates = []
        H, W = image_np.shape[:2]
        print(f"\n🔍 Processing {len(decisions)} decisions...")
        for i, dec in enumerate(decisions):
            print(f"  - Decision {i}: accepted={dec.get('accepted')}, class='{dec.get('class')}', confidence={dec.get('confidence', 0.0):.3f}")
            if not dec.get('accepted'):
                print(f"    -> SKIPPED (not accepted)")
                continue
            original_idx = idx_map[i]
            print(f"    -> ACCEPTED. Original mask index: {original_idx}")
            mask_dict = masks[original_idx].copy()
            confidence = float(np.clip(dec.get('confidence', 0.0), 0.0, 1.0))
            mask_dict['confidence'] = confidence
            mask_dict['class'] = dec.get('class')
            if 'area' not in mask_dict and 'segmentation' in mask_dict:
                mask_dict['area'] = int(np.sum(mask_dict['segmentation']))
            bx = mask_dict.get('bbox', [0,0,0,0])
            if len(bx) == 4 and (bx[2] <= W and bx[3] <= H):
                x1, y1, w, h = bx
                bbox_xyxy = [int(x1), int(y1), int(x1 + w), int(y1 + h)]
            else:
                bbox_xyxy = [int(bx[0]), int(bx[1]), int(bx[2]), int(bx[3])]
            cls_label = dec.get('class')
            try:
                cls_label = str(cls_label) if cls_label is not None else "__unknown__"
            except Exception:
                cls_label = "__unknown__"
            print(f"    -> Appending candidate: class='{cls_label}', confidence={confidence:.3f}")
            candidates.append({
                'mask': mask_dict['segmentation'].astype(bool),
                'bbox_xyxy': bbox_xyxy,
                'confidence': confidence,
                'area': int(mask_dict['area']),
                'class': cls_label,
            })
        from collections import Counter
        print("NMS candidates by class:", Counter([c.get('class') for c in candidates]))
        kept = self._nms(candidates, class_aware=True)
        for e in kept:
            seg = e['mask']
            x1, y1, x2, y2 = e['bbox_xyxy']
            bbox_xywh = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
            mask_dict = {
                'segmentation': seg,
                'bbox': bbox_xywh,
                'area': int(seg.sum()),
                'confidence': float(e['confidence']),
                'class': e.get('class')
            }
            found.append({
                'mask': mask_dict,
                'confidence': float(e['confidence']),
                'bbox': bbox_xywh,
                'class': e.get('class')
            })
            result_masks.append(mask_dict)
        timing_info['result_formatting'] = time.time() - t_result
        print("1️⃣5️⃣ Шаг 11: ResultSaver.save_all_results() - сохранение файлов")
        t_saving = time.time()
        image_name = Path(image_path).name
        saved_files = self.result_saver.save_all_results(
            image_np, 
            result_masks, 
            output_dir, 
            image_name,
            pipeline_config={"backend": self.mask_backend}
        )
        timing_info['result_saving'] = time.time() - t_saving
        total_time = time.time() - t_total
        timing_info['total_time'] = total_time
        print(f"🎯 Принято масок: {len(found)} (после правил и NMS)")
        print(f"⏱️ Общее время: {total_time:.2f} сек")
        print(f"💾 Результаты сохранены в: {output_dir}")
        print(f"📁 Сохранено файлов: {len(saved_files)}")
        self._print_timing_statistics(timing_info)
        return {
            "found_elements": found, 
            "masks": result_masks,
            "timing_info": timing_info,
            "output_directory": output_dir,
            "saved_files": saved_files
        }

    def _mask_iou(self, mask_a, mask_b):
        """Быстрое вычисление IoU для масок."""
        intersection = np.logical_and(mask_a, mask_b).sum()
        union = np.logical_or(mask_a, mask_b).sum()
        return intersection / union if union > 0 else 0.0

    def _nms(self, elements, class_aware=True, class_thresholds=None):
        """Эффективная реализация NMS с использованием векторизованных операций."""
        if not elements:
            return []
        
        try:
            import torch
            import torchvision.ops as ops
            use_torch = True
        except ImportError:
            use_torch = False
            
        from collections import defaultdict
        
        def _class_key(v):
            if v is None:
                return "__unknown__"
            try:
                return str(v)
            except Exception:
                return repr(v)
        
        groups = defaultdict(list)
        if class_aware:
            for el in elements:
                groups[_class_key(el.get('class'))].append(el)
        else:
            groups["__all__"] = list(elements)
        
        kept_all = []
        for cls_key, group in groups.items():
            if not group:
                continue
                
            iou_thr = (class_thresholds or {}).get(cls_key, self.nms_iou)
            
            if use_torch and len(group) > 10:  # Используем torch для больших групп
                kept_cls = self._nms_torch(group, iou_thr)
            else:
                kept_cls = self._nms_numpy(group, iou_thr)
                
            kept_all.extend(kept_cls)
        
        return kept_all
    
    def _nms_torch(self, elements, iou_threshold):
        """NMS с использованием torchvision для bbox + маски."""
        import torch
        import torchvision.ops as ops
        
        # Извлекаем bbox и scores
        boxes = []
        scores = []
        for el in elements:
            x1, y1, x2, y2 = el['bbox_xyxy']
            boxes.append([x1, y1, x2, y2])
            scores.append(el.get('confidence', 0.0))
        
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        scores_tensor = torch.tensor(scores, dtype=torch.float32)
        
        # Применяем bbox NMS
        keep_indices = ops.nms(boxes_tensor, scores_tensor, iou_threshold)
        bbox_kept = [elements[i] for i in keep_indices.tolist()]
        
        # Дополнительная фильтрация по маскам для оставшихся элементов
        if len(bbox_kept) <= 1:
            return bbox_kept
            
        final_kept = []
        for i, current in enumerate(bbox_kept):
            should_keep = True
            current_mask = current['mask']
            
            for j in range(i):
                if j < len(final_kept):
                    other_mask = final_kept[j]['mask']
                    if self._mask_iou(current_mask, other_mask) >= iou_threshold:
                        should_keep = False
                        break
            
            if should_keep:
                final_kept.append(current)
                
        return final_kept
    
    def _nms_numpy(self, elements, iou_threshold):
        """Оптимизированная numpy реализация NMS."""
        if len(elements) <= 1:
            return elements
            
        # Сортируем по confidence
        sorted_elements = sorted(elements, key=lambda e: float(e.get('confidence', 0.0)), reverse=True)
        
        kept = []
        masks_kept = []
        
        for current in sorted_elements:
            current_mask = current['mask']
            should_keep = True
            
            # Векторизованная проверка IoU с уже принятыми масками
            for kept_mask in masks_kept:
                if self._mask_iou(current_mask, kept_mask) >= iou_threshold:
                    should_keep = False
                    break
            
            if should_keep:
                kept.append(current)
                masks_kept.append(current_mask)
                
        return kept

    def _load_example_images(self, dir_path):

        if not dir_path or not Path(dir_path).exists():
            return []
        images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            for img_path in Path(dir_path).glob(ext):
                try:
                    img = Image.open(img_path).convert('RGB')
                    images.append(img)
                except Exception as e:
                    print(f"   ⚠️ Не удалось загрузить {img_path}: {e}")
        return images
    
    
    def _load_positive_by_class(self, dir_path):
        from pathlib import Path
        from PIL import Image
        result = {}
        if not dir_path or not Path(dir_path).exists():
            return result
        subdirs = [p for p in Path(dir_path).iterdir() if p.is_dir()]
        if not subdirs:
            result['object'] = self._load_example_images(dir_path)
            return result
        for sd in subdirs:
            imgs = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                for img_path in sd.glob(ext):
                    try:
                        imgs.append(Image.open(img_path).convert('RGB'))
                    except Exception as e:
                        print(f"   ⚠️ Не удалось загрузить {img_path}: {e}")
            result[sd.name] = imgs
        return result
    def _print_timing_statistics(self, timing_info):
        print("\n" + "="*60)
        print("⏱️ ДЕТАЛЬНАЯ СТАТИСТИКА ВРЕМЕНИ ВЫПОЛНЕНИЯ:")
        print("="*60)
        total_time = timing_info['total_time']
        stages = [
            ('image_loading', '📁 Загрузка изображения'),
            ('examples_loading', '🖼️ Загрузка примеров'),
            ('mask_generation', '🎯 Генерация масок (SAM/FastSAM)'),
            ('mask_filtering', '🔍 Фильтрация масок'),
            ('embedding_extraction', '🧠 Извлечение эмбеддингов'),
            ('scoring_and_decisions', '📊 Скоринг и решения'),
            ('result_formatting', '📋 Формирование результата'),
            ('result_saving', '💾 Сохранение файлов')
        ]
        
        for stage_key, stage_name in stages:
            if stage_key in timing_info:
                stage_time = timing_info[stage_key]
                percentage = (stage_time / total_time * 100) if total_time > 0 else 0
                print(f"{stage_name:<40}: {stage_time:>6.3f}с ({percentage:>5.1f}%)")
        print("-" * 60)
        print(f"{'🚀 ОБЩЕЕ ВРЕМЯ':<40}: {total_time:>6.3f}с (100.0%)")
        print("="*60)
        stage_times = [(name, timing_info.get(key, 0)) for key, name in stages if key in timing_info]
        stage_times.sort(key=lambda x: x[1], reverse=True)
        if len(stage_times) > 1:
            print("\n🐌 САМЫЕ МЕДЛЕННЫЕ ЭТАПЫ:")
            for i, (name, stage_time) in enumerate(stage_times[:3]):
                percentage = (stage_time / total_time * 100) if total_time > 0 else 0
                print(f"   {i+1}. {name}: {stage_time:.3f}с ({percentage:.1f}%)")
        if 'mask_generation' in timing_info and timing_info['mask_generation'] > total_time * 0.5:
            print("\n💡 РЕКОМЕНДАЦИИ ПО ОПТИМИЗАЦИИ:")
            print("   • Генерация масок занимает >50% времени")
            print("   • Попробуйте FastSAM вместо SAM-HQ для ускорения")
            print("   • Или уменьшите параметры SAM (points_per_side, imgsz)")
        if 'embedding_extraction' in timing_info and timing_info['embedding_extraction'] > total_time * 0.3:
            print("\n💡 РЕКОМЕНДАЦИИ ПО ОПТИМИЗАЦИИ:")
            print("   • Извлечение эмбеддингов занимает >30% времени")
            print("   • Проверьте размер feature map (SEARCHDET_FEAT_SHORT_SIDE)")
            print("   • Убедитесь что используется быстрый метод извлечения")
        print()
class Detector:
    def __init__(self, sam_encoder: str, sam_checkpoint: str, sam2_checkpoint: str, 
                 backbone: str = 'dinov2_b', layer: str = 'layer3', 
                 feat_short_side: int = 512, dinov3_ckpt: str = None):
        # Загрузка SAM
        self.sam_predictor = SAMPredictor(sam_encoder, sam_checkpoint, sam2_checkpoint)

        # Загрузка модели эмбеддингов
        if backbone == 'resnet101':
            self.embedding_model = ResNet101Embedding(layer=layer)
        elif backbone.startswith('dinov2'):
            self.embedding_model = DINOv2Embedding(backbone=backbone)
        elif backbone == 'dinov3_convnext_base':
            self.embedding_model = DINOv3Embedding(dinov3_ckpt)
        else:
            raise ValueError(f"Неизвестный бэкенд: {backbone}")

        self.backbone = backbone