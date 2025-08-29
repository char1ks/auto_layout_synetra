

"""
Автономный SearchDetDetector для модульного пайплайна.
"""

import argparse
import os
import sys
import cv2
import time
import numpy as np
from pathlib import Path
from PIL import Image
sys.path.append('./searchdet-main')
try:
    from mask_withsearch import initialize_sam as init_searchdet
    SEARCHDET_AVAILABLE = True
except Exception as e:
    print(f"⚠️ SearchDet недоступен: {e}")
    SEARCHDET_AVAILABLE = False
from .mask_generation import MaskGenerator
from .filtering import MaskFilter  
from .embeddings import EmbeddingExtractor
from .scoring import ScoreCalculator
from .step7_result_saving import ResultSaver
from .sam_predictor import SAMPredictor
from .utils import get_image_size, get_feature_map_size, upsample_feature_map
from .dinov3_encoder import DinoV3Encoder
import torch


class SearchDetDetector:
    def __init__(self, **kwargs):
        if not SEARCHDET_AVAILABLE:
            raise RuntimeError("SearchDet не найден")
        self.params = kwargs
        self.args = argparse.Namespace(**kwargs)
        self.sam_encoder = self.params.get('sam_encoder', 'vit_l')
        self.sam_model = self.params.get('sam_model', None)
        self.device = self.params.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.half = self.params.get('half', False)
        self.mask_backend = self.params.get('backend', 'sam-hq')
        self.backbone = self.params.get('backbone', 'dinov2_b')
        self.dinov3_ckpt = self.params.get('dinov3_ckpt', None)
        self.nms_iou = self.params.get('nms_iou', 0.5)
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
            self.searchdet_transform = None
        self.sam_predictor = SAMPredictor(self.searchdet_sam)
        
        self.dinov3_backbone = getattr(self.args, "dinov3_backbone", None) or getattr(self.args, "backbone", None)
        self.dinov3_ckpt     = getattr(self.args, "dinov3_ckpt", None)
        self.dinov3_device   = "cuda" if (getattr(self.args, "device", "cuda") == "cuda" and torch.cuda.is_available()) else "cpu"
        self.dinov3_half     = bool(getattr(self.args, "half", False))
        self.vit_pooling     = getattr(self.args, "vit_pooling", "cls").lower()
        self.pos_agg_mode    = getattr(self.args, "pos_agg_mode", "max").lower()

        self.dinov3_encoder = DinoV3Encoder(
            backbone=self.dinov3_backbone,
            ckpt=self.dinov3_ckpt,
            device=self.dinov3_device,
            half=self.dinov3_half,
            vit_pooling=self.vit_pooling,
        )

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
        class_pos, q_neg = self.embedding_extractor.build_queries_multiclass(pos_by_class, neg_imgs, pos_as_query_masks=False)
        timing_info['embedding_extraction'] = time.time() - t_embeddings

        online_negatives = None
        # Если нет явных негативных примеров, используем онлайн-негативы
        if q_neg is None or q_neg.shape[0] == 0:
            print("   ⚠️ Нет явных негативных примеров, генерируем онлайн-негативы...")
            
            # 1. Собрать все позитивные запросы в один тензор
            pos_queries_tensors = [torch.from_numpy(v) for v in class_pos.values() if v.shape[0] > 0]

            if not pos_queries_tensors:
                print("   ❌ Нет эмбеддингов для positive-классов, невозможно сгенерировать онлайн-негативы.")
            else:
                all_pos_queries = torch.cat(pos_queries_tensors, dim=0)

                if all_pos_queries.shape[0] > 0 and mask_vecs.shape[0] > 0:
                    # 2. Рассчитать косинусное сходство между масками и всеми позитивными запросами
                    mask_vecs_torch = torch.from_numpy(mask_vecs)
                    
                    # Используем torch для расчета косинусной близости
                    sim_matrix = torch.nn.functional.cosine_similarity(mask_vecs_torch.unsqueeze(1), all_pos_queries.unsqueeze(0), dim=2)

                    # 3. Найти лучший позитивный скор для каждой маски
                    best_pos_scores, _ = torch.max(sim_matrix, dim=1)
                    
                    # 4. Определить количество для онлайн-негативов (нижние 40%)
                    num_online_negatives = int(mask_vecs.shape[0] * 0.4)
                    
                    if num_online_negatives > 0:
                        # 5. Найти индексы масок с наименьшими скорами
                        k = min(num_online_negatives, len(best_pos_scores))
                        if k > 0:
                            _, bottom_indices = torch.topk(best_pos_scores, k=k, largest=False)
                            
                            # 6. Собрать эмбеддинги для онлайн-негативов
                            online_negatives = mask_vecs[bottom_indices.numpy()]
                            print(f"   💡 Создано {online_negatives.shape[0]} онлайн-негативов из масок с наихудшими positive-скорами.")

        print("1️⃣4️⃣ Шаг 10: ScoreCalculator.score_multiclass() - скоринг и принятие решений")
        print("🔍 ЭТАП 3: Сопоставление с positive/negative по классам...")
        t_scoring = time.time()
        decisions = self.score_calculator.score_multiclass(
            mask_vecs, 
            class_pos, 
            q_neg,
            online_negatives=online_negatives
        )
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
        """Рекурсивно загружает все изображения из директории."""
        from pathlib import Path
        from PIL import Image

        images = []
        if not dir_path:
            return images
            
        p_dir = Path(dir_path)
        if not p_dir.exists() or not p_dir.is_dir() or p_dir.name.startswith('.'):
            return images

        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        for item in p_dir.iterdir():
            if item.name.startswith('.'):
                continue
            
            if item.is_dir():
                images.extend(self._load_example_images(item))  # Рекурсивный вызов
            elif item.is_file() and item.suffix.lower() in valid_extensions:
                try:
                    img = Image.open(item).convert('RGB')
                    images.append(img)
                except Exception as e:
                    print(f"   ⚠️ Не удалось загрузить {item}: {e}")
        return images

    def _load_positive_by_class(self, dir_path):
        """Загружает позитивные примеры, распределяя их по классам."""
        from pathlib import Path
        result = {}
        if not dir_path:
            return result
            
        p_dir = Path(dir_path)
        if not p_dir.exists() or not p_dir.is_dir() or p_dir.name.startswith('.'):
            return result

        # Ищем нескрытые поддиректории
        subdirs = [p for p in p_dir.iterdir() if p.is_dir() and not p.name.startswith('.')]
        
        if subdirs:
            # Режим 1: Поддиректории являются классами
            print(f"   📂 Режим мульти-класса: подпапки в '{p_dir.name}' считаются классами.")
            for class_dir in subdirs:
                class_name = class_dir.name
                loaded_images = self._load_example_images(class_dir)
                if loaded_images:
                    result[class_name] = loaded_images
                    print(f"     -> Класс '{class_name}': найдено {len(loaded_images)} изображений.")
                else:
                    print(f"     -> Класс '{class_name}': 0 изображений.")
        else:
            # Режим 2: Нет поддиректорий, вся папка - один класс
            print(f"   📂 Режим одного класса: все изображения в '{p_dir.name}' будут принадлежать классу '{p_dir.name}'.")
            class_name = p_dir.name
            loaded_images = self._load_example_images(p_dir)
            if loaded_images:
                result[class_name] = loaded_images
                print(f"     -> Класс '{class_name}': найдено {len(loaded_images)} изображений.")
            else:
                print(f"     -> Класс '{class_name}': 0 изображений.")

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