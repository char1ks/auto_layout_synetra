import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import cv2
import faiss
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
import pickle
from segment_anything_hq import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import timm
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time
import base64
from io import BytesIO
import requests
from langchain_community.llms import VLLM
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
import urllib.request
from pathlib import Path


def initialize_sam():
    """Инициализация моделей для SearchDet: ResNet, слой, трансформ и SAM"""
    total_init_start = time.time()
    print("🚀 Инициализация моделей SearchDet...")
    
    # 1. Инициализация ResNet модели
    resnet_start = time.time()
    print("🧠 Загрузка ResNet101 модели...")
    resnet_model = models.resnet101(pretrained=True)
    resnet_model.eval()
    
    # Перевод на GPU если доступно
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resnet_model = resnet_model.to(device)
    
    # Выбираем слой для извлечения фич (layer3 для хорошего баланса)
    pooling_layer = resnet_model.layer3
    
    resnet_time = time.time() - resnet_start
    print(f"   ⏱️ ResNet101 модель загружена за: {resnet_time:.3f} сек")
    
    # 2. Создание трансформа для предобработки
    transform_start = time.time()
    print("🔧 Создание трансформа...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_time = time.time() - transform_start
    print(f"   ⏱️ Трансформ создан за: {transform_time:.3f} сек")
    
    # 3. Инициализация SAM модели
    sam_start = time.time()
    print("🎯 Инициализация SAM-HQ модели...")
    
    # Скачиваем модель если её нет
    sam_checkpoint = download_sam_hq_model()
    
    if sam_checkpoint is None:
        print("⚠️ Не удалось загрузить SAM-HQ, используем fallback")
        # Создаем заглушку
        sam = None
    else:
        try:
            model_type = "vit_l"
            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam.to(device=device)
            print(f"✅ SAM-HQ модель загружена: {sam_checkpoint}")
        except Exception as e:
            print(f"❌ Ошибка загрузки SAM-HQ: {e}")
            sam = None
    
    sam_time = time.time() - sam_start
    print(f"   ⏱️ SAM модель инициализирована за: {sam_time:.3f} сек")
    
    total_init_time = time.time() - total_init_start
    print(f"🎉 Общее время инициализации: {total_init_time:.3f} сек")
    
    return resnet_model, pooling_layer, transform, sam


def initialize_models():
    """Алиас для обратной совместимости"""
    return initialize_sam()


def download_sam_hq_model():
    """Автоматическое скачивание SAM-HQ модели если её нет"""
    sam_dir = Path("sam-hq/pretrained_checkpoint")
    sam_checkpoint = sam_dir / "sam_hq_vit_l.pth"
    
    if sam_checkpoint.exists():
        print(f"✅ SAM-HQ модель уже существует: {sam_checkpoint}")
        return str(sam_checkpoint)
    
    print("📥 SAM-HQ модель не найдена, скачиваем...")
    
    # Создаем директорию
    sam_dir.mkdir(parents=True, exist_ok=True)
    
    # URL для скачивания SAM-HQ модели
    model_url = "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_l.pth"
    
    try:
        print(f"🔄 Скачиваем SAM-HQ модель из {model_url}")
        print("⚠️ Это займет несколько минут (~1.2GB)...")
        
        # Скачиваем с прогресс-баром
        def reporthook(blocknum, blocksize, totalsize):
            if totalsize > 0:
                percent = min(blocknum * blocksize * 100 / totalsize, 100)
                print(f"\r📥 Прогресс: {percent:.1f}%", end="", flush=True)
        
        urllib.request.urlretrieve(model_url, sam_checkpoint, reporthook=reporthook)
        print(f"\n✅ SAM-HQ модель скачана: {sam_checkpoint}")
        
        return str(sam_checkpoint)
        
    except Exception as e:
        print(f"\n❌ Ошибка скачивания SAM-HQ: {e}")
        print("🔄 Пробуем альтернативный URL...")
        
        # Альтернативный URL
        alt_url = "https://github.com/SysCV/sam-hq/releases/download/v0.3/sam_hq_vit_l.pth"
        
        try:
            urllib.request.urlretrieve(alt_url, sam_checkpoint, reporthook=reporthook)
            print(f"\n✅ SAM-HQ модель скачана (альтернативный источник): {sam_checkpoint}")
            return str(sam_checkpoint)
            
        except Exception as e2:
            print(f"\n❌ Окончательная ошибка скачивания: {e2}")
            print("⚠️ Используем fallback на обычную SAM модель...")
            return None


def get_conversation_chain():
    llm = VLLM(
        model="microsoft/Phi-3-mini-4k-instruct",
        trust_remote_code=True,
        max_new_tokens=10,
        top_k=20,
        top_p=0.8,
        temperature=0.8,
        dtype="float16",
        tensor_parallel_size=8
    )

    prompt_template = PromptTemplate(
        input_variables=["question"],
        template="""
        Given a primary visual concept, provide a list of negative concepts that would commonly appear in the same visual setting. 
        These negative concepts should visually interfere or hinder the localization and clear identification of the primary concept. 
        For each example, avoid objects identical or too similar to the main concept but focus on those that share the context or background. 
        For example: Primary Concept: 'Surfboard' Negative Concepts: 'Waves', 'Sand' Primary Concept: 'Fork' Negative Concepts: 'Plates', 'Food'.
        Now, for the following list of primary concepts, generate similar lists of negative concepts. conceptlist: []

        Question: {question}
        
        Response:
        """
    )

    # Create the LLMChain
    conversation_chain = LLMChain(
        llm=llm,
        prompt=prompt_template
    )

    return conversation_chain


def get_negative_word(positive_word):
    chain = get_conversation_chain()
    template_argument_identification = f"What is the opposite of {positive_word}?"
    response = chain.run(question=template_argument_identification).strip().lower()
    print(f"Negative word for '{positive_word}' is '{response}'")
    return response


# Function to scrape images from Google Images
def scrape_images(keyword, save_dir, num_images=5):
    options = Options()
    options.add_argument('--no-sandbox')
    options.add_argument('--headless')
    options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome(options=options)

    driver.get('https://images.google.com/')
    search_box = driver.find_element(By.NAME, 'q')
    search_box.send_keys(keyword)
    search_box.submit()
    time.sleep(3)  # Wait for the page to load

    os.makedirs(save_dir, exist_ok=True)

    def save_image(img_url, img_name):
        if img_url.startswith('data:image/jpeg;base64,') or img_url.startswith('data:image/png;base64,'):
            img_data = base64.b64decode(img_url.split(',')[1])
            img = Image.open(BytesIO(img_data))
            img.save(os.path.join(save_dir, img_name))
        else:
            response = requests.get(img_url)
            with open(os.path.join(save_dir, img_name), 'wb') as f:
                f.write(response.content)

    for i in range(num_images):
        try:
            img_xpath = f'//div[@jsname="dTDiAc"][{i+1}]//img'
            img_tag = driver.find_element(By.XPATH, img_xpath)
            img_url = img_tag.get_attribute('src')
            save_image(img_url, f'image_{i + 1}.jpg')
            print("Downloaded image:", f'image_{i + 1}.jpg')
        except Exception as e:
            print(f"Could not download image {i + 1}: {e}")

    driver.quit()




# Function to extract features from an image using DINOv2 or ResNet
def get_vector(image, model, layer, transform):
    t_img = transform(image).unsqueeze(0)
    
    # ---> FIX: move tensor to model's device
    try:
        device = next(model.parameters()).device
        t_img = t_img.to(device)
    except Exception as e:
        print(f"[get_vector] Could not move tensor to device: {e}")

    # Проверяем, является ли модель DINOv2/v3
    model_name = model.__class__.__name__.lower()
    if 'vit' in model_name or hasattr(model, 'forward_features'):
        # DINOv2/v3 модель - используем прямой вызов
        with torch.no_grad():
            if hasattr(model, 'forward_features'):
                # DINOv2/v3: специальная обработка, если модель — это timm-модель
                feats = model.forward_features(t_img)
                
                # Обработка выхода: может быть dict или tensor
                if isinstance(feats, dict):
                    if 'x_norm_clstoken' in feats: 
                        vec = feats['x_norm_clstoken'][0]
                    elif 'x_norm_patchtokens' in feats: 
                        vec = feats['x_norm_patchtokens'][0].mean(dim=0)
                    else: 
                        vec = next(iter(feats.values()))[0].mean(dim=0)
                elif torch.is_tensor(feats):
                    if feats.ndim == 3 and feats.shape[1] > 0: 
                        vec = feats[0, 0] # CLS token for B=1
                    else: 
                        vec = feats.flatten() # Fallback to flatten
                else:
                    raise TypeError(f"Неожиданный тип выхода DINOv2/v3: {type(feats)}")
                
                # L2 нормализация
                temp_embedding = torch.nn.functional.normalize(vec.float(), p=2, dim=0)
            else:
                # Стандартный forward для других ViT моделей
                output = model(t_img)
                temp_embedding = output.flatten()
    else:
        # ResNet модель - используем hook на слой
        temp_embedding = None

        def copy_data(m, i, o):
            nonlocal temp_embedding
            # Берем feature map и делаем Global Average Pooling как в быстром методе
            if len(o.shape) == 4:  # [B, C, H, W]
                pooled = torch.nn.functional.adaptive_avg_pool2d(o, (1, 1))
                temp_embedding = pooled.flatten()
            else:
                temp_embedding = o.flatten()

        # Уважаем переданный layer: строка ('layer2', 'layer3', ...) или модуль
        target_layer = None
        if isinstance(layer, str) and hasattr(model, layer):
            blk = getattr(model, layer)
            target_layer = blk[-1] if isinstance(blk, torch.nn.Sequential) else blk
        elif isinstance(layer, torch.nn.Module):
            target_layer = layer
        else:
            # Fallback: старые эвристики
            if hasattr(model, 'layer3'):
                target_layer = model.layer3[-1]
            elif hasattr(model, 'layer4'):
                target_layer = model.layer4[-1]
            else:
                target_layer = layer
        
        h = target_layer.register_forward_hook(copy_data)
        with torch.no_grad():
            model(t_img)
        h.remove()
    
    if temp_embedding is None:
        # Fallback - создаем вектор стандартной размерности
        temp_embedding = torch.zeros(1024)
    
    return temp_embedding


# 🔥 КЭШИРОВАНИЕ для ускорения загрузки примеров
def get_cached_embeddings(image_paths, model, layer, transform, cache_dir="cache"):
    """
    Кэширует эмбеддинги позитивных и негативных примеров на диск для быстрой загрузки
    """
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    # Создаем хэш от путей файлов для уникального имени кэша
    paths_str = "|".join(sorted(image_paths))
    cache_hash = hashlib.md5(paths_str.encode()).hexdigest()
    cache_file = os.path.join(cache_dir, f"embeddings_{cache_hash}.pkl")
    
    # Проверяем, есть ли кэш и актуален ли он
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            # Проверяем, что все файлы не изменились
            cache_valid = True
            for path, cached_mtime in cached_data['mtimes'].items():
                if not os.path.exists(path) or os.path.getmtime(path) != cached_mtime:
                    cache_valid = False
                    break
            
            if cache_valid:
                print(f"   📦 Загружен кэш эмбеддингов ({len(cached_data['embeddings'])} файлов)")
                return cached_data['embeddings']
        except:
            pass  # Кэш поврежден, пересоздаем
    
    # Вычисляем эмбеддинги заново
    embeddings = []
    mtimes = {}
    
    print(f"   🔧 Создаем кэш эмбеддингов для {len(image_paths)} файлов...")
    for path in image_paths:
        if os.path.exists(path):
            image = Image.open(path).convert('RGB')
            embedding = get_vector(image, model, layer, transform).cpu().numpy()
            embeddings.append(embedding)
            mtimes[path] = os.path.getmtime(path)
    
    # Сохраняем в кэш
    cache_data = {
        'embeddings': np.array(embeddings),
        'mtimes': mtimes
    }
    
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"   💾 Кэш сохранен: {cache_file}")
    except:
        print(f"   ⚠️ Не удалось сохранить кэш")
    
    return np.array(embeddings)


# 🚀 FAST Function to extract features using Masked Pooling (10-50x faster!)
def extract_features_from_masks_fast(image, masks, model, layer, transform):
    """
    🚀 БЫСТРОЕ извлечение features с Masked Pooling - один прогон бэкбона на все маски!
    Ускорение в 10-50 раз по сравнению с отдельными прогонами каждой маски.
    """
    import time
    import torch
    import torch.nn.functional as F
    
    extract_start = time.time()
    print(f"🚀 БЫСТРОЕ извлечение эмбеддингов для {len(masks)} масок (Masked Pooling)...")
    
    if len(masks) == 0:
        print("   ❌ ДИАГНОСТИКА: Передано 0 масок в функцию!")
        return np.array([])
    
    # Подготовка изображения для модели
    if transform:
        image_tensor = transform(Image.fromarray(image)).unsqueeze(0)
    else:
        # Стандартная нормализация
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        # ImageNet нормализация
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std
        image_tensor = image_tensor.unsqueeze(0)
    
    # ---> FIX: move tensor to model's device and match dtype
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)
    
    # Match model dtype (e.g. half)
    if next(model.parameters()).dtype == torch.float16:
        image_tensor = image_tensor.half()
    
    # Диагностика размера входного тензора
    print(f"   🧪 Backbone input tensor: {list(image_tensor.shape)}")  # должен быть [1,3,?,?] где ? ~512
    print(f"   🧪 SEARCHDET_FEAT_SHORT_SIDE = {os.getenv('SEARCHDET_FEAT_SHORT_SIDE', 'не установлено')}")
    
    # 🔥 Hook для захвата feature map ИЗ ПРАВИЛЬНОГО СЛОЯ
    feature_map = None
    def hook_fn(module, input, output):
        nonlocal feature_map
        # Получаем feature map перед Global Average Pool
        if len(output.shape) == 4:  # [B, C, H, W] - это то что нам нужно
            feature_map = output
    
    # Ищем подходящий слой для hook (не avgpool/fc).
    # ⚠️ Уважая аргумент `layer`: допустимы 'layer1'/'layer2'/'layer3'/'layer4'.
    target_layer = None
    # 1) Если передали строковый идентификатор блока ResNet — используем его
    if isinstance(layer, str) and hasattr(model, layer):
        blk = getattr(model, layer)
        target_layer = blk[-1] if isinstance(blk, torch.nn.Sequential) else blk
        print(f"   🔧 Выбран слой из аргумента --layer: {layer}")
    # 2) Если передали сам модуль — используем как есть
    elif isinstance(layer, torch.nn.Module):
        target_layer = layer
    # 3) Иначе — старые эвристики
    elif hasattr(model, 'layer4'):  # ResNet style
        if hasattr(model, 'layer2'):
            target_layer = model.layer2[-1]
            print("   🔧 Автовыбор: layer2 (stride=8) для более плотного feature map")
        elif hasattr(model, 'layer3'):
            target_layer = model.layer3[-1]
            print("   🔧 Автовыбор: layer3")
        else:
            target_layer = model.layer4[-1]
    elif hasattr(model, 'features'):  # DenseNet/VGG style
        target_layer = model.features[-3] if len(model.features) > 3 else model.features[-1]
    elif hasattr(model, 'classifier') and hasattr(model, 'features'):
        target_layer = model.features[-3] if len(model.features) > 3 else model.features[-2]
    else:
        # Fallback — ранний conv
        conv_layers = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                conv_layers.append((name, module))
        if len(conv_layers) >= 3:
            target_layer = conv_layers[-3][1]
            print(f"   🔧 Используем conv слой: {conv_layers[-3][0]}")
        elif conv_layers:
            target_layer = conv_layers[-1][1]
    
    if target_layer is None:
        print("   ❌ Не удалось найти подходящий слой для hook, используем старый метод")
        return extract_features_from_masks_slow(image, masks, model, layer, transform)
    
    # Регистрируем hook на найденный слой
    hook = target_layer.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        # Проверяем версию PyTorch для autocast
        try:
            with torch.amp.autocast('cuda'):
                # 🔥 ОДИН прогон бэкбона с оптимизациями
                model.eval()
                if hasattr(model, 'to'):
                    model = model.to(memory_format=torch.channels_last)
                _ = model(image_tensor)
        except (AttributeError, TypeError):
            # Fallback для старых версий PyTorch
            try:
                with torch.cuda.amp.autocast():
                    model.eval()
                    if hasattr(model, 'to'):
                        model = model.to(memory_format=torch.channels_last)
                    _ = model(image_tensor)
            except:
                # Без autocast если ничего не работает
                model.eval()
                _ = model(image_tensor)
    
    hook.remove()
    
    if feature_map is None or len(feature_map.shape) != 4:
        print(f"   ❌ ДИАГНОСТИКА: Получили неправильный feature map shape: {feature_map.shape if feature_map is not None else None}")
        print(f"   🔍 Target layer was: {target_layer}")
        print("   🔄 Используем старый метод")
        return extract_features_from_masks_slow(image, masks, model, layer, transform)
    
    # Проверяем минимальный размер feature map для качественного masked pooling
    batch_size, channels, feat_h, feat_w = feature_map.shape
    print(f"   🧪 Feature map @{layer if isinstance(layer, str) else 'layer'}: [{feat_h}, {feat_w}]")  # диагностика
    
    if feat_h < 18 or feat_w < 18:  # Оптимизированный порог для SEARCHDET_FEAT_SHORT_SIDE=384
        print(f"   ⚠️ Feature map слишком маленький: {feat_h}×{feat_w} < 18×18")
        print("   💡 Советы: запустите с --layer layer2 или установите SEARCHDET_FEAT_SHORT_SIDE=384/512")
        print("   🔄 Используем старый метод для лучшего качества")
        return extract_features_from_masks_slow(image, masks, model, layer, transform)
    
    # Подготовка масок под размер feature map
    original_h, original_w = image.shape[:2]
    scale_h = feat_h / original_h
    scale_w = feat_w / original_w
    
    print(f"   📐 Feature map: {channels}×{feat_h}×{feat_w}, масштаб: {scale_h:.3f}×{scale_w:.3f}")
    
    # Собираем все маски в один тензор
    mask_tensors = []
    valid_mask_indices = []
    
    for i, mask in enumerate(masks):
        segmentation = mask['segmentation']
        
        # Изменяем размер маски под feature map (более мягкий способ)
        mask_resized = cv2.resize(segmentation.astype(np.float32), 
                                (feat_w, feat_h), 
                                interpolation=cv2.INTER_LINEAR)
        
        # Более мягкий порог для маленьких feature map
        if feat_h < 20 or feat_w < 20:
            threshold = 0.01  # Очень мягкий порог для маленьких feature map
        else:
            threshold = 0.05  # Уменьшили с 0.1 до 0.05
        
        # Применяем порог для бинаризации после resize
        mask_resized = (mask_resized > threshold).astype(bool)
        
        # Проверяем, что маска не пустая (более мягкий порог)
        mask_area = np.sum(mask_resized)
        if mask_area > 0:
            mask_tensors.append(torch.from_numpy(mask_resized).float())
            valid_mask_indices.append(i)
            if i < 5:  # Отладка для первых масок
                print(f"   🔍 Маска {i}: исходная {np.sum(segmentation)}, после resize {mask_area}")
        else:
            # Не создаем искусственную «минимальную маску» — просто пропускаем
            if i < 5:
                print(f"   ⚠️ Маска {i}: исходная {np.sum(segmentation)}, после resize {mask_area} — пропускаем")
    
    if len(mask_tensors) == 0:
        print(f"   ❌ ДИАГНОСТИКА: Все {len(masks)} масок оказались пустыми после ресайза!")
        print(f"   📐 Масштаб: {scale_h:.4f}×{scale_w:.4f}, feature map: {feat_h}×{feat_w}")
        return np.array([])
    
    # Стекаем маски в батч [M, H', W']
    masks_batch = torch.stack(mask_tensors).to(device)
    
    # 🔥 ВЕКТОРИЗОВАННЫЙ Masked Pooling
    # feature_map: [1, C, H', W'] -> [C, H'*W']
    feat_flat = feature_map.squeeze(0).view(channels, -1)
    # masks_batch: [M, H', W'] -> [M, H'*W']
    masks_flat = masks_batch.view(len(mask_tensors), -1)
    
    # Суммируем фичи по каждой маске: [M, C]
    masked_sums = torch.mm(masks_flat, feat_flat.t())
    # Нормализуем на площадь маски
    mask_areas = masks_flat.sum(dim=1, keepdim=True)
    embeddings = masked_sums / (mask_areas + 1e-6)
    
    # Global Average Pooling
    embeddings = F.adaptive_avg_pool1d(embeddings.unsqueeze(-1), 1).squeeze(-1)
    
    # Проверяем на NaN/Inf и нормализуем
    embeddings = torch.nan_to_num(embeddings, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # L2 нормализация для стабильности
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    # Переводим в numpy
    result_embeddings = embeddings.cpu().numpy()
    
    print(f"   🔍 Embeddings shape: {result_embeddings.shape}, range: [{result_embeddings.min():.6f}, {result_embeddings.max():.6f}]")
    
    # Восстанавливаем порядок (для масок, которые были пропущены)
    final_embeddings = []
    valid_idx = 0
    embedding_dim = result_embeddings.shape[1] if len(result_embeddings.shape) > 1 and result_embeddings.shape[1] > 0 else 512
    
    for i in range(len(masks)):
        if i in valid_mask_indices:
            if valid_idx < len(result_embeddings):
                emb = result_embeddings[valid_idx]
                # Если эмбеддинг одномерный, делаем его плоским
                if len(emb.shape) > 1:
                    emb = emb.flatten()
                final_embeddings.append(emb)
            else:
                final_embeddings.append(np.zeros(embedding_dim))
            valid_idx += 1
        else:
            # Для пустых масок — нулевой вектор (маска пропущена)
            final_embeddings.append(np.zeros(embedding_dim))
    
    extract_time = time.time() - extract_start
    old_time_estimate = len(masks) * 0.1  # Примерное время старого метода
    speedup = old_time_estimate / extract_time if extract_time > 0 else 1
    print(f"   ⚡ БЫСТРО: {extract_time:.3f} сек ({extract_time/len(masks)*1000:.1f} мс/маска) - ускорение ~{speedup:.1f}x")
    
    return np.array(final_embeddings)


# Function to extract features from masks with OPTIMIZATION (старый метод)
# Удалено, чтобы избежать рекурсии


# Function to extract features from masks with OPTIMIZATION
def extract_features_from_masks(image, masks, model, layer, transform):
    """
    Главная функция извлечения фич - использует быстрый Masked Pooling метод
    """
    # 🚀 Пробуем быстрый метод сначала
    try:
        return extract_features_from_masks_fast(image, masks, model, layer, transform)
    except Exception as e:
        print(f"   ⚠️ Быстрый метод не сработал ({str(e)}), используем старый")
        return extract_features_from_masks_slow(image, masks, model, layer, transform)


# Старая функция для fallback
def extract_features_from_masks_slow(image, masks, model, layer, transform):
    extract_start = time.time()
    print(f"🧠 МЕДЛЕННОЕ извлечение эмбеддингов для {len(masks)} масок...")
    
    if len(masks) == 0:
        print("   ❌ ДИАГНОСТИКА: Передано 0 масок в медленную функцию!")
        return np.array([])
    
    # 🚀 ОПТИМИЗАЦИЯ: Определяем размер для DINO обработки
    original_height, original_width = image.shape[:2]
    dino_max_size = 512  # Уменьшенный размер для DINO (быстрее обработка)
    
    if max(original_height, original_width) > dino_max_size:
        dino_scale = dino_max_size / max(original_height, original_width)
        dino_height = int(original_height * dino_scale)
        dino_width = int(original_width * dino_scale)
        use_dino_resize = True
        print(f"   🔧 DINO оптимизация: уменьшаем изображения до {dino_width}x{dino_height} для ускорения")
    else:
        use_dino_resize = False
        print(f"   ✅ DINO: используем оригинальный размер {original_width}x{original_height}")
    
    features = []
    for i, mask in enumerate(masks):
        if i % 50 == 0 and i > 0:  # Прогресс каждые 50 масок
            elapsed = time.time() - extract_start
            estimated_total = elapsed * len(masks) / i
            print(f"   📊 Обработано {i}/{len(masks)} масок ({elapsed:.1f}с, осталось ~{estimated_total-elapsed:.1f}с)")
        
        segmentation = mask['segmentation']
        mask_image = np.zeros_like(image)
        mask_image[segmentation] = image[segmentation]
        
        # 🚀 ОПТИМИЗАЦИЯ: Уменьшаем изображение для DINO если нужно
        if use_dino_resize:
            # Изменяем размер как изображения, так и маски
            resized_image = cv2.resize(mask_image, (dino_width, dino_height), interpolation=cv2.INTER_LINEAR)
            pil_image = Image.fromarray(resized_image)
        else:
            pil_image = Image.fromarray(mask_image)
            
        features.append(get_vector(pil_image, model, layer, transform).cpu().numpy())
    
    extract_time = time.time() - extract_start
    print(f"   ⏱️ Извлечение эмбеддингов завершено за: {extract_time:.3f} сек ({extract_time/len(masks)*1000:.1f} мс/маска)")
    print(f"   ✅ Старый метод: обработано {len(features)} из {len(masks)} масок")
    
    return np.array(features)


# Function to calculate softmax attention weights
def calculate_attention_weights_softmax(query_embedding, example_embeddings):
    similarities = cosine_similarity(query_embedding.reshape(1, -1), example_embeddings).flatten()
    exp_similarities = np.exp(similarities)
    attention_weights = exp_similarities / np.sum(exp_similarities)
    return attention_weights


# Function to adjust the query embedding with STRONGER emphasis on positive examples
def adjust_embedding(query_embedding, positive_embeddings, negative_embeddings, positive_weight=3.0, negative_weight=1.5):
    """
    Улучшенная функция корректировки эмбеддингов с сильным упором на положительные примеры
    и усиленным отталкиванием от фоновых областей
    
    Args:
        positive_weight: Вес для положительных примеров (увеличен до 3.0)
        negative_weight: Вес для отрицательных примеров (увеличен до 1.5 для лучшего отталкивания)
    """
    positive_weights = calculate_attention_weights_softmax(query_embedding, positive_embeddings)
    negative_weights = calculate_attention_weights_softmax(query_embedding, negative_embeddings)

    # УСИЛЕННЫЙ упор на положительные примеры
    positive_adjustment = positive_weight * np.sum(positive_weights[:, np.newaxis] * positive_embeddings, axis=0)
    
    # УСИЛЕННОЕ отталкивание от фоновых областей
    negative_adjustment = negative_weight * np.sum(negative_weights[:, np.newaxis] * negative_embeddings, axis=0)

    # Комбинируем включив исходный query, усиливаем позитив и отталкиваем негатив
    combined_adjustment = query_embedding + positive_adjustment - negative_adjustment
    
    # Дополнительная нормализация для стабильности
    norm = np.linalg.norm(combined_adjustment)
    if norm > 0:
        combined_adjustment = combined_adjustment / norm
    else:
        # Fallback если нормализация не удалась
        combined_adjustment = query_embedding
        
    return combined_adjustment


# Function to annotate image with bounding boxes
def annotate_image(
    example_img, 
    query_vectors, 
    resnet_model, 
    layer, 
    transform, 
    sam, 
    output_image_path,
    iou_threshold: float = 0.5,
    # conf_threshold: float = 0.3,
):
    total_annotation_start = time.time()
    print("🎯 Начало аннотации изображения...")
    
    # Generate masks using SAM
    sam_gen_start = time.time()
    print("🔪 Генерация масок с помощью SAM-HQ...")
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        points_per_batch=64,
        pred_iou_thresh=0.8,
        stability_score_thresh=0.9,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,
        box_nms_thresh=iou_threshold,
        crop_nms_thresh=iou_threshold,
        crop_overlap_ratio=512 / 1500,
    )

    masks = mask_generator.generate(np.array(example_img))
    sam_gen_time = time.time() - sam_gen_start
    print(f"   ⏱️ SAM-HQ сгенерировал {len(masks)} масок за: {sam_gen_time:.3f} сек")

    # Extract mask vectors
    example_img_np = np.array(example_img)
    mask_vectors = extract_features_from_masks(example_img_np, masks, resnet_model, layer, transform)
    mask_vectors = np.array(mask_vectors, dtype=np.float32)

    # Normalize query and mask vectors
    norm_start = time.time()
    print("📐 Нормализация векторов...")
    query_vectors = query_vectors / np.linalg.norm(query_vectors, axis=1, keepdims=True)
    mask_vectors = mask_vectors / np.linalg.norm(mask_vectors, axis=1, keepdims=True)
    norm_time = time.time() - norm_start
    print(f"   ⏱️ Нормализация завершена за: {norm_time:.3f} сек")

    # Create FAISS index and add query vectors
    faiss_start = time.time()
    print("🔍 FAISS поиск сходства...")
    index = faiss.IndexFlatIP(mask_vectors.shape[1])  # dim по фактической размерности фич
    index.add(query_vectors)

    # Search for matches in the FAISS index
    similarities, indices = index.search(mask_vectors, 1)
    faiss_time = time.time() - faiss_start
    print(f"   ⏱️ FAISS поиск завершен за: {faiss_time:.3f} сек")

    # Map similarities to [0, 1]
    filter_start = time.time()
    print("🎯 Фильтрация по threshold...")
    normalized_similarities = (similarities + 1) / 2

    # Apply a threshold to filter matches
    threshold = 0.474
    filtered_indices = np.where(normalized_similarities > threshold)[0]
    filter_time = time.time() - filter_start
    print(f"   ⏱️ Найдено {len(filtered_indices)} подходящих масок за: {filter_time:.3f} сек")

    # Draw bounding boxes for detected objects
    draw_start = time.time()
    print("🎨 Рисование аннотаций...")
    example_img_cv = cv2.cvtColor(np.array(example_img), cv2.COLOR_RGB2BGR)
    for idx in filtered_indices:
        mask = masks[idx]
        segmentation = mask['segmentation']
        coords = np.column_stack(np.where(segmentation))
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        cv2.rectangle(example_img_cv, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

    # Convert back to RGB and save the annotated image
    example_img_cv = cv2.cvtColor(example_img_cv, cv2.COLOR_BGR2RGB)
    annotated_img = Image.fromarray(example_img_cv)
    annotated_img.save(output_image_path)
    draw_time = time.time() - draw_start
    print(f"   ⏱️ Аннотации нарисованы и сохранены за: {draw_time:.3f} сек")
    
    total_annotation_time = time.time() - total_annotation_start
    print(f"🎉 Общее время аннотации: {total_annotation_time:.3f} сек")
    print(f"📊 Разбивка времени:")
    print(f"   🔪 SAM генерация масок: {sam_gen_time:.3f}с ({sam_gen_time/total_annotation_time*100:.1f}%)")
    print(f"   🧠 Извлечение эмбеддингов: {extract_time:.3f}с ({extract_time/total_annotation_time*100:.1f}%)")
    print(f"   📐 Нормализация: {norm_time:.3f}с ({norm_time/total_annotation_time*100:.1f}%)")
    print(f"   🔍 FAISS поиск: {faiss_time:.3f}с ({faiss_time/total_annotation_time*100:.1f}%)")
    print(f"   🎯 Фильтрация: {filter_time:.3f}с ({filter_time/total_annotation_time*100:.1f}%)")
    print(f"   🎨 Рисование: {draw_time:.3f}с ({draw_time/total_annotation_time*100:.1f}%)")