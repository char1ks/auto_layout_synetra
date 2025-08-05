import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import cv2
import faiss
from sklearn.metrics.pairwise import cosine_similarity
from segment_anything_hq import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
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


# Initialize models
def initialize_models():
    print("🚀 Инициализация SearchDet моделей...")
    
    # Load ResNet model
    print("📦 Загружаем ResNet101...")
    resnet_model = models.resnet101(pretrained=True)
    layer = resnet_model._modules.get('avgpool')
    resnet_model.eval()

    # Define transformation pipeline
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Initialize SAM model with automatic download
    print("🎯 Инициализация SAM-HQ модели...")
    
    try:
        # Попытка скачать SAM-HQ модель
        sam_checkpoint = download_sam_hq_model()
        
        if sam_checkpoint and os.path.exists(sam_checkpoint):
            # Используем SAM-HQ
            model_type = "vit_l"
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            print(f"🔄 Загружаем SAM-HQ модель на {device}...")
            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam.to(device=device)
            print("✅ SAM-HQ модель загружена успешно")
            
        else:
            # Fallback: используем обычную SAM модель
            print("⚠️ SAM-HQ недоступна, используем fallback...")
            raise Exception("SAM-HQ не найдена, переходим к fallback")
            
    except Exception as e:
        print(f"❌ Ошибка загрузки SAM-HQ: {e}")
        print("🔄 Fallback: пробуем использовать обычную SAM модель...")
        
        try:
            # Попытка использовать стандартную SAM модель
            from segment_anything import sam_model_registry as sam_registry_fallback
            from segment_anything import SamAutomaticMaskGenerator as SamFallback
            
            # Скачиваем стандартную SAM модель
            sam_fallback_dir = Path("models")
            sam_fallback_dir.mkdir(exist_ok=True)
            sam_fallback_checkpoint = sam_fallback_dir / "sam_vit_l_0b3195.pth"
            
            if not sam_fallback_checkpoint.exists():
                print("📥 Скачиваем стандартную SAM модель...")
                sam_fallback_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
                urllib.request.urlretrieve(sam_fallback_url, sam_fallback_checkpoint)
                print("✅ Стандартная SAM модель скачана")
            
            model_type = "vit_l"
            device = "cuda" if torch.cuda.is_available() else "cpu"
            sam = sam_registry_fallback[model_type](checkpoint=str(sam_fallback_checkpoint))
            sam.to(device=device)
            print("✅ Fallback SAM модель загружена успешно")
            
        except Exception as e2:
            print(f"❌ Критическая ошибка: не удалось загрузить ни одну SAM модель: {e2}")
            print("💡 Решение: установите segment-anything или segment-anything-hq вручную")
            raise e2

    return resnet_model, layer, transform, sam


# Function to extract features from an image using ResNet
def get_vector(image, model, layer, transform):
    t_img = transform(image).unsqueeze(0)
    my_embedding = torch.zeros(2048)

    def copy_data(m, i, o):
        my_embedding.copy_(o.flatten())

    h = layer.register_forward_hook(copy_data)
    with torch.no_grad():
        model(t_img)
    h.remove()
    return my_embedding


# Function to extract features from masks
def extract_features_from_masks(image, masks, model, layer, transform):
    features = []
    for mask in masks:
        segmentation = mask['segmentation']
        mask_image = np.zeros_like(image)
        mask_image[segmentation] = image[segmentation]
        pil_image = Image.fromarray(mask_image)
        features.append(get_vector(pil_image, model, layer, transform).numpy())
    return np.array(features)


# Function to calculate softmax attention weights
def calculate_attention_weights_softmax(query_embedding, example_embeddings):
    similarities = cosine_similarity(query_embedding.reshape(1, -1), example_embeddings).flatten()
    exp_similarities = np.exp(similarities)
    attention_weights = exp_similarities / np.sum(exp_similarities)
    return attention_weights


# Function to adjust the query embedding
def adjust_embedding(query_embedding, positive_embeddings, negative_embeddings):
    positive_weights = calculate_attention_weights_softmax(query_embedding, positive_embeddings)
    negative_weights = calculate_attention_weights_softmax(query_embedding, negative_embeddings)

    # Compute weighted sums of positive and negative embeddings
    positive_adjustment = np.sum(positive_weights[:, np.newaxis] * positive_embeddings, axis=0)
    negative_adjustment = np.sum(negative_weights[:, np.newaxis] * negative_embeddings, axis=0)

    # Subtract negative adjustment from positive adjustment
    combined_adjustment = positive_adjustment - negative_adjustment
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
    # Generate masks using SAM
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

    # Extract mask vectors
    example_img_np = np.array(example_img)
    mask_vectors = extract_features_from_masks(example_img_np, masks, resnet_model, layer, transform)
    mask_vectors = np.array(mask_vectors, dtype=np.float32)

    # Normalize query and mask vectors
    query_vectors = query_vectors / np.linalg.norm(query_vectors, axis=1, keepdims=True)
    mask_vectors = mask_vectors / np.linalg.norm(mask_vectors, axis=1, keepdims=True)

    # Create FAISS index and add query vectors
    index = faiss.IndexFlatIP(2048)  # Using inner product for cosine similarity
    index.add(query_vectors)

    # Search for matches in the FAISS index
    similarities, indices = index.search(mask_vectors, 1)

    # Map similarities to [0, 1]
    normalized_similarities = (similarities + 1) / 2

    # Apply a threshold to filter matches
    threshold = 0.474
    filtered_indices = np.where(normalized_similarities > threshold)[0]

    # Draw bounding boxes for detected objects
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


# Function to annotate image with polygon overlays and area filtering (min & max)
def annotate_image_v2(
    example_img: Image.Image,
    query_vectors: np.ndarray,
    resnet_model,
    layer,
    transform,
    sam,
    output_image_path: str,
    iou_threshold: float = 0.5,
    min_area_threshold: int = 1000,
    max_area_threshold: int = 1000000,
    similarity_thrsh: float = 0.474,
    # conf_threshold: float = 0.3,  # optionally add if using confidence filtering
):
    # Generate masks using SAM
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

    # Run SAM to get mask dicts
    masks = mask_generator.generate(np.array(example_img))

    # Extract features for each mask
    example_img_np = np.array(example_img)
    mask_vectors = extract_features_from_masks(
        example_img_np, masks, resnet_model, layer, transform
    )
    mask_vectors = np.array(mask_vectors, dtype=np.float32)

    # Normalize for cosine similarity
    query_vectors = query_vectors / np.linalg.norm(query_vectors, axis=1, keepdims=True)
    mask_vectors = mask_vectors / np.linalg.norm(mask_vectors, axis=1, keepdims=True)

    # Build FAISS index
    index = faiss.IndexFlatIP(mask_vectors.shape[1])  # inner product on feature dim
    index.add(query_vectors)

    # Search nearest query for each mask
    similarities, indices = index.search(mask_vectors, 1)
    normalized_similarities = (similarities + 1) / 2  # map to [0,1]

    # Threshold matches
    filtered_idxs = np.where(normalized_similarities.flatten() > similarity_thrsh)[0]

    # Prepare image for drawing
    canvas = cv2.cvtColor(np.array(example_img), cv2.COLOR_RGB2BGR)

    for idx in filtered_idxs:
        mask = masks[idx]
        seg = mask['segmentation']  # boolean mask HxW

        # Compute mask area and filter by min/max thresholds
        area = seg.sum()
        if area < min_area_threshold or area > max_area_threshold:
            continue

        # Find contours
        binary = seg.astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Draw each contour as polygon
        for cnt in contours:
            c_area = cv2.contourArea(cnt)
            if c_area < min_area_threshold or c_area > max_area_threshold:
                continue
            cv2.polylines(canvas, [cnt], isClosed=True, color=(0, 255, 0), thickness=2)
            # Optionally fill polygon:
            # cv2.drawContours(canvas, [cnt], -1, (0, 255, 0), thickness=cv2.FILLED)

    # Convert back to PIL and save
    annotated = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    annotated_img = Image.fromarray(annotated)
    annotated_img.save(output_image_path)


# Main function to process images
def detect_and_annotate_objects(example_image_path, keyword, output_image_path='annotated_output.jpg', num_query_images=5):
    # Initialize models
    resnet_model, layer, transform, sam = initialize_models()

    # Scrape images from the web for positive keyword
    query_image_folder = f"queried_downloads/query_images_{keyword}"
    scrape_images(keyword, query_image_folder, num_query_images)

    # Get negative keyword and scrape images for it
    # negative_keyword = get_negative_word(keyword)
    negative_keyword = "black hair"
    negative_image_folder = f"queried_downloads/negative_images_{negative_keyword}"
    scrape_images(negative_keyword, negative_image_folder, num_query_images)

    # Load example image, query images, and negative images
    example_img = Image.open(example_image_path).convert("RGB")
    query_imgs = [Image.open(os.path.join(query_image_folder, img)).convert("RGB") 
                  for img in os.listdir(query_image_folder) if img.endswith('.jpg')]
    negative_imgs = [Image.open(os.path.join(negative_image_folder, img)).convert("RGB") 
                     for img in os.listdir(negative_image_folder) if img.endswith('.jpg')]

    # Extract query vectors and negative vectors
    positive_embeddings = [get_vector(img, resnet_model, layer, transform).numpy() for img in query_imgs]
    positive_embeddings = np.array(positive_embeddings, dtype=np.float32)

    negative_embeddings = [get_vector(img, resnet_model, layer, transform).numpy() for img in negative_imgs]
    negative_embeddings = np.array(negative_embeddings, dtype=np.float32)

    # Adjust the query embedding for each query image
    adjusted_query_vectors = np.array([
        adjust_embedding(embedding, positive_embeddings, negative_embeddings)
        for embedding in positive_embeddings
    ])

    # Annotate the example image
    annotate_image(example_img, adjusted_query_vectors, resnet_model, layer, transform, sam, output_image_path)


if __name__ == "__main__":
    # Example usage
    detect_and_annotate_objects('/surfboard_with_ocean.jpg', 'surfboard')