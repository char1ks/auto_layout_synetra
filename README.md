# 🚀 LLaVA + SAM2 Pipeline для автоматической разметки дефектов

Интеллектуальная система определения материалов и сегментации дефектов с использованием LLaVA для анализа и SAM2 для точной сегментации.

**🚨 Важно:** Модели НЕ включены в репозиторий! Они скачиваются автоматически при первом запуске (~8GB).

## 🔄 Новая архитектура

Система использует **двойной запрос к LLaVA**:
1. **Запрос 1**: Определение типа материала (metal, wood, plastic, etc.)
2. **Запрос 2**: Анализ дефектов с указанием типов и локаций
3. **SAM2**: Точная сегментация на основе подсказок от LLaVA

**Преимущества над FastSAM:**
- ✅ Более точная сегментация благодаря направленным подсказкам
- ✅ Интеллектуальный анализ дефектов через LLaVA
- ✅ Лучшая работа с различными типами материалов
- ✅ Автоматическая генерация точек-подсказок для SAM2

## 📋 Системные требования

- **Python 3.10+** 
- **RAM**: 12GB+ (16GB рекомендуется для SAM2)
- **Свободное место**: 18GB (LLaVA ~7GB + SAM2 ~1GB + зависимости)
- **GPU** (настоятельно рекомендуется): NVIDIA с CUDA 11.8+ или 12.x
- **CPU**: Intel i7/AMD Ryzen 7+ (для работы без GPU)

### Требования по устройствам:
- **Минимум (CPU)**: 16GB RAM, Intel i7-8700K / AMD Ryzen 7 2700X
- **Рекомендуется (GPU)**: 8GB+ VRAM, RTX 3060+ / RTX 4060+
- **Оптимально**: 16GB+ RAM, RTX 4070+ / RTX 4090

## ⚡ ПРОСТАЯ УСТАНОВКА И ЗАПУСК

### 🚀 **Автоматическая установка (Linux/macOS) - РЕКОМЕНДУЕТСЯ:**

```bash
# 1. Клонируйте проект
git clone https://github.com/char1ks/auto_layout_synetra.git
cd auto_layout_synetra

# 2. Запустите автоматическую установку (установит всё сразу)
chmod +x setup_llava_sam2.sh
./setup_llava_sam2.sh

# 3. Запустите анализ
./run_sam2_pipeline.sh input/test_metal.jpg
```

### 🖥️ **Пошаговая установка (любая ОС):**

#### **Шаг 1: Клонирование проекта**
```bash
# Клонируйте проект
git clone https://github.com/char1ks/auto_layout_synetra.git
cd auto_layout_synetra
```

#### **Шаг 2: Создание Python окружения**
```bash
# Создайте виртуальное окружение
python3 -m venv venv

# Активируйте окружение
# На Linux/macOS:
source venv/bin/activate
# На Windows:
# venv\Scripts\activate

# Обновите pip
pip install --upgrade pip
```

#### **Шаг 3: Установка PyTorch**
```bash
# Для GPU (NVIDIA CUDA 11.8):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Для GPU (NVIDIA CUDA 12.1):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Для CPU (если нет GPU):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Для Apple Silicon (M1/M2):
pip install torch torchvision torchaudio
```

#### **Шаг 4: Установка зависимостей**
```bash
# Установите основные зависимости
pip install -r requirements_llava_sam2.txt
```

#### **Шаг 5: Установка LLaVA**
```bash
# Клонируйте и установите LLaVA
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
pip install -e .
cd ..
```

#### **Шаг 6: Установка SAM2**
```bash
# Клонируйте и установите SAM2
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2
pip install -e .
cd ..
```

#### **Шаг 7: Создание папок**
```bash
# Создайте необходимые папки
mkdir -p models input output
```

## 🎯 ЗАПУСК АНАЛИЗА

### **Простой запуск:**
```bash
# 1. Активируйте окружение (если не активно)
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# 2. Поместите изображение в папку input/
cp your_image.jpg input/

# 3. Запустите анализ
python3 llava_sam2_pipeline.py --image input/your_image.jpg --output output/

# 4. Результаты появятся в папке output/
```

### **Примеры команд:**
```bash
# Анализ конкретного файла
python3 llava_sam2_pipeline.py --image input/test_metal.jpg --output output/

# Анализ с указанием выходной папки
python3 llava_sam2_pipeline.py --image input/defect_sample.png --output results/

# Использование готового скрипта (Linux/macOS)
./run_sam2_pipeline.sh input/your_image.jpg
```

### **Параметры запуска:**
```bash
python3 llava_sam2_pipeline.py --help

# Опции:
# --image IMAGE    Путь к изображению для анализа (обязательно)
# --output OUTPUT  Директория для сохранения результатов (по умолчанию: ./output)
```

## 📥 ЧТО ПРОИСХОДИТ ПРИ ПЕРВОМ ЗАПУСКЕ

**⚠️ ВАЖНО:** Первый запуск занимает 5-15 минут для скачивания моделей!

При первом запуске автоматически скачаются:
1. **LLaVA-1.5-7B модель** (~7GB) - для анализа материалов и дефектов
2. **SAM2-Hiera-Large модель** (~1GB) - для точной сегментации
3. **Tokenizer и конфиги** (~500MB) - дополнительные файлы

**Общий размер:** ~8.5GB  
**Где хранятся:** `~/.cache/huggingface/` и `models/`  
**Интернет:** Нужен только при первом запуске

## 🔍 Как работает pipeline

```
📸 Ваше изображение
         ↓
🔬 LLaVA: "Какой материал?" → metal/wood/plastic
         ↓
🔍 LLaVA: "Где дефекты?" → scratch at center, dent at top
         ↓
🎯 SAM2: Точная сегментация → маски дефектов
         ↓
🎨 OpenCV: Постобработка → финальные результаты
         ↓
📊 Сохранение результатов
```

## 📊 РЕЗУЛЬТАТЫ

После анализа в папке `output/` появятся:

### **Файлы результатов:**
- **`image_result.jpg`** - Визуализация с цветными масками дефектов
- **`image_analysis.json`** - Полный анализ в формате JSON
- **`image_mask_1.png`**, **`image_mask_2.png`** - Отдельные маски дефектов

### **JSON содержит:**
```json
{
  "material": {
    "material": "metal",
    "confidence": 0.9,
    "description": "Metallic surface with wear patterns"
  },
  "defect_analysis": {
    "defects_found": true,
    "defect_types": ["scratch", "corrosion"],
    "defect_locations": ["center", "bottom-right"],
    "severity": "moderate",
    "prompt_points": [[320, 240], [480, 360]]
  },
  "defects": [
    {
      "id": 1,
      "category": "scratch",
      "bbox": [310, 230, 20, 80],
      "confidence": 0.85,
      "severity": "moderate"
    }
  ]
}
```

## ⚡ ПРОИЗВОДИТЕЛЬНОСТЬ

| Устройство | Время анализа |
|---|---|
| **RTX 4090** | 3-6 секунд |
| **RTX 4070** | 5-8 секунд |
| **RTX 3060** | 8-12 секунд |
| **M2 Pro** | 20-35 секунд |
| **M1** | 35-50 секунд |
| **Intel i7** | 25-45 секунд |

## 🐛 РЕШЕНИЕ ПРОБЛЕМ

### ❌ "Модули не найдены"
```bash
# Переустановите зависимости
pip install -r requirements_llava_sam2.txt --force-reinstall
```

### ❌ "CUDA out of memory"
```bash
# Используйте CPU версию
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### ❌ "Медленная работа"
```bash
# Уменьшите размер изображения до 640x640
# Или используйте GPU если доступен
```

### ❌ "Ошибки установки на Windows"
```cmd
# Установите Visual Studio Build Tools
# Скачайте с: https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

## 📞 БЫСТРАЯ ПОМОЩЬ

**Если ничего не работает:**
1. Убедитесь что Python 3.10+
2. Активируйте виртуальное окружение
3. Проверьте интернет соединение
4. Освободите место на диске (нужно 18GB)
5. Закройте другие программы (нужно 12GB RAM)

**Тестовая команда:**
```bash
python3 -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA доступна:', torch.cuda.is_available())
try:
    from llava.model.builder import load_pretrained_model
    print('LLaVA: ✅')
except:
    print('LLaVA: ❌')
try:
    from sam2.build_sam import build_sam2
    print('SAM2: ✅')
except:
    print('SAM2: ❌')
"
```

**Готово! Теперь можно анализировать дефекты! 🚀**

---

# I этап: Определение материала по фото 

**Текущее решение: LLaVA (Large Language and Vision Assistant)**

*1. CLIP (OpenAI)* - Обучена на огромном количестве фоток и данных, используется для zero-shot классификации

*2. BLIP/BLIP-2* - имеет языковое понимание+зрение, можно задавать вопросы по типу: "What Material on photo?"

*3. AWS Rekognition* - Надежный, но требуется обучение custom модели

*4. Google Cloud Vision AI* - чисто гипотетически можно попробовать юзать general object detection, но может потребоваться дообучение 

*5. OpenAI (через GPT-4o Vision или GPT-4 + Vision API)* - часто достаточно четкий результат, но при этом достаточно трудно использовать в real-time режиме, что для нас главный ред флаг.

**6. LLaVA (Large Language and Vision Assistant)** ⭐ **ИСПОЛЬЗУЕМ** - Open source альтернатива GPT-4V, можно запускать локально, хорошо понимает материалы и текстуры

*7. MiniGPT-4* - Локальная модель с vision capabilities, быстрая для определения типа материала

*8. InstructBLIP* - Отличная open source модель для visual question answering, можно спрашивать "What material is this?"

## Преимущества LLaVA для нашего случая:
- ✅ **Локальная работа** - не нужен интернет после установки
- ✅ **Бесплатность** - никаких API ключей и лимитов
- ✅ **Точность** - отличное понимание материалов
- ✅ **Гибкость** - можно задавать сложные вопросы
- ✅ **Приватность** - данные не покидают вашу систему

# II этап: Обнаружение и анализ дефектов

**Текущее решение: LLaVA с специализированными промптами**

*1. Grounding DINO + SAM (Grounded-SAM)* - Находят объект по тексту: "crack", "corrosion" и т.д, но при этом могут ошибаться в достаточно сложных сценариях.

*2. Detectron2 (Meta)* - Очень надежно работает на производственном уровне, уже имеет встроенное выделение масок, что решает проблему III Этап (фаворит, на мой взгляд) (Потестировал, оказалось, что это не юзабельная херь в нашем контексте, так что ее отбрасываем)

*3. YOLOv8 + Segmentation* - Быстрый и точный для real-time обнаружения дефектов, есть готовые веса для промышленных дефектов

*4. Anomalib (Intel)* - Профессиональная библиотека для обнаружения аномалий, множество алгоритмов (PaDiM, STFPM, PatchCore), работает без разметки

*5. FastSAM* - Быстрая версия SAM для real-time сегментации, хорошо подходит для выделения дефектных областей

*6. SegmentAnything 2 (SAM2)* - Новая версия от Meta, лучше работает с видео и сложными сценариями

*7. YOLO-World* - Zero-shot object detection, можно детектить дефекты по текстовому описанию без дообучения

**8. LLaVA с анализом дефектов** ⭐ **ИСПОЛЬЗУЕМ** - Интеллектуальный анализ через специализированные промпты

## Как работает наш подход:
- **Специализированный промпт** для поиска дефектов
- **Типизация дефектов**: scratches, cracks, dents, corrosion, stains, wear, chips
- **Локализация**: top-left, center, bottom-right, etc.
- **Оценка серьезности**: minor, moderate, severe
- **Генерация подсказок** для следующего этапа

# III этап: Автоматическая сегментация дефектов

**Текущее решение: SAM2 (Segment Anything Model 2) с подсказками от LLaVA**

## Open Source модели (локальный запуск):

**1. Segment Anything Model 2 (SAM2)** ⭐ **ИСПОЛЬЗУЕМ** - Самая мощная модель сегментации от Meta, работает с промптами и автоматически, отличная точность на дефектах

*2. FastSAM* - Быстрая альтернатива SAM, основана на YOLOv8, real-time сегментация, идеально для производства

*3. MobileSAM* - Легкая версия SAM для мобильных устройств и edge computing, быстрая сегментация

*4. EfficientSAM* - Оптимизированная версия SAM, баланс между скоростью и качеством

*5. Grounding DINO + SAM Pipeline* - Связка детекции по тексту + сегментация, можно писать "rust on metal", "crack in wood"

*6. Mask2Former* - Универсальная модель сегментации от Meta, хорошо настраивается под специфические дефекты

*7. OneFormer* - Единая модель для semantic, instance и panoptic сегментации

*8. CLIPSeg* - Сегментация по текстовым промптам, основана на CLIP, zero-shot сегментация дефектов

## API решения:
*1. Roboflow Universe API* - Огромная база готовых моделей для дефектов, API для кастомных моделей, есть автосегментация

*2. Clarifai Custom Models API* - Обучение кастомных моделей сегментации дефектов, API для inference

*3. Azure Custom Vision + Computer Vision API* - Microsoft решение для кастомной сегментации, интеграция с production

*4. AWS SageMaker + Rekognition Custom Labels* - Amazon решение для автоматической сегментации дефектов

*5. Google Vertex AI Vision API* - AutoML для сегментации, можно обучать на дефектах материалов

*6. Ultralytics HUB API* - Облачная платформа для YOLO моделей, есть сегментация дефектов

## Преимущества SAM2 для нашего случая:
- ✅ **Направляемая сегментация** - работает с подсказками от LLaVA
- ✅ **Высочайшая точность** - state-of-the-art качество масок
- ✅ **Универсальность** - работает с любыми объектами
- ✅ **Автоматический режим** - может работать без подсказок
- ✅ **Быстрая работа** - оптимизирован для production

# IV этап: End-to-End Pipeline автосегментации

## Наше решение: LLaVA + SAM2 + OpenCV ⭐

**Архитектура:**
```
LLaVA (материал) → LLaVA (дефекты) → SAM2 (сегментация) → OpenCV (постобработка)
```

## Готовые решения:
*1. Industrial AI Inspection Toolkit* - Комплексное решение: материал → дефекты → сегментация → отчет

*2. OpenCV AI Kit (OAK)* - Hardware + software решение для промышленной инспекции в real-time

*3. Intel OpenVINO Toolkit* - Оптимизация моделей для edge deployment, быстрая сегментация на CPU

*4. NVIDIA Triton + TensorRT* - Высокопроизводительный inference сервер для комплексной обработки

## Альтернативные гибридные подходы:
*1. CLIP (материал) → Anomalib (дефекты) → SAM2 (сегментация)* - Точный pipeline с open source компонентами

*2. LLaVA (материал) → YOLOv8-seg (дефекты+сегментация)* - Быстрый pipeline для production

*3. API (материал) → Local models (дефекты+сегментация)* - Баланс между точностью и контролем данных

*4. Grounding DINO + SAM2 (все в одном)* - Универсальное решение по текстовым промптам

# Рекомендации по выбору:

## Для прототипа:
- **LLaVA + SAM2** ⭐ **НАШ ВЫБОР** - интеллектуальный и точный
- **FastSAM + простые фильтры** - быстро запустить и протестировать
- **Grounding DINO + SAM** - если есть время на настройку

## Для production:
- **LLaVA + SAM2 + оптимизация** ⭐ **РЕКОМЕНДУЕМ** - лучший баланс точности и интеллекта
- **YOLOv8-seg кастомная модель** - если есть данные для обучения  
- **Anomalib + SAM2** - если нет размеченных данных
- **API решения** - если нужна стабильность без поддержки

## Для research:
- **SAM2 + CLIP** - максимальная гибкость
- **Полный pipeline с LLaVA** ⭐ **ИСПОЛЬЗУЕМ** - автоматизация всех этапов

---

# Техническая архитектура нашего решения

## I этап: Определение материала (LLaVA)
- **LLaVA-1.5-7B** - Анализ типа материала по изображению
- Поддержка: metal, wood, plastic, concrete, fabric, glass, ceramic
- Описание состояния поверхности и качества

## II этап: Анализ дефектов (LLaVA)  
- **Специализированный промпт** для поиска дефектов
- Определение типов: scratches, cracks, dents, corrosion, stains, wear, chips
- Локализация: top-left, center, bottom-right, etc.
- Оценка серьезности: minor, moderate, severe

## III этап: Сегментация (SAM2)
- **SAM2-Hiera-Large** - Точная сегментация по подсказкам
- Генерация точек-подсказок на основе анализа LLaVA
- Автоматическая сегментация для дополнительных дефектов
- Фильтрация дублирующихся масок (IoU > 0.7)

## IV этап: Постобработка (OpenCV)
- Морфологические операции для очистки масок
- Фильтрация по размеру в зависимости от материала
- Генерация аннотаций в формате COCO
- Создание визуализации с типами дефектов

## Конфигурация моделей:
- **LLaVA**: liuhaotian/llava-v1.5-7b
- **SAM2**: facebook/sam2-hiera-large
- **Автоматическое скачивание** при первом запуске