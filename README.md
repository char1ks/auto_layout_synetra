# 🚀 LLaVA + FastSAM Pipeline для автоматической разметки дефектов

Автоматическая система определения материалов и сегментации дефектов с использованием LLaVA и FastSAM.

**🚨 Важно:** Модели НЕ включены в репозиторий! Они скачиваются автоматически при первом запуске (~7.5GB).

## 📋 Требования

- **Python 3.10+** 
- **RAM**: 8GB+ (16GB рекомендуется)
- **Свободное место**: 15GB
- **GPU** (опционально): NVIDIA с CUDA для ускорения

## ⚡ Полная установка (Python venv)

### 🖥️ **Windows:**

#### Шаг 1: Установка Python
```cmd
# Скачайте Python 3.10+ с https://python.org
# При установке отметьте "Add Python to PATH"

# Проверьте установку
python --version
pip --version
```

#### Шаг 2: Создание виртуального окружения
```cmd
# Перейдите в папку проекта
cd путь\к\проекту

# Создайте виртуальное окружение
python -m venv venv

# Активируйте окружение
venv\Scripts\activate

# Обновите pip
python -m pip install --upgrade pip
```

#### Шаг 3: Установка зависимостей
```cmd
# Для GPU (NVIDIA CUDA):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Для CPU (если нет GPU):
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Основные зависимости
pip install transformers==4.37.2
pip install accelerate==0.21.0
pip install opencv-python opencv-contrib-python
pip install numpy==1.26.4 pillow scipy matplotlib
pip install protobuf sentencepiece==0.1.99 einops==0.6.1
pip install gradio==4.16.0 gradio_client==0.8.1
pip install requests tqdm pyyaml regex
pip install safetensors tokenizers huggingface-hub
pip install ultralytics
```

#### Шаг 4: Установка LLaVA
```cmd
# Клонируйте репозиторий LLaVA (автоматически скачается ~200MB)
git clone https://github.com/haotian-liu/LLaVA.git

# Установите LLaVA
cd LLaVA
pip install -e .
cd ..
```

#### Шаг 5: Создание папок
```cmd
mkdir models output input
```

**⚠️ Важно:** Все модели (LLaVA ~7GB, FastSAM ~140MB) будут **автоматически скачаны** при первом запуске. Подключение к интернету обязательно!

### 🐧 **Linux/macOS:**

#### Шаг 1: Проверка Python
```bash
python3 --version  # Должно быть 3.10+
pip3 --version
```

#### Шаг 2: Создание виртуального окружения
```bash
# Перейдите в папку проекта
cd /путь/к/проекту

# Создайте виртуальное окружение
python3 -m venv venv

# Активируйте окружение
source venv/bin/activate

# Обновите pip
pip install --upgrade pip
```

#### Шаг 3: Установка зависимостей
```bash
# Для GPU (NVIDIA CUDA):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Для CPU или macOS:
pip install torch torchvision torchaudio

# Основные зависимости (те же что для Windows)
pip install transformers==4.37.2 accelerate==0.21.0
pip install opencv-python opencv-contrib-python
pip install numpy==1.26.4 pillow scipy matplotlib
pip install protobuf sentencepiece==0.1.99 einops==0.6.1
pip install gradio==4.16.0 gradio_client==0.8.1
pip install requests tqdm pyyaml regex
pip install safetensors tokenizers huggingface-hub
pip install ultralytics
```

#### Шаг 4: Установка LLaVA
```bash
# Клонируйте репозиторий LLaVA (автоматически скачается ~200MB)
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
pip install -e .
cd ..
```

#### Шаг 5: Создание папок
```bash
mkdir -p models output input
```

**⚠️ Важно:** Все модели (LLaVA ~7GB, FastSAM ~140MB) будут **автоматически скачаны** при первом запуске. Подключение к интернету обязательно!

## 🎯 Запуск анализа

### **Активация окружения и запуск:**

#### Windows:
```cmd
# 1. Активировать окружение
venv\Scripts\activate

# 2. ПЕРВЫЙ запуск (скачает модели ~7GB, займет 5-10 минут)
python llava_fastsam_pipeline.py --image 001.png --output output

# 3. Последующие запуски (быстро)
python llava_fastsam_pipeline.py --image "путь\к\изображению.jpg" --output output
```

#### Linux/macOS:
```bash
# 1. Активировать окружение
source venv/bin/activate

# 2. ПЕРВЫЙ запуск (скачает модели ~7GB, займет 5-10 минут)
python llava_fastsam_pipeline.py --image 001.png --output output

# 3. Последующие запуски (быстро)
python llava_fastsam_pipeline.py --image "путь/к/изображению.jpg" --output output
```

## 📥 Что происходит при первом запуске

При первом запуске автоматически скачаются:
1. **LLaVA модель** (~7GB) - для определения материалов
2. **FastSAM модель** (~140MB) - для сегментации дефектов  
3. **Дополнительные файлы** (~500MB) - tokenizer, конфиги

**Общий размер:** ~7.5GB  
**Время скачивания:** 5-10 минут (зависит от интернета)  
**Где хранятся:** В кэше HuggingFace и папке models/

**⚠️ Важно:** Не прерывайте первый запуск! Все последующие запуски будут быстрыми.

## 📊 Результаты

После запуска в папке `output/` появятся:
- **`image_result.jpg`** - Визуализация с цветными масками дефектов
- **`image_analysis.json`** - JSON с координатами и метаданными
- **`image_mask_*.png`** - Отдельные маски для каждого дефекта

## 🔧 Создание bat/sh скриптов для удобства

### Windows (create `run.bat`):
```batch
@echo off
echo Активируем окружение...
call venv\Scripts\activate
echo Запускаем анализ...
python llava_fastsam_pipeline.py --image %1 --output output
pause
```

**Использование:** `run.bat "image.jpg"`

### Linux/macOS (create `run.sh`):
```bash
#!/bin/bash
echo "Активируем окружение..."
source venv/bin/activate
echo "Запускаем анализ..."
python llava_fastsam_pipeline.py --image "$1" --output output
```

**Использование:** `./run.sh "image.jpg"`

## 🐛 Решение проблем

### ❌ "python не найден"
```cmd
# Windows: Переустановите Python с python.org
# Отметьте "Add Python to PATH"

# Linux: 
sudo apt install python3 python3-venv python3-pip

# macOS:
brew install python
```

### ❌ "venv не активируется"
```cmd
# Windows - убедитесь что используете правильный слэш:
venv\Scripts\activate

# Linux/macOS:
source venv/bin/activate
```

### ❌ "CUDA out of memory"
```cmd
# Используйте CPU версию PyTorch:
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### ❌ "LLaVA не найдена"
```cmd
# Переустановите LLaVA:
cd LLaVA
pip install -e . --force-reinstall
cd ..
```

## ⚙️ Оптимизация производительности

### Для GPU:
- Используйте изображения до 1024x1024
- Закройте другие GPU приложения
- Убедитесь что CUDA версия совпадает с PyTorch

### Для CPU:
- Используйте изображения до 640x640  
- Закройте ненужные программы
- Рассмотрите использование более мощного CPU

## 📈 Производительность

- **RTX 4090**: ~2-4 секунды
- **RTX 3080**: ~4-6 секунд
- **CPU Intel i7**: ~15-25 секунд

## 🔄 Что делает система

1. **🔍 LLaVA** - Определяет тип материала на изображении
2. **⚡ FastSAM** - Быстро сегментирует потенциальные дефекты  
3. **🎨 OpenCV** - Очищает и фильтрует результаты по типу материала

**Готово к использованию! 🚀**

---

# I этап: Определение материала по фото 
  *1.CLIP(OpenAi)*-Обучена на огромном количестве фоток и данных,используется для zero-shot классификации

  *2.BLIP/BLIP-2*-имеет языковое понимание+зрение,можно задавать вопросы по типу :"What Material on photo?"

  *3.AWS Rekognition*-Надежный ,но требуется обучение custom модели

  *4.Google Cloud Vision AI*-чисто гипотетически можно попробовать юзать general object detection ,но может потребоваться дообучение 

  *5. OpenAI (через GPT-4o Vision или GPT-4 + Vision API)*-часто достаточно четкий результат,но при этом достаточно трудно использовать в real-time режиме,что для нас главный ред флаг.

  *6. LLaVA (Large Language and Vision Assistant)*-Open source альтернатива GPT-4V, можно запускать локально, хорошо понимает материалы и текстуры

  *7. MiniGPT-4*-Локальная модель с vision capabilities, быстрая для определения типа материала

  *8. InstructBLIP*-Отличная open source модель для visual question answering, можно спрашивать "What material is this?"

# II этап: Обнаружение дефектов
  *1.Grounding DINO + SAM (Grounded-SAM)*-Находят объект по тексту: "crack", "corrosion" и т.д,но при этом могут ошибаться в достаточно сложных сценариях.

  *2.Detectron2 (Meta)*-Очень надежно работает на производственном уровне,уже имеет встроенное выделение масок,что решает проблему III Этап(фаворит,на мой взгляд)(Потестировал,оказалось,что это не юзабельная херь в нашем контексте ,так что ее отбрасываем)

  *3. YOLOv8 + Segmentation*-Быстрый и точный для real-time обнаружения дефектов, есть готовые веса для промышленных дефектов

  *4. Anomalib (Intel)*-Профессиональная библиотека для обнаружения аномалий, множество алгоритмов (PaDiM, STFPM, PatchCore), работает без разметки

  *5. FastSAM*-Быстрая версия SAM для real-time сегментации, хорошо подходит для выделения дефектных областей

  *6. SegmentAnything 2 (SAM2)*-Новая версия от Meta, лучше работает с видео и сложными сценариями

  *7. YOLO-World*-Zero-shot object detection, можно детектить дефекты по текстовому описанию без дообучения

# III этап: Автоматическая сегментация дефектов

## Open Source модели (локальный запуск):
  *1. Segment Anything Model 2 (SAM2)*-Самая мощная модель сегментации от Meta, работает с промптами и автоматически, отличная точность на дефектах

  *2. FastSAM*-Быстрая альтернатива SAM, основана на YOLOv8, real-time сегментация, идеально для производства

  *3. MobileSAM*-Легкая версия SAM для мобильных устройств и edge computing, быстрая сегментация

  *4. EfficientSAM*-Оптимизированная версия SAM, баланс между скоростью и качеством

  *5. Grounding DINO + SAM Pipeline*-Связка детекции по тексту + сегментация, можно писать "rust on metal", "crack in wood"

  *6. Mask2Former*-Универсальная модель сегментации от Meta, хорошо настраивается под специфические дефекты

  *7. OneFormer*-Единая модель для semantic, instance и panoptic сегментации

  *8. CLIPSeg*-Сегментация по текстовым промптам, основана на CLIP, zero-shot сегментация дефектов

## API решения:
  *1. Roboflow Universe API*-Огромная база готовых моделей для дефектов, API для кастомных моделей, есть автосегментация

  *2. Clarifai Custom Models API*-Обучение кастомных моделей сегментации дефектов, API для inference

  *3. Azure Custom Vision + Computer Vision API*-Microsoft решение для кастомной сегментации, интеграция с production

  *4. AWS SageMaker + Rekognition Custom Labels*-Amazon решение для автоматической сегментации дефектов

  *5. Google Vertex AI Vision API*-AutoML для сегментации, можно обучать на дефектах материалов

  *6. Ultralytics HUB API*-Облачная платформа для YOLO моделей, есть сегментация дефектов

# IV этап: End-to-End Pipeline автосегментации

## Готовые решения:
  *1. Industrial AI Inspection Toolkit*-Комплексное решение: материал → дефекты → сегментация → отчет

  *2. OpenCV AI Kit (OAK)*-Hardware + software решение для промышленной инспекции в real-time

  *3. Intel OpenVINO Toolkit*-Оптимизация моделей для edge deployment, быстрая сегментация на CPU

  *4. NVIDIA Triton + TensorRT*-Высокопроизводительный inference сервер для комплексной обработки

## Гибридные подходы (наш случай):
  *1. CLIP (материал) → Anomalib (дефекты) → SAM2 (сегментация)*-Точный pipeline с open source компонентами

  *2. LLaVA (материал) → YOLOv8-seg (дефекты+сегментация)*-Быстрый pipeline для production

  *3. API (материал) → Local models (дефекты+сегментация)*-Баланс между точностью и контролем данных

  *4. Grounding DINO + SAM2 (все в одном)*-Универсальное решение по текстовым промптам

# Рекомендации по выбору:

## Для прототипа:
- **FastSAM + простые фильтры** - быстро запустить и протестировать
- **Grounding DINO + SAM** - если есть время на настройку

## Для production:
- **YOLOv8-seg кастомная модель** - если есть данные для обучения  
- **Anomalib + SAM2** - если нет размеченных данных
- **API решения** - если нужна стабильность без поддержки

## Для research:
- **SAM2 + CLIP** - максимальная гибкость
- **Полный pipeline с LLaVA** - автоматизация всех этапов