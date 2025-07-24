#!/bin/bash
set -e

echo "🚀 Установка LLaVA + SAM2 + OpenCV Pipeline"
echo "============================================"

# Проверка Python версии
echo "🐍 Проверка Python версии..."
python3 --version

# Проверка pip
echo "📦 Обновление pip..."
python3 -m pip install --upgrade pip

# Создание виртуального окружения (опционально)
echo "🌍 Создание виртуального окружения..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Виртуальное окружение создано"
else
    echo "✅ Виртуальное окружение уже существует"
fi

# Активация виртуального окружения
echo "🔌 Активация виртуального окружения..."
source venv/bin/activate

# Установка PyTorch (с поддержкой CUDA если доступна)
echo "🔥 Установка PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    echo "   🎯 CUDA найдена, устанавливаем PyTorch с CUDA"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "   💻 CUDA не найдена, устанавливаем CPU версию PyTorch"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Установка базовых зависимостей
echo "📚 Установка базовых зависимостей..."
pip install -r requirements_llava_sam2.txt

# Установка LLaVA
echo "🔬 Установка LLaVA..."
if [ ! -d "LLaVA" ]; then
    echo "   📥 Клонирование LLaVA репозитория..."
    git clone https://github.com/haotian-liu/LLaVA.git
    cd LLaVA
    pip install -e .
    cd ..
    echo "   ✅ LLaVA установлена"
else
    echo "   ✅ LLaVA уже установлена"
fi

# Установка SAM2
echo "🎯 Установка SAM2..."
if [ ! -d "segment-anything-2" ]; then
    echo "   📥 Клонирование SAM2 репозитория..."
    git clone https://github.com/facebookresearch/segment-anything-2.git
    cd segment-anything-2
    pip install -e .
    cd ..
    echo "   ✅ SAM2 установлен"
else
    echo "   ✅ SAM2 уже установлен"
fi

# Создание директорий
echo "📁 Создание необходимых директорий..."
mkdir -p models
mkdir -p input
mkdir -p output

# Скачивание моделей
echo "📥 Скачивание моделей..."

# Проверка наличия моделей SAM2
if [ ! -f "models/sam2_hiera_large.pt" ]; then
    echo "   🔽 Скачивание SAM2 модели..."
    cd models
    # Используем gdown для скачивания с Google Drive или wget/curl для прямых ссылок
    # Модели SAM2 будут скачиваться автоматически при первом использовании
    echo "   ℹ️  SAM2 модели будут скачаны автоматически при первом запуске"
    cd ..
else
    echo "   ✅ SAM2 модель уже существует"
fi

# Проверка установки
echo "🧪 Проверка установки..."
python3 -c "
try:
    import torch
    print('✅ PyTorch:', torch.__version__)
    print('✅ CUDA доступна:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('✅ GPU:', torch.cuda.get_device_name(0))
except ImportError as e:
    print('❌ PyTorch не найден:', e)

try:
    import cv2
    print('✅ OpenCV:', cv2.__version__)
except ImportError as e:
    print('❌ OpenCV не найден:', e)

try:
    import transformers
    print('✅ Transformers:', transformers.__version__)
except ImportError as e:
    print('❌ Transformers не найден:', e)

try:
    from llava.model.builder import load_pretrained_model
    print('✅ LLaVA успешно импортирована')
except ImportError as e:
    print('❌ LLaVA не найдена:', e)

try:
    from sam2.build_sam import build_sam2
    print('✅ SAM2 успешно импортирован')
except ImportError as e:
    print('❌ SAM2 не найден:', e)
"

echo ""
echo "🎉 Установка завершена!"
echo ""
echo "📝 Для запуска анализа используйте:"
echo "   python3 llava_sam2_pipeline.py --image input/your_image.jpg --output output/"
echo ""
echo "📖 Пример:"
echo "   python3 llava_sam2_pipeline.py --image input/test_metal.jpg --output output/"
echo ""
echo "💡 Не забудьте активировать виртуальное окружение:"
echo "   source venv/bin/activate" 