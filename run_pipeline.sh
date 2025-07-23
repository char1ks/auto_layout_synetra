#!/bin/bash
# Скрипт для запуска pipeline

ENV_NAME="llava_fastsam"

# Активация окружения
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

# Проверка аргументов
if [ "$#" -eq 0 ]; then
    echo "🔍 Запускаем анализ тестового изображения..."
    python llava_fastsam_pipeline.py --image input/test_metal.jpg --output output
else
    echo "🔍 Запускаем анализ изображения: $1"
    python llava_fastsam_pipeline.py --image "$1" --output output
fi

echo "📁 Результаты сохранены в директории: output/"
echo "🖼️  Визуализация: output/*_result.jpg"
echo "📄 JSON аннотации: output/*_analysis.json"
