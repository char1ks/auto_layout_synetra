#!/bin/bash

echo "🚀 Запуск LLaVA + SAM2 Pipeline для анализа дефектов"
echo "===================================================="

# Проверка наличия файла
if [ $# -eq 0 ]; then
    echo "❌ Ошибка: Не указан путь к изображению"
    echo ""
    echo "Использование:"
    echo "  $0 <путь_к_изображению> [директория_вывода]"
    echo ""
    echo "Примеры:"
    echo "  $0 input/test_metal.jpg"
    echo "  $0 input/test_metal.jpg output/results"
    echo "  $0 /path/to/your/image.jpg"
    exit 1
fi

IMAGE_PATH="$1"
OUTPUT_DIR="${2:-output}"

# Проверка существования файла
if [ ! -f "$IMAGE_PATH" ]; then
    echo "❌ Ошибка: Файл '$IMAGE_PATH' не найден"
    exit 1
fi

# Активация виртуального окружения если существует
if [ -d "venv" ]; then
    echo "🔌 Активация виртуального окружения..."
    source venv/bin/activate
fi

# Создание выходной директории
mkdir -p "$OUTPUT_DIR"

echo "📁 Входное изображение: $IMAGE_PATH"
echo "📁 Выходная директория: $OUTPUT_DIR"
echo ""

# Запуск анализа
echo "🔍 Запускаем анализ..."
python3 llava_sam2_pipeline.py --image "$IMAGE_PATH" --output "$OUTPUT_DIR"

# Проверка успешности выполнения
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Анализ завершен успешно!"
    echo "📁 Результаты сохранены в: $OUTPUT_DIR"
    
    # Показать созданные файлы
    echo ""
    echo "📄 Созданные файлы:"
    ls -la "$OUTPUT_DIR"/*$(basename "$IMAGE_PATH" | cut -d. -f1)*
else
    echo ""
    echo "❌ Ошибка при выполнении анализа"
    exit 1
fi 