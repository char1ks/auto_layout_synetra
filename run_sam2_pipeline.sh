#!/bin/bash

echo "🚀 Запуск LLaVA + SAM2 Pipeline для анализа дефектов"
echo "===================================================="

# Проверка наличия файла
if [ $# -eq 0 ]; then
    echo "❌ Ошибка: Не указан путь к изображению"
    echo ""
    echo "Использование:"
    echo "  $0 <путь_к_изображению> [директория_вывода] [модель]"
    echo ""
    echo "Модели LLaVA:"
    echo "  detailed  - LLaVA-1.5-13B (максимальная детализация, медленнее)"
    echo "  standard  - LLaVA-1.5-7B (стандартная скорость)"
    echo "  latest    - LLaVA-1.6-13B (новейшая архитектура)"
    echo "  onevision - LLaVA-OneVision-7B (специально для детального анализа)"
    echo ""
    echo "Примеры:"
    echo "  $0 input/test_metal.jpg                    # detailed модель"
    echo "  $0 input/test_metal.jpg output             # detailed модель"
    echo "  $0 input/test_metal.jpg output detailed    # detailed модель"
    echo "  $0 input/test_metal.jpg output standard    # быстрая 7B модель"
    echo "  $0 input/cables.jpg output onevision       # для кабелей/проводов"
    exit 1
fi

IMAGE_PATH="$1"
OUTPUT_DIR="${2:-output}"
MODEL_TYPE="${3:-detailed}"

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
echo "🤖 Модель LLaVA: $MODEL_TYPE"
echo ""

# Информация о выбранной модели
case $MODEL_TYPE in
    "detailed")
        echo "🔬 Используем LLaVA-1.5-13B для максимально детального анализа"
        echo "   ⚡ Лучше всего для поиска мелких дефектов и проводов"
        echo "   💾 Требует: ~26GB RAM или 12GB+ VRAM"
        echo "   ⏱️ Время: ~1.5-2x медленнее стандартной модели"
        ;;
    "standard")
        echo "⚡ Используем LLaVA-1.5-7B для быстрого анализа"
        echo "   🔬 Подходит для общих дефектов"
        echo "   💾 Требует: ~14GB RAM или 7GB+ VRAM"
        echo "   ⏱️ Время: базовая скорость"
        ;;
    "latest")
        echo "🆕 Используем LLaVA-1.6-13B новейшую версию"
        echo "   🔬 Улучшенная архитектура для детального анализа"
        echo "   💾 Требует: ~26GB RAM или 12GB+ VRAM"
        echo "   ⏱️ Время: ~1.5-2x медленнее, но качественнее"
        ;;
    "onevision")
        echo "👁️ Используем LLaVA-OneVision для сверхдетального анализа"
        echo "   🔍 Специально для мелких деталей и проводов"
        echo "   💾 Требует: ~14GB RAM или 8GB+ VRAM"
        echo "   ⏱️ Время: сравнимо со стандартной, но точнее"
        ;;
esac

# Запуск анализа
echo ""
echo "🔍 Запускаем анализ..."
python3 llava_sam2_pipeline.py --image "$IMAGE_PATH" --output "$OUTPUT_DIR" --model "$MODEL_TYPE"

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