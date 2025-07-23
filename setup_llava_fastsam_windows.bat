@echo off
REM Установка LLaVA + FastSAM + OpenCV Pipeline для Windows
setlocal enabledelayedexpansion

echo 🚀 Устанавливаем LLaVA + FastSAM + OpenCV Pipeline для Windows...

REM Проверка conda
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ Conda не найдена. Установите Anaconda или Miniconda сначала.
    echo 📥 Скачать можно тут: https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)

REM Проверка CUDA
echo 🔍 Проверяем CUDA...
where nvcc >nul 2>nul
if %errorlevel% equ 0 (
    echo ✅ CUDA найдена
    set "CUDA_AVAILABLE=1"
) else (
    echo ⚠️ CUDA не найдена, будет установлена CPU версия
    set "CUDA_AVAILABLE=0"
)

REM Удаление старого окружения если существует
set ENV_NAME=llava_fastsam
conda env list | findstr "!ENV_NAME!" >nul
if %errorlevel% equ 0 (
    echo ⚠️ Окружение !ENV_NAME! уже существует. Удаляем...
    conda env remove -n !ENV_NAME! -y
)

REM Создание нового окружения
echo 📦 Создаем conda окружение: !ENV_NAME!
conda create -n !ENV_NAME! python=3.10 -y

REM Активация окружения
echo 🔄 Активируем окружение...
call conda activate !ENV_NAME!

REM Обновление pip
echo 📦 Обновляем pip...
python -m pip install --upgrade pip

REM Установка PyTorch
echo 🔥 Устанавливаем PyTorch...
if !CUDA_AVAILABLE! equ 1 (
    echo 🎯 Устанавливаем PyTorch с CUDA поддержкой...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
) else (
    echo 💻 Устанавливаем CPU версию PyTorch...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
)

REM Установка основных зависимостей
echo 📦 Устанавливаем основные зависимости...
pip install -r requirements_llava_fastsam.txt

REM Клонирование и установка LLaVA
echo 📥 Клонируем LLaVA репозиторий...
if not exist "LLaVA" (
    git clone https://github.com/haotian-liu/LLaVA.git
)

cd LLaVA
echo 🔧 Устанавливаем LLaVA...
pip install -e .
cd ..

REM Создание директорий
echo 📁 Создаем рабочие директории...
if not exist "models" mkdir models
if not exist "output" mkdir output
if not exist "input" mkdir input

REM Создание тестового изображения
echo 🖼️ Создаем тестовое изображение...
python -c "
import cv2
import numpy as np
from pathlib import Path

# Создание тестового изображения с имитацией металлической поверхности и дефектов
img = np.ones((600, 800, 3), dtype=np.uint8) * 180  # Серый фон (металл)

# Добавляем текстуру металла
noise = np.random.normal(0, 10, (600, 800, 3))
img = np.clip(img + noise, 0, 255).astype(np.uint8)

# Добавляем 'царапины' (линии)
cv2.line(img, (100, 200), (300, 250), (80, 80, 80), 3)
cv2.line(img, (400, 150), (500, 400), (90, 90, 90), 2)

# Добавляем 'ржавчину' (коричневые пятна)
cv2.circle(img, (200, 400), 30, (50, 100, 150), -1)
cv2.circle(img, (600, 300), 25, (60, 110, 160), -1)

# Добавляем 'вмятины' (темные области)
cv2.ellipse(img, (500, 500), (40, 60), 45, 0, 360, (120, 120, 120), -1)

# Сохранение
Path('input').mkdir(exist_ok=True)
cv2.imwrite('input/test_metal.jpg', img)
print('✅ Тестовое изображение создано: input/test_metal.jpg')
"

REM Создание скрипта запуска для Windows
echo 📄 Создаем скрипт запуска...
(
echo @echo off
echo REM Скрипт для запуска pipeline
echo.
echo set ENV_NAME=llava_fastsam
echo.
echo REM Активация окружения
echo call conda activate %%ENV_NAME%%
echo.
echo REM Проверка аргументов
echo if "%%1"=="" ^(
echo     echo 🔍 Запускаем анализ тестового изображения...
echo     python llava_fastsam_pipeline.py --image input/test_metal.jpg --output output
echo ^) else ^(
echo     echo 🔍 Запускаем анализ изображения: %%1
echo     python llava_fastsam_pipeline.py --image "%%1" --output output
echo ^)
echo.
echo echo 📁 Результаты сохранены в директории: output/
echo echo 🖼️  Визуализация: output/*_result.jpg
echo echo 📄 JSON аннотации: output/*_analysis.json
echo pause
) > run_pipeline.bat

REM Создание инструкции
echo 📖 Создаем инструкцию по использованию...
(
echo # LLaVA + FastSAM + OpenCV Pipeline для Windows
echo.
echo ## 🚀 Запуск
echo.
echo ### Быстрый тест:
echo ```
echo run_pipeline.bat
echo ```
echo.
echo ### Анализ своего изображения:
echo ```
echo run_pipeline.bat "путь\к\изображению.jpg"
echo ```
echo.
echo ### Прямой запуск:
echo ```
echo # Активация окружения
echo conda activate llava_fastsam
echo.
echo # Запуск анализа
echo python llava_fastsam_pipeline.py --image input/test_metal.jpg --output output
echo ```
echo.
echo ## 📁 Структура результатов
echo.
echo После анализа в папке `output/` появятся:
echo - `{имя}_result.jpg` - Визуализация с цветными масками дефектов
echo - `{имя}_analysis.json` - JSON файл с координатами и метаданными
echo - `{имя}_mask_*.png` - Отдельные маски для каждого дефекта
echo.
echo ## 🔧 Pipeline этапы
echo.
echo 1. **LLaVA** - Определение материала ^("metal", "wood", "plastic", ...^)
echo 2. **FastSAM** - Быстрая сегментация всех объектов
echo 3. **OpenCV** - Постобработка и фильтрация по материалу
echo.
echo ## 🐛 Решение проблем
echo.
echo ### CUDA ошибки:
echo - Убедитесь что установлены драйверы NVIDIA
echo - Проверьте совместимость CUDA версии
echo.
echo ### Медленная работа:
echo - Уменьшите размер изображения
echo - Закройте другие GPU приложения
echo.
echo ## 📊 Производительность
echo.
echo - **С GPU**: ~1-3 секунды на изображение 1024x1024
echo - **CPU**: ~5-15 секунд на изображение 640x640
echo - **Память**: 6-12 GB ^(зависит от GPU^)
) > ИНСТРУКЦИЯ_WINDOWS.md

echo.
echo ✅ Установка завершена успешно!
echo.
echo 🎉 Следующие шаги:
echo 1. Запустите тест: run_pipeline.bat
echo 2. Анализируйте свои изображения: run_pipeline.bat "path\to\image.jpg"
echo.
echo 📖 Подробная инструкция: ИНСТРУКЦИЯ_WINDOWS.md
echo.
pause 