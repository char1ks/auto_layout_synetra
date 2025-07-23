# LLaVA + FastSAM + OpenCV Pipeline Setup для Windows (PowerShell)
# Запустите из PowerShell как Администратор: Set-ExecutionPolicy RemoteSigned -Scope CurrentUser

Write-Host "🚀 Устанавливаем LLaVA + FastSAM + OpenCV Pipeline для Windows..." -ForegroundColor Green

# Проверка conda
try {
    $condaVersion = conda --version
    Write-Host "✅ Conda найдена: $condaVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Conda не найдена. Установите Anaconda или Miniconda сначала." -ForegroundColor Red
    Write-Host "📥 Скачать можно тут: https://docs.conda.io/en/latest/miniconda.html" -ForegroundColor Yellow
    Read-Host "Нажмите Enter для выхода"
    exit 1
}

# Проверка CUDA
Write-Host "🔍 Проверяем CUDA..." -ForegroundColor Cyan
try {
    $nvccVersion = nvcc --version
    Write-Host "✅ CUDA найдена" -ForegroundColor Green
    $cudaAvailable = $true
} catch {
    Write-Host "⚠️ CUDA не найдена, будет установлена CPU версия" -ForegroundColor Yellow
    $cudaAvailable = $false
}

# Проверка GPU
if ($cudaAvailable) {
    try {
        $gpuInfo = nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
        Write-Host "🔥 GPU найдена: $gpuInfo" -ForegroundColor Green
    } catch {
        Write-Host "⚠️ nvidia-smi не доступна" -ForegroundColor Yellow
    }
}

# Переменные
$envName = "llava_fastsam"

# Проверка существующего окружения
$existingEnv = conda env list | Select-String $envName
if ($existingEnv) {
    Write-Host "⚠️ Окружение $envName уже существует. Удаляем..." -ForegroundColor Yellow
    conda env remove -n $envName -y
}

# Создание нового окружения
Write-Host "📦 Создаем conda окружение: $envName" -ForegroundColor Cyan
conda create -n $envName python=3.10 -y

# Активация окружения
Write-Host "🔄 Активируем окружение..." -ForegroundColor Cyan
& conda activate $envName

# Обновление pip
Write-Host "📦 Обновляем pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip

# Установка PyTorch
Write-Host "🔥 Устанавливаем PyTorch..." -ForegroundColor Cyan
if ($cudaAvailable) {
    Write-Host "🎯 Устанавливаем PyTorch с CUDA поддержкой..." -ForegroundColor Green
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
} else {
    Write-Host "💻 Устанавливаем CPU версию PyTorch..." -ForegroundColor Yellow
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
}

# Установка основных зависимостей
Write-Host "📦 Устанавливаем основные зависимости..." -ForegroundColor Cyan
pip install -r requirements_llava_fastsam.txt

# Клонирование и установка LLaVA
Write-Host "📥 Клонируем LLaVA репозиторий..." -ForegroundColor Cyan
if (-not (Test-Path "LLaVA")) {
    git clone https://github.com/haotian-liu/LLaVA.git
}

Set-Location LLaVA
Write-Host "🔧 Устанавливаем LLaVA..." -ForegroundColor Cyan
pip install -e .
Set-Location ..

# Создание директорий
Write-Host "📁 Создаем рабочие директории..." -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path "models", "output", "input"

# Создание тестового изображения
Write-Host "🖼️ Создаем тестовое изображение..." -ForegroundColor Cyan
$pythonScript = @"
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
"@

python -c $pythonScript

# Создание PowerShell скрипта запуска
Write-Host "📄 Создаем PowerShell скрипт запуска..." -ForegroundColor Cyan
$runScript = @"
# PowerShell скрипт для запуска pipeline
param([string]`$ImagePath = "input/test_metal.jpg")

`$envName = "llava_fastsam"

Write-Host "🔄 Активируем окружение `$envName..." -ForegroundColor Cyan
& conda activate `$envName

if (`$ImagePath -eq "input/test_metal.jpg") {
    Write-Host "🔍 Запускаем анализ тестового изображения..." -ForegroundColor Green
} else {
    Write-Host "🔍 Запускаем анализ изображения: `$ImagePath" -ForegroundColor Green
}

python llava_fastsam_pipeline.py --image "`$ImagePath" --output output

Write-Host "📁 Результаты сохранены в директории: output/" -ForegroundColor Green  
Write-Host "🖼️  Визуализация: output/*_result.jpg" -ForegroundColor Cyan
Write-Host "📄 JSON аннотации: output/*_analysis.json" -ForegroundColor Cyan
"@

$runScript | Out-File -FilePath "run_pipeline.ps1" -Encoding UTF8

Write-Host "✅ Установка завершена успешно!" -ForegroundColor Green
Write-Host ""
Write-Host "🎉 Следующие шаги:" -ForegroundColor Yellow
Write-Host "1. Запустите тест: .\run_pipeline.ps1" -ForegroundColor White
Write-Host "2. Анализируйте свои изображения: .\run_pipeline.ps1 -ImagePath 'path\to\image.jpg'" -ForegroundColor White
Write-Host ""
Write-Host "📖 Подробная инструкция: ИНСТРУКЦИЯ_WINDOWS.md" -ForegroundColor Cyan

Read-Host "Нажмите Enter для завершения" 