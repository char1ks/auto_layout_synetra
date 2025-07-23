# 🚀 Полная инструкция по установке LLaVA + FastSAM для Windows

## 📋 Требования

### Минимальные требования:
- **Windows 10/11** (64-bit)
- **RAM**: 8GB (16GB рекомендуется)
- **Свободное место**: 15GB
- **Python**: 3.10+ (устанавливается автоматически)

### Рекомендуемые требования:
- **GPU**: NVIDIA GTX 1060+ или RTX серии (6GB+ VRAM)
- **RAM**: 16GB+ 
- **SSD**: Для быстрой загрузки моделей

## 📥 Шаг 1: Установка Anaconda/Miniconda

### Вариант A: Miniconda (рекомендуется)
1. Скачайте Miniconda: https://docs.conda.io/en/latest/miniconda.html
2. Выберите **"Miniconda3 Windows 64-bit"**
3. Запустите установщик как **Администратор**
4. ✅ Отметьте **"Add Miniconda3 to my PATH environment variable"**
5. Завершите установку

### Вариант B: Anaconda (полная версия)
1. Скачайте Anaconda: https://www.anaconda.com/products/distribution
2. Установите как **Администратор**

## 🔥 Шаг 2: Установка CUDA (для GPU)

### Если у вас NVIDIA GPU:

1. **Проверьте совместимость GPU:**
   ```cmd
   nvidia-smi
   ```

2. **Скачайте CUDA 12.1:**
   - Перейдите: https://developer.nvidia.com/cuda-downloads
   - Выберите: Windows → x86_64 → 11 → exe (local)

3. **Установите CUDA:**
   - Запустите установщик как **Администратор**
   - Выберите **"Express Installation"**

4. **Проверьте установку:**
   ```cmd
   nvcc --version
   ```

### Если нет NVIDIA GPU:
- Пропустите этот шаг
- Будет использоваться CPU версия

## 📂 Шаг 3: Подготовка проекта

1. **Скачайте проект:**
   ```cmd
   git clone <ваш-репозиторий>
   cd Task_I-Find-models
   ```

2. **Или создайте папку и скопируйте файлы:**
   ```cmd
   mkdir LLaVA-Pipeline
   cd LLaVA-Pipeline
   ```

## ⚡ Шаг 4: Автоматическая установка

### Вариант A: BAT скрипт (простой)
```cmd
setup_llava_fastsam_windows.bat
```

### Вариант B: PowerShell (современный)
```powershell
# Откройте PowerShell как Администратор
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
.\setup_llava_fastsam_windows.ps1
```

### Что происходит при установке:
1. 🔍 Проверка conda и CUDA
2. 📦 Создание окружения Python 3.10
3. 🔥 Установка PyTorch (GPU или CPU версия)
4. 📥 Скачивание и установка LLaVA
5. ⚡ Установка FastSAM через ultralytics
6. 🖼️ Создание тестового изображения
7. 📄 Создание скриптов запуска

## 🎯 Шаг 5: Первый запуск

### Быстрый тест:
```cmd
run_pipeline.bat
```

### Или PowerShell:
```powershell
.\run_pipeline.ps1
```

### Анализ своего изображения:
```cmd
run_pipeline.bat "C:\path\to\your\image.jpg"
```

## 📊 Ожидаемые результаты

### Успешный запуск покажет:
```
🔥 GPU найдено: NVIDIA GeForce RTX 3080 (10.0GB)
🎯 Устройство: cuda
🖥️ Операционная система: Windows
✅ FastSAM через ultralytics найден
✅ LLaVA модель загружена на cuda
✅ FastSAM модель загружена успешно

🔍 Анализируем изображение: 001.png
🔬 Этап 1: Определение материала...
   ✅ Материал: metal (уверенность: 0.95)
   ⏱️ Время: 2.1 сек
🎯 Этап 2: Сегментация дефектов...
   ✅ Найдено сегментов: 5
   ⏱️ Время: 1.8 сек
🎨 Этап 3: Постобработка...
   ✅ Финальных дефектов: 3
   ⏱️ Время: 0.3 сек

🎉 Анализ завершен за 4.2 секунды
```

### Файлы результатов:
- `output/001_result.jpg` - Визуализация с масками
- `output/001_analysis.json` - JSON с координатами
- `output/001_mask_1.png` - Маски дефектов

## 🐛 Решение проблем

### ❌ "conda не найдена"
**Решение:**
1. Перезапустите командную строку
2. Проверьте PATH: `echo %PATH%`
3. Переустановите conda с опцией PATH

### ❌ "CUDA не найдена"
**Решение:**
1. Установите драйверы NVIDIA
2. Установите CUDA 12.1
3. Перезагрузите компьютер

### ❌ "Out of memory" ошибки
**Решение:**
1. Закройте другие программы
2. Уменьшите размер изображения
3. Используйте CPU версию:
   ```python
   DEVICE = 'cpu'  # В llava_fastsam_pipeline.py
   ```

### ❌ LLaVA медленно загружается
**Причина:** Скачивание модели 7GB  
**Решение:** Дождитесь первого скачивания (один раз)

### ❌ "Module not found" ошибки
**Решение:**
```cmd
conda activate llava_fastsam
pip install -r requirements_llava_fastsam.txt
```

### ❌ FastSAM ошибки
**Решение:**
```cmd
pip install ultralytics --upgrade
```

## ⚙️ Настройка производительности

### Для GPU оптимизации:
```python
# В llava_fastsam_pipeline.py найдите:
max_tokens = 512 if DEVICE == 'cuda' else 128  # Увеличьте если много GPU памяти
imgsz = 1280 if DEVICE == 'cuda' else 640     # Увеличьте размер обработки
```

### Для экономии памяти:
```python
torch_dtype=torch.float16  # Вместо float32 для GPU
```

## 📈 Производительность

### RTX 3080 (10GB):
- **LLaVA**: ~2-3 сек
- **FastSAM**: ~1-2 сек  
- **Общее время**: ~4-6 сек

### RTX 4090 (24GB):
- **LLaVA**: ~1-2 сек
- **FastSAM**: ~0.5-1 сек
- **Общее время**: ~2-4 сек

### CPU (Intel i7):
- **LLaVA**: ~8-15 сек
- **FastSAM**: ~3-5 сек
- **Общее время**: ~12-25 сек

## 🔄 Обновление

### Обновить все компоненты:
```cmd
conda activate llava_fastsam
pip install --upgrade torch torchvision ultralytics transformers
```

### Обновить только FastSAM:
```cmd
pip install ultralytics --upgrade
```

### Обновить только LLaVA:
```cmd
cd LLaVA
git pull
pip install -e . --upgrade
```

## 🆘 Поддержка

### Логи ошибок:
```cmd
python llava_fastsam_pipeline.py --image image.jpg > log.txt 2>&1
```

### Проверка GPU:
```python
import torch
print(f"CUDA доступна: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Память: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
```

### Полезные команды:
```cmd
# Информация об окружении
conda info

# Список пакетов
conda list

# Проверка GPU
nvidia-smi

# Освобождение GPU памяти
taskkill /f /im python.exe
```

## ✅ Готово!

Теперь у вас полностью настроенная система для автоматической разметки дефектов с использованием LLaVA и FastSAM! 