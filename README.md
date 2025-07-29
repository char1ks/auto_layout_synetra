
## ⚡ БЫСТРАЯ УСТАНОВКА

### 🐧 Linux / macOS (рекомендуется):

```bash
# 1. Клонирование проекта
git clone https://github.com/your-repo/hybrid-searchdet-pipeline.git
cd hybrid-searchdet-pipeline

# 2. Создание окружения
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# 3. Установка PyTorch (GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 4. Установка основных зависимостей
pip install -r requirements_llava_sam2.txt

# 5. Установка SearchDet зависимостей
pip install -r searchdet-main/requirements.txt

# 6. Установка LLaVA
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA && pip install -e . && cd ..

# 7. Установка SAM2
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2 && pip install -e . && cd ..

# 8. Создание папок
mkdir -p models input output examples/positive examples/negative

# ✅ Готово! Теперь можно запускать анализ
```

### 🖥️ Windows:

```cmd
# 1. Клонирование проекта
git clone https://github.com/your-repo/hybrid-searchdet-pipeline.git
cd hybrid-searchdet-pipeline

# 2. Создание окружения  
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip

# 3. Установка PyTorch (GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 4-8. Повторить шаги как для Linux
```


## 🎯 ЗАПУСК АНАЛИЗА
=
```bash
python3 hybrid_searchdet_pipeline.py \
  --image image.png \
  --positive examples/negative \
  --negative examples/positive \
  --output output/ \
  --ground-truth /content/files/image.png
```
