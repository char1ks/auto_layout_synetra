#!/usr/bin/env python3
"""
Быстрый запуск валидации на MVTec AD

Установка зависимостей:
pip install -r requirements_llava_sam2.txt

Использование:
python run_validation.py quick    # Быстрая валидация (10 образцов)
python run_validation.py          # Полная валидация
"""

from validate_mvtec import MVTecValidator

def run_quick_validation():
    """Быстрая валидация на 10 образцах"""
    
    print("🚀 БЫСТРАЯ ВАЛИДАЦИЯ (10 образцов)")
    print("=" * 40)
    
    validator = MVTecValidator()
    
    # Скачиваем датасет
    dataset_path = validator.download_dataset()
    if not dataset_path:
        print("❌ Не удалось скачать датасет")
        return
    
    # Инициализируем детектор  
    validator.setup_detector("standard")
    
    # Запускаем валидацию на 10 образцах
    results = validator.run_validation(
        output_dir="./quick_validation", 
        max_samples=10
    )
    
    if results:
        overall = results["summary"]["overall"]
        print(f"\n🎯 БЫСТРЫЕ РЕЗУЛЬТАТЫ:")
        print(f"   Dice: {overall['mean_dice']:.3f}")
        print(f"   IoU: {overall['mean_iou']:.3f}")
        print(f"   F1: {overall['mean_f1']:.3f}")

def run_full_validation():
    """Полная валидация на всех образцах"""
    
    print("🔬 ПОЛНАЯ ВАЛИДАЦИЯ")
    print("=" * 40)
    
    validator = MVTecValidator()
    
    # Скачиваем датасет
    dataset_path = validator.download_dataset()
    if not dataset_path:
        print("❌ Не удалось скачать датасет")
        return
    
    # Инициализируем детектор
    validator.setup_detector("standard")
    
    # Запускаем полную валидацию
    results = validator.run_validation(
        output_dir="./full_validation"
    )
    
    if results:
        overall = results["summary"]["overall"]
        print(f"\n🏆 ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ:")
        print(f"   Dice Score: {overall['mean_dice']:.3f}")
        print(f"   IoU: {overall['mean_iou']:.3f}")
        print(f"   F1-Score: {overall['mean_f1']:.3f}")
        print(f"   Precision: {overall['mean_precision']:.3f}")
        print(f"   Recall: {overall['mean_recall']:.3f}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        run_quick_validation()
    else:
        run_full_validation() 