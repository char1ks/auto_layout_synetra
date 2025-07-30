#!/usr/bin/env python3
"""
Валидация гибридного детектора на датасете MVTec AD
Специально для категории "wood" с расчетом метрик качества
"""

import os
import sys
import cv2
import numpy as np
import json
from pathlib import Path
from PIL import Image
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Импорт нашего детектора
from hybrid_searchdet_pipeline import HybridDefectDetector

# Для скачивания датасета
try:
    import kagglehub
    KAGGLEHUB_AVAILABLE = True
except ImportError:
    print("⚠️ kagglehub не установлен. Установите: pip install kagglehub")
    KAGGLEHUB_AVAILABLE = False


class MVTecValidator:
    """Валидатор для датасета MVTec AD"""
    
    def __init__(self, dataset_path=None):
        self.dataset_path = Path(dataset_path) if dataset_path else None
        self.detector = None
        self.results = {
            "metrics": [],
            "detailed_results": {},
            "summary": {}
        }
    
    def download_dataset(self):
        """Скачивание датасета MVTec AD"""
        if not KAGGLEHUB_AVAILABLE:
            raise ImportError("kagglehub недоступен")
        
        print("📥 Скачиваем датасет MVTec AD...")
        try:
            path = kagglehub.dataset_download("ipythonx/mvtec-ad")
            self.dataset_path = Path(path)
            
            print(f"✅ Датасет скачан в: {self.dataset_path}")
            
            # Проверяем структуру
            wood_path = self.dataset_path / "wood"
            if wood_path.exists():
                print(f"✅ Папка wood найдена: {wood_path}")
                print(f"📂 Содержимое wood: {list(wood_path.iterdir())}")
            else:
                print(f"❌ Папка wood не найдена в {self.dataset_path}")
                print(f"📂 Доступные папки: {list(self.dataset_path.iterdir())}")
            
            return self.dataset_path
        except Exception as e:
            print(f"❌ Ошибка скачивания: {e}")
            return None
    
    def setup_detector(self, model_type="standard"):
        """Инициализация детектора"""
        print("🚀 Инициализация детектора...")
        self.detector = HybridDefectDetector(model_type=model_type)
        print("✅ Детектор готов")
    
    def get_wood_test_samples(self):
        """Получение списка тестовых изображений дерева"""
        wood_test_path = self.dataset_path / "wood" / "test"
        wood_gt_path = self.dataset_path / "wood" / "ground_truth"
        
        print(f"🔍 Поиск образцов в: {wood_test_path}")
        print(f"🔍 Ground truth в: {wood_gt_path}")
        
        # Проверяем существование папок
        if not wood_test_path.exists():
            print(f"❌ Папка test не найдена: {wood_test_path}")
            return []
        
        if not wood_gt_path.exists():
            print(f"❌ Папка ground_truth не найдена: {wood_gt_path}")
            return []
        
        samples = []
        
        # Проходим по всем подпапкам с дефектами
        print(f"📂 Содержимое test: {list(wood_test_path.iterdir())}")
        
        for defect_folder in wood_test_path.iterdir():
            if defect_folder.is_dir():
                defect_name = defect_folder.name
                
                # Пропускаем папку "good" - там нет дефектов
                if defect_name == "good":
                    print(f"   ⚪ Пропускаем '{defect_name}' - это хорошие образцы без дефектов")
                    continue
                
                print(f"   🔍 Найден тип дефекта: {defect_name}")
                
                gt_defect_path = wood_gt_path / defect_name
                
                if gt_defect_path.exists():
                    # Собираем пары изображение-маска
                    img_files = list(defect_folder.glob("*.png"))
                    gt_files = list(gt_defect_path.glob("*.png"))
                    
                    print(f"      📷 Изображений в test: {len(img_files)}")
                    print(f"      📷 Масок в ground_truth: {len(gt_files)}")
                    
                    if len(gt_files) > 0:
                        print(f"      📂 Примеры GT файлов: {[f.name for f in gt_files[:3]]}")
                        print(f"      📂 Примеры test файлов: {[f.name for f in img_files[:3]]}")
                    
                    # Ищем совпадающие пары файлов с учетом суффикса _mask
                    matched_pairs = 0
                    for img_file in img_files:
                        img_stem = img_file.stem  # 001
                        found_match = False
                        
                        # Вариант 1: Точное совпадение (001.png -> 001.png)
                        gt_file = gt_defect_path / img_file.name
                        if gt_file.exists():
                            samples.append({
                                "image_path": img_file,
                                "gt_path": gt_file,
                                "defect_type": defect_name,
                                "sample_id": img_stem
                            })
                            matched_pairs += 1
                            found_match = True
                        
                        # Вариант 2: С суффиксом _mask (001.png -> 001_mask.png)
                        if not found_match:
                            gt_mask_file = gt_defect_path / f"{img_stem}_mask.png"
                            if gt_mask_file.exists():
                                samples.append({
                                    "image_path": img_file,
                                    "gt_path": gt_mask_file,
                                    "defect_type": defect_name,
                                    "sample_id": img_stem
                                })
                                matched_pairs += 1
                                found_match = True
                        
                        # Вариант 3: Альтернативные расширения с _mask
                        if not found_match:
                            alt_extensions = ['.jpg', '.jpeg', '.bmp']
                            for ext in alt_extensions:
                                alt_gt_file = gt_defect_path / f"{img_stem}_mask{ext}"
                                if alt_gt_file.exists():
                                    samples.append({
                                        "image_path": img_file,
                                        "gt_path": alt_gt_file,
                                        "defect_type": defect_name,
                                        "sample_id": img_stem
                                    })
                                    matched_pairs += 1
                                    found_match = True
                                    print(f"      ✅ Найден GT с суффиксом _mask{ext}: {alt_gt_file.name}")
                                    break
                        
                        # Диагностика для первого файла если не найден
                        if not found_match and img_file == img_files[0]:
                            print(f"      ⚠️ Не найден GT для: {img_file.name}")
                            print(f"         Проверяли: {img_file.name}, {img_stem}_mask.png")
                            print(f"         Доступно: {[f.name for f in gt_files[:5]]}")
                    
                    print(f"      ✅ Найдено совпадающих пар: {matched_pairs}/{len(img_files)}")
                else:
                    print(f"      ❌ Ground truth папка не найдена: {gt_defect_path}")
        
        print(f"📊 Найдено {len(samples)} тестовых образцов дерева")
        return samples
    
    def calculate_metrics(self, predicted_mask, gt_mask):
        """Расчет метрик качества сегментации"""
        
        # Приводим к бинарному виду
        if len(predicted_mask.shape) == 3:
            predicted_mask = cv2.cvtColor(predicted_mask, cv2.COLOR_BGR2GRAY)
        if len(gt_mask.shape) == 3:
            gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_BGR2GRAY)
        
        # Бинаризация
        pred_binary = (predicted_mask > 127).astype(np.uint8)
        gt_binary = (gt_mask > 127).astype(np.uint8)
        
        # Приводим к одному размеру
        if pred_binary.shape != gt_binary.shape:
            pred_binary = cv2.resize(pred_binary, (gt_binary.shape[1], gt_binary.shape[0]))
        
        # Вычисления
        intersection = np.logical_and(pred_binary, gt_binary).sum()
        union = np.logical_or(pred_binary, gt_binary).sum()
        
        pred_area = pred_binary.sum()
        gt_area = gt_binary.sum()
        
        # Метрики
        iou = intersection / union if union > 0 else 0.0
        dice = (2 * intersection) / (pred_area + gt_area) if (pred_area + gt_area) > 0 else 0.0
        precision = intersection / pred_area if pred_area > 0 else 0.0
        recall = intersection / gt_area if gt_area > 0 else 0.0
        
        # F1-score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Pixel Accuracy
        total_pixels = gt_binary.size
        correct_pixels = np.sum(pred_binary == gt_binary)
        pixel_accuracy = correct_pixels / total_pixels
        
        return {
            "iou": iou,
            "dice": dice,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "pixel_accuracy": pixel_accuracy,
            "pred_area": int(pred_area),
            "gt_area": int(gt_area),
            "intersection": int(intersection)
        }
    
    def create_positive_negative_examples(self, defect_type):
        """Создание примеров для SearchDet на основе типа дефекта"""
        
        # Пути к обучающим данным
        wood_train_path = self.dataset_path / "wood" / "train" / "good"
        wood_test_path = self.dataset_path / "wood" / "test" / defect_type
        
        # Создаем временные папки
        temp_dir = Path("./temp_examples")
        positive_dir = temp_dir / "positive"
        negative_dir = temp_dir / "negative"
        
        positive_dir.mkdir(parents=True, exist_ok=True)
        negative_dir.mkdir(parents=True, exist_ok=True)
        
        # Положительные примеры - хорошие образцы дерева (6 штук)
        good_samples = list(wood_train_path.glob("*.png"))
        if len(good_samples) >= 6:
            # Берем равномерно распределенные образцы
            step = len(good_samples) // 6
            selected_good = [good_samples[i * step] for i in range(6)]
        else:
            selected_good = good_samples  # Если мало, берем все
        
        for i, sample in enumerate(selected_good):
            dst = positive_dir / f"good_{i:02d}.png"
            img = cv2.imread(str(sample))
            if img is not None:
                cv2.imwrite(str(dst), img)
        
        print(f"   ✅ Положительные примеры: {len(selected_good)} хороших образцов")
        
        # Отрицательные примеры - образцы ТЕКУЩЕГО дефекта (5 штук)
        if wood_test_path.exists():
            defect_samples = list(wood_test_path.glob("*.png"))
            if len(defect_samples) >= 5:
                # Берем равномерно распределенные дефектные образцы
                step = len(defect_samples) // 5
                selected_defects = [defect_samples[i * step] for i in range(5)]
            else:
                selected_defects = defect_samples  # Если мало, берем все
            
            for i, sample in enumerate(selected_defects):
                dst = negative_dir / f"defect_{defect_type}_{i:02d}.png"
                img = cv2.imread(str(sample))
                if img is not None:
                    cv2.imwrite(str(dst), img)
            
            print(f"   ✅ Отрицательные примеры: {len(selected_defects)} образцов дефекта '{defect_type}'")
        else:
            print(f"   ⚠️ Папка с дефектом '{defect_type}' не найдена: {wood_test_path}")
        
        return str(positive_dir), str(negative_dir)
    
    def process_sample(self, sample, output_dir):
        """Обработка одного образца"""
        
        image_path = sample["image_path"]
        gt_path = sample["gt_path"]
        defect_type = sample["defect_type"]
        sample_id = sample["sample_id"]
        
        try:
            # Создаем примеры для SearchDet
            positive_dir, negative_dir = self.create_positive_negative_examples(defect_type)
            
            # Запускаем анализ
            results = self.detector.analyze_with_examples(
                str(image_path), 
                positive_dir, 
                negative_dir, 
                str(output_dir),
                str(gt_path)  # Ground truth для сравнения
            )
            
            # Загружаем ground truth
            gt_mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
            
            # Создаем предсказанную маску из результатов SearchDet
            predicted_mask = self.create_prediction_mask(results, image_path)
            
            # Вычисляем метрики
            metrics = self.calculate_metrics(predicted_mask, gt_mask)
            
            # Добавляем метаданные
            metrics.update({
                "sample_id": sample_id,
                "defect_type": defect_type,
                "image_path": str(image_path),
                "searchdet_detections": len(results.get("stages", {}).get("searchdet_analysis", {}).get("result", {}).get("missing_elements", [])),
                "processing_time": sum(stage.get("duration", 0) for stage in results.get("stages", {}).values())
            })
            
            # Очищаем временные файлы
            self.cleanup_temp_files()
            
            return metrics, results
            
        except Exception as e:
            print(f"❌ Ошибка обработки {sample_id}: {e}")
            return None, None
    
    def create_prediction_mask(self, results, image_path):
        """Создание предсказанной маски из результатов анализа"""
        
        # Загружаем оригинальное изображение для размера
        img = cv2.imread(str(image_path))
        h, w = img.shape[:2]
        
        # Создаем пустую маску
        prediction_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Извлекаем SearchDet детекции
        searchdet_elements = results.get("stages", {}).get("searchdet_analysis", {}).get("result", {}).get("missing_elements", [])
        
        for element in searchdet_elements:
            bbox = element.get("bbox", [])
            if len(bbox) == 4:
                # Денормализуем координаты
                x_min = int(bbox[0] * w)
                y_min = int(bbox[1] * h)
                x_max = int(bbox[2] * w)
                y_max = int(bbox[3] * h)
                
                # Заполняем область дефекта
                prediction_mask[y_min:y_max, x_min:x_max] = 255
        
        return prediction_mask
    
    def cleanup_temp_files(self):
        """Очистка временных файлов"""
        import shutil
        temp_dir = Path("./temp_examples")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
    
    def run_validation(self, output_dir="./mvtec_validation", max_samples=None):
        """Запуск полной валидации"""
        
        print("🔬 ВАЛИДАЦИЯ НА ДАТАСЕТЕ MVTec AD")
        print("=" * 50)
        
        # Создаем выходную папку
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Получаем образцы
        samples = self.get_wood_test_samples()
        
        if max_samples:
            samples = samples[:max_samples]
            print(f"📝 Ограничиваем до {max_samples} образцов")
        
        # Группируем по типам дефектов
        defect_types = {}
        for sample in samples:
            defect_type = sample["defect_type"]
            if defect_type not in defect_types:
                defect_types[defect_type] = []
            defect_types[defect_type].append(sample)
        
        print(f"📊 Типы дефектов: {list(defect_types.keys())}")
        
        # Обрабатываем образцы
        all_metrics = []
        detailed_results = {}
        
        start_time = time.time()
        
        with tqdm(total=len(samples), desc="Валидация") as pbar:
            for sample in samples:
                metrics, results = self.process_sample(sample, output_path)
                
                if metrics:
                    all_metrics.append(metrics)
                    detailed_results[sample["sample_id"]] = {
                        "metrics": metrics,
                        "defect_type": sample["defect_type"]
                    }
                
                pbar.update(1)
                pbar.set_postfix({
                    'Defect': sample["defect_type"][:8],
                    'ID': sample["sample_id"]
                })
        
        total_time = time.time() - start_time
        
        # Сохраняем результаты
        self.results = {
            "metrics": all_metrics,
            "detailed_results": detailed_results,
            "summary": self.calculate_summary_metrics(all_metrics, defect_types),
            "validation_info": {
                "total_samples": len(samples),
                "processed_samples": len(all_metrics),
                "total_time": total_time,
                "avg_time_per_sample": total_time / len(all_metrics) if all_metrics else 0
            }
        }
        
        # Сохраняем в файл
        results_file = output_path / "validation_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        # Создаем отчет
        self.create_validation_report(output_path)
        
        print(f"\n🎉 Валидация завершена за {total_time:.2f} секунд")
        print(f"📁 Результаты сохранены в: {output_path}")
        
        return self.results
    
    def calculate_summary_metrics(self, all_metrics, defect_types):
        """Расчет сводных метрик"""
        
        if not all_metrics:
            return {}
        
        df = pd.DataFrame(all_metrics)
        
        # Общие метрики
        overall = {
            "mean_dice": df["dice"].mean(),
            "mean_iou": df["iou"].mean(),
            "mean_f1": df["f1"].mean(),
            "mean_precision": df["precision"].mean(),
            "mean_recall": df["recall"].mean(),
            "mean_pixel_accuracy": df["pixel_accuracy"].mean(),
            "std_dice": df["dice"].std(),
            "std_iou": df["iou"].std()
        }
        
        # Метрики по типам дефектов
        by_defect_type = {}
        for defect_type in defect_types.keys():
            defect_metrics = df[df["defect_type"] == defect_type]
            if len(defect_metrics) > 0:
                by_defect_type[defect_type] = {
                    "count": len(defect_metrics),
                    "mean_dice": defect_metrics["dice"].mean(),
                    "mean_iou": defect_metrics["iou"].mean(),
                    "mean_f1": defect_metrics["f1"].mean(),
                    "mean_precision": defect_metrics["precision"].mean(),
                    "mean_recall": defect_metrics["recall"].mean()
                }
        
        return {
            "overall": overall,
            "by_defect_type": by_defect_type
        }
    
    def create_validation_report(self, output_path):
        """Создание детального отчета валидации"""
        
        summary = self.results["summary"]
        
        # Текстовый отчет
        report_file = output_path / "validation_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("🔬 ОТЧЕТ ПО ВАЛИДАЦИИ MVTec AD (WOOD)\n")
            f.write("=" * 50 + "\n\n")
            
            # Проверяем наличие результатов
            if not summary or "overall" not in summary:
                f.write("❌ ВАЛИДАЦИЯ НЕ ВЫПОЛНЕНА\n")
                f.write("   Причина: Не найдено образцов для валидации\n")
                f.write("   Проверьте:\n")
                f.write("   - Правильность пути к датасету\n")
                f.write("   - Наличие папки wood/test/ с подпапками дефектов\n")
                f.write("   - Наличие папки wood/ground_truth/ с масками\n\n")
                
                val_info = self.results.get("validation_info", {})
                f.write(f"⏱️ ИНФОРМАЦИЯ О ПОПЫТКЕ ВАЛИДАЦИИ:\n")
                f.write(f"   Всего найдено образцов: {val_info.get('total_samples', 0)}\n")
                f.write(f"   Обработано успешно: {val_info.get('processed_samples', 0)}\n")
                f.write(f"   Время выполнения: {val_info.get('total_time', 0):.2f} сек\n")
                
                print(f"📄 Отчет об ошибке сохранен: {report_file}")
                return
            
            # Общие метрики
            overall = summary["overall"]
            f.write("📊 ОБЩИЕ МЕТРИКИ:\n")
            f.write(f"   Dice Score: {overall['mean_dice']:.3f} ± {overall['std_dice']:.3f}\n")
            f.write(f"   IoU: {overall['mean_iou']:.3f} ± {overall['std_iou']:.3f}\n")
            f.write(f"   F1-Score: {overall['mean_f1']:.3f}\n")
            f.write(f"   Precision: {overall['mean_precision']:.3f}\n")
            f.write(f"   Recall: {overall['mean_recall']:.3f}\n")
            f.write(f"   Pixel Accuracy: {overall['mean_pixel_accuracy']:.3f}\n\n")
            
            # По типам дефектов
            f.write("📝 МЕТРИКИ ПО ТИПАМ ДЕФЕКТОВ:\n")
            by_defect_type = summary.get("by_defect_type", {})
            if by_defect_type:
                for defect_type, metrics in by_defect_type.items():
                    f.write(f"\n   {defect_type.upper()} ({metrics['count']} образцов):\n")
                    f.write(f"      Dice: {metrics['mean_dice']:.3f}\n")
                    f.write(f"      IoU: {metrics['mean_iou']:.3f}\n")
                    f.write(f"      F1: {metrics['mean_f1']:.3f}\n")
                    f.write(f"      Precision: {metrics['mean_precision']:.3f}\n")
                    f.write(f"      Recall: {metrics['mean_recall']:.3f}\n")
            else:
                f.write("   Нет данных по типам дефектов\n")
            
            # Информация о валидации
            val_info = self.results["validation_info"]
            f.write(f"\n⏱️ ИНФОРМАЦИЯ О ВАЛИДАЦИИ:\n")
            f.write(f"   Всего образцов: {val_info['total_samples']}\n")
            f.write(f"   Обработано: {val_info['processed_samples']}\n")
            f.write(f"   Общее время: {val_info['total_time']:.2f} сек\n")
            f.write(f"   Среднее время на образец: {val_info['avg_time_per_sample']:.2f} сек\n")
        
        print(f"📄 Отчет сохранен: {report_file}")
        
        # Создаем графики если есть данные
        if summary and "overall" in summary and len(self.results["metrics"]) > 0:
            try:
                self.create_validation_plots(output_path)
            except Exception as e:
                print(f"⚠️ Не удалось создать графики: {e}")
        else:
            print("⚠️ Недостаточно данных для создания графиков")
    
    def create_validation_plots(self, output_path):
        """Создание графиков результатов валидации"""
        
        df = pd.DataFrame(self.results["metrics"])
        
        # Настройка стиля
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        
        # График 1: Распределение метрик
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        metrics_to_plot = ["dice", "iou", "f1", "precision", "recall", "pixel_accuracy"]
        
        for i, metric in enumerate(metrics_to_plot):
            row, col = i // 3, i % 3
            axes[row, col].hist(df[metric], bins=20, alpha=0.7, edgecolor='black')
            axes[row, col].set_title(f'{metric.upper()} Distribution')
            axes[row, col].set_xlabel(metric.upper())
            axes[row, col].set_ylabel('Frequency')
            axes[row, col].axvline(df[metric].mean(), color='red', linestyle='--', 
                                 label=f'Mean: {df[metric].mean():.3f}')
            axes[row, col].legend()
        
        plt.tight_layout()
        plt.savefig(output_path / "metrics_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # График 2: Метрики по типам дефектов
        if len(df["defect_type"].unique()) > 1:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            defect_types = df["defect_type"].unique()
            x_pos = np.arange(len(defect_types))
            
            dice_means = [df[df["defect_type"] == dt]["dice"].mean() for dt in defect_types]
            iou_means = [df[df["defect_type"] == dt]["iou"].mean() for dt in defect_types]
            
            width = 0.35
            ax.bar(x_pos - width/2, dice_means, width, label='Dice Score', alpha=0.8)
            ax.bar(x_pos + width/2, iou_means, width, label='IoU', alpha=0.8)
            
            ax.set_xlabel('Defect Type')
            ax.set_ylabel('Score')
            ax.set_title('Performance by Defect Type')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(defect_types, rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_path / "performance_by_defect_type.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"📈 Графики сохранены в {output_path}")


def main():
    """Основная функция валидации"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Валидация на MVTec AD")
    parser.add_argument("--dataset-path", default=None, help="Путь к датасету (если уже скачан)")
    parser.add_argument("--output", default="./mvtec_validation", help="Папка для результатов")
    parser.add_argument("--max-samples", type=int, default=None, help="Максимум образцов для тестирования")
    parser.add_argument("--model", default="standard", choices=["detailed", "standard", "latest"], 
                       help="Тип модели LLaVA (по умолчанию: standard = 7B)")
    
    args = parser.parse_args()
    
    # Создаем валидатор
    validator = MVTecValidator(args.dataset_path)
    
    # Скачиваем датасет если нужно
    if not validator.dataset_path:
        dataset_path = validator.download_dataset()
        if not dataset_path:
            print("❌ Не удалось скачать датасет")
            return
    
    # Проверяем наличие папки wood
    wood_path = validator.dataset_path / "wood"
    if not wood_path.exists():
        print(f"❌ Папка wood не найдена в {validator.dataset_path}")
        return
    
    # Инициализируем детектор
    validator.setup_detector(args.model)
    
    # Запускаем валидацию
    results = validator.run_validation(args.output, args.max_samples)
    
    if results:
        # Выводим краткие результаты
        summary = results.get("summary", {})
        if summary and "overall" in summary:
            overall = summary["overall"]
            print(f"\n📊 ИТОГОВЫЕ РЕЗУЛЬТАТЫ:")
            print(f"   🎯 Dice Score: {overall['mean_dice']:.3f}")
            print(f"   🎯 IoU: {overall['mean_iou']:.3f}")
            print(f"   🎯 F1-Score: {overall['mean_f1']:.3f}")
            print(f"   🎯 Precision: {overall['mean_precision']:.3f}")
            print(f"   🎯 Recall: {overall['mean_recall']:.3f}")
            print(f"   🎯 Pixel Accuracy: {overall['mean_pixel_accuracy']:.3f}")
        else:
            print(f"\n❌ ВАЛИДАЦИЯ НЕ УДАЛАСЬ:")
            val_info = results.get("validation_info", {})
            print(f"   📊 Найдено образцов: {val_info.get('total_samples', 0)}")
            print(f"   ✅ Обработано: {val_info.get('processed_samples', 0)}")
            print(f"   ⚠️ Проверьте структуру датасета и пути к файлам")


if __name__ == "__main__":
    main() 