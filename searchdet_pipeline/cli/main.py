"""
CLI интерфейс для SearchDet Pipeline.

Предоставляет удобный интерфейс командной строки для всех функций пайплайна.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional, List

# Добавляем корневую директорию в path для импортов
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    # Пробуем относительный импорт
    from ..core.pipeline import PipelineProcessor
    from ..utils.config import Config, DEFAULT_CONFIG
except ImportError:
    # Fallback на абсолютные импорты
    try:
        from searchdet_pipeline.core.pipeline import PipelineProcessor
        from searchdet_pipeline.utils.config import Config, DEFAULT_CONFIG
    except ImportError:
        # Если и это не работает, создаем заглушки
        print("⚠️ Модули конфигурации недоступны, используем упрощенный режим")
        PipelineProcessor = None
        Config = None
        DEFAULT_CONFIG = None


def create_parser() -> argparse.ArgumentParser:
    """
    Создаёт парсер аргументов командной строки.
    
    Returns:
        Настроенный ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog='searchdet-pipeline',
        description='SearchDet Pipeline - профессиональный инструмент для детекции объектов на изображениях',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:

  # Базовая детекция
  searchdet-pipeline detect image.jpg --positive examples/positive/

  # С отрицательными примерами  
  searchdet-pipeline detect image.jpg --positive examples/pos/ --negative examples/neg/

  # Пакетная обработка
  searchdet-pipeline batch images/*.jpg --positive examples/ --output results/

  # Быстрая детекция (автоопределение positive/negative)
  searchdet-pipeline quick image.jpg examples/ --output results/

  # Настройка конфигурации
  searchdet-pipeline detect image.jpg --positive examples/ --backend sam2 --confidence 0.4

  # Использование кастомной конфигурации
  searchdet-pipeline detect image.jpg --positive examples/ --config my_config.json

Для подробной справки по команде используйте: searchdet-pipeline КОМАНДА --help
        """
    )
    
    # Основные подкоманды
    subparsers = parser.add_subparsers(dest='command', help='Доступные команды')
    
    # Команда detect - обработка одного изображения
    detect_parser = subparsers.add_parser(
        'detect', 
        help='Детекция объектов на одном изображении',
        description='Выполняет детекцию объектов на указанном изображении'
    )
    _add_detect_arguments(detect_parser)
    
    # Команда batch - пакетная обработка
    batch_parser = subparsers.add_parser(
        'batch',
        help='Пакетная обработка нескольких изображений', 
        description='Выполняет детекцию на нескольких изображениях'
    )
    _add_batch_arguments(batch_parser)
    
    # Команда quick - быстрая детекция
    quick_parser = subparsers.add_parser(
        'quick',
        help='Быстрая детекция с автоопределением примеров',
        description='Упрощённый режим детекции с автоматическим поиском positive/negative примеров'
    )
    _add_quick_arguments(quick_parser)
    
    # Команда config - управление конфигурацией
    config_parser = subparsers.add_parser(
        'config',
        help='Управление конфигурацией',
        description='Создание, просмотр и валидация конфигурации пайплайна'
    )
    _add_config_arguments(config_parser)
    
    # Команда info - информация о пайплайне
    info_parser = subparsers.add_parser(
        'info',
        help='Информация о пайплайне и системе',
        description='Показывает информацию о версии, конфигурации и доступных компонентах'
    )
    
    return parser


def _add_detect_arguments(parser: argparse.ArgumentParser):
    """Добавляет аргументы для команды detect."""
    # Обязательные аргументы
    parser.add_argument('image', help='Путь к изображению для обработки')
    
    # Примеры
    parser.add_argument('--positive', '-p', help='Директория с положительными примерами')
    parser.add_argument('--negative', '-n', help='Директория с отрицательными примерами')
    
    # Выходные файлы
    parser.add_argument('--output', '-o', help='Директория для сохранения результатов')
    parser.add_argument('--no-save', action='store_true', help='Не сохранять результаты на диск')
    
    # Конфигурация
    parser.add_argument('--config', '-c', help='Путь к файлу конфигурации JSON')
    parser.add_argument('--backend', choices=['sam-hq', 'sam2', 'fastsam'], 
                       help='Бэкенд для генерации масок')
    parser.add_argument('--confidence', type=float, help='Минимальный порог уверенности (0-1)')
    parser.add_argument('--max-masks', type=int, help='Максимальное количество детекций')
    
    # Пути к моделям
    parser.add_argument('--sam-checkpoint', help='Путь к checkpoint SAM-HQ')
    parser.add_argument('--sam2-checkpoint', help='Путь к checkpoint SAM2')
    parser.add_argument('--sam2-config', help='Путь к конфигурации SAM2')
    parser.add_argument('--fastsam-checkpoint', help='Путь к checkpoint FastSAM')
    
    # Дополнительные опции
    parser.add_argument('--verbose', '-v', action='store_true', help='Подробный вывод')
    parser.add_argument('--quiet', '-q', action='store_true', help='Минимальный вывод')


def _add_batch_arguments(parser: argparse.ArgumentParser):
    """Добавляет аргументы для команды batch."""
    # Изображения
    parser.add_argument('images', nargs='+', help='Пути к изображениям (можно использовать wildcards)')
    
    # Примеры (общие для всей пачки)
    parser.add_argument('--positive', '-p', help='Директория с положительными примерами')
    parser.add_argument('--negative', '-n', help='Директория с отрицательными примерами')
    
    # Выходные файлы
    parser.add_argument('--output', '-o', required=True, help='Базовая директория для результатов')
    
    # Конфигурация
    parser.add_argument('--config', '-c', help='Путь к файлу конфигурации JSON')
    parser.add_argument('--backend', choices=['sam-hq', 'sam2', 'fastsam'], 
                       help='Бэкенд для генерации масок')
    parser.add_argument('--confidence', type=float, help='Минимальный порог уверенности (0-1)')
    
    # Пакетная обработка
    parser.add_argument('--parallel', action='store_true', help='Параллельная обработка (экспериментально)')
    parser.add_argument('--continue-on-error', action='store_true', 
                       help='Продолжать обработку при ошибках')
    
    # Пути к моделям
    parser.add_argument('--sam-checkpoint', help='Путь к checkpoint SAM-HQ')
    parser.add_argument('--sam2-checkpoint', help='Путь к checkpoint SAM2')
    parser.add_argument('--sam2-config', help='Путь к конфигурации SAM2')
    
    # Дополнительные опции
    parser.add_argument('--verbose', '-v', action='store_true', help='Подробный вывод')


def _add_quick_arguments(parser: argparse.ArgumentParser):
    """Добавляет аргументы для команды quick."""
    parser.add_argument('image', help='Путь к изображению для обработки')
    parser.add_argument('examples', help='Директория с примерами (ищет подпапки positive/negative)')
    parser.add_argument('--output', '-o', help='Директория для сохранения результатов')
    parser.add_argument('--backend', choices=['sam-hq', 'sam2', 'fastsam'], 
                       default='sam-hq', help='Бэкенд для генерации масок')
    parser.add_argument('--verbose', '-v', action='store_true', help='Подробный вывод')


def _add_config_arguments(parser: argparse.ArgumentParser):
    """Добавляет аргументы для команды config."""
    config_subparsers = parser.add_subparsers(dest='config_action', help='Действия с конфигурацией')
    
    # Создание дефолтной конфигурации
    create_parser = config_subparsers.add_parser('create', help='Создать дефолтную конфигурацию')
    create_parser.add_argument('output', help='Путь для сохранения конфигурации')
    
    # Валидация конфигурации
    validate_parser = config_subparsers.add_parser('validate', help='Проверить конфигурацию')
    validate_parser.add_argument('config', help='Путь к файлу конфигурации')
    
    # Показ конфигурации
    show_parser = config_subparsers.add_parser('show', help='Показать текущую конфигурацию')
    show_parser.add_argument('--config', help='Путь к файлу конфигурации (по умолчанию - дефолтная)')


def main():
    """Основная функция CLI."""
    print("🚀 ЗАПУСК МОДУЛЬНОГО SEARCHDET ПАЙПЛАЙНА")
    print("=" * 60)
    print("📁 Точка входа: searchdet_pipeline/cli/main.py → main()")
    
    parser = create_parser()
    args = parser.parse_args()
    
    # Если команда не указана, показываем help
    if not args.command:
        parser.print_help()
        return 0
    
    try:
        # Выполняем соответствующую команду
        if args.command == 'detect':
            return _execute_detect(args)
        elif args.command == 'batch':
            return _execute_batch(args)
        elif args.command == 'quick':
            return _execute_quick(args)
        elif args.command == 'config':
            return _execute_config(args)
        elif args.command == 'info':
            return _execute_info(args)
        else:
            print(f"❌ Неизвестная команда: {args.command}")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️ Обработка прервана пользователем")
        return 130
    except Exception as e:
        print(f"❌ Неожиданная ошибка: {e}")
        if hasattr(args, 'verbose') and args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def _execute_detect(args) -> int:
    """Выполняет команду detect."""
    print("\n" + "="*70)
    print("📋 ПОРЯДОК ВЫПОЛНЕНИЯ МОДУЛЬНОГО ПАЙПЛАЙНА:")
    print("="*70)
    print("1️⃣ main.py → searchdet_pipeline.cli.main.main()")
    print("2️⃣ searchdet_pipeline/cli/main.py → _execute_detect()")
    
    # Используем гибридный пайплайн напрямую
    import sys
    import os
    
    # Добавляем корневую директорию в path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    try:
        print("3️⃣ Импорт: searchdet_pipeline.core.detector → SearchDetDetector (автономная версия)")
        # Импортируем автономный модульный детектор
        from ..core.detector import SearchDetDetector
        
        print("4️⃣ Инициализация: SearchDetDetector.__init__()")
        print("🔧 Инициализация SearchDet детектора...")
        
        # Параметры для детектора
        detector_params = {
            'mask_backend': args.backend or 'fastsam',
        }
        
        # Добавляем пути к моделям если указаны
        if hasattr(args, 'sam_checkpoint') and args.sam_checkpoint:
            detector_params['sam_model'] = args.sam_checkpoint
        if hasattr(args, 'sam2_checkpoint') and args.sam2_checkpoint:
            detector_params['sam2_weights'] = args.sam2_checkpoint
        if hasattr(args, 'fastsam_checkpoint') and args.fastsam_checkpoint:
            detector_params['fastsam_model'] = args.fastsam_checkpoint
            
        # Параметры скоринга
        if hasattr(args, 'confidence') and args.confidence is not None:
            detector_params['min_confidence'] = args.confidence
        if hasattr(args, 'max_masks') and args.max_masks is not None:
            detector_params['max_masks'] = args.max_masks
        
        print("5️⃣ Создание: detector = SearchDetDetector(**params)")
        # Создаём детектор
        detector = SearchDetDetector(**detector_params)
        
        print("6️⃣ Вызов: detector.find_present_elements()")
        print("   ↳ Это запустит весь пайплайн из hybrid_searchdet_pipeline.py:")
        print("   ↳ _load_example_images() → _generate_sam_masks() → _filter_*() → _extract_mask_embeddings() → _score_masks()")
        
        # Выполняем детекцию
        result = detector.find_present_elements(
            args.image,
            args.positive,
            args.negative
        )
        
        # Выводим результат
        _print_result(result, args)
        
        # Проверяем успешность (новый или старый формат)
        if 'success' in result:
            return 0 if result['success'] else 1
        else:
            # Старый формат - считаем успешным если есть found_elements
            return 0 if 'found_elements' in result else 1
        
    except ImportError as e:
        print(f"❌ Ошибка импорта модульного пайплайна: {e}")
        print("   Убедитесь что модули searchdet_pipeline/core/ доступны")
        return 1
    except Exception as e:
        print(f"❌ Ошибка выполнения детекции: {e}")
        if hasattr(args, 'verbose') and args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def _execute_batch(args) -> int:
    """Выполняет команду batch."""
    import glob
    
    # Разворачиваем wildcards в путях
    all_images = []
    for pattern in args.images:
        matched_files = glob.glob(pattern)
        if matched_files:
            all_images.extend(matched_files)
        else:
            all_images.append(pattern)  # Добавляем как есть, если не matched
    
    print(f"🔄 Найдено {len(all_images)} изображений для обработки")
    
    # Проверяем существование файлов
    valid_images = []
    for img_path in all_images:
        if Path(img_path).exists():
            valid_images.append(img_path)
        else:
            print(f"⚠️ Файл не найден: {img_path}")
    
    if not valid_images:
        print("❌ Не найдено валидных изображений")
        return 1
    
    print(f"✅ Будет обработано {len(valid_images)} изображений")
    
    # Загружаем конфигурацию и настраиваем пайплайн
    config = _load_config(args)
    config = _apply_cli_args_to_config(config, args)
    
    processor = PipelineProcessor(config)
    model_paths = _extract_model_paths(args)
    if not processor.setup(**model_paths):
        print("❌ Не удалось настроить пайплайн")
        return 1
    
    # Выполняем пакетную обработку
    results = processor.process_batch(
        valid_images,
        args.positive,
        args.negative,
        args.output
    )
    
    # Выводим сводную статистику
    _print_batch_results(results, args)
    
    # Возвращаем код ошибки если были неуспешные обработки
    failed_count = sum(1 for r in results if not r['success'])
    return 1 if failed_count > 0 and not args.continue_on_error else 0


def _execute_quick(args) -> int:
    """Выполняет команду quick."""
    print("\n" + "="*70)
    print("📋 ПОРЯДОК ВЫПОЛНЕНИЯ QUICK РЕЖИМА:")
    print("="*70)
    print("1️⃣ main.py → searchdet_pipeline.cli.main.main()")
    print("2️⃣ searchdet_pipeline/cli/main.py → _execute_quick()")
    
    import sys
    
    # Добавляем корневую директорию в path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    try:
        print("3️⃣ Импорт: searchdet_pipeline.core.detector → SearchDetDetector (автономная версия)")
        # Импортируем автономный модульный детектор
        from ..core.detector import SearchDetDetector
        
        print("4️⃣ Инициализация: SearchDetDetector.__init__()")
        print("🔧 Быстрая инициализация SearchDet детектора...")
        
        print("5️⃣ Автопоиск папок: examples/positive и examples/negative")
        # Автоопределение positive/negative папок
        examples_path = Path(args.examples)
        positive_path = examples_path / "positive"
        negative_path = examples_path / "negative"
        
        # Проверяем наличие подпапок
        if not positive_path.exists():
            print(f"⚠️ Папка positive не найдена в {examples_path}")
            positive_path = examples_path  # Используем саму папку как positive
            negative_path = None
        
        if negative_path and not negative_path.exists():
            print(f"⚠️ Папка negative не найдена в {examples_path}")
            negative_path = None
        
        # Параметры для детектора
        detector_params = {
            'mask_backend': args.backend or 'fastsam',
        }
        
        print("6️⃣ Создание: detector = SearchDetDetector(**params)")
        # Создаём детектор
        detector = SearchDetDetector(**detector_params)
        
        print("7️⃣ Вызов: detector.find_present_elements()")
        print("   ↳ Запуск полного пайплайна из hybrid_searchdet_pipeline.py")
        # Выполняем детекцию
        result = detector.find_present_elements(
            args.image,
            str(positive_path) if positive_path else None,
            str(negative_path) if negative_path else None
        )
        
        # Выводим результат
        _print_result(result, args)
        
        # Проверяем успешность (новый или старый формат)
        if 'success' in result:
            return 0 if result['success'] else 1
        else:
            # Старый формат - считаем успешным если есть found_elements
            return 0 if 'found_elements' in result else 1
        
    except ImportError as e:
        print(f"❌ Ошибка импорта модульного пайплайна: {e}")
        return 1
    except Exception as e:
        print(f"❌ Ошибка выполнения детекции: {e}")
        if hasattr(args, 'verbose') and args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def _execute_config(args) -> int:
    """Выполняет команды управления конфигурацией."""
    if not args.config_action:
        print("❌ Не указано действие с конфигурацией. Используйте --help для справки")
        return 1
    
    if args.config_action == 'create':
        return _create_default_config(args.output)
    elif args.config_action == 'validate':
        return _validate_config_file(args.config)
    elif args.config_action == 'show':
        return _show_config(args.config if hasattr(args, 'config') else None)
    
    return 1


def _execute_info(args) -> int:
    """Выполняет команду info."""
    print("🔍 SearchDet Pipeline - Информация о системе")
    print("=" * 60)
    print(f"📋 Версия: 2.0.0")
    print(f"🐍 Python: {sys.version}")
    print(f"📁 Рабочая директория: {Path.cwd()}")
    
    # Проверяем доступность зависимостей
    print("\n📦 Проверка зависимостей:")
    _check_dependencies()
    
    # Показываем дефолтную конфигурацию
    if DEFAULT_CONFIG is not None:
        print("\n⚙️ Конфигурация по умолчанию:")
        config_dict = DEFAULT_CONFIG.to_dict()
        print(json.dumps(config_dict, indent=2, ensure_ascii=False))
    else:
        print("\n⚙️ Работает в упрощенном режиме (без модульной конфигурации)")
    
    return 0


def _load_config(args):
    """Загружает конфигурацию из файла или использует дефолтную."""
    if DEFAULT_CONFIG is None:
        return None  # Упрощенный режим
        
    if hasattr(args, 'config') and args.config:
        if PipelineProcessor:
            config = PipelineProcessor.load_config_from_file(args.config)
            if config is None:
                print(f"⚠️ Не удалось загрузить конфигурацию из {args.config}, используется дефолтная")
                return DEFAULT_CONFIG
            return config
    return DEFAULT_CONFIG


def _apply_cli_args_to_config(config, args):
    """Применяет аргументы CLI к конфигурации."""
    if config is None or Config is None:
        return None  # Упрощенный режим
        
    # Создаём копию конфигурации
    config_dict = config.to_dict()
    
    # Применяем аргументы CLI (они имеют приоритет над файлом конфигурации)
    if hasattr(args, 'backend') and args.backend:
        config_dict['mask_generation']['backend'] = args.backend
    
    if hasattr(args, 'confidence') and args.confidence is not None:
        config_dict['scoring']['min_confidence'] = args.confidence
    
    if hasattr(args, 'max_masks') and args.max_masks is not None:
        config_dict['post_processing']['max_masks'] = args.max_masks
    
    # Пути к моделям
    if hasattr(args, 'sam_checkpoint') and args.sam_checkpoint:
        config_dict['sam_hq_checkpoint'] = args.sam_checkpoint
    
    if hasattr(args, 'sam2_checkpoint') and args.sam2_checkpoint:
        config_dict['sam2_checkpoint'] = args.sam2_checkpoint
    
    if hasattr(args, 'sam2_config') and args.sam2_config:
        config_dict['sam2_config'] = args.sam2_config
    
    if hasattr(args, 'fastsam_checkpoint') and args.fastsam_checkpoint:
        config_dict['fastsam_checkpoint'] = args.fastsam_checkpoint
    
    # Настройка сохранения
    if hasattr(args, 'no_save') and args.no_save:
        config_dict['save_all'] = False
    
    return Config.from_dict(config_dict)


def _extract_model_paths(args) -> dict:
    """Извлекает пути к моделям из аргументов."""
    model_paths = {}
    
    if hasattr(args, 'sam_checkpoint') and args.sam_checkpoint:
        model_paths['sam_hq_checkpoint'] = args.sam_checkpoint
    
    if hasattr(args, 'sam2_checkpoint') and args.sam2_checkpoint:
        model_paths['sam2_checkpoint'] = args.sam2_checkpoint
    
    if hasattr(args, 'sam2_config') and args.sam2_config:
        model_paths['sam2_config'] = args.sam2_config
    
    if hasattr(args, 'fastsam_checkpoint') and args.fastsam_checkpoint:
        model_paths['fastsam_checkpoint'] = args.fastsam_checkpoint
    
    return model_paths


def _print_result(result: dict, args):
    """Выводит результат обработки."""
    # Проверяем формат результата (новый или старый)
    if 'success' in result:
        # Новый формат
        if result['success']:
            print(f"\n✅ Детекция завершена успешно!")
            print(f"⏱️ Время обработки: {result['processing_time']:.2f} сек")
            print(f"🔍 Найдено объектов: {len(result['detections'])}")
            
            if result['detections']:
                confidences = [d['confidence'] for d in result['detections']]
                print(f"📊 Средняя уверенность: {sum(confidences)/len(confidences):.3f}")
            
            if result.get('saved_files'):
                print(f"💾 Сохранено файлов: {len(result['saved_files'])}")
                if hasattr(args, 'verbose') and args.verbose:
                    for file_type, path in result['saved_files'].items():
                        print(f"   • {file_type}: {path}")
        else:
            print(f"\n❌ Ошибка детекции: {result.get('error', 'Неизвестная ошибка')}")
    else:
        # Старый формат (из hybrid_searchdet_pipeline.py)
        if 'found_elements' in result:
            found_elements = result['found_elements']
            print(f"\n✅ Детекция завершена успешно!")
            print(f"🔍 Найдено объектов: {len(found_elements)}")
            
            if found_elements:
                confidences = [elem.get('confidence', 0.0) for elem in found_elements]
                if any(c > 0 for c in confidences):
                    print(f"📊 Средняя уверенность: {sum(confidences)/len(confidences):.3f}")
            
            if hasattr(args, 'output') and args.output:
                print(f"💾 Результаты будут сохранены в: {args.output}")
        else:
            print(f"\n❌ Неожиданный формат результата: {result}")


def _print_batch_results(results: List[dict], args):
    """Выводит результаты пакетной обработки."""
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"\n📊 РЕЗУЛЬТАТЫ ПАКЕТНОЙ ОБРАБОТКИ:")
    print(f"✅ Успешно: {len(successful)}/{len(results)}")
    print(f"❌ Ошибки: {len(failed)}/{len(results)}")
    
    if successful:
        total_detections = sum(len(r['detections']) for r in successful)
        total_time = sum(r['processing_time'] for r in successful)
        print(f"🔍 Всего найдено объектов: {total_detections}")
        print(f"⏱️ Общее время: {total_time:.2f} сек")
        print(f"⚡ Среднее время на изображение: {total_time/len(successful):.2f} сек")
    
    if failed and (hasattr(args, 'verbose') and args.verbose):
        print(f"\n❌ Неуспешные обработки:")
        for result in failed[:5]:  # Показываем только первые 5 ошибок
            print(f"   • {result.get('image_path', '?')}: {result.get('error', 'Неизвестная ошибка')}")


def _create_default_config(output_path: str) -> int:
    """Создаёт дефолтную конфигурацию."""
    try:
        processor = PipelineProcessor()
        if processor.save_config_to_file(output_path):
            print(f"✅ Дефолтная конфигурация создана: {output_path}")
            return 0
        else:
            return 1
    except Exception as e:
        print(f"❌ Ошибка создания конфигурации: {e}")
        return 1


def _validate_config_file(config_path: str) -> int:
    """Валидирует файл конфигурации."""
    config = PipelineProcessor.load_config_from_file(config_path)
    if config:
        print(f"✅ Конфигурация {config_path} валидна")
        return 0
    else:
        print(f"❌ Конфигурация {config_path} невалидна")
        return 1


def _show_config(config_path: Optional[str]) -> int:
    """Показывает конфигурацию."""
    if config_path:
        config = PipelineProcessor.load_config_from_file(config_path)
        if not config:
            return 1
    else:
        config = DEFAULT_CONFIG
    
    config_dict = config.to_dict()
    print(json.dumps(config_dict, indent=2, ensure_ascii=False))
    return 0


def _check_dependencies():
    """Проверяет доступность основных зависимостей."""
    dependencies = [
        ('numpy', 'NumPy'),
        ('cv2', 'OpenCV'),
        ('PIL', 'Pillow'),
    ]
    
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name} - не найден")
    
    # Проверяем SearchDet
    try:
        import sys
        sys.path.append('./searchdet-main')
        import mask_withsearch
        print(f"   ✅ SearchDet")
    except ImportError:
        print(f"   ❌ SearchDet - не найден")
    
    # Проверяем SAM модели
    try:
        import segment_anything
        print(f"   ✅ Segment Anything")
    except ImportError:
        print(f"   ⚠️ Segment Anything - не найден (опционально)")
    
    try:
        import sam2
        print(f"   ✅ SAM2")
    except ImportError:
        print(f"   ⚠️ SAM2 - не найден (опционально)")
    
    try:
        import ultralytics
        print(f"   ✅ Ultralytics (FastSAM)")
    except ImportError:
        print(f"   ⚠️ Ultralytics - не найден (опционально)")


if __name__ == '__main__':
    sys.exit(main())
