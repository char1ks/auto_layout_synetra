#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Главная точка входа в SearchDet Pipeline.

Этот файл служит основной точкой запуска приложения и может использоваться как:
1. Прямой запуск: python main.py
2. Модульный запуск: python -m searchdet_pipeline
3. После установки пакета: searchdet-pipeline

Проверяет системные требования и перенаправляет управление в CLI модуль.
"""

import sys
import os
from pathlib import Path

# Добавляем корневую директорию проекта в PYTHONPATH
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def check_python_version():
    """Проверяет версию Python."""
    min_version = (3, 8)
    current_version = sys.version_info[:2]
    
    if current_version < min_version:
        print(f"❌ Требуется Python {min_version[0]}.{min_version[1]} или новее")
        print(f"   Текущая версия: {current_version[0]}.{current_version[1]}")
        print(f"   Обновите Python: https://www.python.org/downloads/")
        return False
    
    return True


def check_working_directory():
    """Проверяет, что мы находимся в правильной директории проекта."""
    expected_files = [
        'searchdet_pipeline',
        'searchdet-main', 
        'requirements_llava_sam2.txt',
        'README.md'
    ]
    
    missing_files = []
    for file_path in expected_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"⚠️ Похоже, вы находитесь не в корневой директории проекта")
        print(f"   Не найдены файлы/папки: {missing_files}")
        print(f"   Текущая директория: {Path.cwd()}")
        print(f"   Перейдите в корневую директорию проекта SearchDet Pipeline")
        return False
    
    return True


def check_dependencies():
    """Проверяет основные зависимости."""
    required_modules = ['numpy', 'cv2', 'PIL']
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print(f"❌ Отсутствуют обязательные зависимости: {missing_modules}")
        print(f"   Установите их командой:")
        print(f"   pip install -r requirements_llava_sam2.txt")
        return False
    
    return True


def print_banner():
    """Выводит баннер приложения."""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                         🔍 SearchDet Pipeline v2.0                          ║
║                                                                              ║
║              Профессиональный инструмент детекции объектов                   ║
║                        на базе Segment Anything Models                       ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def main():
    """Основная функция запуска."""
    # Показываем баннер
    print_banner()
    
    # Проверяем системные требования
    print("🔧 Проверка системных требований...")
    
    if not check_python_version():
        return 1
    print("   ✅ Версия Python подходит")
    
    if not check_working_directory():
        return 1  
    print("   ✅ Рабочая директория корректна")
    
    if not check_dependencies():
        return 1
    print("   ✅ Основные зависимости найдены")
    
    print("\n🚀 Запуск SearchDet Pipeline CLI...")
    print("🔗 Передача управления в: searchdet_pipeline.cli.main.main()")
    
    # Импортируем и запускаем CLI
    try:
        from searchdet_pipeline.cli.main import main as cli_main
        return cli_main()
        
    except ImportError as e:
        print(f"❌ Ошибка импорта CLI модуля: {e}")
        print("   Проверьте структуру пакета и зависимости")
        return 1
    
    except KeyboardInterrupt:
        print("\n⚠️ Выполнение прервано пользователем")
        return 130
    
    except Exception as e:
        print(f"❌ Неожиданная ошибка: {e}")
        return 1


def show_quick_help():
    """Показывает быструю справку."""
    help_text = """
🔍 SearchDet Pipeline - Быстрая справка

Основные команды:
  python main.py detect image.jpg --positive examples/     # Базовая детекция
  python main.py quick image.jpg examples/                 # Быстрый режим  
  python main.py batch *.jpg --positive examples/ -o out/  # Пакетная обработка
  python main.py config create config.json                 # Создать конфигурацию
  python main.py info                                       # Информация о системе

Для полной справки:
  python main.py --help
  python main.py КОМАНДА --help

Документация и примеры:
  README.md в корневой директории проекта
    """
    print(help_text)


if __name__ == '__main__':
    # Если запущен без аргументов, показываем быструю справку
    if len(sys.argv) == 1:
        print_banner()
        show_quick_help()
        sys.exit(0)
    
    # Обрабатываем специальные аргументы
    if len(sys.argv) == 2:
        if sys.argv[1] in ['--help', '-h', 'help']:
            print_banner()
            show_quick_help()
            sys.exit(0)
        elif sys.argv[1] in ['--version', '-v', 'version']:
            print("SearchDet Pipeline v2.0.0")
            sys.exit(0)
    
    # Запускаем основную функцию
    sys.exit(main())
