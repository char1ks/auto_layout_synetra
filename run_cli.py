#!/usr/bin/env python3
"""
Простая обертка для запуска CLI модуля напрямую.
"""

import sys
from pathlib import Path

# Добавляем корневую директорию в path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

if __name__ == '__main__':
    # Импортируем и запускаем CLI
    from searchdet_pipeline.cli.main import main
    sys.exit(main())
