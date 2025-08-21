#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup script для SearchDet Pipeline.

Позволяет установить пакет и создать entry points для удобного запуска из командной строки.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Читаем описание из README
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()

# Читаем зависимости из requirements
requirements_path = Path(__file__).parent / "requirements_llava_sam2.txt"
requirements = []
if requirements_path.exists():
    with open(requirements_path, "r", encoding="utf-8") as f:
        requirements = [
            line.strip() 
            for line in f.readlines() 
            if line.strip() and not line.startswith("#")
        ]

# Основные зависимости (обязательные)
install_requires = [
    "numpy>=1.21.0",
    "opencv-python>=4.5.0", 
    "Pillow>=8.0.0",
    "tqdm>=4.60.0",
]

# Опциональные зависимости для различных компонентов
extras_require = {
    # SAM-HQ поддержка
    "sam": [
        "segment-anything @ git+https://github.com/facebookresearch/segment-anything.git",
        "torch>=1.7.0",
        "torchvision>=0.8.0",
    ],
    
    # SAM2 поддержка  
    "sam2": [
        "sam2 @ git+https://github.com/facebookresearch/sam2.git",
        "torch>=1.7.0",
        "torchvision>=0.8.0",
    ],
    
    # FastSAM поддержка
    "fastsam": [
        "ultralytics>=8.1.0",
    ],
    
    # Разработка и тестирование
    "dev": [
        "pytest>=6.0.0",
        "pytest-cov>=2.10.0",
        "black>=21.0.0",
        "flake8>=3.8.0",
        "mypy>=0.910",
    ],
    
    # Все опциональные компоненты
    "all": [
        "segment-anything @ git+https://github.com/facebookresearch/segment-anything.git",
        "sam2 @ git+https://github.com/facebookresearch/sam2.git", 
        "ultralytics>=8.1.0",
        "torch>=1.7.0",
        "torchvision>=0.8.0",
    ]
}

setup(
    # Основная информация о пакете
    name="searchdet-pipeline",
    version="2.0.0",
    description="Профессиональный пайплайн для детекции объектов на базе Segment Anything Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    # Автор и контакты
    author="SearchDet Team",
    author_email="searchdet@example.com",
    url="https://github.com/your-org/searchdet-pipeline",
    
    # Лицензия и классификаторы
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9", 
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    
    # Требования к Python
    python_requires=">=3.8",
    
    # Пакеты и зависимости
    packages=find_packages(exclude=["tests*", "docs*", "examples*"]),
    install_requires=install_requires,
    extras_require=extras_require,
    
    # Включаем дополнительные файлы
    include_package_data=True,
    package_data={
        "searchdet_pipeline": [
            "*.txt",
            "*.md", 
            "*.json",
        ],
    },
    
    # Entry points для CLI
    entry_points={
        "console_scripts": [
            "searchdet-pipeline=searchdet_pipeline.cli.main:main",
            "searchdet=searchdet_pipeline.cli.main:main",
        ],
    },
    
    # Ключевые слова для поиска
    keywords=[
        "computer-vision",
        "object-detection", 
        "segmentation",
        "segment-anything",
        "sam",
        "image-processing",
        "deep-learning",
        "ai",
        "machine-learning",
    ],
    
    # Проектные URL
    project_urls={
        "Bug Reports": "https://github.com/your-org/searchdet-pipeline/issues",
        "Source": "https://github.com/your-org/searchdet-pipeline",
        "Documentation": "https://github.com/your-org/searchdet-pipeline/blob/main/README.md",
    },
    
    # Дополнительные опции
    zip_safe=False,
)
