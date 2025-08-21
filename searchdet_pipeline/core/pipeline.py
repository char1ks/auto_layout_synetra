"""
PipelineProcessor - высокоуровневый интерфейс для управления пайплайном.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path

from ..utils.config import Config, DEFAULT_CONFIG
from .detector import SearchDetDetector


class PipelineProcessor:
    """
    Высокоуровневый процессор пайплайна с упрощённым интерфейсом.
    
    Предоставляет удобные методы для различных сценариев использования:
    - Быстрый запуск с минимальными настройками
    - Пакетная обработка
    - Настройка и валидация конфигурации
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Инициализация процессора пайплайна.
        
        Args:
            config: Конфигурация пайплайна (если None, используется дефолтная)
        """
        self.config = config or DEFAULT_CONFIG
        self.detector = None
        self._is_initialized = False
    
    def setup(self, **model_paths) -> bool:
        """
        Настройка и инициализация всех компонентов пайплайна.
        
        Args:
            **model_paths: Пути к моделям (sam_hq_checkpoint, sam2_checkpoint, etc.)
            
        Returns:
            True если настройка прошла успешно
        """
        try:
            print("🔧 Настройка пайплайна...")
            
            # Создаём детектор
            self.detector = SearchDetDetector(self.config)
            
            # Инициализируем модели
            self.detector.initialize_models(**model_paths)
            
            # Проверяем валидность настройки
            validation_results = self.detector.validate_setup()
            
            if all(validation_results.values()):
                self._is_initialized = True
                print("✅ Пайплайн успешно настроен и готов к работе")
                return True
            else:
                print("❌ Не удалось полностью настроить пайплайн")
                return False
                
        except Exception as e:
            print(f"❌ Ошибка настройки пайплайна: {e}")
            return False
    
    def process_single(self, image_path: str,
                      positive_dir: Optional[str] = None,
                      negative_dir: Optional[str] = None,
                      output_dir: Optional[str] = None,
                      **kwargs) -> Dict[str, Any]:
        """
        Обработка одного изображения.
        
        Args:
            image_path: Путь к изображению
            positive_dir: Директория с положительными примерами
            negative_dir: Директория с отрицательными примерами
            output_dir: Директория для сохранения результатов
            **kwargs: Дополнительные параметры
            
        Returns:
            Результат обработки
        """
        if not self._check_initialization():
            return self._create_error_result("Пайплайн не инициализирован")
        
        # Валидация входных данных
        validation_error = self._validate_inputs(image_path, positive_dir, negative_dir)
        if validation_error:
            return self._create_error_result(validation_error)
        
        try:
            result = self.detector.find_present_elements(
                image_path, positive_dir, negative_dir, output_dir
            )
            return result
            
        except Exception as e:
            return self._create_error_result(f"Ошибка обработки: {e}")
    
    def process_batch(self, image_paths: List[str],
                     positive_dir: Optional[str] = None,
                     negative_dir: Optional[str] = None,
                     output_base_dir: Optional[str] = None,
                     **kwargs) -> List[Dict[str, Any]]:
        """
        Пакетная обработка изображений.
        
        Args:
            image_paths: Список путей к изображениям
            positive_dir: Директория с положительными примерами
            negative_dir: Директория с отрицательными примерами
            output_base_dir: Базовая директория для результатов
            **kwargs: Дополнительные параметры
            
        Returns:
            Список результатов обработки
        """
        if not self._check_initialization():
            error_result = self._create_error_result("Пайплайн не инициализирован")
            return [error_result] * len(image_paths)
        
        # Валидация списка изображений
        if not image_paths:
            error_result = self._create_error_result("Пустой список изображений")
            return [error_result]
        
        invalid_paths = [path for path in image_paths if not Path(path).exists()]
        if invalid_paths:
            error_msg = f"Не найдены файлы: {invalid_paths[:3]}..." if len(invalid_paths) > 3 else f"Не найдены файлы: {invalid_paths}"
            error_result = self._create_error_result(error_msg)
            return [error_result] * len(image_paths)
        
        try:
            results = self.detector.detect_batch(
                image_paths, positive_dir, negative_dir, output_base_dir
            )
            return results
            
        except Exception as e:
            error_result = self._create_error_result(f"Ошибка пакетной обработки: {e}")
            return [error_result] * len(image_paths)
    
    def quick_detect(self, image_path: str, examples_dir: str, 
                    output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Быстрая детекция с автоматическим определением positive/negative примеров.
        
        Args:
            image_path: Путь к изображению
            examples_dir: Директория с примерами (ищет подпапки 'positive' и 'negative')
            output_dir: Директория для сохранения результатов
            
        Returns:
            Результат детекции
        """
        examples_path = Path(examples_dir)
        
        # Автоматически определяем positive и negative директории
        positive_dir = None
        negative_dir = None
        
        if (examples_path / "positive").exists():
            positive_dir = str(examples_path / "positive")
        elif (examples_path / "pos").exists():
            positive_dir = str(examples_path / "pos")
        else:
            # Если нет подпапок, используем саму директорию как positive
            positive_dir = str(examples_path)
        
        if (examples_path / "negative").exists():
            negative_dir = str(examples_path / "negative")
        elif (examples_path / "neg").exists():
            negative_dir = str(examples_path / "neg")
        
        return self.process_single(image_path, positive_dir, negative_dir, output_dir)
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """
        Возвращает информацию о состоянии пайплайна.
        
        Returns:
            Словарь с информацией о пайплайне
        """
        info = {
            "initialized": self._is_initialized,
            "config": self.config.to_dict() if self.config else None,
            "backend": self.config.mask_generation.backend if self.config else None,
            "version": "2.0.0",
        }
        
        if self.detector:
            info["validation"] = self.detector.validate_setup()
        
        return info
    
    def update_config(self, config_updates: Dict[str, Any]) -> bool:
        """
        Обновляет конфигурацию пайплайна.
        
        Args:
            config_updates: Словарь с обновлениями конфигурации
            
        Returns:
            True если обновление прошло успешно
        """
        try:
            # Создаём новую конфигурацию на основе текущей
            new_config_dict = self.config.to_dict()
            new_config_dict.update(config_updates)
            
            # Создаём новый объект конфигурации
            self.config = Config.from_dict(new_config_dict)
            
            # Если пайплайн был инициализирован, нужно пересоздать детектор
            if self._is_initialized:
                print("⚠️ Конфигурация изменена. Необходимо вызвать setup() заново.")
                self._is_initialized = False
                self.detector = None
            
            return True
            
        except Exception as e:
            print(f"❌ Ошибка обновления конфигурации: {e}")
            return False
    
    def _check_initialization(self) -> bool:
        """Проверяет, инициализирован ли пайплайн."""
        if not self._is_initialized or not self.detector:
            print("❌ Пайплайн не инициализирован. Вызовите setup() сначала.")
            return False
        return True
    
    def _validate_inputs(self, image_path: str, 
                        positive_dir: Optional[str],
                        negative_dir: Optional[str]) -> Optional[str]:
        """
        Валидирует входные параметры.
        
        Args:
            image_path: Путь к изображению
            positive_dir: Директория с positive примерами  
            negative_dir: Директория с negative примерами
            
        Returns:
            Сообщение об ошибке или None если всё корректно
        """
        # Проверяем изображение
        if not image_path:
            return "Не указан путь к изображению"
        
        image_file = Path(image_path)
        if not image_file.exists():
            return f"Файл изображения не найден: {image_path}"
        
        # Проверяем расширение
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
        if image_file.suffix.lower() not in valid_extensions:
            return f"Неподдерживаемый формат изображения: {image_file.suffix}"
        
        # Проверяем positive примеры
        if positive_dir:
            pos_path = Path(positive_dir)
            if not pos_path.exists():
                return f"Директория positive примеров не найдена: {positive_dir}"
            
            # Проверяем, есть ли изображения в директории
            image_files = list(pos_path.glob("*.jpg")) + list(pos_path.glob("*.png")) + list(pos_path.glob("*.jpeg"))
            if not image_files:
                return f"В директории positive примеров не найдено изображений: {positive_dir}"
        else:
            print("⚠️ Не указаны positive примеры - детекция может быть неточной")
        
        # Проверяем negative примеры (необязательные)
        if negative_dir:
            neg_path = Path(negative_dir)
            if not neg_path.exists():
                print(f"⚠️ Директория negative примеров не найдена: {negative_dir}")
        
        return None
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """
        Создаёт стандартный результат с ошибкой.
        
        Args:
            error_message: Сообщение об ошибке
            
        Returns:
            Словарь с результатом ошибки
        """
        return {
            'success': False,
            'error': error_message,
            'processing_time': 0,
            'detections': [],
            'statistics': {},
            'saved_files': {},
            'config': self.config.to_dict() if self.config else {},
        }
    
    @staticmethod
    def create_default_config() -> Config:
        """
        Создаёт конфигурацию по умолчанию.
        
        Returns:
            Объект конфигурации с настройками по умолчанию
        """
        return DEFAULT_CONFIG
    
    @staticmethod
    def load_config_from_file(config_path: str) -> Optional[Config]:
        """
        Загружает конфигурацию из JSON файла.
        
        Args:
            config_path: Путь к файлу конфигурации
            
        Returns:
            Объект конфигурации или None при ошибке
        """
        try:
            import json
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            return Config.from_dict(config_dict)
        except Exception as e:
            print(f"❌ Ошибка загрузки конфигурации из {config_path}: {e}")
            return None
    
    def save_config_to_file(self, config_path: str) -> bool:
        """
        Сохраняет текущую конфигурацию в JSON файл.
        
        Args:
            config_path: Путь для сохранения конфигурации
            
        Returns:
            True если сохранение прошло успешно
        """
        try:
            import json
            config_dict = self.config.to_dict()
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, ensure_ascii=False, indent=2)
            print(f"✅ Конфигурация сохранена: {config_path}")
            return True
        except Exception as e:
            print(f"❌ Ошибка сохранения конфигурации: {e}")
            return False
