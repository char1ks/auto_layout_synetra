from typing import Dict, Any, Optional, List
from pathlib import Path

from ..utils.config import Config, DEFAULT_CONFIG
from .detector import SearchDetDetector


class PipelineProcessor:
    def __init__(self, config: Optional[Config] = None):
        self.config = config or DEFAULT_CONFIG
        self.detector = None
        self._is_initialized = False
    
    def setup(self, **model_paths) -> bool:
        try:
            print("🔧 Настройка пайплайна...")
            
            self.detector = SearchDetDetector(self.config)
            
            self.detector.initialize_models(**model_paths)
            
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
        if not self._check_initialization():
            return self._create_error_result("Пайплайн не инициализирован")
        
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
        if not self._check_initialization():
            error_result = self._create_error_result("Пайплайн не инициализирован")
            return [error_result] * len(image_paths)
        
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
        examples_path = Path(examples_dir)
        
        positive_dir = None
        negative_dir = None
        
        if (examples_path / "positive").exists():
            positive_dir = str(examples_path / "positive")
        elif (examples_path / "pos").exists():
            positive_dir = str(examples_path / "pos")
        else:
            positive_dir = str(examples_path)
        
        if (examples_path / "negative").exists():
            negative_dir = str(examples_path / "negative")
        elif (examples_path / "neg").exists():
            negative_dir = str(examples_path / "neg")
        
        return self.process_single(image_path, positive_dir, negative_dir, output_dir)
    
    def get_pipeline_info(self) -> Dict[str, Any]:
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
        try:
            new_config_dict = self.config.to_dict()
            new_config_dict.update(config_updates)
            
            self.config = Config.from_dict(new_config_dict)
            
            if self._is_initialized:
                print("⚠️ Конфигурация изменена. Необходимо вызвать setup() заново.")
                self._is_initialized = False
                self.detector = None
            
            return True
            
        except Exception as e:
            print(f"❌ Ошибка обновления конфигурации: {e}")
            return False
    
    def _check_initialization(self) -> bool:
        if not self._is_initialized or not self.detector:
            print("❌ Пайплайн не инициализирован. Вызовите setup() сначала.")
            return False
        return True
    
    def _validate_inputs(self, image_path: str, 
                        positive_dir: Optional[str],
                        negative_dir: Optional[str]) -> Optional[str]:
        if not image_path:
            return "Не указан путь к изображению"
        
        image_file = Path(image_path)
        if not image_file.exists():
            return f"Файл изображения не найден: {image_path}"
        
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
        if image_file.suffix.lower() not in valid_extensions:
            return f"Неподдерживаемый формат изображения: {image_file.suffix}"
        
        if positive_dir:
            pos_path = Path(positive_dir)
            if not pos_path.exists():
                return f"Директория positive примеров не найдена: {positive_dir}"
            
            image_files = list(pos_path.glob("*.jpg")) + list(pos_path.glob("*.png")) + list(pos_path.glob("*.jpeg"))
            if not image_files:
                return f"В директории positive примеров не найдено изображений: {positive_dir}"
        else:
            print("⚠️ Не указаны positive примеры - детекция может быть неточной")
        
        if negative_dir:
            neg_path = Path(negative_dir)
            if not neg_path.exists():
                print(f"⚠️ Директория negative примеров не найдена: {negative_dir}")
        
        return None
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
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
        return DEFAULT_CONFIG
    
    @staticmethod
    def load_config_from_file(config_path: str) -> Optional[Config]:
        try:
            import json
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            return Config.from_dict(config_dict)
        except Exception as e:
            print(f"❌ Ошибка загрузки конфигурации из {config_path}: {e}")
            return None
    
    def save_config_to_file(self, config_path: str) -> bool:
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
