from importlib import import_module
from typing import Dict, Any
from hyperadvanced_ai.core.logging import get_logger

logger = get_logger("ModuleManager")

class ModuleManager:
    """Central manager for dynamic AI modules."""
    def __init__(self):
        self.modules: Dict[str, Any] = {}

    def load_module(self, module_path: str, class_name: str = ""):
        try:
            mod = import_module(module_path)
            if class_name:
                obj = getattr(mod, class_name)
                self.modules[class_name] = obj
                logger.info(f"Loaded {class_name} from {module_path}")
                return obj
            logger.info(f"Loaded module {module_path}")
            return mod
        except Exception as e:
            logger.error(f"Failed to load module {module_path}: {e}")
            return None

    def get_module(self, name: str):
        return self.modules.get(name)
