import json
import yaml
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from .config_validator import ConfigValidator

logger = logging.getLogger(__name__)

class ConfigManager:
    def __init__(self, config_dir: str = None):
        self.config_dir = Path(config_dir or Path(__file__).parent)
        self.validator = ConfigValidator()
        self.config: Dict[str, Any] = {}
        self._load_all_configs()

    def _load_all_configs(self):
        """Load all configuration files"""
        try:
            # Load main config
            main_config = self._load_json_config("defaults/main.json")
            self.config.update(main_config)

            # Load module configs
            for config_file in (self.config_dir / "defaults").glob("*.json"):
                if config_file.stem != "main":
                    module_config = self._load_json_config(f"defaults/{config_file.name}")
                    self.config[config_file.stem] = module_config

            # Convert to YAML for LLM
            self._save_yaml_config()
            
        except Exception as e:
            logger.error(f"Failed to load configurations: {e}")
            raise

    def _load_json_config(self, relative_path: str) -> Dict[str, Any]:
        """Load a JSON configuration file"""
        file_path = self.config_dir / relative_path
        try:
            with open(file_path) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config {file_path}: {e}")
            return {}

    def _save_yaml_config(self):
        """Save consolidated config as YAML for LLM"""
        yaml_path = self.config_dir.parent / "data" / "config.yaml"
        yaml_path.parent.mkdir(exist_ok=True)
        
        try:
            with open(yaml_path, 'w') as f:
                yaml.dump(self.config, f)
        except Exception as e:
            logger.error(f"Failed to save YAML config: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value by dot notation key"""
        try:
            value = self.config
            for k in key.split('.'):
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any) -> bool:
        """Set config value by dot notation key"""
        try:
            keys = key.split('.')
            target = self.config
            for k in keys[:-1]:
                target = target[k]
            target[keys[-1]] = value
            self._save_yaml_config()
            return True
        except Exception as e:
            logger.error(f"Failed to set config {key}: {e}")
            return False
