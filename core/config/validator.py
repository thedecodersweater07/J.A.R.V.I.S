from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class ConfigValidator:
    def validate(self, config: Dict[str, Any], section_name: str = None) -> bool:
        """Validate configuration with section support"""
        try:
            if section_name:
                return self._validate_section(config, section_name)
            return self._validate_full_config(config)
        except Exception as e:
            logger.error(f"Config validation failed: {e}")
            return False
            
    def _validate_section(self, section_config: Dict[str, Any], section_name: str) -> bool:
        """Validate a specific configuration section"""
        validators = {
            "llm": self._validate_llm_config,
            "ui": self._validate_ui_config,
            "nlp": self._validate_nlp_config
        }
        
        validator = validators.get(section_name)
        if validator:
            return validator(section_config)
        return True
        
    def _validate_llm_config(self, config: Dict[str, Any]) -> bool:
        required = ["model", "max_length"]
        return all(key in config for key in required)
        
    def _validate_ui_config(self, config: Dict[str, Any]) -> bool:
        required = ["width", "height", "title", "theme"]
        return all(key in config for key in required)
        
    def _validate_nlp_config(self, config: Dict[str, Any]) -> bool:
        required = ["language", "model", "models"]
        return all(key in config for key in required)
