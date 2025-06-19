import yaml
import os
from dataclasses import dataclass, asdict
from typing import Dict, Any

@dataclass
class LLMConfig:
    """Configuration for the LLM service."""
    model_name: str = "jarvis-base"
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0
    num_return_sequences: int = 1
    device: str = "cpu"
    max_tokens: int = 150
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

class ConfigManager:
    def __init__(self, config_path: str = 'config.yaml'):
        self.config_path = config_path
        self.default_config = LLMConfig()
        self.config = self.load_config()

    def load_config(self) -> LLMConfig:
        """Loads configuration from a YAML file, falling back to defaults."""
        config_data = self._read_yaml()
        if config_data and 'llm' in config_data:
            # Update default config with values from file
            llm_config_data = {k: v for k, v in config_data['llm'].items() if hasattr(self.default_config, k)}
            return LLMConfig(**llm_config_data)
        return self.default_config

    def _read_yaml(self) -> Dict[str, Any]:
        """Reads the YAML configuration file."""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                try:
                    return yaml.safe_load(f)
                except yaml.YAMLError as e:
                    print(f"Error loading YAML config: {e}")
        return {}

    def get_config(self) -> LLMConfig:
        """Returns the current configuration."""
        return self.config
