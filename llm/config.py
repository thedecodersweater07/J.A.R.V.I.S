from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class LLMConfig:
    model_name: str
    temperature: float
    max_tokens: int
    top_p: float
    frequency_penalty: float
    presence_penalty: float

class ConfigManager:
    def __init__(self):
        self.default_config = LLMConfig(
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=150,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
    
    def load_config(self) -> LLMConfig:
        # Load from config file or use default
        return self.default_config
