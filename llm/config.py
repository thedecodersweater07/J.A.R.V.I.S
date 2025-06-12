from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class LLMConfig:
    """Configuration for the (dummy) LLM service.

    Includes a superset of parameters required by *models.jarvis* so that
    instantiation never fails even if some fields are unused by the dummy
    backend.
    """
    model_name: str = "jarvis-base"
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0
    num_return_sequences: int = 1
    device: str = "cpu"
    # Legacy / optional parameters for compatibility with older code paths
    max_tokens: int = 150
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

class ConfigManager:
    def __init__(self):
        self.default_config = LLMConfig(
            model_name="jarvis-base",
            temperature=0.7,
            max_length=512,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.0,
            num_return_sequences=1,
            device="cpu"
        )
    
    def load_config(self) -> LLMConfig:
        # Load from config file or use default
        return self.default_config
