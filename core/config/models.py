from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from pathlib import Path

@dataclass
class UIConfig:
    width: int = 800
    height: int = 600
    theme: str = "dark"
    font_size: int = 12
    opacity: float = 1.0
    use_opengl: bool = True

@dataclass
class ChatConfig:
    window_title: str = "JARVIS Chat"
    input_placeholder: str = "Type your message..."
    history_max_messages: int = 1000
    enable_markdown: bool = True
    enable_code_highlighting: bool = True
    timestamp_format: str = "%H:%M:%S"

@dataclass
class LLMConfig:
    model_name: str = "gpt2"
    temperature: float = 0.7
    max_tokens: int = 150
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    batch_size: int = 1

@dataclass
class SystemConfig:
    name: str = "JARVIS"
    version: str = "2.5.0"
    language: str = "nl-NL"
    log_level: str = "INFO"
    data_dir: Path = Path("data")
    cache_dir: Path = Path("cache")
    config_dir: Path = Path("config")
