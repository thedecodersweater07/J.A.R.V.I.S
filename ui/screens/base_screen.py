from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseScreen(ABC):
    def __init__(self):
        self.initialized = False
        
    @abstractmethod
    def init(self) -> bool:
        pass
        
    @abstractmethod
    def render(self, frame_data: Dict[str, Any]) -> None:
        pass
        
    @abstractmethod
    def handle_input(self, input_data: Dict[str, Any]) -> None:
        pass
        
    def cleanup(self) -> None:
        self.initialized = False
