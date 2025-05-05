from abc import ABC, abstractmethod
from typing import Dict, Any

class RendererBase(ABC):
    """Base class for all renderers"""
    
    @abstractmethod
    def init(self) -> bool:
        """Initialize the renderer"""
        pass
        
    @abstractmethod
    def render(self, frame_data: Dict[str, Any]) -> None:
        """Render a frame"""
        pass
        
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources"""
        pass
