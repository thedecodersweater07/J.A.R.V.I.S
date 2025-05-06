from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

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
    
    @contextmanager
    def frame(self):
        """Context manager for frame rendering"""
        try:
            self.begin_frame()
            yield
            self.end_frame()
        except Exception as e:
            logging.error(f"Frame rendering error: {e}")
            raise
    @abstractmethod
    def begin_frame(self):
        """Setup for frame rendering"""
        pass
        
    @abstractmethod
    def end_frame(self):
        """Cleanup after frame rendering"""
        pass
