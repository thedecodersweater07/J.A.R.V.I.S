import logging
from typing import Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)

class RenderMode(Enum):
    OPENGL = "opengl"
    TEXT = "text"

class RendererFactory:
    @staticmethod
    def create_renderer(mode: RenderMode, config: Dict[str, Any]):
        if mode == RenderMode.OPENGL:
            try:
                from .opengl_renderer import OpenGLRenderer
                return OpenGLRenderer(config)
            except ImportError:
                logger.warning("OpenGL not available")
                mode = RenderMode.TEXT
                
        if mode == RenderMode.TEXT:
            from .text_renderer import TextRenderer
            return TextRenderer(config)
        
        raise ValueError(f"Unsupported render mode: {mode}")
