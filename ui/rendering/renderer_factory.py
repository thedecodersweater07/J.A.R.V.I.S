import logging
from enum import Enum
from typing import Dict, Any, Optional

# Import renderers
from .opengl_renderer import OpenGLRenderer
from .text_renderer import TextRenderer

logger = logging.getLogger(__name__)

class RenderMode(Enum):
    """Rendering modes supported by the application"""
    OPENGL = "opengl"
    TEXT = "text"

class RendererFactory:
    """Factory class for creating appropriate renderers"""
    
    @staticmethod
    def create_renderer(mode: RenderMode, config: Dict[str, Any]):
        """
        Create and return appropriate renderer based on mode
        
        Args:
            mode: The rendering mode to use
            config: Configuration options for the renderer
            
        Returns:
            A renderer instance
        """
        try:
            if mode == RenderMode.OPENGL:
                logger.info("Creating OpenGL renderer")
                return OpenGLRenderer(
                    width=config.get("width", 800),
                    height=config.get("height", 600),
                    title=config.get("title", "JARVIS")
                )
            elif mode == RenderMode.TEXT:
                logger.info("Creating text renderer")
                return TextRenderer(
                    width=config.get("width", 80),
                    height=config.get("height", 24),
                    title=config.get("title", "JARVIS")
                )
            else:
                logger.error(f"Unsupported render mode: {mode}")
                # Fallback to text renderer
                return TextRenderer(
                    width=config.get("width", 80),
                    height=config.get("height", 24),
                    title=config.get("title", "JARVIS")
                )
        except Exception as e:
            logger.error(f"Error creating renderer: {e}")
            # Fallback to minimal text renderer
            return TextRenderer(
                width=config.get("width", 80),
                height=config.get("height", 24),
                title=config.get("title", "JARVIS")
            )