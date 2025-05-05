import logging
from typing import Optional, Dict, Any
from .rendering import RendererFactory, RenderMode

logger = logging.getLogger(__name__)

class Screen:
    def __init__(self, width: int = 800, height: int = 600, title: str = "JARVIS Interface"):
        self.width = width
        self.height = height
        self.title = title
        self.renderer = None
        self.render_mode = RenderMode.OPENGL
        self.is_initialized = False
        
    def init(self) -> bool:
        try:
            # Try OpenGL first
            self.renderer = RendererFactory.create_renderer(
                RenderMode.OPENGL,
                {"width": self.width, "height": self.height, "title": self.title}
            )
            self.is_initialized = self.renderer.init()
        except ImportError as e:
            logger.warning(f"OpenGL not available: {e}, falling back to text mode")
            self.render_mode = RenderMode.TEXT
            self.renderer = RendererFactory.create_renderer(
                RenderMode.TEXT,
                {"width": self.width, "height": self.height}
            )
            self.is_initialized = self.renderer.init()
            
        return self.is_initialized

    def render(self, frame_data: Dict[str, Any]) -> None:
        if not self.is_initialized:
            return
        self.renderer.render(frame_data)

    def cleanup(self) -> None:
        if self.renderer:
            self.renderer.cleanup()
        self.is_initialized = False
