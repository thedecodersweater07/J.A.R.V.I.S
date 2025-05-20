import os
import sys
import logging
from typing import Optional, Dict, Any
from contextlib import contextmanager
from pathlib import Path

from .rendering.renderer_factory import RendererFactory, RenderMode
from .rendering.imGUI_manager import ImGuiManager

logger = logging.getLogger(__name__)

class UIState:
    def __init__(self):
        self.imgui_manager = None
        self.renderer = None
        self.is_initialized = False

class Screen:
    def __init__(self, width: int = 800, height: int = 600, title: str = "JARVIS"):
        self.width = width
        self.height = height
        self.title = title
        self.ui_state = UIState()
        self.is_initialized = False
        self.should_quit = False
        self.active_screen = "main"
        self.screens = {}

    def init(self) -> bool:
        """Initialize the screen and renderers"""
        try:
            # Create renderer first
            self.ui_state.renderer = RendererFactory.create_renderer(
                RenderMode.OPENGL if self._check_opengl() else RenderMode.TEXT,
                {
                    "width": self.width,
                    "height": self.height,
                    "title": self.title
                }
            )
            
            if not self.ui_state.renderer.init():
                logger.error("Failed to initialize renderer")
                return False
                
            # Create ImGui manager after renderer
            self.ui_state.imgui_manager = ImGuiManager()
            if not self.ui_state.imgui_manager.init(self.ui_state.renderer.window):
                logger.error("Failed to initialize ImGui manager")
                return False
                
            self._init_screens()
            self.is_initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Screen initialization failed: {e}")
            return False

    def _check_opengl(self) -> bool:
        try:
            import glfw
            return glfw.init()
        except ImportError:
            return False

    def render(self, frame_data: Dict[str, Any]) -> None:
        if not self.is_initialized:
            return

        try:
            # Use ImGui manager's frame context
            with self.ui_state.imgui_manager.frame():
                current_screen = self.screens.get(self.active_screen)
                if current_screen:
                    current_screen.render({
                        **frame_data,
                        "imgui_manager": self.ui_state.imgui_manager
                    })
                    
        except Exception as e:
            logger.error(f"Render error: {e}")

    def cleanup(self) -> None:
        """Clean up resources in correct order"""
        try:
            logger.info("Starting screen cleanup...")
            
            # Clean up ImGui manager first
            if self.ui_state.imgui_manager:
                self.ui_state.imgui_manager.cleanup()
                self.ui_state.imgui_manager = None
                
            # Then clean up renderer
            if self.ui_state.renderer:
                self.ui_state.renderer.cleanup()
                self.ui_state.renderer = None
                
            self.is_initialized = False
            logger.info("Screen cleanup complete")
            
        except Exception as e:
            logger.error(f"Error during screen cleanup: {e}")

    def _init_screens(self):
        """Initialize different screen views"""
        self.screens = {
            'main': {'active': True},
            'settings': {'active': False},
            'debug': {'active': False}
        }

    def process_frame(self, data: dict) -> bool:
        """Process a single frame with the given data"""
        try:
            # Basic frame processing
            if self.ui_state.renderer:
                self.ui_state.renderer.render(data)
            return not self.should_quit
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return True