import os
import sys
import logging
import tkinter as tk
from tkinter import ttk
import imgui
from typing import Optional, Dict, Any
from contextlib import contextmanager
from pathlib import Path

# Fix relative imports
from ..base_screen import BaseScreen
from ..login_screen import LoginScreen  
from ..chat_screen import ChatScreen
from ..settings_screen import SettingsScreen
from ..data_screen import DataScreen

from ui.rendering import RendererFactory, RenderMode
from ui.themes.theme_manager import ThemeManager
from security.auth.auth_service import AuthService
from security.config.security_config import SecurityConfig
from core.session import SessionManager
from core.constants import SECURITY
from llm.pipeline import LLMPipeline

logger = logging.getLogger(__name__)

class UIState:
    def __init__(self):
        self.imgui_initialized = False
        self.context = None
        self.renderer = None
        self.is_initialized = False

class Screen:
    def __init__(self, width: int = 800, height: int = 600, title: str = "JARVIS"):
        self.width = width
        self.height = height
        self.title = title
        self.theme_manager = ThemeManager()
        self.renderer_manager = RendererFactory()
        self.auth_service = None
        self.session_manager = SessionManager()
        self.llm_pipeline = LLMPipeline()
        self.active_screen = "login"
        self.screens = {}
        self.interrupt_received = False
        self.should_quit = False
        self.current_screen: Optional[BaseScreen] = None
        self.is_initialized = False

    def initialize(self) -> bool:
        try:
            self.renderer_manager.initialize(self.width, self.height, self.title)
            self.theme_manager.load_themes()
            self.theme_manager.apply_theme("dark")
            self._init_screens()
            self.is_initialized = True
            return True
        except Exception as e:
            logger.error(f"Screen initialization failed: {e}")
            return self._fallback_init()

    def _fallback_init(self) -> bool:
        """Fallback to basic text mode"""
        try:
            self.renderer_manager.initialize_text_mode()
            self.is_initialized = True
            return True
        except Exception as e:
            logger.critical(f"Fallback initialization failed: {e}")
            return False

    def _init_screens(self):
        # Zorg dat LoginScreen de juiste dependency krijgt
        if self.auth_service is None:
            self.auth_service = AuthService()
        self.register_screen("login", LoginScreen(self.auth_service))
        self.register_screen("chat", ChatScreen(self.llm_pipeline))
        self.register_screen("settings", SettingsScreen())
        self.register_screen("data", DataScreen())
        self.active_screen = "login"

    def register_screen(self, name: str, screen: 'BaseScreen'):
        self.screens[name] = screen

    def switch_screen(self, screen_name: str):
        if screen_name in self.screens:
            if self.session_manager.validate_session() or screen_name == "login":
                self.active_screen = screen_name

    @contextmanager
    def frame(self):
        """Context manager for ImGui frames"""
        try:
            if self.ui_state.renderer and self.ui_state.imgui_initialized:
                self.ui_state.renderer.process_inputs()
                imgui.new_frame()
                yield
                imgui.render()
                self.ui_state.renderer.render(imgui.get_draw_data())
        except Exception as e:
            logger.error(f"Frame error: {e}")
            raise

    def render(self, frame_data: Dict[str, Any]) -> None:
        """Improved render with error handling"""
        if not self.ui_state.is_initialized:
            return

        try:
            with self.frame():
                current_screen = self.screens.get(self.active_screen)
                if current_screen:
                    current_screen.render(frame_data)
        except Exception as e:
            logger.error(f"Render error: {e}")

    def handle_input(self, input_data: Dict[str, Any]) -> None:
        current_screen = self.screens.get(self.active_screen)
        if current_screen:
            current_screen.handle_input(input_data)

    def add_message(self, message: str, is_user: bool) -> None:
        """Add message to current screen"""
        try:
            current_screen = self.screens.get(self.active_screen)
            if current_screen and hasattr(current_screen, 'add_message'):
                current_screen.add_message(message, is_user)
        except Exception as e:
            logger.error(f"Error adding message: {e}")

    def cleanup(self) -> None:
        if self.renderer:
            self.renderer.cleanup()
        self.is_initialized = False
        self.session_manager.cleanup()
        try:
            if imgui.get_current_context():
                imgui.destroy_context()
        except Exception as e:
            logger.error(f"Error destroying ImGui context: {e}")
        # Additional cleanup for tkinter if used
        try:
            if self.mode == "graphical" and self.is_initialized:
                for screen in self.screens.values():
                    screen.cleanup()
                if hasattr(self, 'root'):
                    self.root.destroy()
            self.is_initialized = False
        except Exception as e:
            logger.error(f"Error during screen cleanup: {e}")
