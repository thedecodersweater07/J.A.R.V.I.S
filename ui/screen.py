import os
import sys
import logging
import tkinter as tk
from tkinter import ttk
import imgui
from typing import Optional, Dict, Any
from contextlib import contextmanager
from pathlib import Path

from components.chat_history import ChatHistory
from components.chat_input import ChatInput
from components.status_bar import StatusBar
from themes.theme_manager import ThemeManager
from components.typing_interface import TypingInterface
from llm.pipeline import LLMPipeline
from llm.core.llm_core import LLMCore

# Import local modules
from .screens.base_screen import BaseScreen
from .screens.login_screen import LoginScreen
from .screens.chat_screen import ChatScreen
from .screens.settings_screen import SettingsScreen
from .screens.data_screen import DataScreen
from .screens.main_screen import MainScreen
from ui.rendering import RendererFactory, RenderMode
from security.auth.auth_service import AuthService
from security.config.security_config import SecurityConfig
from core.session import SessionManager
from core.constants import SECURITY
from ml.models.model_manager import ModelManager

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
        self.renderer = None
        self.ui_state = UIState()
        self.is_initialized = False
        self.should_quit = False
        self.interrupt_received = False
        self.auth_service = None
        self.session_manager = SessionManager()
        self.llm_pipeline = LLMPipeline()
        self.screens = {}
        self.active_screen = "main"  # Change default to main screen
        self.current_screen = None
        self.llm: Optional[LLMCore] = None
        self.model_manager: Optional[ModelManager] = None
        self.typing_interface = None
        self.chat_history = None

    def init(self) -> bool:
        try:
            self._init_imgui()
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
        
        self._init_security()
        self._init_screens()
        self._init_components()
        return self.is_initialized

    def _init_imgui(self) -> bool:
        """Initialize ImGui with proper error handling"""
        try:
            if not self.ui_state.imgui_initialized:
                self.ui_state.context = imgui.create_context()
                imgui.set_current_context(self.ui_state.context)
                self.ui_state.imgui_initialized = True
            return True
        except Exception as e:
            logger.error(f"ImGui initialization failed: {e}")
            return False

    def _init_security(self):
        security_config = SecurityConfig(
            jwt_secret=SECURITY["jwt_secret"],
            token_expiry_hours=SECURITY["token_expiry_hours"],
            max_login_attempts=SECURITY["max_login_attempts"],
            lockout_duration_minutes=SECURITY["lockout_duration_minutes"]
        )
        self.auth_service = AuthService(security_config)

    def _init_screens(self):
        """Initialize all screens"""
        self.register_screen("main", MainScreen())  # Register main first
        self.register_screen("login", LoginScreen(self.auth_service))
        self.register_screen("chat", ChatScreen(self.llm_pipeline))
        self.register_screen("settings", SettingsScreen())
        self.register_screen("data", DataScreen())
        
        # Set initial screen
        if self.screens:
            self.current_screen = self.screens["main"]

    def _init_components(self):
        self.typing_interface = TypingInterface(self.frame, self.handle_typed_input)
        self.typing_interface.pack(fill="x", pady=5)
        
        self.chat_history = ChatHistory(self.frame)
        self.chat_history.pack(fill="both", expand=True)

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

    def set_llm(self, llm: LLMCore):
        self.llm = llm

    def set_model_manager(self, model_manager: ModelManager):
        self.model_manager = model_manager

    def handle_typed_input(self, text: str):
        if self.llm:
            response = self.llm.generate_response(text)
            self.chat_history.add_message(text, is_user=True)
            self.typing_interface.simulate_typing(response)
            self.chat_history.add_message(response, is_user=False)

    def cleanup(self) -> None:
        """Clean up resources"""
        try:
            if hasattr(self, 'renderer') and self.renderer:
                self.renderer.cleanup()
            self.is_initialized = False
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
        if hasattr(self, 'session_manager'):
            self.session_manager.cleanup()
        try:
            if imgui.get_current_context():
                imgui.destroy_context()
        except Exception as e:
            logger.error(f"Error destroying ImGui context: {e}")
        
        # Cleanup screens
        if hasattr(self, 'screens'):
            for screen in self.screens.values():
                screen.cleanup()

    def should_exit(self) -> bool:
        """Check if application should exit"""
        return self.should_quit or self.interrupt_received
