import os
import sys
import logging
from typing import Optional, Dict, Any
from contextlib import contextmanager
from pathlib import Path

# Replace main import with core.constants
from core.constants import OPENGL_AVAILABLE

# Import our ImGui manager for safer handling
from .rendering.imGUI_manager import ImGuiManager

# Rest of the imports
from .components.chat_history import ChatHistory
from .components.chat_input import ChatInput
from .components.status_bar import StatusBar
from .components.typing_interface import TypingInterface
from .themes.theme_manager import ThemeManager
from .rendering import RendererFactory, RenderMode

# Import screens
from .screens.base_screen import BaseScreen
from .screens.login_screen import LoginScreen
from .screens.chat_screen import ChatScreen
from .screens.settings_screen import SettingsScreen
from .screens.data_screen import DataScreen
from .screens.main_screen import MainScreen

# Import LLM components
from llm.core import LLMCore
from llm.pipeline import LLMPipeline

# Import other dependencies
from security.auth.auth_service import AuthService
from security.config.security_config import SecurityConfig
from core.session import SessionManager
from core.constants import SECURITY
from ml.models.model_manager import ModelManager

logger = logging.getLogger(__name__)

class UIState:
    def __init__(self):
        self.imgui_manager = ImGuiManager()
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
        self.active_screen = "main"
        self.current_screen = None
        self.llm: Optional[LLMCore] = None
        self.model_manager: Optional[ModelManager] = None
        self.typing_interface = None
        self.chat_history = None
        
        # Check early if OpenGL is available to avoid errors
        self.render_mode = RenderMode.OPENGL if self._check_opengl() else RenderMode.TEXT
        logger.info(f"Using render mode: {self.render_mode}")

    def _check_opengl(self) -> bool:
        """Check if OpenGL is actually available by trying to import GLFW."""
        try:
            import glfw
            return OPENGL_AVAILABLE and glfw.init()
        except ImportError:
            logger.warning("GLFW not available, falling back to text mode")
            return False
        except Exception as e:
            logger.warning(f"OpenGL initialization error: {e}, falling back to text mode")
            return False

    def init(self) -> bool:
        try:
            # Initialize appropriate rendering mode first
            if self.render_mode == RenderMode.OPENGL:
                if not self._init_opengl():
                    logger.warning("OpenGL initialization failed, falling back to text mode")
                    self.render_mode = RenderMode.TEXT
                    return self._init_text_mode()
                return True
            else:
                return self._init_text_mode()
                
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            return self._fallback_init()
            
    def _init_opengl(self) -> bool:
        """Initialize OpenGL rendering with ImGui"""
        try:
            # Initialize ImGui first using our manager
            if not self.ui_state.imgui_manager.init():
                logger.error("ImGui initialization failed")
                return False
            
            # Create OpenGL renderer
            self.renderer = RendererFactory.create_renderer(
                RenderMode.OPENGL,
                {"width": self.width, "height": self.height, "title": self.title}
            )
            
            if not self.renderer.init():
                return False
                
            self._init_security()
            self._init_screens()
            
            self.is_initialized = True
            return True
        except Exception as e:
            logger.error(f"OpenGL initialization failed: {e}")
            return False

    def _init_text_mode(self) -> bool:
        """Initialize text-based UI without OpenGL/ImGui"""
        try:
            # Create text-based renderer
            self.renderer = RendererFactory.create_renderer(
                RenderMode.TEXT,
                {"width": self.width, "height": self.height, "title": self.title}
            )
            
            if not self.renderer.init():
                return False
                
            self._init_security()
            self._init_screens()
            
            self.is_initialized = True
            return True
        except Exception as e:
            logger.error(f"Text mode initialization failed: {e}")
            return False
            
    def _fallback_init(self) -> bool:
        """Last resort minimal initialization"""
        try:
            logger.warning("Using minimal fallback initialization")
            # Minimal init without any UI components
            self._init_security()
            self.is_initialized = True
            return True
        except Exception as e:
            logger.error(f"Fallback initialization failed: {e}")
            return False

    def _init_security(self):
        try:
            security_config = SecurityConfig(
                jwt_secret=SECURITY["jwt_secret"],
                token_expiry_hours=SECURITY["token_expiry_hours"],
                max_login_attempts=SECURITY["max_login_attempts"],
                lockout_duration_minutes=SECURITY["lockout_duration_minutes"]
            )
            self.auth_service = AuthService(security_config)
        except Exception as e:
            logger.error(f"Security initialization error: {e}")
            # Continue without auth if necessary
            self.auth_service = None

    def _init_screens(self):
        """Initialize all screens"""
        try:
            self.register_screen("main", MainScreen())
            self.register_screen("login", LoginScreen(self.auth_service))
            self.register_screen("chat", ChatScreen(self.llm_pipeline))
            self.register_screen("settings", SettingsScreen())
            self.register_screen("data", DataScreen())
            
            # Set initial screen
            if self.screens:
                self.current_screen = self.screens["main"]
        except Exception as e:
            logger.error(f"Screen initialization error: {e}")

    def register_screen(self, name: str, screen: 'BaseScreen'):
        self.screens[name] = screen
        screen.set_parent(self)  # Ensure screens can access the parent Screen

    def switch_screen(self, screen_name: str):
        if screen_name in self.screens:
            if self.session_manager.validate_session() or screen_name == "login":
                self.active_screen = screen_name
                logger.info(f"Switched to screen: {screen_name}")

    @contextmanager
    def frame(self):
        """Context manager for rendering frames based on mode"""
        frame_processed = False
        try:
            if self.render_mode == RenderMode.OPENGL and self.ui_state.imgui_manager.is_initialized:
                # Process inputs before starting the frame
                if self.renderer:
                    self.renderer.process_inputs()
                
                # Use the ImGui manager for safe frame handling
                with self.ui_state.imgui_manager.frame():
                    frame_processed = True
                    yield
                    
                # Only render if we have a renderer and ImGui is properly initialized
                import imgui
                if self.renderer and imgui.get_draw_data():
                    self.renderer.render(imgui.get_draw_data())
            else:
                # Text mode rendering
                frame_processed = True
                yield
                if self.renderer:
                    self.renderer.render(None)  # No ImGui draw data in text mode
        except Exception as e:
            logger.error(f"Frame error: {e}")
            # Always yield to avoid breaking context manager if not already yielded
            if not frame_processed:
                yield

    def render(self, frame_data: Dict[str, Any]) -> None:
        """Render the current screen"""
        if not self.is_initialized:
            return

        try:
            with self.frame():
                current_screen = self.screens.get(self.active_screen)
                if current_screen:
                    current_screen.render(frame_data)
        except Exception as e:
            logger.error(f"Render error: {e}")

    def handle_input(self, input_data: Dict[str, Any]) -> None:
        # Add quit handling
        if input_data.get("type") == "quit":
            self.should_quit = True
            
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

    def process_frame(self, frame_data: Dict[str, Any]) -> bool:
        """Process a single frame. Returns False if window should close."""
        if not self.is_initialized:
            return False

        try:
            # Process window events
            if self.renderer:
                self.renderer.process_events()
                if self.renderer.window_should_close():
                    self.should_quit = True
                    return False
                
            # Handle queued inputs
            self._process_inputs()
                
            # Render frame
            self.render(frame_data)
                    
            return not self.should_quit
            
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            return False
            
    def _process_inputs(self):
        """Process queued input events"""
        try:
            if self.renderer:
                for event in self.renderer.get_input_events():
                    if event.type == "quit":
                        self.should_quit = True
                    elif event.type == "keydown" and event.key == "escape":
                        self.should_quit = True
                    else:
                        self.handle_input(event)
        except Exception as e:
            logger.error(f"Input processing error: {e}")
            
    def handle_typed_input(self, text: str):
        if self.llm and not self.should_quit:
            try:
                response = self.llm.generate_response(text)
                if hasattr(self, 'chat_history') and self.chat_history:
                    self.chat_history.add_message(text, is_user=True)
                    if hasattr(self, 'typing_interface') and self.typing_interface:
                        self.typing_interface.simulate_typing(response)
                    self.chat_history.add_assistant_message(response)
            except Exception as e:
                logger.error(f"Error handling typed input: {e}")

    def cleanup(self) -> None:
        """Clean up resources in a safe manner"""
        try:
            logger.info("Starting cleanup...")
            
            # Clean up renderer first
            if self.renderer:
                try:
                    self.renderer.cleanup()
                except Exception as e:
                    logger.error(f"Error cleaning up renderer: {e}")
            
            # Clean up ImGui after renderer
            if self.ui_state.imgui_manager:
                try:
                    self.ui_state.imgui_manager.cleanup()
                except Exception as e:
                    logger.error(f"Error cleaning up ImGui manager: {e}")
                    
            self.is_initialized = False
            
            # Clean up session
            if self.session_manager:
                try:
                    self.session_manager.cleanup()
                except Exception as e:
                    logger.error(f"Error cleaning up session manager: {e}")
                    
            logger.info("Cleanup complete")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def set_should_quit(self, value: bool):
        """Set the should_quit flag"""
        self.should_quit = value