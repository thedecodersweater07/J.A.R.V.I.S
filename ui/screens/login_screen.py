import imgui
from typing import Optional, Callable, Dict, Any
import logging
from datetime import datetime, timedelta
import socket
from security.auth.auth_service import AuthService
from ui.themes.theme_manager import ThemeManager
from .base_screen import BaseScreen

logger = logging.getLogger(__name__)

class LoginScreen(BaseScreen):
    def __init__(self, auth_service: AuthService):
        super().__init__()
        self.auth_service = auth_service
        self.username = ""
        self.password = ""
        self.error_message: Optional[str] = None
        self.success_callback: Optional[Callable] = None
        self.attempt_count = 0
        self.locked_until: Optional[datetime] = None
        self.theme = ThemeManager()
        self._setup_styling()

    def init(self) -> bool:
        self.initialized = True
        return True
    
    def handle_input(self, input_data: Dict[str, Any]) -> None:
        pass  # Login handled through imgui interface

    def render(self, frame_data: Dict[str, Any]) -> None:
        imgui.set_next_window_size(400, 200)
        imgui.set_next_window_centered()
        
        imgui.begin("Nova Industries Security", flags=imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE)
        
        try:
            if self._check_lockout():
                return

            self._render_logo()
            self._render_inputs()
            self._render_login_button()
            self._render_error()
            
        except Exception as e:
            logger.error(f"Error rendering login screen: {e}")
            self.error_message = "System error occurred"
            
        finally:
            imgui.end()

    def _setup_styling(self):
        """Initialize custom styling for login screen"""
        self.colors = {
            "normal": (0.2, 0.6, 1.0, 1.0),
            "error": (1.0, 0.3, 0.3, 1.0),
            "success": (0.3, 0.8, 0.3, 1.0)
        }

    def _check_lockout(self) -> bool:
        if self.locked_until and datetime.now() < self.locked_until:
            remaining = (self.locked_until - datetime.now()).seconds
            imgui.text_colored(f"Account locked. Try again in {remaining} seconds", *self.colors["error"])
            imgui.end()
            return True
        return False

    def _render_logo(self):
        imgui.dummy(0, 10)
        imgui.text_centered("Nova Industries")
        imgui.dummy(0, 10)

    def _render_inputs(self):
        # Username input with validation
        changed, self.username = imgui.input_text(
            "Username", 
            self.username, 
            256,
            flags=imgui.INPUT_TEXT_NO_SPACES
        )
        
        # Password input with masking
        changed, self.password = imgui.input_text(
            "Password", 
            self.password, 
            256,
            flags=imgui.INPUT_TEXT_PASSWORD
        )

    def _render_login_button(self):
        if imgui.button("Login", width=120):
            try:
                ip_address = socket.gethostbyname(socket.gethostname())
                token = self.auth_service.authenticate(self.username, self.password, ip_address)
                
                if token:
                    self._handle_success(token)
                else:
                    self._handle_failure()
                    
            except Exception as e:
                logger.error(f"Login error: {e}")
                self.error_message = "Authentication error occurred"

    def _handle_success(self, token: str):
        if self.success_callback:
            self.success_callback(token)
        self.attempt_count = 0
        self.error_message = None
        logger.info(f"Successful login for user: {self.username}")

    def _handle_failure(self):
        self.attempt_count += 1
        self.error_message = "Invalid credentials"
        if self.attempt_count >= 3:
            self.locked_until = datetime.now() + timedelta(minutes=15)
            logger.warning(f"Account locked for user: {self.username}")

    def _render_error(self):
        if self.error_message:
            imgui.text_colored(self.error_message, *self.colors["error"])
