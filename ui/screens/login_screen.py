import imgui
from typing import Optional, Callable
import logging

class LoginScreen:
    def __init__(self, auth_service):
        self.auth_service = auth_service
        self.username = ""
        self.password = ""
        self.error_message: Optional[str] = None
        self.success_callback: Optional[Callable] = None
        
    def render(self):
        imgui.set_next_window_size(400, 200)
        imgui.set_next_window_centered()
        
        imgui.begin("JARVIS Login", flags=imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE)
        
        # Username input
        changed, self.username = imgui.input_text("Username", self.username, 256)
        
        # Password input
        changed, self.password = imgui.input_text(
            "Password", self.password, 256,
            flags=imgui.INPUT_TEXT_PASSWORD
        )
        
        if imgui.button("Login", width=120):
            token = self.auth_service.authenticate(self.username, self.password)
            if token:
                if self.success_callback:
                    self.success_callback(token)
            else:
                self.error_message = "Invalid credentials"
                
        if self.error_message:
            imgui.text_colored(self.error_message, 1, 0, 0)
            
        imgui.end()
