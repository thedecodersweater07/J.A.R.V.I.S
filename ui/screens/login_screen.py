import tkinter as tk
from tkinter import messagebox
from security.auth.auth_service import AuthService
from ui.themes.theme_manager import ThemeManager
from .base_screen import BaseScreen
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class LoginScreen(BaseScreen):
    def __init__(self, auth_service: AuthService, master=None):
        super().__init__()
        self.master = master
        self.initialized = False
        self.auth_service = auth_service
        self.username = ""
        self.password = ""
        self.error_message = None
        self.success_callback = None
        self.attempt_count = 0
        self.locked_until = None
        self.theme = ThemeManager()
        self.frame = None
        self.username_entry = None
        self.password_entry = None
        self.status_label = None

    def init(self) -> bool:
        try:
            self.frame = tk.Frame(self.master)
            self.frame.pack(fill=tk.BOTH, expand=True)
            tk.Label(self.frame, text="Login", font=("Arial", 16, "bold")).pack(pady=10)
            tk.Label(self.frame, text="Username:").pack()
            self.username_entry = tk.Entry(self.frame)
            self.username_entry.pack()
            tk.Label(self.frame, text="Password:").pack()
            self.password_entry = tk.Entry(self.frame, show="*")
            self.password_entry.pack()
            tk.Button(self.frame, text="Login", command=self._on_login).pack(pady=10)
            self.status_label = tk.Label(self.frame, text="", fg="red")
            self.status_label.pack()
            self.initialized = True
            return True
        except Exception as e:
            logger.error(f"LoginScreen init failed: {e}")
            return False

    def _on_login(self):
        if not self.username_entry or not self.password_entry or not self.status_label:
            return
        username = self.username_entry.get()
        password = self.password_entry.get()
        if self.locked_until and datetime.now() < self.locked_until:
            self.status_label.config(text=f"Locked. Try again at {self.locked_until.strftime('%H:%M:%S')}")
            return
        if self.auth_service.authenticate(username, password):
            self.status_label.config(text="Login successful!", fg="green")
            if self.success_callback:
                self.success_callback()
        else:
            self.attempt_count += 1
            self.status_label.config(text="Login failed.", fg="red")
            if self.attempt_count >= 3:
                self.locked_until = datetime.now() + timedelta(minutes=1)
                self.status_label.config(text="Too many attempts. Locked for 1 minute.")

    def handle_input(self, input_data: dict) -> None:
        pass

    def render(self, frame_data: dict) -> None:
        pass

    def cleanup(self) -> None:
        if self.frame:
            self.frame.destroy()
        self.initialized = False
