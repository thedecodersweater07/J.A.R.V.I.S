import tkinter as tk
from tkinter import ttk
from .base_screen import BaseScreen
import logging

logger = logging.getLogger(__name__)

class MainScreen(BaseScreen):
    def __init__(self, master=None):
        super().__init__()
        self.master = master
        self.initialized = False
        self.width = 800
        self.height = 600
        self.frame = None

    def init(self) -> bool:
        try:
            self.frame = tk.Frame(self.master, width=self.width, height=self.height)
            self.frame.pack(fill=tk.BOTH, expand=True)
            label = tk.Label(self.frame, text="Welcome to JARVIS", font=("Arial", 18, "bold"))
            label.pack(pady=30)
            sep = ttk.Separator(self.frame, orient='horizontal')
            sep.pack(fill=tk.X, padx=20, pady=10)
            self.initialized = True
            return True
        except Exception as e:
            logger.error(f"MainScreen init failed: {e}")
            return False

    def render(self, frame_data: dict) -> None:
        pass

    def cleanup(self) -> None:
        if self.frame:
            self.frame.destroy()
        self.initialized = False
