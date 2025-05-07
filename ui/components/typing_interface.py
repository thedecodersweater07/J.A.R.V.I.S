import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional
import time

class TypingInterface(ttk.Frame):
    def __init__(self, master, callback: Optional[Callable] = None):
        super().__init__(master)
        self.callback = callback
        self.typing_speed = 50  # ms between characters
        self.setup_ui()

    def setup_ui(self):
        self.text_widget = tk.Text(self, height=3, wrap=tk.WORD)
        self.text_widget.pack(fill=tk.BOTH, expand=True)
        self.text_widget.bind('<Return>', self.handle_return)
        
        self.status_label = ttk.Label(self, text="")
        self.status_label.pack()

    def simulate_typing(self, text: str):
        """Simulate typing animation"""
        self.text_widget.delete(1.0, tk.END)
        for char in text:
            self.text_widget.insert(tk.END, char)
            self.text_widget.see(tk.END)
            self.text_widget.update()
            time.sleep(self.typing_speed / 1000)

    def handle_return(self, event):
        if self.callback:
            text = self.text_widget.get(1.0, tk.END).strip()
            self.callback(text)
            self.text_widget.delete(1.0, tk.END)
        return 'break'  # Prevent default newline
