from typing import Callable, Optional
import queue
import logging
import tkinter as tk

class TextInputHandler:
    def __init__(self):
        self.input_queue = queue.Queue()
        self.callbacks = []
        self.logger = logging.getLogger(__name__)
        
    def register_callback(self, callback: Callable[[str], None]):
        """Registreer callback voor tekstinvoer verwerking"""
        self.callbacks.append(callback)
        
    def handle_input(self, text: str) -> bool:
        """Verwerk binnenkomende tekstinvoer"""
        try:
            self.input_queue.put(text)
            for callback in self.callbacks:
                callback(text)
            return True
        except Exception:
            return False

    def get_next_input(self) -> Optional[str]:
        """Haal volgende tekstinvoer op uit queue"""
        try:
            return self.input_queue.get_nowait()
        except queue.Empty:
            return None

class TextInput(tk.Frame):
    def __init__(self, master=None, on_submit=None, **kwargs):
        super().__init__(master, **kwargs)
        self.input_var = tk.StringVar()
        self.entry = tk.Entry(self, textvariable=self.input_var, font=("Consolas", 12), bg="#23272e", fg="#e8eaed", insertbackground="#e8eaed", bd=0, relief=tk.FLAT)
        self.entry.pack(fill=tk.X, expand=True, ipady=6)
        self.entry.bind('<Return>', self._on_submit)
        self.on_submit = on_submit

    def _on_submit(self, event=None):
        text = self.input_var.get().strip()
        if text and self.on_submit:
            self.on_submit(text)
        self.input_var.set("")
