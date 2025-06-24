import tkinter as tk
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class ChatInput(tk.Frame):
    def __init__(self, master=None, theme_manager=None, on_send=None, on_voice=None, **kwargs):
        super().__init__(master, **kwargs)
        self.theme_manager = theme_manager
        self.input_var = tk.StringVar()
        self.entry = tk.Entry(self, textvariable=self.input_var, font=("Consolas", 12), bg="#23272e", fg="#e8eaed", insertbackground="#e8eaed", bd=0, relief=tk.FLAT)
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,8), ipady=6)
        self.entry.bind('<Return>', self._on_send)
        self.send_button = tk.Button(self, text="Verzend", command=self._on_send, font=("Consolas", 12, "bold"), bg="#00d4ff", fg="#181c20", bd=0, relief=tk.FLAT, activebackground="#0099cc")
        self.send_button.pack(side=tk.RIGHT)
        self.on_send = on_send
        self.on_voice = on_voice

    def _on_send(self, event=None):
        message = self.input_var.get().strip()
        if message:
            if self.on_send:
                self.on_send(message)
            self.input_var.set("")

    def apply_theme(self):
        # Optionally update colors/fonts
        pass
