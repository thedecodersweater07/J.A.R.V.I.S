import logging
from typing import Optional
from datetime import datetime
import tkinter as tk

class ConversationScreen:
    def __init__(self, master=None):
        self.master = master
        self.frame = None
        self.header_label = None
        self.prompt_label = None
        self.response_label = None
        self.error_label = None
        self.logger = logging.getLogger(__name__)
        
    def init(self) -> bool:
        try:
            self.frame = tk.Frame(self.master)
            self.frame.pack(fill=tk.BOTH, expand=True)
            self.header_label = tk.Label(self.frame, text=f"JARVIS Conversation Interface - {datetime.now().strftime('%H:%M:%S')}", font=("Arial", 12, "bold"))
            self.header_label.pack(pady=10)
            self.prompt_label = tk.Label(self.frame, text="JARVIS > ", font=("Arial", 10))
            self.prompt_label.pack(pady=5)
            self.response_label = tk.Label(self.frame, text="", font=("Arial", 10))
            self.response_label.pack(pady=5)
            self.error_label = tk.Label(self.frame, text="", fg="red", font=("Arial", 10))
            self.error_label.pack(pady=5)
            return True
        except Exception as e:
            print(f"ConversationScreen init failed: {e}")
            return False

    def display_header(self):
        if self.header_label:
            self.header_label.config(text=f"JARVIS Conversation Interface - {datetime.now().strftime('%H:%M:%S')}")

    def display_prompt(self):
        if self.prompt_label:
            self.prompt_label.config(text="JARVIS > ")

    def display_response(self, response: str):
        if self.response_label:
            self.response_label.config(text=f"JARVIS: {response}")

    def display_error(self, error: Optional[str] = None):
        if self.error_label:
            if error:
                self.error_label.config(text=f"Error: {error}")
            else:
                self.error_label.config(text="JARVIS: Sorry, er is een fout opgetreden.")

    def clear_screen(self):
        if self.response_label:
            self.response_label.config(text="")
        if self.error_label:
            self.error_label.config(text="")

    def cleanup(self):
        if self.frame:
            self.frame.destroy()
