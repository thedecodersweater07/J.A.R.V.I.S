import tkinter as tk
from tkinter import ttk, messagebox
from .base_screen import BaseScreen
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class SettingsScreen(BaseScreen):
    def __init__(self, master=None):
        super().__init__()
        self.master = master
        self.initialized = False
        self.settings = {
            "llm": {
                "model": "gpt2",
                "max_length": 1024,
                "temperature": 0.7,
                "gpu_enabled": True
            },
            "ui": {
                "theme": "dark",
                "font_size": 12,
                "window_opacity": 1.0
            },
            "audio": {
                "input_device": "default",
                "output_device": "default",
                "volume": 1.0
            },
            "security": {
                "session_timeout": 12,
                "max_login_attempts": 3
            }
        }
        self.config_path = Path(__file__).parent.parent.parent / "config" / "settings.json"
        self.frame = None
        self.entries = {}
        self.load_settings()

    def init(self) -> bool:
        try:
            self.frame = tk.Frame(self.master)
            self.frame.pack(fill=tk.BOTH, expand=True)
            row = 0
            for section, opts in self.settings.items():
                tk.Label(self.frame, text=section.upper(), font=("Arial", 12, "bold")).grid(row=row, column=0, sticky="w", pady=(10,0))
                row += 1
                for key, value in opts.items():
                    tk.Label(self.frame, text=key).grid(row=row, column=0, sticky="e")
                    entry = tk.Entry(self.frame)
                    entry.insert(0, str(value))
                    entry.grid(row=row, column=1, sticky="w")
                    self.entries[f"{section}.{key}"] = entry
                    row += 1
            tk.Button(self.frame, text="Save", command=self.save_settings).grid(row=row, column=0, pady=10)
            tk.Button(self.frame, text="Reset", command=self.reset_settings).grid(row=row, column=1, pady=10)
            self.initialized = True
            return True
        except Exception as e:
            logger.error(f"SettingsScreen init failed: {e}")
            return False

    def load_settings(self):
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    self.settings = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load settings: {e}")

    def save_settings(self) -> bool:
        try:
            for section, opts in self.settings.items():
                for key in opts:
                    entry = self.entries.get(f"{section}.{key}")
                    if entry:
                        val = entry.get()
                        try:
                            val = json.loads(val)
                        except Exception:
                            pass
                        self.settings[section][key] = val
            with open(self.config_path, 'w') as f:
                json.dump(self.settings, f, indent=2)
            messagebox.showinfo("Settings", "Settings saved successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to save settings: {e}")
            messagebox.showerror("Settings", f"Failed to save settings: {e}")
            return False

    def reset_settings(self) -> bool:
        self.load_settings()
        for section, opts in self.settings.items():
            for key, value in opts.items():
                entry = self.entries.get(f"{section}.{key}")
                if entry:
                    entry.delete(0, tk.END)
                    entry.insert(0, str(value))
        messagebox.showinfo("Settings", "Settings reset to last saved values.")
        return True

    def render(self, frame_data: dict) -> None:
        pass

    def cleanup(self) -> None:
        if self.frame:
            self.frame.destroy()
        self.initialized = False
