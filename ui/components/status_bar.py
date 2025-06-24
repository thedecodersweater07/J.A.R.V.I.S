import tkinter as tk

class StatusBar(tk.Frame):
    def __init__(self, master=None, theme_manager=None, **kwargs):
        super().__init__(master, **kwargs)
        self.theme_manager = theme_manager
        self.status_var = tk.StringVar(value="Status: Ready")
        self.label = tk.Label(self, textvariable=self.status_var, anchor='w', font=("Consolas", 10), bg="#23272e", fg="#00d4ff", bd=0, relief=tk.FLAT)
        self.label.pack(fill=tk.X)

    def set_status(self, text: str):
        self.status_var.set(text)

    def show_error(self, error: str):
        self.status_var.set(f"FOUT: {error}")
        self.label.config(fg="#ff4444")

    def apply_theme(self):
        # Optionally update colors/fonts
        pass
