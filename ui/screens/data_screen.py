import tkinter as tk
from tkinter import ttk
from .base_screen import BaseScreen
from db.manager import DatabaseManager
import logging

logger = logging.getLogger(__name__)

class DataScreen(BaseScreen):
    def __init__(self, master=None):
        super().__init__()
        self.master = master
        self.db = DatabaseManager()
        self.metrics = {}
        self.initialized = False
        self.frame = None
        self.metrics_tree = None

    def init(self) -> bool:
        try:
            self.frame = tk.Frame(self.master)
            self.frame.pack(fill=tk.BOTH, expand=True)
            tk.Button(self.frame, text="Refresh Data", command=self.load_data).pack(pady=10)
            self.metrics_tree = ttk.Treeview(self.frame, columns=("Metric", "Value"), show="headings")
            self.metrics_tree.heading("Metric", text="Metric")
            self.metrics_tree.heading("Value", text="Value")
            self.metrics_tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            self.initialized = True
            return True
        except Exception as e:
            logger.error(f"Failed to initialize data screen: {e}")
            return False

    def load_data(self):
        try:
            # Example: fetch metrics from db (fallback: show connection status)
            self.metrics = {"DB Connection": "OK" if self.db.get_connection() else "Unavailable"}
            self.refresh_metrics()
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            self.metrics = {"DB Connection": "Error"}
            self.refresh_metrics()

    def refresh_metrics(self):
        if not self.metrics_tree:
            return
        for i in self.metrics_tree.get_children():
            self.metrics_tree.delete(i)
        if self.metrics:
            for key, value in self.metrics.items():
                self.metrics_tree.insert('', 'end', values=(key, value))
        else:
            self.metrics_tree.insert('', 'end', values=("No data available", ""))

    def render(self, frame_data: dict) -> None:
        self.refresh_metrics()

    def cleanup(self) -> None:
        if self.frame:
            self.frame.destroy()
        self.initialized = False
