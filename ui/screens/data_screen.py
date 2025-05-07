from .base_screen import BaseScreen
from db.database_manager import DatabaseManager
import tkinter as tk
from tkinter import ttk

class DataScreen(BaseScreen):
    def __init__(self, master=None):
        super().__init__(master)
        self.db = DatabaseManager()
        self.setup_ui()
        
    def setup_ui(self):
        # Stats section
        self.stats_frame = ttk.LabelFrame(self, text="Statistics")
        self.stats_frame.pack(pady=10, padx=5, fill="x")
        
        # Data viewer
        self.data_tree = ttk.Treeview(self)
        self.data_tree.pack(pady=10, fill="both", expand=True)
        
        # Load data button
        self.load_btn = ttk.Button(self, text="Load Data", command=self.load_data)
        self.load_btn.pack(pady=5)
        
    def load_data(self):
        metrics = self.db.load_metrics()
        self.update_stats(metrics)
