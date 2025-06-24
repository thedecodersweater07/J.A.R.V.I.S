import tkinter as tk
from tkinter import ttk
from ui.screens.base_screen import BaseScreen

class VirtualEnvScreen(BaseScreen):
    def __init__(self, master=None):
        super().__init__()
        self.master = master
        self.initialized = False
        self.simulations = []
        self.error = None
        self.active_simulations = {}
        self.overall_progress = 0.0
        self.frame = None
        self.progress_var = tk.DoubleVar(value=0.0)
        self.sim_tree = None

    def init(self) -> bool:
        try:
            self.frame = tk.Frame(self.master)
            self.frame.pack(fill=tk.BOTH, expand=True)
            # Overall progress bar
            tk.Label(self.frame, text="Overall Progress").pack(pady=(10, 0))
            progress = ttk.Progressbar(self.frame, variable=self.progress_var, maximum=1.0, length=400)
            progress.pack(pady=(0, 10))
            # Active simulations tree
            self.sim_tree = ttk.Treeview(self.frame, columns=("Topic", "Steps", "Progress"), show="headings")
            self.sim_tree.heading("Topic", text="Topic")
            self.sim_tree.heading("Steps", text="Steps")
            self.sim_tree.heading("Progress", text="Progress")
            self.sim_tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            self.initialized = True
            return True
        except Exception as e:
            print(f"VirtualEnvScreen init failed: {e}")
            return False

    def update_simulation_status(self, sim_id: str, status: dict):
        self.active_simulations[sim_id] = status
        self.refresh_simulations()

    def refresh_simulations(self):
        if not self.sim_tree:
            return
        for i in self.sim_tree.get_children():
            self.sim_tree.delete(i)
        for sim_id, status in self.active_simulations.items():
            topic = status.get('topic', 'Unknown')
            steps = f"{status.get('current_step', 0)}/{status.get('total_steps', 0)}"
            progress = f"{status.get('progress', 0.0) * 100:.1f}%"
            self.sim_tree.insert('', 'end', values=(topic, steps, progress))
        self.progress_var.set(self.overall_progress)

    def render(self, frame_data: dict) -> None:
        self.refresh_simulations()

    def cleanup(self) -> None:
        if self.frame:
            self.frame.destroy()
        self.initialized = False
