import imgui
from typing import Dict, Any
import glfw
from OpenGL import GL
from ui.screens.base_screen import BaseScreen

class VirtualEnvScreen(BaseScreen):
    def __init__(self):
        super().__init__()
        self.initialized = False
        self.simulations = []
        self.error = None
        self.active_simulations = {}
        self.overall_progress = 0.0
        
    def update_simulation_status(self, sim_id: str, status: Dict[str, Any]):
        self.active_simulations[sim_id] = status
        
    def render(self, frame_data: Dict[str, Any]) -> None:
        imgui.begin("Virtual Environment Monitor", flags=imgui.WINDOW_NO_COLLAPSE)
        
        # Overall progress
        imgui.text("Overall Progress")
        imgui.progress_bar(
            self.overall_progress, 
            (-1, 0), 
            f"{self.overall_progress * 100:.1f}%"
        )
        
        imgui.separator()
        
        # Active simulations
        if imgui.collapsing_header("Active Simulations")[0]:
            for sim_id, status in self.active_simulations.items():
                if imgui.tree_node(f"Simulation {sim_id}"):
                    imgui.text(f"Topic: {status.get('topic', 'Unknown')}")
                    imgui.text(f"Steps: {status.get('current_step', 0)}/{status.get('total_steps', 0)}")
                    imgui.progress_bar(
                        status.get('progress', 0.0),
                        (-1, 0),
                        f"{status.get('progress', 0.0) * 100:.1f}%"
                    )
                    imgui.text(f"Status: {status.get('status', 'Unknown')}")
                    imgui.tree_pop()
                    
        imgui.end()
