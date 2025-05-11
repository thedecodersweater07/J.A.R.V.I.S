from typing import Dict, Any
from pathlib import Path
import json
import logging
import imgui

logger = logging.getLogger(__name__)

class ThemeManager:
    def __init__(self):
        self.themes = {}
        self.current_theme = "dark"
        self.theme_path = Path(__file__).parent / "themes"
        
    def load_themes(self):
        """Load all theme configurations"""
        try:
            for theme_file in self.theme_path.glob("*.json"):
                with open(theme_file) as f:
                    theme_data = json.load(f)
                    self.themes[theme_file.stem] = theme_data
        except Exception as e:
            logger.error(f"Failed to load themes: {e}")
            self._load_fallback_theme()
            
    def apply_theme(self, theme_name: str) -> bool:
        if theme_name in self.themes:
            try:
                self._apply_theme_colors(self.themes[theme_name])
                self.current_theme = theme_name
                return True
            except Exception as e:
                logger.error(f"Failed to apply theme {theme_name}: {e}")
        return False

    def _apply_theme_colors(self, theme_data: Dict[str, Any]):
        style = imgui.get_style()
        for name, color in theme_data.get("colors", {}).items():
            try:
                color_idx = getattr(imgui, f"Col_{name}")
                style.colors[color_idx] = color
            except Exception as e:
                logger.error(f"Failed to set color {name}: {e}")
                
    @staticmethod
    def text_centered(text: str):
        """Center text in the current window"""
        try:
            window_width = imgui.get_window_width()
            text_width = imgui.calc_text_size(text).x
            imgui.set_cursor_pos_x((window_width - text_width) * 0.5)
            imgui.text(text)
        except Exception as e:
            logger.error(f"Error centering text: {e}")
            imgui.text(text)  # Fallback to regular text
