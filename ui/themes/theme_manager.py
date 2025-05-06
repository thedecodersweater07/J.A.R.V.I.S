from typing import Dict, Any, Optional
import imgui
from .dark_theme import DarkTheme
from .light_theme import LightTheme
from .minimal_theme import MinimalTheme
from .stark_theme import StarkTheme

class ThemeManager:
    def __init__(self):
        self.themes = {
            "dark": DarkTheme(),
            "light": LightTheme(),
            "minimal": MinimalTheme(),
            "stark": StarkTheme()
        }
        self.current_theme = "dark"
        self.imgui_style = imgui.get_style()

    def switch_theme(self, theme_name: str) -> bool:
        """Switch to a different theme"""
        if theme_name in self.themes:
            self.current_theme = theme_name
            self._apply_theme()
            return True
        return False

    def _apply_theme(self):
        """Apply current theme to ImGui"""
        theme = self.themes[self.current_theme]
        colors = theme.get_colors()
        metrics = theme.get_metrics()

        # Apply colors
        for name, color in colors.items():
            color_id = getattr(imgui, f"COL_{name}", None)
            if color_id is not None:
                self.imgui_style.colors[color_id] = color

        # Apply metrics
        for name, value in metrics.items():
            if hasattr(self.imgui_style, name):
                setattr(self.imgui_style, name, value)

    def apply_login_theme(self, colors: Dict[int, Any]):
        """Apply specific theme for login screen"""
        theme = self.themes[self.current_theme]
        theme_colors = theme.get_colors()
        
        # Set login-specific colors
        colors[imgui.COLOR_TEXT] = theme_colors["TEXT"]
        colors[imgui.COLOR_WINDOW_BACKGROUND] = theme_colors["WINDOW_BACKGROUND"]
        colors[imgui.COLOR_TITLE_BACKGROUND_ACTIVE] = theme_colors["TITLE_BACKGROUND_ACTIVE"]
        colors[imgui.COLOR_FRAME_BACKGROUND] = theme_colors["FRAME_BACKGROUND"]
        colors[imgui.COLOR_FRAME_BACKGROUND_HOVERED] = theme_colors["FRAME_BACKGROUND_HOVERED"]
        colors[imgui.COLOR_BUTTON] = theme_colors["BUTTON"]
        colors[imgui.COLOR_BUTTON_HOVERED] = theme_colors["BUTTON_HOVERED"]

    def get_theme_style(self) -> Dict[str, Any]:
        """Get current theme's complete style dictionary"""
        return self.themes[self.current_theme].get_style_dict()

    def get_current_theme(self) -> str:
        """Get name of current theme"""
        return self.current_theme

    def get_available_themes(self) -> list:
        """Get list of available themes"""
        return list(self.themes.keys())
