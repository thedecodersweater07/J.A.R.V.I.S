from typing import Dict

class MinimalTheme:
    def __init__(self):
        self.colors = {
            "primary": "#2196F3",  # Blauw
            "secondary": "#757575",  # Grijs
            "background": "#FFFFFF",  # Wit
            "text": "#212121",  # Donkergrijs
            "accent": "#FF4081"  # Roze
        }
        
        self.fonts = {
            "main": "Roboto",
            "display": "Roboto Light",
            "monospace": "Roboto Mono"
        }
        
        self.animations = {
            "transition_speed": "0.2s",
            "hover_effect": "subtle-lift",
            "loading_animation": "minimal-spin"
        }

    def get_style_dict(self) -> Dict:
        """Return complete stijl dictionary"""
        return {
            "colors": self.colors,
            "fonts": self.fonts,
            "animations": self.animations
        }

    def get_colors(self):
        return {
            "TEXT": (0.90, 0.90, 0.90, 1.0),
            "WINDOW_BACKGROUND": (0.10, 0.10, 0.10, 0.95),
            "TITLE_BACKGROUND_ACTIVE": (0.15, 0.15, 0.15, 1.0),
            "FRAME_BACKGROUND": (0.20, 0.20, 0.20, 1.0),
            "FRAME_BACKGROUND_HOVERED": (0.25, 0.25, 0.25, 1.0),
            "BUTTON": (0.25, 0.25, 0.25, 0.8),
            "BUTTON_HOVERED": (0.30, 0.30, 0.30, 1.0)
        }

    def get_metrics(self):
        return {
            "window_padding": (8, 8),
            "frame_padding": (4, 4),
            "item_spacing": (6, 4),
            "scrollbar_size": 12,
            "frame_rounding": 2,
            "grab_rounding": 2
        }
