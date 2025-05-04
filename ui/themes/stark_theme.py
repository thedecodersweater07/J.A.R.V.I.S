from typing import Dict

class StarkTheme:
    def __init__(self):
        self.colors = {
            "primary": "#FF0000",  # Stark rood
            "secondary": "#FFD700",  # Goud
            "background": "#1A1A1A",  # Donkergrijs
            "text": "#FFFFFF",  # Wit
            "accent": "#00FF00"  # Fel groen voor HUD elementen
        }
        
        self.fonts = {
            "main": "Stark Sans",
            "display": "Stark Display",
            "monospace": "Stark Mono"
        }
        
        self.animations = {
            "transition_speed": "0.3s",
            "hover_effect": "glow",
            "loading_animation": "arc-reactor"
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
            "TEXT": (0.95, 0.95, 0.95, 1.0),
            "WINDOW_BACKGROUND": (0.05, 0.05, 0.08, 0.95),
            "TITLE_BACKGROUND_ACTIVE": (0.20, 0.40, 0.70, 1.0),
            "FRAME_BACKGROUND": (0.15, 0.15, 0.20, 1.0),
            "FRAME_BACKGROUND_HOVERED": (0.25, 0.25, 0.30, 1.0),
            "BUTTON": (0.20, 0.40, 0.70, 0.8),
            "BUTTON_HOVERED": (0.30, 0.50, 0.80, 1.0)
        }

    def get_metrics(self):
        return {
            "window_padding": (12, 12),
            "frame_padding": (6, 6),
            "item_spacing": (8, 4),
            "scrollbar_size": 14,
            "frame_rounding": 6,
            "grab_rounding": 6
        }
