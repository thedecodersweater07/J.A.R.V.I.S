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
