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
