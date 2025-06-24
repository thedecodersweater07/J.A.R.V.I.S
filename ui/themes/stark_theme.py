from typing import Dict, Tuple
from .base_theme import BaseTheme

class StarkTheme(BaseTheme):
    """Futuristische theme geïnspireerd door high-tech interfaces"""
    
    def __init__(self):
        super().__init__()
        self.name = "Stark Futuristic"
        self.version = "2.0.0"
        self.author = "UI Design Team"
        
        # Theme specifieke eigenschappen
        self.primary_color = "#0A0A0F"
        self.accent_color = "#00D4FF"
        self.warning_color = "#FF6B00"
        self.success_color = "#00FF88"
        
    def get_colors(self) -> Dict[str, Tuple[float, float, float, float]]:
        """Futuristische kleurenschema met neon accenten"""
        return {
            # Tekst kleuren - hoog contrast
            "TEXT_PRIMARY": (0.95, 0.95, 1.00, 1.0),
            "TEXT_SECONDARY": (0.70, 0.80, 0.95, 1.0),
            "TEXT_DISABLED": (0.40, 0.45, 0.60, 1.0),
            
            # Achtergrond kleuren - deep space
            "WINDOW_BACKGROUND": (0.04, 0.04, 0.06, 0.98),
            "PANEL_BACKGROUND": (0.08, 0.08, 0.12, 1.0),
            "CARD_BACKGROUND": (0.12, 0.12, 0.18, 1.0),
            
            # Interactieve elementen - glow effects
            "BUTTON_DEFAULT": (0.15, 0.20, 0.35, 1.0),
            "BUTTON_HOVERED": (0.20, 0.30, 0.50, 1.0),
            "BUTTON_PRESSED": (0.25, 0.40, 0.65, 1.0),
            "BUTTON_DISABLED": (0.10, 0.10, 0.15, 0.4),
            
            # Neon accent kleuren
            "ACCENT_PRIMARY": (0.00, 0.83, 1.00, 1.0),
            "ACCENT_SECONDARY": (0.30, 0.90, 1.00, 1.0),
            "WARNING": (1.00, 0.42, 0.00, 1.0),
            "SUCCESS": (0.00, 1.00, 0.53, 1.0),
            "ERROR": (1.00, 0.20, 0.40, 1.0),
            
            # Glow borders
            "BORDER_DEFAULT": (0.20, 0.25, 0.40, 1.0),
            "BORDER_FOCUSED": (0.00, 0.83, 1.00, 1.0),
            "SEPARATOR": (0.15, 0.20, 0.30, 1.0),
            
            # Special effects
            "GLOW_LIGHT": (0.00, 0.83, 1.00, 0.3),
            "GLOW_MEDIUM": (0.00, 0.83, 1.00, 0.5),
            "GLOW_HEAVY": (0.00, 0.83, 1.00, 0.8)
        }
        
    def get_metrics(self):
        return super().get_metrics()

    def get_style_properties(self):
        return super().get_style_properties()