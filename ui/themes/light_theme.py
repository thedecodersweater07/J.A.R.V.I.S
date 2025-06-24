from typing import Dict, Tuple
from .base_theme import BaseTheme

class LightTheme(BaseTheme):
    """Licht theme voor heldere omgevingen - professioneel en schoon"""
    
    def __init__(self):
        super().__init__()
        self.name = "Light Professional"
        self.version = "2.0.0"
        self.author = "UI Design Team"
        
        # Theme specifieke eigenschappen
        self.primary_color = "#F5F5F5"
        self.accent_color = "#1976D2"
        self.warning_color = "#FF9800"
        self.success_color = "#4CAF50"
        
    def get_colors(self) -> Dict[str, Tuple[float, float, float, float]]:
        """Lichte kleurenschema - geoptimaliseerd voor heldere omgevingen"""
        return {
            # Tekst kleuren
            "TEXT_PRIMARY": (0.13, 0.13, 0.13, 1.0),
            "TEXT_SECONDARY": (0.38, 0.38, 0.38, 1.0),
            "TEXT_DISABLED": (0.62, 0.62, 0.62, 1.0),
            
            # Achtergrond kleuren
            "WINDOW_BACKGROUND": (0.98, 0.98, 0.98, 0.95),
            "PANEL_BACKGROUND": (0.95, 0.95, 0.95, 1.0),
            "CARD_BACKGROUND": (1.00, 1.00, 1.00, 1.0),
            
            # Interactieve elementen
            "BUTTON_DEFAULT": (0.90, 0.90, 0.90, 1.0),
            "BUTTON_HOVERED": (0.85, 0.85, 0.85, 1.0),
            "BUTTON_PRESSED": (0.80, 0.80, 0.80, 1.0),
            "BUTTON_DISABLED": (0.95, 0.95, 0.95, 0.5),
            
            # Accent kleuren
            "ACCENT_PRIMARY": (0.10, 0.46, 0.82, 1.0),
            "ACCENT_SECONDARY": (0.25, 0.58, 0.90, 1.0),
            "WARNING": (1.00, 0.60, 0.00, 1.0),
            "SUCCESS": (0.30, 0.69, 0.31, 1.0),
            "ERROR": (0.96, 0.26, 0.21, 1.0),
            
            # Borders en separators
            "BORDER_DEFAULT": (0.78, 0.78, 0.78, 1.0),
            "BORDER_FOCUSED": (0.10, 0.46, 0.82, 1.0),
            "SEPARATOR": (0.88, 0.88, 0.88, 1.0)
        }
        
    def get_metrics(self):
        """Afmetingen en spacing voor light theme"""
        return super().get_metrics()

    def get_style_properties(self):
        """Extra style eigenschappen voor light theme"""
        return super().get_style_properties()