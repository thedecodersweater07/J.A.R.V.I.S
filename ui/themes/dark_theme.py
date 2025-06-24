from typing import Dict, Tuple
from .base_theme import BaseTheme

class DarkTheme(BaseTheme):
    """Donker theme voor moderne UI - comfortabel voor de ogen"""
    
    def __init__(self):
        super().__init__()
        self.name = "Dark Professional"
        self.version = "2.0.0"
        self.author = "UI Design Team"
        
        # Theme specifieke eigenschappen
        self.primary_color = "#2D2D30"
        self.accent_color = "#007ACC"
        self.warning_color = "#FF6B35"
        self.success_color = "#4CAF50"
        
    def get_colors(self) -> Dict[str, Tuple[float, float, float, float]]:
        """Donkere kleurenschema - geoptimaliseerd voor lage lichtomstandigheden"""
        return {
            # Tekst kleuren
            "TEXT_PRIMARY": (0.92, 0.92, 0.92, 1.0),
            "TEXT_SECONDARY": (0.75, 0.75, 0.75, 1.0),
            "TEXT_DISABLED": (0.45, 0.45, 0.45, 1.0),
            
            # Achtergrond kleuren
            "WINDOW_BACKGROUND": (0.08, 0.08, 0.08, 0.95),
            "PANEL_BACKGROUND": (0.12, 0.12, 0.12, 1.0),
            "CARD_BACKGROUND": (0.15, 0.15, 0.15, 1.0),
            
            # Interactieve elementen
            "BUTTON_DEFAULT": (0.18, 0.18, 0.18, 1.0),
            "BUTTON_HOVERED": (0.22, 0.22, 0.22, 1.0),
            "BUTTON_PRESSED": (0.25, 0.25, 0.25, 1.0),
            "BUTTON_DISABLED": (0.10, 0.10, 0.10, 0.5),
            
            # Accent kleuren
            "ACCENT_PRIMARY": (0.00, 0.48, 0.80, 1.0),
            "ACCENT_SECONDARY": (0.20, 0.60, 0.90, 1.0),
            "WARNING": (1.00, 0.42, 0.21, 1.0),
            "SUCCESS": (0.30, 0.69, 0.31, 1.0),
            "ERROR": (0.96, 0.26, 0.21, 1.0),
            
            # Borders en separators
            "BORDER_DEFAULT": (0.25, 0.25, 0.25, 1.0),
            "BORDER_FOCUSED": (0.00, 0.48, 0.80, 1.0),
            "SEPARATOR": (0.20, 0.20, 0.20, 1.0)
        }
        
    def get_metrics(self):
        """Afmetingen en spacing voor dark theme"""
        return super().get_metrics()

    def get_style_properties(self):
        """Extra style eigenschappen voor dark theme"""
        return super().get_style_properties()
