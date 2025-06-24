from typing import Dict, Tuple
from .base_theme import BaseTheme

class MinimalTheme(BaseTheme):
    """Minimalistisch theme - focus op content met subtiele UI elementen"""
    
    def __init__(self):
        super().__init__()
        self.name = "Minimal Clean"
        self.version = "2.0.0"
        self.author = "UI Design Team"
        
        # Theme specifieke eigenschappen
        self.primary_color = "#FAFAFA"
        self.accent_color = "#2196F3"
        self.warning_color = "#FF5722"
        self.success_color = "#4CAF50"
        
    def get_colors(self) -> Dict[str, Tuple[float, float, float, float]]:
        """Minimale kleurenschema - minder afleiding, meer focus"""
        return {
            # Tekst kleuren
            "TEXT_PRIMARY": (0.20, 0.20, 0.20, 1.0),
            "TEXT_SECONDARY": (0.50, 0.50, 0.50, 1.0),
            "TEXT_DISABLED": (0.70, 0.70, 0.70, 1.0),
            
            # Achtergrond kleuren
            "WINDOW_BACKGROUND": (1.00, 1.00, 1.00, 0.98),
            "PANEL_BACKGROUND": (0.99, 0.99, 0.99, 1.0),
            "CARD_BACKGROUND": (1.00, 1.00, 1.00, 1.0),
            
            # Interactieve elementen - zeer subtiel
            "BUTTON_DEFAULT": (0.97, 0.97, 0.97, 1.0),
            "BUTTON_HOVERED": (0.94, 0.94, 0.94, 1.0),
            "BUTTON_PRESSED": (0.91, 0.91, 0.91, 1.0),
            "BUTTON_DISABLED": (0.98, 0.98, 0.98, 0.3),
            
            # Accent kleuren - spaarzaam gebruikt
            "ACCENT_PRIMARY": (0.13, 0.59, 0.95, 1.0),
            "ACCENT_SECONDARY": (0.25, 0.68, 0.98, 1.0),
            "WARNING": (1.00, 0.34, 0.13, 1.0),
            "SUCCESS": (0.30, 0.69, 0.31, 1.0),
            "ERROR": (0.96, 0.26, 0.21, 1.0),
            
            # Borders en separators - bijna onzichtbaar
            "BORDER_DEFAULT": (0.92, 0.92, 0.92, 1.0),
            "BORDER_FOCUSED": (0.13, 0.59, 0.95, 1.0),
            "SEPARATOR": (0.95, 0.95, 0.95, 1.0)
        }
        
    def get_metrics(self):
        return super().get_metrics()

    def get_style_properties(self):
        return super().get_style_properties()