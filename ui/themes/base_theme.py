from abc import ABC, abstractmethod
from typing import Dict, Tuple

class BaseTheme(ABC):
    """Basis klasse voor alle UI themes"""
    
    def __init__(self):
        self.name = ""
        self.version = "1.0.0"
        self.author = "Theme Designer"
        
    @abstractmethod
    def get_colors(self) -> Dict[str, Tuple[float, float, float, float]]:
        """Retourneert kleur definities voor UI elementen"""
        pass
        
    @abstractmethod
    def get_metrics(self) -> Dict[str, any]:
        """Retourneert afmetingen en spacing voor UI elementen"""
        pass
        
    @abstractmethod
    def get_style_properties(self) -> Dict[str, any]:
        """Retourneert extra style eigenschappen"""
        pass
        
    def get_theme_info(self) -> Dict[str, str]:
        """Retourneert theme informatie"""
        return {
            "name": self.name,
            "version": self.version,
            "author": self.author
        }
        
    def get_complete_theme(self) -> Dict[str, any]:
        """Retourneert complete theme configuratie"""
        return {
            "info": self.get_theme_info(),
            "colors": self.get_colors(),
            "metrics": self.get_metrics(),
            "style_properties": self.get_style_properties()
        }