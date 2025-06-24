import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import tkinter as tk
from tkinter import ttk
from .stark_theme import StarkTheme

logger = logging.getLogger(__name__)

def rgba_to_hex(rgba):
    # Convert (r,g,b,a) floats (0-1) to hex string, ignore alpha
    if isinstance(rgba, tuple) and len(rgba) >= 3:
        r, g, b = [int(255 * float(x)) for x in rgba[:3]]
        return f"#{r:02x}{g:02x}{b:02x}"
    return "#000000"

class ThemeManager:
    """Geavanceerde theme manager voor JARVIS UI"""
    
    def __init__(self):
        self.themes = {}
        self.current_theme_name = "stark"
        self.current_theme = {}
        self.theme_path = Path(__file__).parent.parent / "themes"
        self.callbacks = []
        self._load_default_themes()
        self._load_class_themes()

    def _load_default_themes(self):
        # Voeg class-based theme toe
        from .minimal_theme import MinimalTheme
        from .light_theme import LightTheme
        from .dark_theme import DarkTheme
        stark = StarkTheme().get_complete_theme()
        minimal = MinimalTheme().get_complete_theme()
        light = LightTheme().get_complete_theme()
        dark = DarkTheme().get_complete_theme()
        # Zet alle kleuren om naar hex
        stark_colors = {k.lower(): rgba_to_hex(v) for k, v in stark["colors"].items()}
        minimal_colors = {k.lower(): rgba_to_hex(v) for k, v in minimal["colors"].items()}
        light_colors = {k.lower(): rgba_to_hex(v) for k, v in light["colors"].items()}
        dark_colors = {k.lower(): rgba_to_hex(v) for k, v in dark["colors"].items()}
        self.themes["stark"] = {
            "name": "Stark Futuristic",
            "colors": stark_colors,
            "fonts": {
                "main": ("Orbitron", 10),
                "heading": ("Orbitron", 14, "bold"),
                "mono": ("Roboto Mono", 10),
                "large": ("Orbitron", 12)
            },
            "spacing": {
                "padding": 12,
                "margin": 6,
                "border_radius": 6
            }
        }
        self.themes["minimal"] = {
            "name": "Minimal Clean",
            "colors": minimal_colors,
            "fonts": {
                "main": ("Segoe UI", 10),
                "heading": ("Segoe UI", 14, "bold"),
                "mono": ("Roboto Mono", 10),
                "large": ("Segoe UI", 12)
            },
            "spacing": {
                "padding": 10,
                "margin": 5,
                "border_radius": 4
            }
        }
        self.themes["light"] = {
            "name": "Light Professional",
            "colors": light_colors,
            "fonts": {
                "main": ("Segoe UI", 10),
                "heading": ("Segoe UI", 14, "bold"),
                "mono": ("Roboto Mono", 10),
                "large": ("Segoe UI", 12)
            },
            "spacing": {
                "padding": 10,
                "margin": 5,
                "border_radius": 4
            }
        }
        self.themes["dark"] = {
            "name": "Dark Professional",
            "colors": dark_colors,
            "fonts": {
                "main": ("Segoe UI", 10),
                "heading": ("Segoe UI", 14, "bold"),
                "mono": ("Roboto Mono", 10),
                "large": ("Segoe UI", 12)
            },
            "spacing": {
                "padding": 10,
                "margin": 5,
                "border_radius": 4
            }
        }
        # ...laad eventueel andere dict-based themes (zoals minimal, dark, light)...
        # ...existing code...

    def _load_class_themes(self):
        # Probeer class-based themes te laden (zoals StarkTheme)
        try:
            stark = StarkTheme()
            theme_dict = {
                "name": stark.name,
                "colors": {k.lower(): rgba_to_hex(v) for k, v in stark.get_colors().items()},
                "fonts": {"main": ("Segoe UI", 10)},
                "spacing": {"padding": 10, "margin": 5, "border_radius": 6}
            }
            self.themes["stark_class"] = theme_dict
        except Exception as e:
            logger.warning(f"Kon class-based StarkTheme niet laden: {e}")

    def load_themes(self):
        """Load externe theme bestanden"""
        try:
            if self.theme_path.exists():
                for theme_file in self.theme_path.glob("*.json"):
                    try:
                        with open(theme_file, 'r', encoding='utf-8') as f:
                            theme_data = json.load(f)
                            self.themes[theme_file.stem] = theme_data
                    except Exception as e:
                        logger.warning(f"Could not load theme {theme_file}: {e}")
            
            logger.info(f"Loaded {len(self.themes)} themes")
            
        except Exception as e:
            logger.error(f"Error loading themes: {e}")

    def get_available_themes(self) -> Dict[str, str]:
        """Get lijst van beschikbare themes"""
        return {key: theme.get("name", key.title()) 
                for key, theme in self.themes.items()}

    def apply_theme(self, theme_name: str) -> bool:
        """Apply theme"""
        if theme_name not in self.themes:
            logger.warning(f"Theme '{theme_name}' not found")
            return False
            
        try:
            self.current_theme_name = theme_name
            self.current_theme = self.themes[theme_name].copy()
            
            # Notify callbacks
            for callback in self.callbacks:
                try:
                    callback(self.current_theme)
                except Exception as e:
                    logger.error(f"Theme callback error: {e}")
            
            logger.info(f"Applied theme: {theme_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error applying theme {theme_name}: {e}")
            return False

    def get_current_theme(self) -> Dict[str, Any]:
        """Get current theme data"""
        return self.current_theme

    def get_color(self, color_name: str) -> str:
        """Get kleur uit current theme"""
        return self.current_theme.get("colors", {}).get(color_name, "#000000")

    def get_font(self, font_name: str) -> tuple:
        """Get font uit current theme"""
        return self.current_theme.get("fonts", {}).get(font_name, ("Arial", 10))

    def get_spacing(self, spacing_name: str) -> int:
        """Get spacing waarde"""
        return self.current_theme.get("spacing", {}).get(spacing_name, 5)

    def register_callback(self, callback):
        """Register callback voor theme changes"""
        self.callbacks.append(callback)

    def unregister_callback(self, callback):
        """Unregister callback"""
        if callback in self.callbacks:
            self.callbacks.remove(callback)

    def create_style_config(self) -> Dict[str, Any]:
        """Create complete style config voor ttk"""
        theme = self.current_theme
        colors = theme.get("colors", {})
        fonts = theme.get("fonts", {})
        
        return {
            # Frame styles
            "TFrame": {
                "configure": {"background": colors.get("background", "#ffffff")}
            },
            
            # Label styles  
            "TLabel": {
                "configure": {
                    "background": colors.get("background", "#ffffff"),
                    "foreground": colors.get("text", "#000000"),
                    "font": fonts.get("main", ("Arial", 10))
                }  
            },
            
            # Button styles
            "TButton": {
                "configure": {
                    "background": colors.get("button_bg", "#007acc"),
                    "foreground": colors.get("button_text", "#ffffff"),
                    "font": fonts.get("main", ("Arial", 10)),
                    "borderwidth": 0,
                    "focuscolor": "none"
                },
                "map": {
                    "background": [("active", colors.get("primary_hover", "#005f99"))]
                }
            },
            
            # Entry styles
            "TEntry": {
                "configure": {
                    "fieldbackground": colors.get("input_bg", "#ffffff"),
                    "foreground": colors.get("text", "#000000"),
                    "insertcolor": colors.get("text", "#000000"),
                    "borderwidth": 1,
                    "relief": "solid"
                }
            },
            
            # Combobox styles
            "TCombobox": {
                "configure": {
                    "fieldbackground": colors.get("input_bg", "#ffffff"),
                    "foreground": colors.get("text", "#000000"),
                    "arrowcolor": colors.get("text", "#000000")
                }
            }
        }

    def apply_ttk_styles(self, style: ttk.Style):
        """Apply styles to ttk Style object"""
        style_config = self.create_style_config()
        
        for widget_class, config in style_config.items():
            if "configure" in config:
                style.configure(widget_class, **config["configure"])
            if "map" in config:
                style.map(widget_class, **config["map"])

    def save_theme(self, theme_name: str, theme_data: Dict[str, Any]):
        """Save custom theme"""
        try:
            self.theme_path.mkdir(exist_ok=True)
            theme_file = self.theme_path / f"{theme_name}.json"
            
            with open(theme_file, 'w', encoding='utf-8') as f:
                json.dump(theme_data, f, indent=2, ensure_ascii=False)
            
            self.themes[theme_name] = theme_data
            logger.info(f"Saved theme: {theme_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving theme {theme_name}: {e}")
            return False

    def export_current_theme(self, filename: str):
        """Export current theme to file"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.current_theme, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Error exporting theme: {e}")
            return False