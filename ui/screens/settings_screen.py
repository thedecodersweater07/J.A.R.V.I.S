import imgui
from typing import Dict, Any, Optional
from pathlib import Path
import json
import logging
import glfw
from OpenGL import GL
from .base_screen import BaseScreen

logger = logging.getLogger(__name__)

class SettingsScreen(BaseScreen):
    def __init__(self):
        super().__init__()
        self.initialized = False
        self.settings: Dict[str, Any] = {
            "llm": {
                "model": "gpt2",
                "max_length": 1024,
                "temperature": 0.7,
                "gpu_enabled": True
            },
            "ui": {
                "theme": "dark",
                "font_size": 12,
                "window_opacity": 1.0
            },
            "audio": {
                "input_device": "default",
                "output_device": "default",
                "volume": 1.0
            },
            "security": {
                "session_timeout": 12,
                "max_login_attempts": 3
            }
        }
        self.config_path = Path(__file__).parent.parent.parent / "config" / "settings.json"
        self.load_settings()

    def init(self) -> bool:
        self.initialized = True
        return True

    def render(self, frame_data: Dict[str, Any]) -> None:
        # Haal de ImGuiManager uit de frame_data
        imgui_manager = frame_data.get("imgui_manager")
        if not imgui_manager:
            logger.error("No ImGuiManager provided in frame_data")
            return
        
        # Stel venstergrootte in voor settings
        window_width = 700
        window_height = 500
        display_w, display_h = glfw.get_window_size(glfw.get_current_context())
        imgui.set_next_window_size(window_width, window_height)
        imgui.set_next_window_position((display_w - window_width) // 2, (display_h - window_height) // 2)
            
        # Verbeterde stijl voor settings venster
        imgui.push_style_var(imgui.STYLE_WINDOW_ROUNDING, 8.0)
        imgui.push_style_var(imgui.STYLE_FRAME_ROUNDING, 4.0)
        imgui.push_style_var(imgui.STYLE_FRAME_BORDER_SIZE, 1.0)
        imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, 0.06, 0.06, 0.12, 0.98)
        imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND_ACTIVE, 0.2, 0.2, 0.4, 1.0)
        
        # Gebruik de veilige contextmanagers van ImGuiManager
        with imgui_manager.window("JARVIS Settings", flags=imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_RESIZE):
            # Toon een korte beschrijving bovenaan
            imgui.push_font(imgui.get_io().fonts.fonts[0])
            imgui.text_colored("Configure JARVIS to your preferences", 0.7, 0.7, 1.0)
            imgui.separator()
            imgui.dummy(0, 5)  # Extra ruimte
            imgui.pop_font()
            
            # Verbeterde tab bar stijl
            imgui.push_style_color(imgui.COLOR_TAB_ACTIVE, 0.2, 0.3, 0.7, 1.0)
            imgui.push_style_color(imgui.COLOR_TAB_HOVERED, 0.3, 0.4, 0.8, 1.0)
            
            with imgui_manager.tab_bar("SettingsTabs"):
                self._render_llm_settings()
                self._render_ui_settings()
                self._render_audio_settings()
                self._render_security_settings()
                self._render_about_tab()
            
            imgui.pop_style_color(2)  # Tab kleuren
            
            # Statusbalk onderaan
            imgui.dummy(0, 10)
            imgui.separator()
            imgui.dummy(0, 5)
            
            # Knoppen met verbeterde stijl
            button_width = 150
            imgui.push_style_color(imgui.COLOR_BUTTON, 0.2, 0.4, 0.8, 0.8)
            imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.3, 0.5, 0.9, 0.9)
            imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, 0.1, 0.3, 0.7, 1.0)
            
            # Centreer de knoppen
            window_width = imgui.get_window_width()
            imgui.set_cursor_pos_x((window_width - (button_width * 2 + 10)) / 2)
            
            if imgui.button("Save Settings", width=button_width):
                success = self.save_settings()
                if success:
                    self.status_message = "Settings saved successfully!"
                    self.status_color = (0.2, 0.8, 0.2, 1.0)
                else:
                    self.status_message = "Failed to save settings!"
                    self.status_color = (0.8, 0.2, 0.2, 1.0)
                self.status_time = 3.0  # Toon bericht voor 3 seconden
                
            imgui.same_line(spacing=10)
            imgui.push_style_color(imgui.COLOR_BUTTON, 0.7, 0.3, 0.3, 0.8)  # Rode knop voor reset
            if imgui.button("Reset to Default", width=button_width):
                self.reset_settings()
                self.status_message = "Settings reset to default"
                self.status_color = (0.8, 0.8, 0.2, 1.0)
                self.status_time = 3.0
            imgui.pop_style_color()
            
            imgui.pop_style_color(3)  # Knop kleuren
            
            # Toon statusbericht indien nodig
            if hasattr(self, 'status_message') and self.status_message:
                if hasattr(self, 'status_time') and self.status_time > 0:
                    imgui.dummy(0, 5)
                    imgui.push_style_color(imgui.COLOR_TEXT, *self.status_color)
                    text_width = imgui.calc_text_size(self.status_message).x
                    imgui.set_cursor_pos_x((window_width - text_width) * 0.5)  # Centreer tekst
                    imgui.text(self.status_message)
                    imgui.pop_style_color()
                    self.status_time -= imgui.get_io().delta_time  # Verminder timer
                    if self.status_time <= 0:
                        self.status_message = ""
        
        # Herstel stijl
        imgui.pop_style_color(2)  # Window kleuren
        imgui.pop_style_var(3)  # Window en frame stijlen

    def _render_llm_settings(self):
        if imgui.begin_tab_item("LLM")[0]:
            changed, value = imgui.combo(
                "Model",
                ["gpt2", "gpt2-medium", "gpt2-large"].index(self.settings["llm"]["model"]),
                ["gpt2", "gpt2-medium", "gpt2-large"]
            )
            if changed:
                self.settings["llm"]["model"] = value

            changed, value = imgui.slider_int(
                "Max Length",
                self.settings["llm"]["max_length"],
                128, 2048
            )
            if changed:
                self.settings["llm"]["max_length"] = value

            changed, value = imgui.checkbox(
                "GPU Enabled",
                self.settings["llm"]["gpu_enabled"]
            )
            if changed:
                self.settings["llm"]["gpu_enabled"] = value

            imgui.end_tab_item()

    def _render_ui_settings(self):
        if imgui.begin_tab_item("UI")[0]:
            changed, value = imgui.combo(
                "Theme",
                ["dark", "light"].index(self.settings["ui"]["theme"]),
                ["dark", "light"]
            )
            if changed:
                self.settings["ui"]["theme"] = value

            changed, value = imgui.slider_int(
                "Font Size",
                self.settings["ui"]["font_size"],
                8, 24
            )
            if changed:
                self.settings["ui"]["font_size"] = value

            imgui.end_tab_item()

    def _render_audio_settings(self):
        if imgui.begin_tab_item("Audio")[0]:
            changed, value = imgui.slider_float(
                "Volume",
                self.settings["audio"]["volume"],
                0.0, 1.0
            )
            if changed:
                self.settings["audio"]["volume"] = value

            imgui.end_tab_item()

    def _render_security_settings(self):
        if imgui.begin_tab_item("Security")[0]:
            changed, value = imgui.slider_int(
                "Session Timeout (hours)",
                self.settings["security"]["session_timeout"],
                1, 24
            )
            if changed:
                self.settings["security"]["session_timeout"] = value

            changed, value = imgui.slider_int(
                "Max Login Attempts",
                self.settings["security"]["max_login_attempts"],
                1, 10
            )
            if changed:
                self.settings["security"]["max_login_attempts"] = value

            imgui.end_tab_item()

    def load_settings(self):
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    loaded_settings = json.load(f)
                    self.settings.update(loaded_settings)
        except Exception as e:
            logger.error(f"Failed to load settings: {e}")

    def save_settings(self) -> bool:
        """Save settings to file and return success status"""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.settings, f, indent=4)
            logger.info(f"Settings saved successfully to {self.config_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save settings: {e}")
            return False

    def reset_settings(self) -> bool:
        """Reset settings to default values"""
        try:
            self.__init__()
            logger.info("Settings reset to default values")
            return True
        except Exception as e:
            logger.error(f"Failed to reset settings: {e}")
            return False
            
    def _render_about_tab(self):
        """Render about tab with system information"""
        if imgui.begin_tab_item("About")[0]:
            # Voeg wat ruimte toe
            imgui.dummy(0, 10)
            
            # JARVIS versie informatie
            imgui.text_colored("JARVIS - Just A Rather Very Intelligent System", 0.7, 0.9, 1.0)
            imgui.text("Version: 1.0.0")
            imgui.text("Build Date: May 2025")
            imgui.dummy(0, 10)
            
            # Systeem informatie
            imgui.separator()
            imgui.text_colored("System Information", 0.7, 0.9, 1.0)
            imgui.dummy(0, 5)
            
            # Toon OpenGL versie
            gl_version = GL.glGetString(GL.GL_VERSION).decode('utf-8')
            imgui.text(f"OpenGL Version: {gl_version}")
            
            # Toon GLFW versie
            glfw_version = f"{glfw.get_version()[0]}.{glfw.get_version()[1]}.{glfw.get_version()[2]}"
            imgui.text(f"GLFW Version: {glfw_version}")
            
            # Toon ImGui versie
            imgui_version = imgui.get_version()
            imgui.text(f"ImGui Version: {imgui_version}")
            
            imgui.dummy(0, 10)
            
            # Credits sectie
            imgui.separator()
            imgui.text_colored("Credits", 0.7, 0.9, 1.0)
            imgui.dummy(0, 5)
            imgui.text("Developed by: Nova Industrie")
            imgui.text("© 2025 All Rights Reserved")
            
            # Licentie informatie
            imgui.dummy(0, 10)
            imgui.separator()
            imgui.text_colored("License Information", 0.7, 0.9, 1.0)
            imgui.dummy(0, 5)
            imgui.push_text_wrap_pos(imgui.get_window_width() - 20)
            imgui.text_wrapped("This software is licensed under the MIT License. See LICENSE file for details.")
            imgui.pop_text_wrap_pos()
            
            imgui.end_tab_item()

    def handle_input(self, input_data: Dict[str, Any]) -> None:
        pass  # Settings are handled through imgui interface
