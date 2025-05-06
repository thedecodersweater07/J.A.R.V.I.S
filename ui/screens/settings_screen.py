import imgui
from typing import Dict, Any, Optional
from pathlib import Path
import json
import logging
from .base_screen import BaseScreen

logger = logging.getLogger(__name__)

class SettingsScreen(BaseScreen):
    def __init__(self):
        super().__init__()
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
        imgui.begin("JARVIS Settings", flags=imgui.WINDOW_NO_COLLAPSE)

        if imgui.begin_tab_bar("SettingsTabs"):
            self._render_llm_settings()
            self._render_ui_settings()
            self._render_audio_settings()
            self._render_security_settings()
            imgui.end_tab_bar()

        if imgui.button("Save Settings"):
            self.save_settings()
            
        imgui.same_line()
        if imgui.button("Reset to Default"):
            self.reset_settings()

        imgui.end()

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

    def save_settings(self):
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.settings, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save settings: {e}")

    def reset_settings(self):
        self.__init__()

    def handle_input(self, input_data: Dict[str, Any]) -> None:
        pass  # Settings are handled through imgui interface
