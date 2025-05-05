import os
import sys
import json
import traceback  # Add this import
from typing import Optional, Literal, Dict, Any
import logging
import glfw
from OpenGL import GL as gl  # OpenGL import for rendering
import signal
from pathlib import Path

# Import core components
from core.brain.cerebrum import Cerebrum
from core.command.command_parser import CommandParser
from core.command.executor import CommandExecutor
from ui.visual.hologram_projector import HologramProjector
from ui.input.voice_input import VoiceInput
from ui.input.text_input import TextInput
from security.authentication.identity_verifier import IdentityVerifier
from ml.model_manager import ModelManager  # Updated path
from nlp.language_processor import LanguageProcessor  # Using the updated version
from nlp.conversation.conversation_handler import ConversationHandler  # Updated path
from ui.screen import Screen
from config.config_validator import ConfigValidator


class JARVIS:
    def __init__(self):
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load and validate configs
        self.config_validator = ConfigValidator()
        self.config = self._load_config()
        
        # Initialize components with validated configs
        self.brain = Cerebrum()
        self.command_parser = CommandParser(schema_path="config/command_schema.json")
        self.executor = CommandExecutor()
        self.ui = HologramProjector(config=self.config.get("ui", {}))
        self.voice_input = VoiceInput()
        self.text_input = TextInput()
        self.input_mode: Literal["voice", "text"] = "text"  # Default to text mode
        self.security = IdentityVerifier()
        self.model_manager = ModelManager()  # Now works without parameters
        self.nlp = LanguageProcessor(language=self.config.get("nlp", {}).get("language", "nl"))
        self.llm = None  # Placeholder for LLM integration
        self.conversation = ConversationHandler(self.nlp)
        self.screen = Screen(
            width=self.config.get("ui", {}).get("width", 1024),
            height=self.config.get("ui", {}).get("height", 768),
            title="JARVIS - Advanced AI Interface"
        )
        self.error_count = 0
        self.max_errors = 3

    def _load_config(self) -> Dict[str, Any]:
        """Load and validate configuration"""
        try:
            config_path = Path("config") / "config.json"
            if not config_path.exists():
                self.logger.warning("Config file not found, using defaults")
                return self._get_default_config()
                
            with open(config_path) as f:
                config = json.load(f)
                
            # Validate different config sections
            for section in ["nlp", "llm", "ui"]:
                if section in config:
                    if not self.config_validator.validate(config[section], section):
                        self.logger.warning(f"Invalid {section} config, using defaults")
                        config[section] = self._get_default_config()[section]
                        
            return config
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return self._get_default_config()
            
    def _get_default_config(self) -> Dict[str, Any]:
        return {
            "nlp": {
                "language": "nl",
                "models": {
                    "sentiment": "default",
                    "intent": "default"
                }
            },
            "llm": {
                "model": "gpt2",
                "inference": {
                    "max_length": 100,
                    "temperature": 0.7
                }
            },
            "ui": {
                "width": 1024,
                "height": 768
            }
        }

    def handle_input(self, text: str):
        """Handle user input with better error handling"""
        try:
            if text.lower() in ["exit", "quit", "stop"]:
                self.screen.add_message("Shutting down...", False)
                self.screen.interrupt_received = True
                return

            self.screen.add_message(text, True)
            
            try:
                response = self.conversation.process_input(text)
                self.screen.add_message(response, False)
                self.error_count = 0  # Reset error count on success
            except Exception as e:
                self.error_count += 1
                self.logger.error(f"Error processing input: {e}\n{traceback.format_exc()}")
                self.screen.add_message(
                    "Sorry, I encountered an error. Could you rephrase that?", 
                    False
                )
                
        except Exception as e:
            self.logger.error(f"Input handler error: {e}\n{traceback.format_exc()}")
            self.screen.add_message("An internal error occurred.", False)

    def cleanup(self):
        """Cleanup resources properly"""
        try:
            self.logger.info("Starting cleanup...")
            if hasattr(self, 'screen'):
                self.screen.cleanup()
            if hasattr(self, 'ui'):
                self.ui.cleanup()
            self.logger.info("Cleanup completed successfully")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}\n{traceback.format_exc()}")

    def run(self):
        """Main run loop with improved initialization and error handling"""
        try:
            self.logger.info("Starting JARVIS initialization...")
            
            # Initialize core components first
            self.brain.initialize()
            self.model_manager.initialize()
            self.nlp.initialize()
            
            # Initialize UI components
            self.ui.start()
            if not self.screen.init():
                raise RuntimeError("Failed to initialize screen interface")

            self.screen.set_callback(self.handle_input)
            self.screen.add_message("JARVIS initialized and ready.", False)
            
            self.logger.info("JARVIS running in %s mode", self.input_mode)
            
            # Main event loop
            while not (self.screen.should_close() or self.screen.interrupt_received):
                try:
                    self.screen.render()
                except Exception as e:
                    self.error_count += 1
                    self.logger.error(f"Render error: {e}\n{traceback.format_exc()}")
                    if self.error_count >= self.max_errors:
                        self.logger.critical("Too many errors, shutting down")
                        break
                    
        except Exception as e:
            self.logger.error(f"Critical error: {e}\n{traceback.format_exc()}")
        finally:
            self.cleanup()


if __name__ == "__main__":
    try:
        jarvis = JARVIS()
        jarvis.run()
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()
