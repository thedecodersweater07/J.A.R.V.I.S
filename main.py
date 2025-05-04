import os
import sys
import json
from typing import Optional, Literal
import logging

# Import core components
from core.brain.cerebrum import Cerebrum
from core.command.command_parser import CommandParser
from core.command.executor import CommandExecutor
from ui.visual.hologram_projector import HologramProjector
from ui.input.voice_input import VoiceInput
from ui.input.text_input import TextInput
from security.authentication.identity_verifier import IdentityVerifier
from ml.model_manager import ModelManager  # Updated path
from nlp.language_processor import LanguageProcessor  # Updated path
from nlp.conversation.conversation_handler import ConversationHandler  # Updated path


class JARVIS:
    def __init__(self):
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load configs
        self.config = self._load_config()
        
        # Initialize components
        self.brain = Cerebrum()
        self.command_parser = CommandParser(schema_path="config/command_schema.json")
        self.executor = CommandExecutor()
        self.ui = HologramProjector(config=self.config.get("ui", {}))
        self.voice_input = VoiceInput()
        self.text_input = TextInput()
        self.input_mode: Literal["voice", "text"] = "text"  # Default to text mode
        self.security = IdentityVerifier()
        self.model_manager = ModelManager()  # Now works without parameters
        self.nlp = LanguageProcessor(language=self.config.get("language", "nl"))
        self.llm = None  # Placeholder for LLM integration
        self.conversation = ConversationHandler(self.nlp)
        self._initialize_components()
        self.logger.info("JARVIS initialized successfully")
        
    def _initialize_components(self):
        """Initialize all system components"""
        try:
            self.brain.initialize()
            # Make command parser initialization optional
            if hasattr(self.command_parser, 'initialize'):
                self.command_parser.initialize()
            if self.nlp:
                self.nlp.initialize()
            if self.llm:
                self.llm.initialize()
            self.model_manager.initialize()
            self.logger.info("All components initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise

    def _load_config(self):
        try:
            with open("config/config.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.warning("Config file not found, using defaults")
            return {}

    def switch_input_mode(self, mode: Literal["voice", "text"]):
        """Switch between voice and text input modes"""
        self.input_mode = mode
        self.logger.info(f"Switched to {mode} input mode")

    def get_input(self) -> Optional[str]:
        """Get input based on current mode"""
        if self.input_mode == "voice":
            return self.voice_input.listen()
        else:
            return self.text_input.listen()

    def initialize(self):
        self.logger.info("Initializing JARVIS...")
        # Initialize core systems
        self.brain.initialize()
        self.ui.start()
        self.nlp.initialize()
        # Only initialize LLM if it exists
        if self.llm:
            self.llm.initialize()
        
    def run(self):
        self.initialize()
        self.logger.info(f"JARVIS running in {self.input_mode} mode")
        while True:
            try:
                # Get user input
                user_input = self.get_input()
                if not user_input:
                    continue

                # Handle mode switching commands
                if user_input == "switch_voice":
                    self.switch_input_mode("voice")
                    continue
                elif user_input == "switch_text":
                    self.switch_input_mode("text")
                    continue

                # Generate response through conversation handler
                response = self.conversation.process_input(user_input)
                print(f"JARVIS: {response}")

            except KeyboardInterrupt:
                self.logger.info("Shutting down JARVIS...")
                break
            except Exception as e:
                self.logger.error(f"Error: {e}")
                print("JARVIS: Sorry, er is een fout opgetreden.")

if __name__ == "__main__":
    jarvis = JARVIS()
    jarvis.run()
