import os
import sys
import json
import signal
import traceback
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import core modules
from core.logging.logger import setup_logging
from core.brain.cognitive.cerebrum import Cerebrum
from core.command.command_parser import CommandParser
from core.command.executor import CommandExecutor

# Import UI modules
from ui.screens.base.screen import Screen
from ui.input.voice_input import VoiceInput
from ui.input.text_input import TextInput

# Import security and ML modules
from security.authentication.identity_verifier import IdentityVerifier
from ml.models import ModelManager

logger = setup_logging()

class JARVIS:
    def __init__(self):
        try:
            # Load configuration first
            self.config = self._load_config()
            
            # Initialize core components with error handling
            self._init_core_components()
            
            # Initialize UI components with fallback
            self._init_ui_components()
            
            # Initialize security and ML components
            self._init_auxiliary_components()
            
        except Exception as e:
            logger.critical(f"Failed to initialize JARVIS: {e}\n{traceback.format_exc()}")
            raise

    def _init_core_components(self):
        """Initialize core system components"""
        try:
            self.brain = Cerebrum()
            self.command_parser = CommandParser()
            self.executor = CommandExecutor()
        except Exception as e:
            logger.error(f"Core components initialization failed: {e}")
            raise

    def _init_ui_components(self):
        """Initialize UI with fallback options"""
        try:
            self.screen = Screen(
                width=self.config.get("ui", {}).get("width", 1024),
                height=self.config.get("ui", {}).get("height", 768),
                title="JARVIS AI Interface"
            )
            self.voice_input = VoiceInput() if self.config.get("use_voice", True) else None
            self.text_input = TextInput()
        except Exception as e:
            logger.error(f"UI initialization failed: {e}")
            self._fallback_to_basic_ui()

    def _init_auxiliary_components(self):
        """Initialize security and ML components"""
        try:
            self.security = IdentityVerifier()
            self.model_manager = ModelManager()
            self.error_count = 0
            self.max_errors = 3
        except Exception as e:
            logger.error(f"Auxiliary components initialization failed: {e}")
            raise

    def _fallback_to_basic_ui(self):
        """Fall back to basic text UI if graphical UI fails"""
        try:
            self.screen = Screen(mode="text")
            self.voice_input = None
            self.text_input = TextInput()
        except Exception as e:
            logger.critical(f"Even basic UI failed: {e}")
            raise

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration with fallback to defaults"""
        try:
            with open("config/main.json", "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")
            return {"use_voice": False, "ui": {"width": 1024, "height": 768}}

    def run(self):
        """Main run loop with improved error handling"""
        try:
            self._setup_signal_handlers()
            self.screen.initialize()
            
            while not self.screen.should_exit():
                try:
                    self.handle_input(self.text_input.get_input())
                except KeyboardInterrupt:
                    logger.info("Received interrupt signal")
                    break
                except Exception as e:
                    self.error_count += 1
                    logger.error(f"Error in main loop: {e}\n{traceback.format_exc()}")
                    if self.error_count >= self.max_errors:
                        logger.critical("Too many errors, shutting down")
                        break
                        
        except Exception as e:
            logger.critical(f"Fatal error: {e}\n{traceback.format_exc()}")
        finally:
            self.cleanup()

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}")
        self.cleanup()
        sys.exit(0)

    def cleanup(self):
        """Cleanup resources"""
        try:
            logger.info("Starting cleanup...")
            if hasattr(self, 'screen'):
                self.screen.cleanup()
            if hasattr(self, 'voice_input'):
                self.voice_input.cleanup()
            logger.info("Cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

if __name__ == "__main__":
    try:
        jarvis = JARVIS()
        jarvis.run()
    except KeyboardInterrupt:
        print("\nShutting down JARVIS...")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)
