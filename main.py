import os
import sys
import traceback
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Now import core modules
from core import setup_logging, Cerebrum
from core.command import CommandParser, CommandExecutor
from ui.screen import Screen
from ui.visual import HologramProjector
from ui.input import VoiceInput, TextInput
from security.authentication import IdentityVerifier
from ml.model_manager import ModelManager

logger = setup_logging()

class JARVIS:
    def __init__(self):
        try:
            # Load configuration
            self.config = self._load_config()
            
            # Initialize core components
            self.brain = Cerebrum()
            self.command_parser = CommandParser()
            self.executor = CommandExecutor()
            
            # Initialize UI with fallback
            self.screen = self._init_screen()
            self.voice_input = VoiceInput() if self.config.get("use_voice", True) else None
            self.text_input = TextInput()
            
            # Initialize other components  
            self.security = IdentityVerifier()
            self.model_manager = ModelManager()
            
            self.error_count = 0
            self.max_errors = 3
            
        except ModuleNotFoundError as e:
            logger.critical(f"Missing module: {e}\nPlease check your Python path and module installation")
            raise
        except Exception as e:
            logger.critical(f"Failed to initialize JARVIS: {e}\n{traceback.format_exc()}")
            raise

    def _init_screen(self) -> Screen:
        """Initialize screen with fallback options"""
        try:
            return Screen(
                width=self.config.get("ui", {}).get("width", 1024),
                height=self.config.get("ui", {}).get("height", 768),
                title="JARVIS AI Interface"
            )
        except Exception as e:
            logger.warning(f"Failed to initialize graphical screen: {e}")
            return Screen(mode="text")

    def _load_config(self) -> Dict:
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
