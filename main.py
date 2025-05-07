import os
import sys 
import json
import signal
import logging
from typing import Dict, Any
from pathlib import Path
from datetime import datetime

# Import core components
from core.logging import setup_logging, get_logger
from core.brain.cerebrum import Cerebrum
from core.command.command_parser import CommandParser
from core.command.executor import CommandExecutor
from ui.screen import Screen
from ui.input.voice_input import VoiceInput
from ui.input.text_input import TextInput
from security.authentication.identity_verifier import IdentityVerifier
from ml.model_manager import ModelManager
from llm.core import LLMCore
from llm.learning import LearningManager
from llm.knowledge import KnowledgeManager
from llm.inference import InferenceEngine
from nlp.language_processor import LanguageProcessor
from nlp.conversation.conversation_handler import ConversationHandler
from config.config_validator import ConfigValidator
from config.validation_schemas import NLP_SCHEMA, LLM_SCHEMA, UI_SCHEMA, DATABASE_SCHEMA
from db.manager import DatabaseManager

# Set up logger
logger = get_logger(__name__)

# Check OpenGL availability
try:
    import OpenGL
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False


class JARVIS:
    def __init__(self):
        self.error_count = 0
        self.max_errors = 3
        self.initialized = False
        self.config = {}
        self.logger = get_logger(__name__)  # Initialize instance logger
        
        try:
            # Setup logging first
            setup_logging(log_dir="logs", level="DEBUG" if "--debug" in sys.argv else "INFO")
            
            # Initialize config manager first
            self.config_validator = ConfigValidator()
            self.config = self._load_config()
            
            # Initialize remaining components
            self._init_core_components()
            self._init_ml_components() 
            self._init_ui()
            
            self.initialized = True
            self.logger.info("JARVIS initialization complete")
            
        except Exception as e:
            self.logger.critical(f"Failed to initialize JARVIS: {e}", exc_info=True)
            raise

        # Set OpenGL config after loading main config
        self.config["use_opengl"] = OPENGL_AVAILABLE and self.config.get("use_opengl", True)

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration settings"""
        return {
            "system": {
                "name": "JARVIS",
                "version": "2.5.0",
                "language": "nl-NL",
                "log_level": "INFO",
                "memory_limit": "16G",
                "gpu_enabled": OPENGL_AVAILABLE
            },
            "nlp": {
                "language": "nl",
                "model": "gpt2",
                "max_length": 100,
                "models": {
                    "sentiment": "distilbert-base-multilingual-cased",
                    "intent": "bert-base-multilingual-cased",
                    "embedding": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                }
            },
            "llm": {
                "model": {
                    "name": "gpt2",
                    "max_length": 100
                }
            },
            "ui": {
                "width": 800,
                "height": 600,
                "title": "JARVIS Interface",
                "theme": "dark"
            },
            "database": {
                "type": "sqlite",
                "path": "data/db",
                "connection": "sqlite:///data/db/jarvis.db"
            }
        }

    def _load_config(self) -> Dict[str, Any]:
        """Load and validate configuration"""
        try:
            config_path = Path("config") / "config.json"
            config = self._get_default_config()
            
            if config_path.exists():
                with open(config_path) as f:
                    user_config = json.load(f)
                    # Deep merge user config with defaults
                    for section, values in user_config.items():
                        if section in config:
                            config[section].update(values)
                
            # Validate configuration sections
            for section in ["nlp", "llm", "ui"]:
                if not self.config_validator.validate(config.get(section, {}), section):
                    self.logger.warning(f"Invalid {section} config, using defaults")
                    config[section] = self._get_default_config()[section]
            
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to initialize config system: {e}", exc_info=True)
            return self._get_default_config()

    def _init_core_components(self):
        """Initialize core AI components"""
        try:
            # Initialize brain first
            self.brain = Cerebrum()
            self._init_systems()
            
            # Initialize LLM with config
            llm_config = self.config.get('llm', self._get_default_config()['llm'])
            self.llm = LLMCore(config=llm_config)
            
            self.logger.info("Core components initialized")
            
        except Exception as e:
            self.logger.error(f"Core components initialization failed: {e}", exc_info=True)
            raise

    def _init_systems(self):
        """Initialize core system components"""
        try:
            # Initialize database first
            self.db_manager = DatabaseManager()
            
            # Initialize knowledge manager
            self.knowledge_manager = KnowledgeManager(self.db_manager)
            
            # Initialize learning components with knowledge manager
            learning_config = self.config.get('learning', {})
            self.learning_manager = LearningManager(
                config=learning_config,
                knowledge_manager=self.knowledge_manager
            )
            
            # Initialize other components
            self.inference_engine = InferenceEngine(self.knowledge_manager)
            
            self.logger.info("System components initialized")
        except Exception as e:
            self.logger.error(f"System initialization failed: {e}")
            raise

    def _init_ml_components(self):
        try:
            self.model_manager = ModelManager()
            self.logger.info("ML components initialized")
        except Exception as e:
            self.logger.error(f"ML initialization failed: {e}")
            raise

    def _init_ui(self):
        """Initialize UI components"""
        try:
            self.screen = Screen(800, 600, "JARVIS Interface")
            success = self.screen.init()
            if not success:
                raise RuntimeError("Failed to initialize screen")
            self.screen.set_llm(self.llm)
            self.screen.set_model_manager(self.model_manager)
            self.logger.info("UI initialized successfully")
        except Exception as e:
            self.logger.error(f"UI initialization failed: {e}")
            raise

    def run(self):
        """Main application loop with improved error handling"""
        if not self.initialized:
            self.logger.error("Cannot run: JARVIS not properly initialized")
            return

        try:
            self.logger.info("Starting main loop...")
            while not self.screen.should_exit:
                try:
                    self.screen.render({
                        "timestamp": datetime.now(),
                        "status": "running",
                        "metrics": self._get_system_metrics()
                    })
                except Exception as e:
                    self.logger.error(f"Error in main loop: {e}")
                    self.error_count += 1
                    if self.error_count > 10:
                        raise RuntimeError("Too many errors in main loop")
                    
        except KeyboardInterrupt:
            self.logger.info("Received shutdown signal")
        except Exception as e:
            self.logger.critical(f"Fatal error in main loop: {e}", exc_info=True)
            raise
        finally:
            self._cleanup()

    def _cleanup(self):
        self.logger.info("Starting cleanup...")
        try:
            if hasattr(self, 'screen'):
                self.screen.cleanup()
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

if __name__ == "__main__":
    jarvis = JARVIS()
    jarvis.run()
