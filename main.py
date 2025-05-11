import os
import sys 
import json
import signal
import logging
from typing import Dict, Any, List
from pathlib import Path
from datetime import datetime
import torch
import yaml

# Import core components
from core.logging import setup_logging, get_logger
from core.brain.cerebrum import Cerebrum
from core.command.command_parser import CommandParser
from core.command.executor import CommandExecutor
from db.manager import DatabaseManager
from llm.core.llm_core import LLMCore
from ui.screen import Screen
from core.constants import OPENGL_AVAILABLE
from core.config import ConfigValidator

# Add new imports
from models.jarvis.model import JarvisModel
from llm.knowledge import KnowledgeManager
from llm.learning.learning_manager import LearningManager
from llm.inference.inference_engine import InferenceEngine
from ml.models import ModelManager

# Set up logger
logger = get_logger(__name__)

class JARVIS:
    def __init__(self):
        self.error_count = 0
        self.max_errors = 3
        self.initialized = False
        self.config = {}
        self.logger = get_logger(__name__)  # Initialize instance logger
        
        try:
            # Run installation/configuration if needed
            if not Path("config/installed.flag").exists():
                from core.setup.installer import SystemInstaller
                installer = SystemInstaller()
                installer.install()
                Path("config/installed.flag").touch()
                
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
                "version": "0.0.0",
                "language": "nl-NL",
                "log_level": "INFO",
                "memory_limit": "16G",
                "gpu_enabled": OPENGL_AVAILABLE
            },
            "nlp": {
                "language": "nl",
                "model": "jarvis-nlp",
                "tokenizer": "bert-base-multilingual-cased",
                "max_length": 100,
                "models": {
                    "sentiment": "distilbert-base-multilingual-cased",
                    "intent": "bert-base-multilingual-cased",
                    "embedding": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                }
            },
            "llm": {
                "model": {
                    "name": "jarvis-llm",
                    "type": "transformer",
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
            
            # Load LLM config with fallback options
            llm_config = self.config.get('llm', {})
            if not llm_config:
                yaml_path = Path("config/llm.yaml")
                if yaml_path.exists():
                    try:
                        with open(yaml_path) as f:
                            llm_config = yaml.safe_load(f)
                    except Exception as e:
                        self.logger.warning(f"Failed to load LLM config: {e}")
                        llm_config = {"model": {"name": "nl_core_news_lg", "type": "spacy"}}
                        
            # Initialize LLM with config
            try:
                self.llm = LLMCore(config=llm_config)
            except Exception as e:
                self.logger.error(f"Failed to initialize primary LLM: {e}")
                # Try fallback configuration
                fallback_config = {"model": {"name": "nl_core_news_sm", "type": "spacy"}}
                self.llm = LLMCore(config=fallback_config)
            
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
            # Initialize model
            model_name = self.config.get("model", "jarvis-base")
            self.model = JarvisModel(model_name)
            
            # Verify CUDA availability
            if torch.cuda.is_available():
                self.model = self.model.to('cuda')
                self.logger.info("Using CUDA for model acceleration")
                
            # Initialize model manager first
            self.model_manager = ModelManager()
            
            # Connect pipelines - with error handling
            try:
                text = "Initialize system check"
                tasks = ["classification", "generation", "qa"]
                results = self.model.process_pipeline(text, tasks)
                self.logger.info("Pipeline test successful")
            except Exception as e:
                self.logger.warning(f"Pipeline test failed: {e}, continuing initialization")
            
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

    def process_input(self, text: str, tasks: List[str]) -> Dict[str, Any]:
        """Process user input through model pipeline"""
        try:
            return self.model.process_pipeline(text, tasks)
        except Exception as e:
            self.logger.error(f"Error processing input: {e}")
            return {"error": str(e)}
            
    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get system metrics for UI display"""
        try:
            metrics = {
                "cpu_usage": 0.0,
                "memory_usage": 0.0,
                "gpu_usage": 0.0 if torch.cuda.is_available() else None,
                "model_loaded": hasattr(self, "model"),
                "llm_loaded": hasattr(self, "llm"),
                "uptime_seconds": 0  # Would need to track start time
            }
            
            # Try to get more detailed metrics if psutil is available
            try:
                import psutil
                process = psutil.Process(os.getpid())
                metrics["cpu_usage"] = process.cpu_percent()
                metrics["memory_usage"] = process.memory_info().rss / (1024 * 1024)  # MB
            except ImportError:
                pass
                
            # Get GPU metrics if available
            if torch.cuda.is_available():
                try:
                    metrics["gpu_usage"] = torch.cuda.memory_allocated() / torch.cuda.memory_reserved() * 100
                except:
                    pass
                    
            return metrics
        except Exception as e:
            self.logger.error(f"Error getting system metrics: {e}")
            return {"error": str(e)}

    def run(self):
        """Main execution loop"""
        if not self.initialized:
            self.logger.error("Cannot run: JARVIS not properly initialized")
            return

        try:
            self.logger.info("Starting main loop...")
            while not self.screen.should_quit:
                try:
                    if not self.screen.process_frame({
                        "timestamp": datetime.now(),
                        "status": "running",
                        "metrics": self._get_system_metrics()
                    }):
                        break  # Exit if window was closed
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
            self.logger.info("JARVIS shutdown complete")

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