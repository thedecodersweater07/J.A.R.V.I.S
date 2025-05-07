import os
import sys
import json
import signal
from typing import Dict, Any
from pathlib import Path
import logging
from datetime import datetime

from ui.screens.base.screen import Screen 
from ui.screens.base_screen import BaseScreen
from core.brain.cognitive.cerebrum import Cerebrum
from core.logging.logger import setup_logging
from config.manager import ConfigManager
from llm.core.llm_core import LLMCore
from ml.models.model_manager import ModelManager

logger = logging.getLogger(__name__)

class JARVIS:
    def __init__(self):
        self.error_count = 0
        self.config_manager = None
        self.initialized = False
        
        try:
            # Setup logging first
            setup_logging(level="DEBUG" if "--debug" in sys.argv else "INFO")
            
            # Initialize components
            self._init_config()
            self._init_core_components()
            self._init_ml_components()
            self._init_ui()
            
            self.initialized = True
            logger.info("JARVIS initialization complete")
            
        except Exception as e:
            logger.critical(f"Failed to initialize JARVIS: {e}", exc_info=True)
            raise

    def _init_config(self):
        """Initialize configuration system"""
        try:
            self.config_manager = ConfigManager()
            logger.info("Configuration system initialized")
        except Exception as e:
            logger.error("Failed to initialize config system", exc_info=True)
            raise

    def _init_core_components(self):
        """Initialize core AI components"""
        try:
            self.brain = Cerebrum()
            
            # Get LLM config with fallback
            llm_config = self.config_manager.get('llm')
            if not llm_config:
                logger.warning("No LLM config found, using defaults")
                llm_config = {
                    "model": {
                        "name": "gpt2",
                        "max_length": 100
                    }
                }
            
            # Initialize LLM with config
            self.llm = LLMCore(config=llm_config)
            self.brain.initialize()
            logger.info("Core components initialized")
            
        except Exception as e:
            logger.error("Core components initialization failed", exc_info=True)
            raise

    def _init_ml_components(self):
        try:
            self.model_manager = ModelManager()
            logger.info("ML components initialized")
        except Exception as e:
            logger.error(f"ML initialization failed: {e}")
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
            logger.info("UI initialized successfully")
        except Exception as e:
            logger.error(f"UI initialization failed: {e}")
            raise

    def run(self):
        """Main application loop with improved error handling"""
        if not self.initialized:
            logger.error("Cannot run: JARVIS not properly initialized")
            return

        try:
            logger.info("Starting main loop...")
            while not self.screen.should_exit:
                try:
                    self.screen.render({
                        "timestamp": datetime.now(),
                        "status": "running",
                        "metrics": self._get_system_metrics()
                    })
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    self.error_count += 1
                    if self.error_count > 10:
                        raise RuntimeError("Too many errors in main loop")
                    
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        except Exception as e:
            logger.critical(f"Fatal error in main loop: {e}", exc_info=True)
            raise
        finally:
            self._cleanup()

    def _cleanup(self):
        logger.info("Starting cleanup...")
        try:
            if hasattr(self, 'screen'):
                self.screen.cleanup()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

if __name__ == "__main__":
    jarvis = JARVIS()
    jarvis.run()
