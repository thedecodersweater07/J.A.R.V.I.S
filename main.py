import logging
import os
import sys
import time
import yaml
from typing import Dict, Any, List, Optional

# Add project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Suppress TensorFlow oneDNN optimization warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Graceful imports for optional dependencies
try:
    import psutil
except ImportError:
    psutil = None
    print("[WARNING] psutil not found. System metric monitoring will be disabled.")

try:
    from core.logging.logger import setup_logging
    from db.manager import DatabaseManager
    from llm.core.llm_core import LLMCore
    from llm.knowledge.knowledge_manager import KnowledgeManager
    from models.jarvis import JarvisModel
    from ui.screen import Screen
except ImportError as e:
    print(f"[ERROR] Failed to import a critical module: {e}. Please check your project structure.")
    sys.exit(1)

class JARVIS:
    """The main orchestrator for the J.A.R.V.I.S. application."""

    def __init__(self):
        # --- Step 1: Load Configuration --- 
        self.config = self._load_config()

        # --- Step 2: Setup Logging (immediately after config) ---
        setup_logging(self.config.get('logging', {}))
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Logger initialized.")

        # --- Step 3: Initialize Core Attributes ---
        self.db_manager: Optional[DatabaseManager] = None
        self.knowledge: Optional[KnowledgeManager] = None
        self.llm: Optional[LLMCore] = None
        self.model: Optional[JarvisModel] = None
        self.ui: Optional[Screen] = None

        # --- Step 4: Initialize Components ---
        try:
            self._init_core_components()
            self._init_ai_components()
            # self._init_ui() # UI is disabled for now
        except Exception as e:
            self.logger.critical(f"A critical error occurred during initialization: {e}", exc_info=True)
            sys.exit(1)

    def _load_config(self) -> Dict[str, Any]:
        """Loads configuration from a YAML file."""
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.yaml')
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except (IOError, yaml.YAMLError) as e:
            print(f"[WARNING] Could not load config.yaml: {e}. Using default/empty config.")
            return {}

    def _init_core_components(self):
        """Initializes database and knowledge manager."""
        self.logger.info("Initializing core components...")
        db_config = self.config.get('database', {})
        self.db_manager = DatabaseManager(db_config)
        self.knowledge = KnowledgeManager(self.db_manager)
        self.logger.info("Database and Knowledge Manager initialized.")

    def _init_ai_components(self):
        """Initializes AI components like LLM and the main model."""
        self.logger.info("Initializing AI components...")
        llm_config = self.config.get('llm', {})
        self.llm = LLMCore(config=llm_config)
        self.logger.info("LLM Core initialized.")

        jarvis_model_config = self.config.get('jarvis_model', {})
        self.model = JarvisModel(
            config=jarvis_model_config,
            llm=self.llm
        )
        # Store knowledge separately if needed
        self.knowledge = self.knowledge
        self.logger.info("JarvisModel orchestrator initialized.")

    def _init_ui(self):
        """Initializes the user interface (currently disabled)."""
        if not self.config.get('ui', {}).get('enabled', False):
            self.logger.info("UI is disabled in the configuration.")
            return
        self.logger.info("Initializing UI...")
        try:
            self.ui = Screen()
            self.logger.info("UI initialized successfully.")
        except Exception as e:
            self.logger.error(f"UI initialization failed: {e}. Running in headless mode.", exc_info=True)
            self.ui = None

    def run(self):
        """Starts the main application loop."""
        self.logger.info("J.A.R.V.I.S. is now running in headless mode.")
        self.logger.info("Press Ctrl+C to exit.")
        try:
            while True:
                # In headless mode, the app will wait here. 
                # Future implementations could handle background tasks.
                time.sleep(1)
        except KeyboardInterrupt:
            self.logger.info("Shutdown signal received.")
        finally:
            self.shutdown()

    def shutdown(self):
        """Gracefully shuts down the application."""
        self.logger.info("Shutting down J.A.R.V.I.S...")
        # Future: Add shutdown logic for components (e.g., DB connection)
        self.logger.info("Shutdown complete.")

if __name__ == '__main__':
    jarvis_app = JARVIS()
    jarvis_app.run()
