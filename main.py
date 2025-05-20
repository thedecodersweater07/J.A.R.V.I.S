import os
import sys 
import json
import signal
import logging
import threading

# Suppress TensorFlow oneDNN warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from typing import Dict, Any, List, Optional
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

# Import new AI architecture components
try:
    from core.ai.coordinator import AICoordinator
    from core.ai.model_registry import ModelRegistry
    from core.ai.pipeline import PipelineManager
    from core.ai.events import EventBus
    from core.ai.resource_manager import ResourceManager
except ImportError:
    print("New AI architecture components not found, using legacy components")
# Import optional components with error handling
APIClient = None
server_launcher = None

try:
    from ui.api_client import APIClient
except ImportError:
    print("APIClient not found, API features will be disabled")

try:
    from server.launcher import start_server
    server_launcher = start_server
except ImportError:
    print("Server launcher not found, server features will be disabled")

# Set up logger
logger = get_logger(__name__)

class JARVIS:
    def __init__(self):
        self.error_count = 0
        self.max_errors = 3
        self.initialized = False
        self.config = {}
        self.logger = get_logger(__name__)  # Initialize instance logger
        self.server_thread = None
        self.api_client = None
        
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
            
            # Start server backend if enabled
            if self.config.get("use_server", True):
                self._start_server_backend()
            
            # Initialize API client
            self._init_api_client()
            
            # Initialize UI last
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
                "version": "1.0.0",
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
            },
            "resources": {
                "memory_limit": "80%",
                "cpu_limit": 0.8,
                "gpu_memory_limit": 0.8,
                "enable_monitoring": True,
                "monitoring_interval": 5.0
            },
            "models": {
                "model_paths": [
                    "llm/models",
                    "ml/models",
                    "nlp/models"
                ]
            },
            "pipelines": {
                "default_timeout": 30.0,
                "enable_metrics": True
            },
            "events": {
                "async_mode": True,
                "max_history": 100
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
            # Initialize AI Coordinator and related components first
            from core.ai.coordinator import AICoordinator
            from core.ai.model_registry import ModelRegistry
            from core.ai.pipeline import PipelineManager
            from core.ai.events import EventBus
            from core.ai.resource_manager import ResourceManager
            
            # Initialize resource manager first to optimize resource allocation
            self.resource_manager = ResourceManager(config=self.config.get('resources', {}))
            
            # Initialize event bus for component communication
            self.event_bus = EventBus(config=self.config.get('events', {}))
            
            # Initialize model registry
            self.model_registry = ModelRegistry(config=self.config.get('models', {}))
            
            # Initialize pipeline manager
            self.pipeline_manager = PipelineManager(config=self.config.get('pipelines', {}))
            
            # Initialize brain
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
                
            # Initialize AI Coordinator last, after all components are ready
            # Skip AI coordinator initialization for now to avoid errors
            self.logger.info("Skipping AI Coordinator initialization for compatibility")
            self.ai_coordinator = None
            
            # Initialize resource manager directly if possible
            try:
                from core.ai.resource_manager import ResourceManager
                self.resource_manager = ResourceManager(config=self.config.get('resources', {}))
                self.logger.info("Resource manager initialized successfully")
            except Exception as e:
                self.logger.warning(f"Resource manager initialization failed: {e}")
                self.resource_manager = None
                
            # Initialize event bus directly if possible
            try:
                from core.ai.events import EventBus
                self.event_bus = EventBus(config=self.config.get('events', {}))
                self.logger.info("Event bus initialized successfully")
            except Exception as e:
                self.logger.warning(f"Event bus initialization failed: {e}")
                self.event_bus = None
                
            # Initialize model registry directly if possible
            try:
                from core.ai.model_registry import ModelRegistry
                self.model_registry = ModelRegistry(config=self.config.get('models', {}))
                self.logger.info("Model registry initialized successfully")
            except Exception as e:
                self.logger.warning(f"Model registry initialization failed: {e}")
                self.model_registry = None
            
            self.logger.info("Core AI components initialized")
            
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
        """Initialize machine learning components with improved error handling"""
        try:
            # Initialize model with error handling
            try:
                model_name = self.config.get("model", "jarvis-base")
                self.model = JarvisModel(model_name)
                
                # Verify CUDA availability
                if torch.cuda.is_available():
                    try:
                        self.model = self.model.to('cuda')
                        self.logger.info("Using CUDA for model acceleration")
                    except Exception as cuda_error:
                        self.logger.warning(f"Failed to move model to CUDA: {cuda_error}")
                        self.logger.info("Continuing with CPU model")
            except Exception as model_error:
                self.logger.error(f"Failed to initialize main model: {model_error}")
                self.model = None
                
            # Initialize AI Coordinator with improved error handling
            try:
                from core.ai.coordinator import AICoordinator
                import traceback
                
                # Create a comprehensive config for the coordinator
                ai_config = {
                    "llm": self.config.get("llm", {}),
                    "nlp": self.config.get("nlp", {}),
                    "ml": {
                        "base_path": os.path.abspath("data/models")
                    },
                    "resources": self.config.get("resources", {}),
                    "events": self.config.get("events", {}),
                    "models": self.config.get("models", {})
                }
                
                # Initialize the AI coordinator
                self.ai_coordinator = AICoordinator(config=ai_config)
                self.ai_coordinator.initialize()
                
                # Get components from the coordinator
                self.model_manager = self.ai_coordinator.get_component("model_manager")
                self.nlp_processor = self.ai_coordinator.get_component("nlp")
                
                self.logger.info("AI Coordinator initialized successfully")
            except Exception as ai_error:
                self.logger.error(f"AI Coordinator initialization failed: {ai_error}")
                self.logger.error(traceback.format_exc())
                self.ai_coordinator = None
                
                # Fall back to direct initialization if coordinator fails
                try:
                    # Initialize model manager directly
                    from ml.model_manager import ModelManager
                    model_path = os.path.abspath("data/models")
                    self.model_manager = ModelManager(base_path=model_path)
                    self.logger.info(f"ModelManager initialized directly with path: {model_path}")
                except Exception as mm_error:
                    self.logger.error(f"Direct ModelManager initialization failed: {mm_error}")
                    self.model_manager = None
                
                try:
                    # Initialize NLP processor directly
                    from nlp.processor import NLPProcessor
                    nlp_model = self.config.get("nlp", {}).get("model", "nl_core_news_sm")
                    self.nlp_processor = NLPProcessor(model_name=nlp_model)
                    self.logger.info(f"NLPProcessor initialized directly with model: {nlp_model}")
                except Exception as nlp_error:
                    self.logger.error(f"Direct NLPProcessor initialization failed: {nlp_error}")
                    self.nlp_processor = None
            
            # Test pipeline if main model is available
            if self.model:
                try:
                    text = "Initialize system check"
                    tasks = ["classification", "generation", "qa"]
                    results = self.model.process_pipeline(text, tasks)
                    self.logger.info("Pipeline test successful")
                except Exception as pipeline_error:
                    self.logger.warning(f"Pipeline test failed: {pipeline_error}, continuing initialization")
            
            # Log initialization status
            components_status = []
            if self.model:
                components_status.append("JarvisModel")
            if self.ai_coordinator:
                components_status.append("AICoordinator")
            if self.model_manager:
                components_status.append("ModelManager")
            if self.nlp_processor:
                components_status.append("NLPProcessor")
                
            if components_status:
                self.logger.info(f"ML components initialized: {', '.join(components_status)}")
            else:
                self.logger.warning("No ML components were successfully initialized")
                
        except Exception as e:
            self.logger.error(f"ML initialization failed: {e}")
            self.logger.error(traceback.format_exc())
            # Don't raise the exception to allow the system to continue with limited functionality
            self.logger.warning("Continuing with limited ML functionality")
            
    def _start_server_backend(self):
        """Start the server backend in a separate thread"""
        # Check if server launcher is available
        if server_launcher is None:
            self.logger.warning("Server launcher not available, skipping server startup")
            return False
            
        try:
            def run_server():
                """Run the server in a separate process"""
                try:
                    # Get server config
                    server_config = self.config.get('server', {})
                    host = server_config.get('host', '127.0.0.1')
                    port = server_config.get('port', 8000)
                    debug = '--debug' in sys.argv
                    
                    # Start server using the launcher
                    server_launcher(host=host, port=port, debug=debug)
                except Exception as e:
                    self.logger.error(f"Server error: {e}")
            
            # Start server in a separate thread
            self.server_thread = threading.Thread(target=run_server, daemon=True)
            self.server_thread.start()
            
            self.logger.info("Server backend started")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start server backend: {e}")
            return False
    
    def _init_api_client(self):
        """Initialize API client for UI-Server communication"""
        # Check if APIClient is available
        if APIClient is None:
            self.logger.warning("APIClient not available, skipping API client initialization")
            return False
            
        try:
            # Get server config
            server_config = self.config.get('server', {})
            host = server_config.get('host', '127.0.0.1')
            port = server_config.get('port', 8000)
            
            # Create API client
            base_url = f"http://{host}:{port}"
            self.api_client = APIClient(base_url=base_url)
            
            self.logger.info(f"API client initialized with URL: {base_url}")
            return True
            
        except Exception as e:
            self.logger.error(f"API client initialization failed: {e}")
            return False

    def _init_ui(self):
        """Initialize UI components"""
        try:
            # Get UI config
            ui_config = self.config.get('ui', {})
            width = ui_config.get('width', 800)
            height = ui_config.get('height', 600)
            title = ui_config.get('title', 'JARVIS Interface')
            
            # Initialize screen with render mode
            from ui.rendering.renderer_factory import RenderMode
            render_mode = RenderMode.OPENGL if OPENGL_AVAILABLE else RenderMode.TEXT
            
            self.screen = Screen(
                width=width, 
                height=height, 
                title=title
            )
            
            # Initialize renderer first
            if not self.screen.init():
                self.logger.error("Failed to initialize Screen")
                return False
            
            # Set components after renderer is initialized
            if self.llm:
                self.screen.set_llm(self.llm)
            if self.model_manager:
                self.screen.set_model_manager(self.model_manager)
            if self.api_client:
                self.screen.set_api_client(self.api_client)
                
            self.logger.info("UI initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"UI initialization failed: {e}")
            return False

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
            self.shutdown()
            self.logger.info("JARVIS shutdown complete")

    def shutdown(self):
        """Properly shut down JARVIS"""
        self.logger.info("Shutting down JARVIS...")
        
        # Stop server if running
        if self.server_thread and self.server_thread.is_alive():
            self.logger.info("Stopping server...")
            # Signal server to stop
            
        # Shut down AI components in the correct order if they were initialized
        if hasattr(self, 'ai_coordinator') and self.ai_coordinator:
            try:
                self.logger.info("Shutting down AI coordinator...")
                self.ai_coordinator.shutdown()
            except Exception as e:
                self.logger.warning(f"Error shutting down AI coordinator: {e}")
            
        if hasattr(self, 'pipeline_manager') and self.pipeline_manager:
            try:
                self.logger.info("Shutting down pipeline manager...")
                # No explicit shutdown needed
            except Exception as e:
                self.logger.warning(f"Error shutting down pipeline manager: {e}")
            
        if hasattr(self, 'event_bus') and self.event_bus:
            try:
                self.logger.info("Shutting down event bus...")
                self.event_bus.shutdown()
            except Exception as e:
                self.logger.warning(f"Error shutting down event bus: {e}")
            
        if hasattr(self, 'resource_manager') and self.resource_manager:
            try:
                self.logger.info("Shutting down resource manager...")
                self.resource_manager.shutdown()
            except Exception as e:
                self.logger.warning(f"Error shutting down resource manager: {e}")
            
        # Close database connections
        if hasattr(self, 'db_manager') and self.db_manager:
            self.logger.info("Closing database connections...")
            self.db_manager.close()
            
        # Shut down UI
        if hasattr(self, 'screen') and self.screen:
            self.logger.info("Shutting down UI...")
            self.screen.shutdown()
            
        self.logger.info("JARVIS shutdown complete")
        logging.shutdown()

    def _cleanup(self):
        self.logger.info("Starting cleanup...")
        try:
            if hasattr(self, 'screen'):
                self.screen.cleanup()
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

if __name__ == "__main__":
    try:
        print("Starting JARVIS initialization...")
        jarvis = JARVIS()
        print("JARVIS initialized successfully, starting run...")
        jarvis.run()
    except Exception as e:
        import traceback
        error_msg = f"Error during JARVIS initialization or run: {e}"
        print(error_msg)
        
        # Write error to a file we can access
        with open("error_log.txt", "w") as f:
            f.write(error_msg + "\n\n")
            f.write(traceback.format_exc())
        
        print(f"Detailed error information has been written to error_log.txt")
        traceback.print_exc()