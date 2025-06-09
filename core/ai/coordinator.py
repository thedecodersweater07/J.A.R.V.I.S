"""
AI Coordinator Module
Centralizes and manages all AI components in the JARVIS system.
"""

import os
import sys
import logging
import traceback
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path

# Import core components
from core.logging import get_logger
from core.knowledge.graph_manager import KnowledgeGraphManager

# Import AI component imports
from llm.core.llm_core import LLMCore

# Import adapters
from core.ai.adapters import NLPProcessorAdapter, ModelManagerAdapter
from server.knowledge_graph.manager import KnowledgeGraphManager

class AICoordinator:
    """
    Central coordinator for all AI components in JARVIS.
    Manages model loading, inference routing, and resource allocation.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the AI Coordinator with configuration.
        
        Args:
            config: Configuration dictionary for AI components
        """
        self.logger = get_logger(__name__)
        self.config = config or {}
        self.components = {}
        self.models = {}
        self.pipelines = {}
        self.initialized = False
        self.initialization_errors = {}
        self._components = {}  # Add components dictionary
        
        # Initialize component registry
        self._init_component_registry()
        self.knowledge_graph = KnowledgeGraphManager()
        
    def _init_component_registry(self):
        """Initialize the component registry with available AI components."""
        self.component_registry = {
            "llm": {
                "class": LLMCore,
                "config_key": "llm",
                "required": True,
                "instance": None,
                "init_params": ["config"],
                "fallback_config": {
                    "model": {
                        "name": "distilgpt2",  # Smaller model
                        "type": "transformer",
                        "low_cpu_mem_usage": True
                    }
                }
            },
            "nlp": {
                "class": NLPProcessorAdapter,
                "config_key": "nlp",
                "required": True,
                "instance": None,
                "init_params": ["model_name"],
                "fallback_config": {"model_name": "en_core_web_sm"}  # Smaller model
            },
            "model_manager": {
                "class": ModelManagerAdapter,
                "config_key": "ml",
                "required": True,
                "instance": None,
                "init_params": ["config"],
                "fallback_config": {
                    "base_path": "data/models",
                    "max_loaded_models": 2,  # Reduce memory usage
                    "use_gpu": False
                }
            }
        }
        
        # Add optional components if available
        try:
            from server.knowledge_graph.manager import KnowledgeGraphManager  # Adjust the import path based on your project structure
            self.component_registry["knowledge_graph"] = {
                "class": KnowledgeGraphManager,
                "config_key": "knowledge_graph",
                "required": False,
                "instance": None,
                "init_params": ["config"]
            }
        except ImportError:
            self.logger.debug("KnowledgeGraphManager not available")
            
        try:
            from llm.inference.inference_engine import InferenceEngine
            self.component_registry["inference"] = {
                "class": InferenceEngine,
                "config_key": "inference",
                "required": False,
                "instance": None,
                "init_params": ["config"]
            }
        except ImportError:
            self.logger.debug("InferenceEngine not available")
    
    def initialize(self):
        """Initialize all AI components based on configuration."""
        if self.initialized:
            self.logger.warning("AI Coordinator already initialized")
            return
            
        self.logger.info("Initializing AI Coordinator")
        initialization_success = True
        
        # Initialize required components first
        for name, component_info in self.component_registry.items():
            if component_info["required"]:
                success = self._init_component(name, component_info)
                if not success and component_info["required"]:
                    initialization_success = False
                
        # Then initialize optional components
        for name, component_info in self.component_registry.items():
            if not component_info["required"] and component_info["instance"] is None:
                self._init_component(name, component_info)
                
        self.initialized = initialization_success
        
        if initialization_success:
            self.logger.info("AI Coordinator initialization complete")
        else:
            self.logger.warning("AI Coordinator initialized with errors: " + 
                             ", ".join([f"{k}: {v}" for k, v in self.initialization_errors.items()]))
        
    def _init_component(self, name: str, component_info: Dict[str, Any]) -> bool:
        """Initialize a specific AI component with improved error handling"""
        success = False
        try:
            config_key = component_info["config_key"]
            component_config = self.config.get(config_key, {})
            
            # Apply default/fallback config if needed
            if not component_config and "fallback_config" in component_info:
                component_config = component_info["fallback_config"]
                self.logger.info(f"Using fallback config for {name}")

            # Initialize component
            component_class = component_info["class"]
            component_instance = component_class(config=component_config)
            
            # Verify component initialization
            if hasattr(component_instance, 'initialize'):
                init_success = component_instance.initialize()
                if not init_success:
                    self.logger.error(f"Component {name} initialization returned False")
                    return False
            
            # Store the initialized component
            component_info["instance"] = component_instance
            self.components[name] = component_instance
            
            self.logger.info(f"Successfully initialized {name} component")
            success = True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize component {name}: {str(e)}", exc_info=True)
            self.initialization_errors[name] = str(e)
            
            if "fallback_config" in component_info:
                try:
                    fallback_config = component_info["fallback_config"]
                    component_class = component_info["class"]
                    component_instance = component_class(config=fallback_config)
                    component_info["instance"] = component_instance 
                    self.components[name] = component_instance
                    success = True
                except Exception as fallback_error:
                    self.logger.error(f"Fallback initialization for {name} failed: {fallback_error}")
        
        return success
    
    def get_component(self, name: str) -> Any:
        """
        Get an initialized AI component by name.
        
        Args:
            name: Name of the component to retrieve
            
        Returns:
            The component instance or None if not found
        """
        if not self.initialized:
            self.logger.warning("AI Coordinator not initialized, initializing now")
            self.initialize()
            
        component = self.components.get(name)
        if component is None:
            self.logger.warning(f"Component {name} not found or not initialized")
            
        return component
    
    def process_input(self, text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process input text through the AI pipeline.
        
        Args:
            text: Input text to process
            context: Additional context for processing
            
        Returns:
            Dictionary with processing results
        """
        if not self.initialized:
            self.logger.warning("AI Coordinator not initialized, initializing now")
            self.initialize()
            
        context = context or {}
        result = {"input": text, "processed": False, "components_used": []}
        
        try:
            # Process through NLP pipeline
            nlp = self.get_component("nlp")
            if nlp:
                try:
                    nlp_result = nlp.process(text, context)
                    result.update({"nlp_result": nlp_result})
                    result["components_used"].append("nlp")
                except Exception as e:
                    self.logger.error(f"Error in NLP processing: {e}")
                    result["nlp_error"] = str(e)
                
            # Process through LLM if available
            llm = self.get_component("llm")
            if llm:
                try:
                    llm_result = llm.process(text, context)
                    result.update({"llm_result": llm_result})
                    result["components_used"].append("llm")
                except Exception as e:
                    self.logger.error(f"Error in LLM processing: {e}")
                    result["llm_error"] = str(e)
                
            # Mark as processed if at least one component was used successfully
            result["processed"] = len(result["components_used"]) > 0
            
        except Exception as e:
            self.logger.error(f"Error processing input: {e}", exc_info=True)
            result["error"] = str(e)
            
        return result
    
    def get_model(self, model_name: str, model_type: str = None) -> Any:
        """
        Get a model by name and optional type.
        
        Args:
            model_name: Name of the model to retrieve
            model_type: Optional type of model (nlp, llm, ml)
            
        Returns:
            The model instance or None if not found
        """
        if not self.initialized:
            self.initialize()
            
        # If type is specified, try to get from specific component
        if model_type:
            if model_type == "nlp":
                component = self.get_component("nlp")
                if component and hasattr(component, "processor"):
                    return component.processor
            elif model_type == "llm":
                component = self.get_component("llm")
                if component and hasattr(component, "model"):
                    return component.model
            elif model_type == "ml":
                component = self.get_component("model_manager")
                if component:
                    return component.get_model(model_name)
        
        # If type not specified or not found, try model manager
        model_manager = self.get_component("model_manager")
        if model_manager:
            model = model_manager.get_model(model_name)
            if model:
                return model
                
        # If still not found, check all components
        for name, component in self.components.items():
            if hasattr(component, "get_model"):
                model = component.get_model(model_name)
                if model:
                    return model
                    
        self.logger.warning(f"Model {model_name} not found")
        return None
        
    def load_model(self, model_name: str, model_path: str, model_type: str = None) -> Any:
        """
        Load a model by name, path and optional type.
        
        Args:
            model_name: Name of the model to load
            model_path: Path to the model
            model_type: Optional type of model (nlp, llm, ml)
            
        Returns:
            The loaded model or None if loading failed
        """
        if not self.initialized:
            self.initialize()
            
        # Try to load through model manager first
        model_manager = self.get_component("model_manager")
        if model_manager:
            try:
                return model_manager.load_model(model_name, model_path)
            except Exception as e:
                self.logger.error(f"Error loading model {model_name} through model manager: {e}")
                
        # If model manager failed or not available, try specific component
        if model_type:
            component = self.get_component(model_type)
            if component and hasattr(component, "load_model"):
                try:
                    return component.load_model(model_name, model_path)
                except Exception as e:
                    self.logger.error(f"Error loading model {model_name} through {model_type} component: {e}")
                    
        self.logger.error(f"Failed to load model {model_name}")
        return None
        
    def get_status(self) -> Dict[str, Any]:
        """
        Get the status of all components.
        
        Returns:
            Dictionary with component status information
        """
        status = {
            "initialized": self.initialized,
            "components": {},
            "errors": self.initialization_errors
        }
        
        for name, component in self.components.items():
            component_status = {
                "available": component is not None,
                "type": component.__class__.__name__ if component else "None"
            }
            
            # Add component-specific status if available
            if hasattr(component, "get_status"):
                try:
                    component_status.update(component.get_status())
                except Exception as e:
                    component_status["status_error"] = str(e)
                    
            status["components"][name] = component_status
            
        return status
    
    def shutdown(self):
        """Shutdown AI coordinator and all components"""
        try:
            for name, component in self._components.items():
                try:
                    if hasattr(component, 'shutdown'):
                        component.shutdown()
                except Exception as e:
                    self.logger.error(f"Error shutting down component {name}: {e}")
            self._components.clear()
        except Exception as e:
            self.logger.error(f"Error during AI coordinator shutdown: {e}")
