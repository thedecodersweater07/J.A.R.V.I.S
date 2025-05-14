"""
Model Registry Module
Provides centralized registration and discovery of all AI models in JARVIS.
"""

import os
import sys
import logging
import json
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import importlib

# Import core components
from core.logging import get_logger

class ModelRegistry:
    """
    Central registry for all AI models in JARVIS.
    Handles model registration, discovery, and metadata management.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Model Registry with configuration.
        
        Args:
            config: Configuration dictionary for the model registry
        """
        self.logger = get_logger(__name__)
        self.config = config or {}
        self.models = {}
        self.model_paths = {}
        self.model_metadata = {}
        
        # Default registry paths
        self.registry_paths = [
            "llm/models",
            "ml/models",
            "nlp/models"
        ]
        
        # Add custom paths from config
        if "model_paths" in self.config:
            self.registry_paths.extend(self.config["model_paths"])
            
        # Initialize registry
        self._init_registry()
        
    def _init_registry(self):
        """Initialize the model registry by discovering available models."""
        self.logger.info("Initializing model registry")
        
        # Scan registry paths for models
        for path in self.registry_paths:
            self._scan_path(path)
            
        self.logger.info(f"Model registry initialized with {len(self.models)} models")
        
    def _scan_path(self, path: str):
        """
        Scan a path for model definitions and metadata.
        
        Args:
            path: Path to scan for models
        """
        full_path = Path(path)
        if not full_path.exists():
            self.logger.debug(f"Model path {path} does not exist")
            return
            
        self.logger.debug(f"Scanning {path} for models")
        
        # Look for model definition files
        for item in full_path.glob("**/*.py"):
            if item.name.startswith("_"):
                continue
                
            # Try to load model metadata
            try:
                module_path = str(item.relative_to(Path("."))).replace("\\", "/").replace(".py", "")
                module_path = module_path.replace("/", ".")
                
                # Check for metadata file
                metadata_file = item.parent / f"{item.stem}_metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, "r") as f:
                        metadata = json.load(f)
                        model_id = metadata.get("id", item.stem)
                        self.model_metadata[model_id] = metadata
                        self.model_paths[model_id] = module_path
                        self.models[model_id] = None  # Lazy loading
                        self.logger.debug(f"Registered model {model_id} from {module_path}")
                        
            except Exception as e:
                self.logger.warning(f"Error scanning model file {item}: {e}")
                
    def get_model(self, model_id: str, load: bool = True) -> Optional[Any]:
        """
        Get a model by ID, optionally loading it if not already loaded.
        
        Args:
            model_id: ID of the model to retrieve
            load: Whether to load the model if not already loaded
            
        Returns:
            The model instance or None if not found
        """
        # Check if model exists in registry
        if model_id not in self.model_paths:
            self.logger.warning(f"Model {model_id} not found in registry")
            return None
            
        # Load model if requested and not already loaded
        if load and self.models[model_id] is None:
            try:
                self._load_model(model_id)
            except Exception as e:
                self.logger.error(f"Error loading model {model_id}: {e}", exc_info=True)
                return None
                
        return self.models[model_id]
        
    def _load_model(self, model_id: str):
        """
        Load a model by ID.
        
        Args:
            model_id: ID of the model to load
        """
        module_path = self.model_paths[model_id]
        metadata = self.model_metadata.get(model_id, {})
        
        try:
            # Import the module
            module = importlib.import_module(module_path)
            
            # Get the model class
            model_class_name = metadata.get("class_name", "Model")
            if not hasattr(module, model_class_name):
                raise AttributeError(f"Module {module_path} has no class {model_class_name}")
                
            model_class = getattr(module, model_class_name)
            
            # Initialize the model
            model_config = metadata.get("config", {})
            model = model_class(**model_config)
            
            # Store the loaded model
            self.models[model_id] = model
            self.logger.info(f"Loaded model {model_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model {model_id} from {module_path}: {e}", exc_info=True)
            raise
            
    def get_available_models(self, model_type: str = None) -> List[str]:
        """
        Get a list of available model IDs, optionally filtered by type.
        
        Args:
            model_type: Optional type to filter models by
            
        Returns:
            List of model IDs
        """
        if model_type:
            return [
                model_id for model_id, metadata in self.model_metadata.items()
                if metadata.get("type") == model_type
            ]
        else:
            return list(self.model_paths.keys())
            
    def get_model_metadata(self, model_id: str) -> Dict[str, Any]:
        """
        Get metadata for a model.
        
        Args:
            model_id: ID of the model to get metadata for
            
        Returns:
            Model metadata dictionary
        """
        return self.model_metadata.get(model_id, {})
        
    def register_model(self, model_id: str, model_instance: Any, metadata: Dict[str, Any] = None):
        """
        Register a model instance with the registry.
        
        Args:
            model_id: ID to register the model under
            model_instance: The model instance to register
            metadata: Optional metadata for the model
        """
        if model_id in self.models:
            self.logger.warning(f"Overwriting existing model {model_id}")
            
        self.models[model_id] = model_instance
        
        if metadata:
            self.model_metadata[model_id] = metadata
            
        self.logger.info(f"Registered model {model_id}")
        
    def unregister_model(self, model_id: str):
        """
        Unregister a model from the registry.
        
        Args:
            model_id: ID of the model to unregister
        """
        if model_id not in self.models:
            self.logger.warning(f"Model {model_id} not found in registry")
            return
            
        # Remove model from registry
        del self.models[model_id]
        
        # Remove metadata and path if they exist
        if model_id in self.model_metadata:
            del self.model_metadata[model_id]
            
        if model_id in self.model_paths:
            del self.model_paths[model_id]
            
        self.logger.info(f"Unregistered model {model_id}")
