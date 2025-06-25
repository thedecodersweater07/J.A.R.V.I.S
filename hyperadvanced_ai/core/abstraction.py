"""
Abstraction layer for seamless communication between JARVIS and hyperadvanced_ai modules.

This module provides the core abstraction for managing AI modules, including loading,
validating, and managing their lifecycle.
"""
import importlib
import inspect
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar, Generic, cast, Union, TYPE_CHECKING
import yaml
import pkg_resources

# Import core types
from .types import AIModule, ModuleMetadata, ModuleInitializationError

# Import BaseAIModule for type checking
if TYPE_CHECKING:
    from .base_module import BaseAIModule

# Define type variable for generic module type
T = TypeVar('T', bound='BaseAIModule')

logger = logging.getLogger("hyperadvanced_ai.core.abstraction")

class HyperAIAbstraction(Generic[T]):
    """
    Abstraction layer for dynamic module loading and communication.
    
    This class provides a unified interface for managing AI modules, including
    loading, initialization, and lifecycle management.
    
    Type Parameters:
        T: The base type of the modules this abstraction will manage.
           Should be a subclass of BaseAIModule.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the HyperAI abstraction layer.
        
        Args:
            config_path: Optional path to a YAML configuration file.
        """
        self.loaded_modules: Dict[str, T] = {}
        self.module_metadata: Dict[str, ModuleMetadata] = {}
        self.config = self._load_config(config_path) if config_path else {}
        self._logger = logging.getLogger("hyperadvanced_ai.abstraction")
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from a YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            return {}
    
    def load_module(self, module_path: str, config: Optional[dict] = None) -> T:
        """Dynamically import and initialize a module.
        
        Args:
            module_path: Dotted path to the module
            config: Optional configuration dictionary
            
        Returns:
            The initialized module instance
            
        Raises:
            ModuleInitializationError: If the module fails to load or initialize
        """
        if module_path in self.loaded_modules:
            return self.loaded_modules[module_path]
            
        try:
            # Import the module
            module = importlib.import_module(module_path)
            self._logger.debug(f"Imported module: {module_path}")
            
            # Find the main class that implements BaseAIModule
            module_class = None
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (issubclass(obj, BaseAIModule) and 
                    obj is not BaseAIModule and 
                    obj.__module__ == module_path):
                    module_class = obj
                    break
            
            if not module_class:
                # Fall back to any class that implements AIModule
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (issubclass(obj, AIModule) and 
                        obj is not AIModule and 
                        obj.__module__ == module_path):
                        module_class = obj
                        break
                
                if not module_class:
                    raise ModuleInitializationError(
                        f"No BaseAIModule or AIModule implementation found in {module_path}")
            
            # Create instance and initialize
            instance = module_class()
            
            # Get default config from the instance if available
            module_config = {}
            if hasattr(instance, 'DEFAULT_CONFIG'):
                default_config = getattr(instance, 'DEFAULT_CONFIG')
                if hasattr(default_config, 'to_dict'):
                    module_config.update(default_config.to_dict())
                elif isinstance(default_config, dict):
                    module_config.update(default_config)
            
            # Apply provided config
            if config:
                module_config.update(config)
            
            # Initialize with config
            instance.initialize(module_config)
            
            # Store the instance and metadata
            self.loaded_modules[module_path] = cast(T, instance)
            
            # Extract metadata from the instance if it has the attributes
            metadata = ModuleMetadata(
                name=getattr(instance, 'NAME', module_path),
                version=getattr(instance, 'VERSION', '0.1.0'),
                description=getattr(instance, 'DESCRIPTION', ''),
                dependencies=dict(getattr(instance, 'REQUIREMENTS', {})),
                initialized=True
            )
            
            self.module_metadata[module_path] = metadata
            
            self._logger.info(f"Successfully loaded and initialized module: {module_path}")
            return cast(T, instance)
            
        except Exception as e:
            error_msg = f"Failed to load module {module_path}: {str(e)}"
            logger.exception(error_msg)
            raise ModuleInitializationError(error_msg) from e
    
    def get_module(self, module_path: str) -> T:
        """Get a module instance, loading it if necessary.
        
        Args:
            module_path: Dotted path to the module
            
        Returns:
            The module instance
            
        Raises:
            ModuleInitializationError: If the module is not loaded and cannot be loaded
        """
        if module_path not in self.loaded_modules:
            # Try to load with default config if available
            module_config = self.config.get('modules', {}).get(module_path, {})
            return self.load_module(module_path, module_config)
        return self.loaded_modules[module_path]
    
    def unload_module(self, module_path: str) -> None:
        """Unload a module and clean up its resources.
        
        Args:
            module_path: Dotted path to the module
        """
        if module_path in self.loaded_modules:
            try:
                module = self.loaded_modules[module_path]
                if hasattr(module, 'shutdown'):
                    module.shutdown()
                del self.loaded_modules[module_path]
                if module_path in self.module_metadata:
                    self.module_metadata[module_path].initialized = False
                logger.info(f"Successfully unloaded module: {module_path}")
            except Exception as e:
                logger.error(f"Error unloading module {module_path}: {e}")
    
    def get_metadata(self, module_path: str) -> Optional[ModuleMetadata]:
        """Get metadata for a loaded module.
        
        Args:
            module_path: Dotted path to the module
            
        Returns:
            ModuleMetadata if the module is loaded, None otherwise
        """
        return self.module_metadata.get(module_path)
    
    def get_all_metadata(self) -> Dict[str, ModuleMetadata]:
        """Get metadata for all loaded modules.
        
        Returns:
            Dictionary mapping module paths to their metadata
        """
        return self.module_metadata.copy()
    
    def check_dependencies(self) -> Dict[str, Dict[str, str]]:
        """Check if all module dependencies are satisfied.
        
        Returns:
            Dictionary with dependency status for each module
        """
        results = {}
        for module_path, metadata in self.module_metadata.items():
            if not metadata.dependencies:
                results[module_path] = {"status": "no_dependencies"}
                continue
            
            missing = {}
            for dep_name, required_version in metadata.dependencies.items():
                try:
                    installed_version = pkg_resources.get_distribution(dep_name).version
                    if pkg_resources.parse_version(installed_version) < pkg_resources.parse_version(required_version):
                        missing[dep_name] = f"installed: {installed_version}, required: {required_version}"
                except pkg_resources.DistributionNotFound:
                    missing[dep_name] = "not_installed"
            
            results[module_path] = {
                "status": "ok" if not missing else "missing_dependencies",
                "missing": missing
            }
            
        return results
    
    def __del__(self):
        """Clean up all modules when the abstraction is garbage collected."""
        for module_path in list(self.loaded_modules.keys()):
            self.unload_module(module_path)
