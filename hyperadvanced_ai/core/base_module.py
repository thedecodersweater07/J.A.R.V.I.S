"""
Base module implementation for hyperadvanced_ai modules.

This module provides a base class that all AI modules should inherit from.
It enforces a consistent interface and provides common functionality.
"""

import logging
from abc import abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, ClassVar, Union, List, TypeVar, Generic, Callable, TYPE_CHECKING

from .types import AIModule, ModuleMetadata, ModuleInitializationError

logger = logging.getLogger("hyperadvanced_ai.core.base_module")

# Type variable for generic module configuration
T = TypeVar('T', bound='ModuleConfig')

@dataclass
class ModuleConfig:
    """Base configuration class for modules.
    
    Attributes:
        enabled: Whether the module is enabled
        debug: Whether to enable debug mode
    """
    enabled: bool = True
    debug: bool = False
    
    def update(self, config: Dict[str, Any]) -> None:
        """Update configuration from a dictionary.
        
        Args:
            config: Dictionary of configuration values to update
        """
        if not config:
            return
            
        for key, value in config.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary."""
        return asdict(self)

class BaseAIModule(AIModule):
    """Base class for all AI modules in the hyperadvanced_ai framework.
    
    This class provides a standard interface and common functionality
    for all AI modules. Subclasses should implement the abstract methods
    and can override the default implementations as needed.
    """
    
    # Module metadata - should be overridden by subclasses
    NAME = "unnamed_module"
    VERSION = "0.1.0"
    DESCRIPTION = "A hyperadvanced_ai module"
    REQUIREMENTS = {}
    
    # Default configuration - can be overridden by subclasses
    DEFAULT_CONFIG = ModuleConfig()
    
    def __init__(self):
        """Initialize the base module."""
        self._config = self.DEFAULT_CONFIG
        self._initialized = False
        self._logger = logging.getLogger(f"hyperadvanced_ai.module.{self.NAME}")
    
    @property
    def is_initialized(self) -> bool:
        """Check if the module has been initialized."""
        return self._initialized
    
    @property
    def config(self) -> ModuleConfig:
        """Get the current configuration."""
        return self._config
    
    @property
    def logger(self) -> logging.Logger:
        """Get the module's logger."""
        return self._logger
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the module with the given configuration.
        
        Args:
            config: Optional configuration dictionary. If None, uses default config.
            
        Raises:
            RuntimeError: If the module is already initialized
        """
        if self._initialized:
            self.logger.warning("Module already initialized, reinitializing...")
            self.shutdown()
        
        try:
            # Create a copy of the default config and update with provided values
            self._config = type(self.DEFAULT_CONFIG)()
            if config:
                self._config.update(config)
            
            # Initialize the module
            self._initialize_impl()
            
            self._initialized = True
            self.logger.info(
                f"Initialized {self.NAME} v{self.VERSION}"
            )
            
        except Exception as e:
            self._initialized = False
            self.logger.error(f"Failed to initialize module: {e}", exc_info=True)
            raise
    
    @abstractmethod
    def _initialize_impl(self) -> None:
        """Module-specific initialization logic.
        
        Subclasses should implement this method to perform any module-specific
        initialization. The base class handles configuration and state management.
        """
        pass
    
    def shutdown(self) -> None:
        """Clean up resources used by the module."""
        if not self._initialized:
            return
            
        try:
            self._shutdown_impl()
            self.logger.info("Module shutdown complete")
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}", exc_info=True)
            raise
        finally:
            self._initialized = False
    
    def _shutdown_impl(self) -> None:
        """Module-specific shutdown logic.
        
        Subclasses should override this method to clean up any resources
        they have allocated. The base class handles state management.
        """
        pass
    
    def health_check(self) -> Dict[str, Any]:
        """Check the health of the module.
        
        Returns:
            A dictionary containing health information about the module.
        """
        try:
            health = self._health_check_impl()
            health.update({
                'status': 'healthy',
                'module': self.NAME,
                'version': self.VERSION,
                'initialized': self._initialized
            })
            return health
        except Exception as e:
            self.logger.error(f"Health check failed: {e}", exc_info=True)
            return {
                'status': 'unhealthy',
                'module': self.NAME,
                'version': self.VERSION,
                'error': str(e),
                'initialized': self._initialized
            }
    
    def _health_check_impl(self) -> Dict[str, Any]:
        """Module-specific health check implementation.
        
        Subclasses should override this method to provide module-specific
        health information. The return value should be a dictionary that can
        be serialized to JSON.
        
        Returns:
            A dictionary containing health information.
        """
        return {}
    
    @property
    def metadata(self) -> ModuleMetadata:
        """Get metadata about the module."""
        return ModuleMetadata(
            name=self.NAME,
            version=self.VERSION,
            description=self.DESCRIPTION,
            dependencies=self.REQUIREMENTS,
            initialized=self._initialized
        )

# Example usage:
if __name__ == "__main__":
    class MyModule(BaseAIModule):
        NAME = "example"
        VERSION = "1.0.0"
        DESCRIPTION = "An example module"
        REQUIREMENTS = {
            "numpy": ">=1.20.0",
            "torch": ">=1.9.0"
        }
        
        def _initialize_impl(self):
            self.logger.info("Initializing example module")
            # Initialize module resources here
            
        def _shutdown_impl(self):
            self.logger.info("Shutting down example module")
            # Clean up resources here
            
        def _health_check_impl(self):
            return {
                'custom_metric': 42,
                'another_metric': 'value'
            }
    
    # Example usage
    module = MyModule()
    module.initialize({"debug": True})
    print(module.health_check())
    module.shutdown()
