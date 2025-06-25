"""
Type definitions for the hyperadvanced_ai framework.

This module contains core type definitions and base classes used throughout
the hyperadvanced_ai framework to avoid circular imports.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Type, TypeVar, Generic, Callable, ClassVar

@dataclass
class ModuleMetadata:
    """Metadata for a loaded module."""
    name: str
    version: str
    description: str = ""
    author: str = ""
    dependencies: Dict[str, str] = field(default_factory=dict)
    config_schema: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to a dictionary."""
        return {
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'author': self.author,
            'dependencies': self.dependencies,
            'config_schema': self.config_schema
        }

class AIModule:
    """Base class for all AI modules.
    
    All AI modules should inherit from this class and implement its abstract methods.
    """
    
    # Module metadata (should be overridden by subclasses)
    NAME: ClassVar[str] = ""
    VERSION: ClassVar[str] = "0.1.0"
    DESCRIPTION: ClassVar[str] = ""
    AUTHOR: ClassVar[str] = ""
    
    @abstractmethod
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the module with optional configuration.
        
        Args:
            config: Optional configuration dictionary
        """
        pass
    
    @abstractmethod
    def shutdown(self) -> None:
        """Clean up resources used by the module."""
        pass
    
    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """Return the health status of the module.
        
        Returns:
            Dictionary containing health status information
        """
        pass
    
    @property
    def metadata(self) -> ModuleMetadata:
        """Return metadata about the module.
        
        Returns:
            ModuleMetadata instance with module information
        """
        return ModuleMetadata(
            name=self.NAME,
            version=self.VERSION,
            description=self.DESCRIPTION,
            author=self.AUTHOR
        )

# Type variable for generic module types
T = TypeVar('T', bound=AIModule)

class ModuleInitializationError(Exception):
    """Raised when a module fails to initialize properly."""
    pass
