"""
Modules package for hyperadvanced_ai.

This package contains various AI modules that can be dynamically loaded
and used by the hyperadvanced_ai framework.
"""

from typing import Dict, Type, Any, Optional
import importlib
import pkgutil
import logging
import sys

# Use relative import for core components
from ..core import BaseAIModule

logger = logging.getLogger("hyperadvanced_ai.modules")

# Dictionary to store registered modules
_MODULES: Dict[str, Type[BaseAIModule]] = {}

def register_module(name: str, module_class: Type[BaseAIModule]) -> None:
    """Register a module class with the given name.
    
    Args:
        name: The name to register the module under
        module_class: The module class to register
        
    Raises:
        TypeError: If the module class is not a subclass of BaseAIModule
    """
    if not issubclass(module_class, BaseAIModule):
        raise TypeError(f"Module {module_class.__name__} must be a subclass of BaseAIModule")
    
    _MODULES[name] = module_class
    logger.debug("Registered module: %s -> %s", name, module_class.__name__)

def get_module(name: str) -> Type[BaseAIModule]:
    """Get a registered module class by name.
    
    Args:
        name: The name of the module to get
        
    Returns:
        The module class
        
    Raises:
        KeyError: If no module is registered with the given name
    """
    if name not in _MODULES:
        raise KeyError(f"No module registered with name: {name}")
    return _MODULES[name]

def list_modules() -> Dict[str, str]:
    """List all registered modules and their descriptions.
    
    Returns:
        A dictionary mapping module names to their descriptions
    """
    return {
        name: getattr(module_class, 'DESCRIPTION', '')
        for name, module_class in _MODULES.items()
    }

def _discover_modules() -> None:
    """Discover and register all modules in this package."""
    # Ensure we have the package path
    package = sys.modules[__name__]
    
    # Find all modules in the package
    for finder, name, is_pkg in pkgutil.iter_modules(package.__path__):
        full_name = f"{package.__name__}.{name}"
        
        # Skip __pycache__ and private modules
        if name.startswith('_') or name == 'py.typed':
            continue
            
        try:
            # Import the module
            module = importlib.import_module(full_name)
            
            # Find all classes that inherit from BaseAIModule
            for item_name in dir(module):
                item = getattr(module, item_name)
                if (isinstance(item, type) and 
                    issubclass(item, BaseAIModule) and 
                    item is not BaseAIModule and 
                    item.__module__ == module.__name__):
                    
                    # Register with the class NAME attribute or the module name
                    module_name = getattr(item, 'NAME', name)
                    register_module(module_name, item)
                    
        except ImportError as e:
            logger.warning("Failed to import module %s: %s", name, e)
        except Exception as e:
            logger.exception("Error discovering modules in %s: %s", name, e)

# Import modules to register them
try:
    from . import nlp_module  # noqa
except ImportError as e:
    logger.warning("Failed to import nlp_module: %s", e)

# Discover and register all modules
_discover_modules()

__all__ = [
    'register_module',
    'get_module',
    'list_modules',
    'BaseAIModule',
]
