"""
JARVIS C++ Extensions

This module provides access to C++ implementations of performance-critical components.
It automatically falls back to Python implementations if the C++ extensions
are not available.
"""
import os
import sys
import importlib
import warnings
from typing import Any, Dict, Optional, Type, TypeVar, Union

# Type variable for generic return types
T = TypeVar('T')

# Global cache for loaded modules
_loaded_modules: Dict[str, Any] = {}

def load_cpp_extension(module_name: str, python_fallback: Optional[Type[T]] = None) -> Union[T, Any]:
    """
    Load a C++ extension module with Python fallback.
    
    Args:
        module_name: Name of the module to load (e.g., 'nlp', 'ml')
        python_fallback: Python class to use as fallback if C++ extension is not available
        
    Returns:
        The loaded C++ module or Python fallback
        
    Raises:
        ImportError: If neither C++ extension nor Python fallback is available
    """
    if module_name in _loaded_modules:
        return _loaded_modules[module_name]
    
    # Try to import the C++ module
    cpp_module_name = f"jarvis.{module_name}._{module_name}"
    try:
        module = importlib.import_module(cpp_module_name)
        _loaded_modules[module_name] = module
        return module
    except ImportError as e:
        if python_fallback is not None:
            warnings.warn(
                f"C++ extension for {module_name} not available, using Python fallback: {e}",
                RuntimeWarning
            )
            _loaded_modules[module_name] = python_fallback()
            return _loaded_modules[module_name]
        else:
            raise ImportError(
                f"Could not import C++ extension '{cpp_module_name}' and no Python fallback provided"
            ) from e

def is_cpp_available(module_name: str) -> bool:
    """
    Check if a C++ extension is available.
    
    Args:
        module_name: Name of the module to check
        
    Returns:
        bool: True if the C++ extension is available, False otherwise
    """
    if module_name in _loaded_modules:
        return hasattr(_loaded_modules[module_name], '__file__') and \
               os.path.splitext(_loaded_modules[module_name].__file__ or '')[1] in {'.so', '.pyd'}
    
    cpp_module_name = f"jarvis.{module_name}._{module_name}"
    try:
        importlib.import_module(cpp_module_name)
        return True
    except ImportError:
        return False

# Export public API
__all__ = ['load_cpp_extension', 'is_cpp_available']
