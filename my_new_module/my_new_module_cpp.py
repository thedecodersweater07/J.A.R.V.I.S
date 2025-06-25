"""
my_new_module - Python wrapper for my_new_module C++ module
"""
import os
import sys
import importlib.util
from typing import Any, Optional, Dict, List, Union

# Try to import the C++ extension
try:
    from ._my_new_module import *
    CPP_AVAILABLE = True
except ImportError as e:
    CPP_AVAILABLE = False

class MyNewModule:
    """
    my_new_module module wrapper that uses C++ implementation when available,
    with fallback to Python implementation.
    """
    def __init__(self, *args, **kwargs):
        """Initialize the my_new_module module."""
        self._cpp_impl = None
        self._py_impl = None
        
        if CPP_AVAILABLE:
            try:
                self._cpp_impl = _MyNewModule(*args, **kwargs)
            except Exception as e:
                print(f"Warning: Failed to initialize C++ my_new_module: {e}", file=sys.stderr)
        
        if self._cpp_impl is None:
            self._py_impl = self._create_py_impl(*args, **kwargs)
    
    def _create_py_impl(self, *args, **kwargs):
        """Create a Python implementation as fallback."""
        # Implement Python fallback here
        class PyImpl:
            def __init__(self, *args, **kwargs):
                pass
                
            def __getattr__(self, name):
                raise NotImplementedError(
                    f"Method {name} is not implemented in Python fallback. "
                    "C++ extension failed to load.")
        
        return PyImpl(*args, **kwargs)
    
    def __getattr__(self, name):
        """Delegate attribute access to the active implementation."""
        if self._cpp_impl is not None:
            return getattr(self._cpp_impl, name)
        return getattr(self._py_impl, name)
    
    @property
    def is_cpp_available(self) -> bool:
        """Check if C++ implementation is available."""
        return self._cpp_impl is not None

# Export the main class
__all__ = ['MyNewModule']

# Add C++ classes if available
if CPP_AVAILABLE:
    __all__.extend([name for name in dir() if name.startswith('_MyNewModule')])
