"""
Python wrapper for JARVIS C++ core functionality.
"""
import os
import sys
import ctypes
from pathlib import Path
from typing import Optional

# Try to import the C++ extension
try:
    from ..cpp._core import Core as _Core
    CPP_AVAILABLE = True
except ImportError:
    _Core = None
    CPP_AVAILABLE = False

class Core:
    """
    Core functionality wrapper that uses C++ implementation when available,
    with fallback to Python implementation.
    """
    def __init__(self):
        self._cpp_core = _Core() if CPP_AVAILABLE else None
        
    def process(self, input_text: str) -> str:
        """Process input text using C++ implementation if available."""
        if self._cpp_core is not None:
            return self._cpp_core.process(input_text)
        return f"[Python Fallback] Processed: {input_text}"
    
    @property
    def is_cpp_available(self) -> bool:
        """Check if C++ implementation is available."""
        return self._cpp_core is not None
