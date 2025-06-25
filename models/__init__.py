"""
JARVIS AI Assistant Models Package

This package contains all the data models and AI components used by the JARVIS AI Assistant.
"""

import warnings

# List of all exported symbols
__all__ = []

# Try to import the C++ wrapper
try:
    from .cpp_wrapper import Model, CPP_AVAILABLE
    __all__.append('Model')
    
    if not CPP_AVAILABLE:
        warnings.warn(
            "C++ extension not available. Using Python fallback. "
            "Performance may be impacted. To build the C++ extension, run: "
            "`python build_ext.py` in the models directory."
        )
except ImportError as e:
    warnings.warn(
        f"Failed to import C++ model wrapper: {e}. "
        "Some functionality may be limited."
    )

# Import and expose other key components
try:
    from .jarvis import JarvisModel, LLMBase, NLPBase, LLMProtocol, NLPProtocol
    __all__.extend([
        'JarvisModel',
        'LLMBase',
        'NLPBase',
        'LLMProtocol',
        'NLPProtocol',
    ])
except ImportError as e:
    warnings.warn(f"Failed to import JARVIS model components: {e}")
