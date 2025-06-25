"""
Hyperadvanced AI - Advanced AI components for JARVIS.

This package provides modular, production-ready AI components for JARVIS,
including natural language processing, computer vision, and machine learning
capabilities.
"""

__version__ = "1.0.0"
__author__ = "Nova Industrie AI Team"

# Use absolute imports for better IDE support
import sys
import os
from pathlib import Path

# Add the package directory to the path if not already there
package_dir = str(Path(__file__).parent.absolute())
if package_dir not in sys.path:
    sys.path.insert(0, package_dir)

# Import core components with try/except for better error handling
try:
    from .core import (
        HyperAIAbstraction,
        BaseAIModule,
        ModuleConfig,
        ModuleMetadata,
        ModuleInitializationError,
        get_logger,
        configure_logging
    )
    
    # Import modules if available
    try:
        from .modules import (
            register_module,
            get_module,
            list_modules
        )
        MODULES_AVAILABLE = True
    except ImportError:
        MODULES_AVAILABLE = False
        
    # Set up logging
    configure_logging()
    
except ImportError as e:
    import warnings
    warnings.warn(f"Failed to import hyperadvanced_ai core components: {e}")

__all__ = [
    # Core
    'HyperAIAbstraction',
    'BaseAIModule',
    'ModuleConfig',
    'ModuleMetadata',
    'ModuleInitializationError',
    
    # Modules
    'register_module',
    'get_module',
    'list_modules',
    
    # Metadata
    '__version__',
    '__author__',
]
