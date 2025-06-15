"""Compatibility layer for different Python and library versions."""
import sys
import warnings

# NumPy compatibility
if 'numpy' in sys.modules:
    import numpy as np
    
    # Patch numpy.bool_ if it doesn't exist
    if not hasattr(np, 'bool_'):
        warnings.warn("Patching numpy.bool_ to use bool")
        np.bool_ = bool
        
    # Add any other compatibility patches here
    
    # Make sure we have the expected attributes
    if not hasattr(np, 'number'):
        class Number:
            pass
        np.number = Number()
        
    if not hasattr(np, 'object_'):
        np.object_ = object

# Apply patches when this module is imported
__all__ = []
