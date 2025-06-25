"""
Math operations module with C++ acceleration.
"""

"""
Math operations module with C++ acceleration.
"""

# Import the Python implementation by default
from ._math_ops_py import add, multiply

# Try to import C++ implementation if available
IS_CPP_IMPLEMENTATION = False

try:
    # First try relative import (when installed as a package)
    from ._math_ops import add as cpp_add, multiply as cpp_multiply
    
    # Override with C++ implementations
    add = cpp_add
    multiply = cpp_multiply
    IS_CPP_IMPLEMENTATION = True
except ImportError:
    try:
        # Then try direct import (for development)
        from _math_ops import add as cpp_add, multiply as cpp_multiply
        
        # Override with C++ implementations
        add = cpp_add
        multiply = cpp_multiply
        IS_CPP_IMPLEMENTATION = True
    except ImportError:
        pass  # Use Python implementation

__all__ = ['add', 'multiply', 'IS_CPP_IMPLEMENTATION']
