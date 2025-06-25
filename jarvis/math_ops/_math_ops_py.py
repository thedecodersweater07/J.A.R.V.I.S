"""
Pure Python implementation of math operations.
Used as a fallback when C++ extensions are not available.
"""

def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

# Indicate that this is the Python implementation
IS_CPP_IMPLEMENTATION = False
