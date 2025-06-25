#!/usr/bin/env python3
"""
Test script for the JARVIS math operations module.
"""
import sys
import os
import unittest
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import the module
from jarvis.math_ops import add, multiply, IS_CPP_IMPLEMENTATION

class TestMathOps(unittest.TestCase):
    """Test cases for math operations."""
    
    def test_add(self):
        """Test addition operation."""
        self.assertEqual(add(2, 3), 5)
        self.assertEqual(add(-1, 1), 0)
        self.assertEqual(add(0, 0), 0)
    
    def test_multiply(self):
        """Test multiplication operation."""
        self.assertEqual(multiply(2, 3), 6)
        self.assertEqual(multiply(-1, 1), -1)
        self.assertEqual(multiply(0, 5), 0)
    
    def test_cpp_available(self):
        """Test if C++ implementation is available."""
        print(f"C++ implementation available: {IS_CPP_IMPLEMENTATION}")

if __name__ == "__main__":
    unittest.main(verbosity=2)
