"""
Python wrapper for the JARVIS core functionality.
"""
import os
import sys
import ctypes
from ctypes import c_char_p, c_int, c_void_p, POINTER, Structure
from typing import Optional, Union

# Platform-specific library loading
if sys.platform == 'win32':
    _lib_name = 'jarvis_core.dll'
elif sys.platform == 'darwin':
    _lib_name = 'libjarvis_core.dylib'
else:
    _lib_name = 'libjarvis_core.so'

# Try to find the library in common locations
_lib_paths = [
    os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'lib', _lib_name),
    os.path.join(os.path.dirname(__file__), '..', '..', 'build', _lib_name),
    os.path.join(os.path.dirname(__file__), _lib_name),
    _lib_name
]

_lib = None
for path in _lib_paths:
    try:
        _lib = ctypes.CDLL(path)
        break
    except (OSError, Exception):
        continue

if _lib is None:
    raise ImportError(f"Could not find {_lib_name} in any of: {_lib_paths}")

# Define function prototypes
_lib.jarvis_initialize.argtypes = [c_char_p]
_lib.jarvis_initialize.restype = c_int

_lib.jarvis_process.argtypes = [c_char_p, POINTER(c_char_p)]
_lib.jarvis_process.restype = c_int

_lib.jarvis_cleanup.argtypes = []
_lib.jarvis_cleanup.restype = None

_lib.jarvis_free_string.argtypes = [c_char_p]
_lib.jarvis_free_string.restype = None

class JarvisStatus:
    SUCCESS = 0
    ERROR = -1
    INVALID_INPUT = -2
    NOT_IMPLEMENTED = -3

class JarvisCore:
    """
    Python interface to the JARVIS core functionality.
    """
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the JARVIS core.
        
        Args:
            config_path: Optional path to configuration file
        """
        config = config_path.encode('utf-8') if config_path else None
        status = _lib.jarvis_initialize(config)
        if status != JarvisStatus.SUCCESS:
            raise RuntimeError(f"Failed to initialize JARVIS core: {status}")
    
    def process(self, input_text: str) -> str:
        """
        Process input text and return the response.
        
        Args:
            input_text: Input text to process
            
        Returns:
            str: The processed response
        """
        if not input_text:
            raise ValueError("Input text cannot be empty")
        
        # Prepare output pointer
        output = c_char_p()
        
        # Call the native function
        status = _lib.jarvis_process(
            input_text.encode('utf-8'),
            ctypes.byref(output)
        )
        
        if status != JarvisStatus.SUCCESS:
            raise RuntimeError(f"Processing failed with status: {status}")
        
        try:
            # Get the string from the output pointer
            result = output.value.decode('utf-8')
            return result
        finally:
            # Free the allocated string
            if output.value is not None:
                _lib.jarvis_free_string(output)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        _lib.jarvis_cleanup()

# Example usage
if __name__ == "__main__":
    with JarvisCore() as jarvis:
        while True:
            try:
                user_input = input("You: ")
                if user_input.lower() in ('exit', 'quit'):
                    break
                response = jarvis.process(user_input)
                print(f"JARVIS: {response}")
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
