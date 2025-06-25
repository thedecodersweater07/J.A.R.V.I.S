# JARVIS NLP Module

This module provides Natural Language Processing (NLP) capabilities for JARVIS, with both Python and optimized C++ implementations.

## Directory Structure

```
nlp/
├── cpp/                    # C++ implementation
│   ├── nlp_engine.hpp      # C++ header file
│   ├── nlp_engine.cpp      # C++ implementation
│   ├── pybindings.cpp      # Python bindings
│   └── CMakeLists.txt      # CMake configuration
├── __init__.py            # Python package initialization
├── cpp_engine.py          # Python wrapper for C++ engine
├── build.py               # Build script
└── setup.py               # Python package setup
```

## Prerequisites

- CMake (3.14 or higher)
- C++17 compatible compiler
- Python 3.7+
- PyBind11 (will be installed automatically)

## Building the C++ Extension

### Using build.py (recommended)

```bash
# From the nlp directory
python build.py
```

### Using setup.py

```bash
# From the nlp directory
pip install -e .
```

## Using the NLP Engine

### Python API

```python
from nlp.cpp_engine import NLEngine

# Initialize the engine
engine = NLEngine(language="nl")  # Supports multiple languages

# Process text
result = engine.process_text("Hallo, hoe gaat het met jou?")
print(result)

# Check if the C++ engine is available
if engine.is_initialized:
    print("Using optimized C++ engine")
else:
    print("Falling back to Python implementation")
```

## Development

### Adding New Features

1. Add new methods to `cpp/nlp_engine.hpp`
2. Implement them in `cpp/nlp_engine.cpp`
3. Update the Python bindings in `cpp/pybindings.cpp`
4. Add Python wrapper methods in `cpp_engine.py`

### Testing

Run the built-in tests:

```bash
python -m nlp.cpp_engine
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
