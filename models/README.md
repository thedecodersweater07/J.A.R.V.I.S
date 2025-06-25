# JARVIS Models

This directory contains the machine learning models and their C++ implementations for the JARVIS AI Assistant.

## C++ Extensions

The C++ extensions provide optimized implementations of performance-critical components. They are optional but recommended for better performance.

### Prerequisites

- Python 3.7+
- C++17 compatible compiler (GCC, Clang, MSVC)
- CMake 3.14+
- pybind11 (will be installed automatically)

### Building the Extensions

1. Install build dependencies:
   ```bash
   pip install pybind11
   ```

2. Build the C++ extensions:
   ```bash
   # From the models directory
   cd models
   python build_ext.py
   ```

   This will:
   - Compile the C++ code
   - Generate Python bindings using pybind11
   - Copy the compiled extension to the correct location

### Using the Models

Import and use the models in your Python code:

```python
from jarvis.models import Model

# Create a model instance
model = Model("onnx")  # or any other supported model type

# Load a model
if model.load("path/to/model.onnx"):
    # Make predictions
    result = model.predict([0.1, 0.2, 0.3])
    print(f"Prediction: {result}")
    
    # Get model info
    print(f"Framework: {model.framework}")
    print(f"Info: {model.model_info}")
else:
    print("Failed to load model")
```

### Development

- The C++ code is in the `cpp/` directory
- Python bindings are in `python_bindings.cpp`
- The Python wrapper is in `cpp_wrapper.py`
- The fallback Python implementation is in `_models.py`

To add a new model type:
1. Add the C++ implementation in `cpp/`
2. Update the Python bindings in `python_bindings.cpp`
3. Add any necessary Python wrapper code in `cpp_wrapper.py`
4. Update the build system if needed

### Testing

Run the test script to verify everything is working:

```bash
python test_model.py
```

This will test both the C++ extension (if available) and the Python fallback implementation.
