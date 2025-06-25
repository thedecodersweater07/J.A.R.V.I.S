# Building JARVIS C++ Components

This document provides instructions for building and installing the C++ components of the JARVIS system, including the NLP module.

## Prerequisites

- CMake 3.14 or higher
- C++17 compatible compiler (GCC 7+, Clang 6+, MSVC 2019+)
- Python 3.7+ with development headers
- pybind11 (will be downloaded automatically)

## Building on Windows

1. Open a command prompt in the project root directory.

2. Create a build directory and configure CMake:
   ```batch
   mkdir build
   cd build
   cmake .. -G "Visual Studio 16 2019" -A x64
   ```

3. Build the project:
   ```batch
   cmake --build . --config Release --target install
   ```

4. (Optional) Run tests:
   ```batch
   ctest -C Release
   ```

## Building on Linux/macOS

1. Open a terminal in the project root directory.

2. Create a build directory and configure CMake:
   ```bash
   mkdir -p build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release
   ```

3. Build the project:
   ```bash
   make -j$(nproc)
   make install
   ```

4. (Optional) Run tests:
   ```bash
   ctest --output-on-failure
   ```

## Using the Build Scripts

For convenience, you can use the provided build scripts:

- Windows: `build.bat`
- Linux/macOS: `build.sh`

These scripts will handle the build process and copy the compiled modules to the correct locations.

## Verifying the Installation

After building, you can verify the installation by running:

```bash
python test_nlp.py
```

## Troubleshooting

### CMake can't find Python

Make sure you have the Python development headers installed:
- Ubuntu/Debian: `sudo apt-get install python3-dev`
- RHEL/CentOS: `sudo yum install python3-devel`
- macOS: `brew install python`

### Module not found after installation

If Python can't find the module, make sure the installation directory is in your `PYTHONPATH`:

```bash
export PYTHONPATH="$(python -c 'import site; print(site.getsitepackages()[0])')"
```

### Building with a specific Python version

You can specify the Python executable to use:

```bash
cmake .. -DPython3_EXECUTABLE=$(which python3.8)
```

## Development Workflow

1. Make your changes to the C++ code
2. Rebuild the project:
   ```bash
   cd build
   make install
   ```
3. Test your changes:
   ```bash
   python test_nlp.py
   ```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
