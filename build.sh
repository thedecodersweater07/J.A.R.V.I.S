#!/bin/bash

# Set build directory
BUILD_DIR="build"

# Create build directory if it doesn't exist
mkdir -p "$BUILD_DIR"

# Configure with CMKAE
echo "Configuring CMake..."
cmake -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=install

if [ $? -ne 0 ]; then
    echo "CMake configuration failed"
    exit 1
fi

# Build the project
echo "Building..."
cmake --build "$BUILD_DIR" --config Release --target install

if [ $? -ne 0 ]; then
    echo "Build failed"
    exit 1
fi

echo "Build completed successfully!"

# Copy the compiled module to the Python package directory
echo "Copying compiled modules..."
mkdir -p nlp
cp "$BUILD_DIR/nlp/_nlp_engine"* nlp/ 2>/dev/null

echo "Build and installation complete!"
