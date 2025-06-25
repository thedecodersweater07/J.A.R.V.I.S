cmake_minimum_required(VERSION 3.14)
project(jarvis_models VERSION 0.1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Find dependencies
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Add library
add_library(jarvis_models
    jarvis.cpp
)

# Include directories
target_include_directories(jarvis_models
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${Python3_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(jarvis_models
    PRIVATE
        pybind11::headers
        Python3::Python
)

# Python module
pybind11_add_module(
    _models
    python_bindings.cpp
)

target_link_libraries(_models 
    PRIVATE 
        jarvis_models
        pybind11::module
        Python3::Python
)

# Set output names
set_target_properties(_models
    PROPERTIES
        OUTPUT_NAME "_models${Python3_SOABI}"
        SUFFIX "${PYTHON3_EXTENSION_SUFFIX}"
)

# Installation
install(TARGETS _models
    LIBRARY 
        DESTINATION ${CMAKE_INSTALL_PREFIX}/jarvis/models
        NAMELINK_SKIP
)

# Install headers
install(
    FILES 
        ${CMAKE_CURRENT_SOURCE_DIR}/jarvis.h
    DESTINATION 
        ${CMAKE_INSTALL_PREFIX}/include/jarvis/models
)

# Install Python package
install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../..
    DESTINATION ${Python3_SITEARCH}/jarvis/models
    FILES_MATCHING
        PATTERN "*.py"
        PATTERN "__pycache__" EXCLUDE
)
