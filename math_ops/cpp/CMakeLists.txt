cmake_minimum_required(VERSION 3.14)
project(jarvis_math_ops VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find Python and pybind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Add the library
pybind11_add_module(
    _math_ops
    math_ops.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../_math_ops.cpp
)

# Set include directories
target_include_directories(_math_ops
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${Python3_INCLUDE_DIRS}
)

# Link against pybind11 and Python
target_link_libraries(_math_ops
    PRIVATE
        pybind11::module
        Python3::Python
)

# Install the module
install(TARGETS _math_ops
    LIBRARY DESTINATION ${Python3_SITEARCH}/jarvis/math_ops
    ARCHIVE DESTINATION ${Python3_SITEARCH}/jarvis/math_ops
    RUNTIME DESTINATION ${Python3_SITEARCH}/jarvis/math_ops
)

# Install Python package
install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
    DESTINATION ${Python3_SITEARCH}/jarvis/math_ops
    FILES_MATCHING
        PATTERN "*.py"
        PATTERN "__pycache__" EXCLUDE
)
