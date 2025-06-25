cmake_minimum_required(VERSION 3.14)
project(jarvis_hyperai_advanced_nn VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and PyBind11
find_package(Python3 COMPONENTS Development REQUIRED)
find_package(pybind11 REQUIRED)

# Add the Advanced Neural Network library
pybind11_add_module(_advanced_nn
    advanced_neural_net.cpp
    pybindings.cpp
)

target_include_directories(_advanced_nn PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(_advanced_nn PRIVATE 
    pybind11::module 
    Python3::Python
)
