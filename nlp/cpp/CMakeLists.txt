cmake_minimum_required(VERSION 3.14)
project(jarvis_nlp VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Find Python and PyBind11
find_package(Python3 COMPONENTS Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Get Python include directories and libraries
python3_get_include_dirs(PYTHON_INCLUDE_DIRS)

# Add the NLP engine library
pybind11_add_module(_nlp_engine
    nlp_engine.cpp
    pybindings.cpp
)

# Set include directories
target_include_directories(_nlp_engine 
    PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PYTHON_INCLUDE_DIRS}
)

# Set C++ standard and compile features
target_compile_features(_nlp_engine PRIVATE cxx_std_17)
set_target_properties(_nlp_engine PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

# Link libraries
target_link_libraries(_nlp_engine 
    PRIVATE 
        pybind11::module 
        pybind11::embed
        Python3::Python
)

# Set output name to match Python module name
set_target_properties(_nlp_engine PROPERTIES OUTPUT_NAME "_nlp_engine")

# Install the module
install(TARGETS _nlp_engine
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Copy the compiled module to the Python package directory
add_custom_command(TARGET _nlp_engine POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:_nlp_engine> 
            ${CMAKE_CURRENT_SOURCE_DIR}/../../nlp/_nlp_engine${CMAKE_SHARED_LIBRARY_SUFFIX}
)
