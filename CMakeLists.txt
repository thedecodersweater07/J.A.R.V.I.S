cmake_minimum_required(VERSION 3.14)
project(jarvis_cpp VERSION 1.0.0 LANGUAGES CXX)

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find Python and pybind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Set output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Enable testing
enable_testing()

# Add subdirectories for each module
set(JARVIS_MODULES
    core
    nlp
    llm
    ml
    models
)

# Build all modules
foreach(MODULE ${JARVIS_MODULES})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${MODULE}/cpp/CMakeLists.txt")
        message(STATUS "Configuring ${MODULE} module")
        add_subdirectory(${MODULE}/cpp ${CMAKE_BINARY_DIR}/${MODULE})
    endif()
endforeach()

# Install configuration
install(
    DIRECTORY ${CMAKE_SOURCE_DIR}/cmake/
    DESTINATION ${CMAKE_INSTALL_PREFIX}/cmake
    FILES_MATCHING PATTERN "*.cmake"
)

# Package configuration
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    ${CMAKE_BINARY_DIR}/JarvisCppConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_BINARY_DIR}/JarvisCppConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_PREFIX}/cmake
)

# Export targets
install(EXPORT JarvisCppTargets
    FILE JarvisCppTargets.cmake
    NAMESPACE Jarvis::
    DESTINATION ${CMAKE_INSTALL_PREFIX}/cmake
)

# Create a config file for the package
configure_package_config_file(
    ${CMAKE_SOURCE_DIR}/cmake/JarvisCppConfig.cmake.in
    ${CMAKE_BINARY_DIR}/JarvisCppConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_PREFIX}/cmake
)

install(FILES
    ${CMAKE_BINARY_DIR}/JarvisCppConfig.cmake
    DESTINATION ${CMAKE_INSTALL_PREFIX}/cmake
)

# Python package building
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
if(BUILD_PYTHON_BINDINGS)
    # Install Python package in development mode
    add_custom_target(
        install_python_package
        COMMAND ${Python3_EXECUTABLE} -m pip install -e .
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Installing Python package in development mode"
    )
    
    # Build C++ extensions
    add_custom_target(
        build_cpp_extensions
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/build_cpp.py
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Building C++ extensions"
    )
    
    add_dependencies(build_cpp_extensions ${PROJECT_NAME}_all)
    add_dependencies(install_python_package build_cpp_extensions)
    
    # Add install target for Python package
    install(CODE "execute_process(COMMAND ${Python3_EXECUTABLE} -m pip install . WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})")
endif()

# Enable coverage if requested
option(ENABLE_COVERAGE "Enable coverage reporting" OFF)
if(ENABLE_COVERAGE)
    add_compile_options(--coverage -O0 -g)
    add_link_options(--coverage)
endif()
