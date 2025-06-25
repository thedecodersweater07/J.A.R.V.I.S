cmake_minimum_required(VERSION 3.14...3.28)

# Set policies to suppress warnings
if(POLICY CMP0177)
    cmake_policy(SET CMP0177 NEW)  # Enable path normalization
endif()

if(POLICY CMP0175)
    cmake_policy(SET CMP0175 NEW)  # Better error reporting for add_custom_command
endif()

# Set default install prefix if not specified
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Install path prefix" FORCE)
endif()

project(jarvis_cpp VERSION 1.0.0 LANGUAGES CXX)

# Set Python installation paths
set(PYTHON_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages")
file(TO_NATIVE_PATH "${PYTHON_INSTALL_DIR}" PYTHON_INSTALL_DIR_NATIVE)

# Create build directory structure
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/python/jarvis/nlp)

# Create __init__.py files
file(WRITE ${CMAKE_BINARY_DIR}/python/jarvis/__init__.py "# JARVIS Python Package\n")
file(WRITE ${CMAKE_BINARY_DIR}/python/jarvis/nlp/__init__.py "# JARVIS NLP Module\n")

# Add the build Python directory to PYTHONPATH
set(ENV{PYTHONPATH} "${CMAKE_BINARY_DIR}/python$<SEMICOLON>$ENV{PYTHONPATH}")

# Print installation paths for debugging
message(STATUS "Python installation directory: ${PYTHON_INSTALL_DIR}")
message(STATUS "Build Python directory: ${CMAKE_BINARY_DIR}/python")

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find Python 3 - use the version from the environment if set
if(DEFINED ENV{Python3_ROOT_DIR} AND EXISTS $ENV{Python3_ROOT_DIR})
    set(Python3_ROOT_DIR $ENV{Python3_ROOT_DIR} CACHE PATH "Path to Python root directory")
    message(STATUS "Using Python root from environment: ${Python3_ROOT_DIR}")
endif()

if(DEFINED ENV{Python3_EXECUTABLE} AND EXISTS $ENV{Python3_EXECUTABLE})
    set(Python3_EXECUTABLE $ENV{Python3_EXECUTABLE} CACHE FILEPATH "Path to Python executable")
    message(STATUS "Using Python executable from environment: ${Python3_EXECUTABLE}")
endif()

# Find Python 3 components
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Set Python module installation path
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Install path prefix" FORCE)
endif()

# Get Python paths
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_path('include'))"
    OUTPUT_VARIABLE PYTHON_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_path('purelib'))"
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Set local Python packages directory
set(LOCAL_PYTHON_PACKAGES "${CMAKE_BINARY_DIR}/python_packages")
set(LOCAL_PYTHON_SITE_PACKAGES "${LOCAL_PYTHON_PACKAGES}/site-packages")

# Create local site-packages directory
file(MAKE_DIRECTORY ${LOCAL_PYTHON_SITE_PACKAGES})

# Create __init__.py in the build tree
file(MAKE_DIRECTORY ${LOCAL_PYTHON_SITE_PACKAGES}/jarvis)
file(WRITE ${LOCAL_PYTHON_SITE_PACKAGES}/jarvis/__init__.py "# JARVIS Python Package\n")

# Add the local Python packages directory to PYTHONPATH
file(TO_NATIVE_PATH "${LOCAL_PYTHON_SITE_PACKAGES}" LOCAL_PYTHON_SITE_PACKAGES_NATIVE)
set(ENV{PYTHONPATH} "${LOCAL_PYTHON_SITE_PACKAGES_NATIVE}$<SEMICOLON>$ENV{PYTHONPATH}")

message(STATUS "Python include directory: ${PYTHON_INCLUDE_DIR}")
message(STATUS "Python library: ${Python3_LIBRARIES}")
message(STATUS "Python executable: ${Python3_EXECUTABLE}")
message(STATUS "Python site-packages: ${PYTHON_SITE_PACKAGES}")
message(STATUS "Local Python packages: ${LOCAL_PYTHON_PACKAGES}")

# Handle pybind11
include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

# Set pybind11 options before finding it
set(PYBIND11_INSTALL ON CACHE BOOL "Install pybind11" FORCE)
set(PYBIND11_TEST OFF CACHE BOOL "Build pybind11 tests" FORCE)
set(PYBIND11_NOPYTHON_LINK OFF CACHE BOOL "Don't link against Python" FORCE)

# First try to find pybind11 via find_package
find_package(pybind11 2.11.1 CONFIG QUIET)

if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found in system, trying to find via Python package...")
    
    # Try to find pybind11 via Python package
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_include())"
        OUTPUT_VARIABLE PYBIND11_INCLUDE
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    
    if(EXISTS ${PYBIND11_INCLUDE})
        set(pybind11_DIR ${PYBIND11_INCLUDE}/../share/cmake/pybind11)
        message(STATUS "Found pybind11 via Python package at: ${pybind11_DIR}")
        find_package(pybind11 2.11.1 CONFIG REQUIRED)
    else()
        message(STATUS "pybind11 not found via Python package, using FetchContent...")
        
        # Use FetchContent to download pybind11
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG v2.11.1
            GIT_SHALLOW TRUE
            GIT_PROGRESS TRUE
        )
        
        # Populate and add to build
        FetchContent_GetProperties(pybind11)
        if(NOT pybind11_POPULATED)
            message(STATUS "Downloading pybind11...")
            FetchContent_Populate(pybind11)
            
            # Set pybind11 options before adding subdirectory
            set(PYBIND11_INSTALL ON CACHE BOOL "" FORCE)
            set(PYBIND11_TEST OFF CACHE BOOL "" FORCE)
            set(PYBIND11_NOPYTHON_LINK OFF CACHE BOOL "" FORCE)
            
            add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR} EXCLUDE_FROM_ALL)
        endif()
        
        set(pybind11_DIR ${pybind11_SOURCE_DIR} CACHE PATH "Path to pybind11")
        set(pybind11_ROOT ${pybind11_SOURCE_DIR} CACHE PATH "Path to pybind11 root")
        message(STATUS "Using pybind11 from: ${pybind11_SOURCE_DIR}")
    endif()
else()
    message(STATUS "Found pybind11: ${pybind11_DIR}")
    
    # Include pybind11 tools if not already included
    if(NOT TARGET pybind11::headers)
        include(${pybind11_DIR}/pybind11Tools.cmake)
        include(${pybind11_DIR}/pybind11Common.cmake)
    endif()
    
    # Set pybind11_ROOT if not set
    if(NOT DEFINED pybind11_ROOT)
        get_filename_component(PARENT_DIR ${pybind11_DIR} DIRECTORY)
        set(pybind11_ROOT ${PARENT_DIR} CACHE PATH "Path to pybind11 root")
    endif()
endif()

# Make sure pybind11_DIR is in CMAKE_PREFIX_PATH
if(pybind11_DIR AND NOT "${CMAKE_PREFIX_PATH}" MATCHES "${pybind11_DIR}")
    list(APPEND CMAKE_PREFIX_PATH ${pybind11_DIR})
    set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} CACHE STRING "" FORCE)
    message(STATUS "Added to CMAKE_PREFIX_PATH: ${pybind11_DIR}")
endif()

message(STATUS "pybind11_DIR: ${pybind11_DIR}")
message(STATUS "pybind11_ROOT: ${pybind11_ROOT}")
message(STATUS "Python3_EXECUTABLE: ${Python3_EXECUTABLE}")

# Export pybind11_DIR for subprojects
set(pybind11_DIR ${pybind11_DIR} CACHE PATH "Path to pybind11" FORCE)

# Add include directories
include_directories(
    ${PYTHON_INCLUDE_DIR}
)

# Set Python module installation path
# Set output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Enable testing
enable_testing()

# Set Python module installation path
set(PYTHON_MODULE_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages/jarvis
    CACHE STRING "Python module installation directory")

# Create the directory in the build tree
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/python/jarvis)

# Only add subdirectories if they exist
set(SUBDIRS "nlp/cpp")

# Check and add each subdirectory
foreach(SUBDIR ${SUBDIRS})
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${SUBDIR}/CMakeLists.txt)
        message(STATUS "Adding subdirectory: ${SUBDIR}")
        add_subdirectory(${SUBDIR})
    else()
        message(STATUS "Skipping ${SUBDIR} - directory or CMakeLists.txt not found")
    endif()
endforeach()

# Create a target for missing directories to prevent build failures
add_custom_target(missing_dirs
    COMMAND ${CMAKE_COMMAND} -E echo "The following directories were not found: ml/cpp, llm/cpp"
    COMMENT "Some optional directories were not found"
    VERBATIM
)

# Install Python package files
install(
    DIRECTORY ${CMAKE_BINARY_DIR}/python/jarvis/
    DESTINATION ${PYTHON_INSTALL_DIR}/jarvis
    FILES_MATCHING
        PATTERN "*.py"
        PATTERN "__pycache__" EXCLUDE
    COMPONENT python
)

# Install compiled modules
install(
    DIRECTORY ${CMAKE_BINARY_DIR}/python/jarvis/
    DESTINATION ${PYTHON_INSTALL_DIR}/jarvis
    FILES_MATCHING
        PATTERN "*.pyd"
        PATTERN "*.so"
    COMPONENT python
)

# Create a setup.py file for development installation
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in
    ${CMAKE_BINARY_DIR}/setup.py
    @ONLY
)

# Add a target to install in development mode
add_custom_target(develop
    COMMAND ${Python3_EXECUTABLE} -m pip install -e ${CMAKE_BINARY_DIR}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Installing JARVIS in development mode"
)

# Print installation information
message(STATUS "\nJARVIS will be installed to: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "Python modules will be installed to: ${PYTHON_INSTALL_DIR}\n")

# Build all modules
foreach(MODULE ${JARVIS_MODULES})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${MODULE}/CMakeLists.txt")
        add_subdirectory(${MODULE})
    else()
        message(STATUS "Skipping non-existent module: ${MODULE}")
    endif()
endforeach()

# Install Python source files from nlp directory
install(
    DIRECTORY ${CMAKE_SOURCE_DIR}/nlp/
    DESTINATION ${LOCAL_PYTHON_SITE_PACKAGES}/jarvis/nlp
    FILES_MATCHING
        PATTERN "*.py"
        PATTERN "__pycache__" EXCLUDE
        PATTERN "*.cpp" EXCLUDE
        PATTERN "*.hpp" EXCLUDE
)

# Install compiled modules from build tree
install(
    DIRECTORY ${CMAKE_BINARY_DIR}/python/jarvis/
    DESTINATION ${LOCAL_PYTHON_SITE_PACKAGES}/jarvis
    FILES_MATCHING
        PATTERN "*.pyd"
        PATTERN "*.so"
)

# Create setup.py for development installation
file(WRITE ${CMAKE_BINARY_DIR}/setup.py.in """
from setuptools import setup, find_packages
import os

setup(
    name="jarvis",
    version="@PROJECT_VERSION@",
    packages=find_packages('${LOCAL_PYTHON_SITE_PACKAGES}'),
    package_dir={'': '${LOCAL_PYTHON_SITE_PACKAGES}'},
    install_requires=[
        'pybind11>=2.11.1',
    ],
)
""")

configure_file(
    ${CMAKE_BINARY_DIR}/setup.py.in
    ${CMAKE_BINARY_DIR}/setup.py
    @ONLY
)

# Add a target to install the package in development mode
add_custom_target(develop
    COMMAND ${Python3_EXECUTABLE} -m pip install -e ${CMAKE_BINARY_DIR}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Installing JARVIS in development mode"
    DEPENDS ${CMAKE_BINARY_DIR}/setup.py
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

# Fix for building with Ninja
if(NINJA)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/NinjaFix.cmake)
    fix_ninja_generated_file_dependencies()
endif()