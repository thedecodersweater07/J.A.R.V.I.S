cmake_minimum_required(VERSION 3.14...3.28)
project(jarvis_nlp VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find Python 3
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Set Python module installation directory
set(PYTHON_MODULE_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages/jarvis/nlp"
    CACHE PATH "Python module installation directory")

# Create output directories in the build tree
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/nlp)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/python/jarvis/nlp)

# Handle pybind11 configuration
# First check if we have pybind11 from parent project
if(TARGET pybind11::module)
    message(STATUS "Using pybind11 from parent project")
    get_target_property(PYBIND11_INCLUDE_DIR pybind11::headers INTERFACE_INCLUDE_DIRECTORIES)
    message(STATUS "Found pybind11 includes at: ${PYBIND11_INCLUDE_DIR}")
    
    # Get the pybind11 directory from the target
    get_target_property(PYBIND11_CONFIG pybind11::module IMPORTED_CONFIGURATIONS)
    get_target_property(PYBIND11_LOCATION pybind11::module IMPORTED_LOCATION_${PYBIND11_CONFIG})
    get_filename_component(PYBIND11_DIR ${PYBIND11_LOCATION} DIRECTORY)
    get_filename_component(PYBIND11_DIR ${PYBIND11_DIR}/.. ABSOLUTE)
    
    # Make sure pybind11_add_module is available
    if(NOT COMMAND pybind11_add_module)
        include(${PYBIND11_DIR}/share/cmake/pybind11/pybind11Tools.cmake)
    endif()
else()
    # Try to find pybind11 in standard locations
    message(STATUS "Looking for pybind11...")
    
    # 1. Check if pybind11_DIR is set
    if(NOT pybind11_DIR)
        # 2. Check common installation paths
        set(POSSIBLE_PYBIND11_PATHS
            "${Python3_SITELIB}/pybind11"
            "${Python3_SITELIB}/../share/cmake/pybind11"
            "/usr/local/share/cmake/pybind11"
            "/usr/share/cmake/pybind11"
        )
        
        foreach(PATH ${POSSIBLE_PYBIND11_PATHS})
            if(EXISTS "${PATH}/pybind11Config.cmake")
                set(pybind11_DIR ${PATH} CACHE PATH "Path to pybind11")
                message(STATUS "Found pybind11 at: ${pybind11_DIR}")
                break()
            endif()
        endforeach()
    endif()
    
    # 3. Try to find via Python package
    if(NOT pybind11_DIR)
        message(STATUS "Trying to find pybind11 via Python package...")
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_include())"
            OUTPUT_VARIABLE PYBIND11_INCLUDE
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE PYBIND11_RESULT
        )
        
        if(PYBIND11_RESULT EQUAL 0 AND EXISTS ${PYBIND11_INCLUDE})
            set(pybind11_DIR ${PYBIND11_INCLUDE}/../share/cmake/pybind11)
            message(STATUS "Found pybind11 via Python package at: ${pybind11_DIR}")
        endif()
    endif()
    
    # 4. Try to find in build directory
    if(NOT pybind11_DIR AND EXISTS "${CMAKE_BINARY_DIR}/_deps/pybind11-src")
        set(pybind11_DIR ${CMAKE_BINARY_DIR}/_deps/pybind11-src)
        message(STATUS "Found pybind11 in build directory: ${pybind11_DIR}")
    endif()
    
    # 5. If still not found, try to find it with find_package
    if(NOT pybind11_DIR)
        find_package(pybind11 2.11.1 CONFIG QUIET)
    endif()
    
    # 6. If still not found, try to find it in the parent directory
    if(NOT pybind11_DIR AND EXISTS "${CMAKE_SOURCE_DIR}/../pybind11")
        set(pybind11_DIR ${CMAKE_SOURCE_DIR}/../pybind11)
        message(STATUS "Found pybind11 in parent directory: ${pybind11_DIR}")
    endif()
    
    # 7. If still not found, try to find it with pkg-config
    if(NOT pybind11_DIR)
        find_package(PkgConfig QUIET)
        if(PKG_CONFIG_FOUND)
            pkg_check_modules(PYBIND11 QUIET pybind11)
            if(PYBIND11_FOUND)
                set(pybind11_DIR ${PYBIND11_PREFIX}/share/cmake/pybind11)
                message(STATUS "Found pybind11 via pkg-config at: ${pybind11_DIR}")
            endif()
        endif()
    endif()
    
    # 8. If still not found, try to download it with FetchContent
    if(NOT pybind11_DIR)
        message(STATUS "pybind11 not found, downloading with FetchContent...")
        include(FetchContent)
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG v2.11.1
            GIT_SHALLOW TRUE
        )
        
        FetchContent_GetProperties(pybind11)
        if(NOT pybind11_POPULATED)
            set(FETCHCONTENT_QUIET OFF)
            FetchContent_Populate(pybind11)
            set(pybind11_DIR ${pybind11_SOURCE_DIR} CACHE PATH "Path to pybind11")
        endif()
    endif()
    
    # Make sure we have a valid pybind11_DIR
    if(NOT EXISTS "${pybind11_DIR}/pybind11Config.cmake" AND 
       NOT EXISTS "${pybind11_DIR}/pybind11-config.cmake" AND
       EXISTS "${pybind11_DIR}/share/cmake/pybind11")
        set(pybind11_DIR ${pybind11_DIR}/share/cmake/pybind11)
    endif()
    
    # Finally, try to find the package
    find_package(pybind11 2.11.1 CONFIG REQUIRED)
    
    # Make sure pybind11_add_module is available
    if(NOT COMMAND pybind11_add_module)
        include(${pybind11_DIR}/pybind11Tools.cmake)
    endif()
    
    message(STATUS "Using pybind11 from: ${pybind11_DIR}")
endif()

# Make sure pybind11_DIR is in CMAKE_PREFIX_PATH
if(pybind11_DIR AND NOT "${CMAKE_PREFIX_PATH}" MATCHES "${pybind11_DIR}")
    list(APPEND CMAKE_PREFIX_PATH ${pybind11_DIR})
    set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} CACHE STRING "" FORCE)
    message(STATUS "Added to CMAKE_PREFIX_PATH: ${pybind11_DIR}")
endif()

message(STATUS "pybind11_DIR: ${pybind11_DIR}")
message(STATUS "Python3_EXECUTABLE: ${Python3_EXECUTABLE}")

# Export pybind11_DIR for subprojects
set(pybind11_DIR ${pybind11_DIR} CACHE PATH "Path to pybind11" FORCE)

# Set include directories
include_directories(
    ${PYTHON_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/preprocessing/tokenization/token_cpp
)

# Set library output directory
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/nlp)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/nlp)

# Get Python include directory and extension suffix
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_path('include'))"
    OUTPUT_VARIABLE PYTHON_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print((sysconfig.get_config_var('EXT_SUFFIX') or '.so').split('.')[-1])"
    OUTPUT_VARIABLE PYTHON_EXT_SUFFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "Python extension suffix: ${PYTHON_EXT_SUFFIX}")

# Add include directories
include_directories(
    ${PYTHON_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/preprocessing/tokenization/token_cpp
)

# Set output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Use pybind11 from parent project
if(NOT TARGET pybind11::module)
    find_package(pybind11 CONFIG REQUIRED)
    if(NOT pybind11_FOUND)
        message(FATAL_ERROR "pybind11 not found. Please install it with: pip install pybind11")
    endif()
endif()

# Add the NLP module
pybind11_add_module(_nlp_engine
    src/nlp_engine.cpp
    src/tokenizer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/preprocessing/tokenization/token_cpp/token.cpp
)

# Set target properties
target_include_directories(_nlp_engine
    PRIVATE
        ${PYTHON_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/preprocessing/tokenization/token_cpp
)

# Set output directory for the module
set_target_properties(_nlp_engine PROPERTIES
    OUTPUT_NAME "_nlp_engine"
    PREFIX ""
    SUFFIX ".${PYTHON_EXT_SUFFIX}"
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/nlp
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/nlp
)

# Create output directory
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/nlp)

# Install the module
install(TARGETS _nlp_engine
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/nlp
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/nlp
)

# Copy Python files to build directory
file(GLOB PYTHON_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.py")
foreach(PYTHON_FILE ${PYTHON_FILES})
    file(COPY ${PYTHON_FILE} DESTINATION ${CMAKE_BINARY_DIR}/nlp)
    install(FILES ${PYTHON_FILE} DESTINATION ${CMAKE_INSTALL_PREFIX}/nlp)
endforeach()

# Create output directories in the build tree
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/python/jarvis/nlp)

# Create __init__.py in the build tree
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/__init__.py "# JARVIS NLP Module\n")
file(WRITE ${CMAKE_BINARY_DIR}/python/jarvis/__init__.py "# JARVIS Python Package\n")
file(WRITE ${CMAKE_BINARY_DIR}/python/jarvis/nlp/__init__.py "# JARVIS NLP Module\n")

# Get the target file name
set(TARGET_FILE "$<TARGET_FILE:${MODULE_NAME}>")
set(TARGET_FILE_NAME "$<TARGET_FILE_NAME:${MODULE_NAME}>")
set(TARGET_DIR "${CMAKE_BINARY_DIR}/python/jarvis/nlp")

# Add custom command to copy the module to the build tree
add_custom_command(
    OUTPUT ${TARGET_DIR}/${TARGET_FILE_NAME}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${TARGET_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy ${TARGET_FILE} ${TARGET_DIR}/
    DEPENDS ${MODULE_NAME}
    COMMENT "Copying ${MODULE_NAME} to Python package directory"
    VERBATIM
)

# Add a custom target that depends on the output file
add_custom_target(
    copy_${MODULE_NAME}_module ALL
    DEPENDS ${TARGET_DIR}/${TARGET_FILE_NAME}
)

# Set the installation paths
set(INSTALL_PYTHON_DIR "${CMAKE_INSTALL_PREFIX}/lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages/jarvis/nlp")

# Install the module
install(TARGETS ${MODULE_NAME}
    LIBRARY DESTINATION ${INSTALL_PYTHON_DIR}
    RUNTIME DESTINATION ${INSTALL_PYTHON_DIR}
    ARCHIVE DESTINATION ${INSTALL_PYTHON_DIR}
    COMPONENT python
)

# Install Python files
install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/__init__.py
    DESTINATION ${INSTALL_PYTHON_DIR}
    COMPONENT python
)

# Create a version file for the package
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/version.py "__version__ = '${PROJECT_VERSION}'\n")
install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/version.py
    DESTINATION ${INSTALL_PYTHON_DIR}
    COMPONENT python
)

# Create a symlink in the build tree for development
if(UNIX AND NOT APPLE)
    add_custom_command(
        OUTPUT ${CMAKE_BINARY_DIR}/python/jarvis/nlp/$<TARGET_FILE_NAME:${MODULE_NAME}>
        COMMAND ${CMAKE_COMMAND} -E create_symlink
            $<TARGET_FILE:${MODULE_NAME}>
            ${CMAKE_BINARY_DIR}/python/jarvis/nlp/$<TARGET_FILE_NAME:${MODULE_NAME}>
        DEPENDS ${MODULE_NAME}
        COMMENT "Creating symlink for ${MODULE_NAME}"
    )
    
    add_custom_target(
        symlink_${MODULE_NAME} ALL
        DEPENDS ${CMAKE_BINARY_DIR}/python/jarvis/nlp/$<TARGET_FILE_NAME:${MODULE_NAME}>
    )
endif()
