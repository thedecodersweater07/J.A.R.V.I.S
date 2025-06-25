# Base configuration for JARVIS C++ modules
set(JARVIS_CPP_VERSION 1.0.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find Python and pybind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Common compiler flags
if(MSVC)
    add_compile_options(/W4 /WX)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Werror)
endif()

# Function to create a JARVIS module
function(jarvis_add_module MODULE_NAME)
    set(options)
    set(oneValueArgs)
    set(multiValueArgs SOURCES HEADERS LINK_LIBS)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # Create shared library
    pybind11_add_module(${MODULE_NAME} ${ARG_SOURCES} ${ARG_HEADERS})
    
    # Set output directory
    set_target_properties(${MODULE_NAME} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/$<CONFIG>
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/$<CONFIG>
    )
    
    # Link libraries
    if(ARG_LINK_LIBS)
        target_link_libraries(${MODULE_NAME} PRIVATE ${ARG_LINK_LIBS})
    endif()
    
    # Install target
    install(TARGETS ${MODULE_NAME}
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}
    )
endfunction()
