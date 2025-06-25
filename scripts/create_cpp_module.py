#!/usr/bin/env python3
"""
Script to create a new C++ module with Python bindings for JARVIS.
"""
import os
import sys
import shutil
import argparse
from pathlib import Path
from string import Template
from typing import Dict, Any

def create_module(module_name: str, base_dir: Path) -> None:
    """Create a new C++ module with Python bindings.
    
    Args:
        module_name: Name of the module to create (e.g., 'nlp', 'ml')
        base_dir: Base directory of the JARVIS project
    """
    # Define paths
    module_dir = base_dir / module_name / "cpp"
    src_dir = module_dir / "src"
    include_dir = module_dir / "include" / "jarvis"
    python_dir = module_dir / "python"
    tests_dir = module_dir / "tests"
    cmake_dir = module_dir / "cmake"
    
    # Create directories
    for directory in [src_dir, include_dir, python_dir, tests_dir, cmake_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Module name in CamelCase
    module_camel = ''.join(word.capitalize() for word in module_name.split('_'))
    
    # Template variables
    template_vars = {
        'MODULE_NAME': module_name,
        'MODULE_NAME_UPPER': module_name.upper(),
        'MODULE_NAME_CAMEL': module_camel,
        'PROJECT_VERSION': '1.0.0',
    }
    
    # Create CMakeLists.txt
    create_from_template(
        base_dir / 'cmake' / 'ModuleCMakeLists.txt.in',
        module_dir / 'CMakeLists.txt',
        template_vars
    )
    
    # Create module config
    create_from_template(
        base_dir / 'cmake' / 'ModuleConfig.cmake.in',
        cmake_dir / f'{module_name}Config.cmake.in',
        template_vars
    )
    
    # Create header file
    header_content = f"""#pragma once

#include <string>
#include <memory>

namespace jarvis {{

class {module_camel} {{
public:
    {module_camel}();
    ~{module_camel}();
    
    std::string process(const std::string& input) const;
    
private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
}};

}} // namespace jarvis
"""
    (include_dir / f"{module_name}.hpp").write_text(header_content)
    
    # Create source file
    source_content = f"""#include <jarvis/{module_name}.hpp>

namespace jarvis {{

class {module_camel}::Impl {{
public:
    std::string process(const std::string& input) {{
        // TODO: Implement {module_name} processing
        return "Processed by {module_name}: " + input;
    }}
}};

{module_camel}::{module_camel}() : pImpl(std::make_unique<Impl>()) {{}}
{module_camel}::~{module_camel}() = default;

std::string {module_camel}::process(const std::string& input) const {{
    return pImpl->process(input);
}}

}} // namespace jarvis
"""
    (src_dir / f"{module_name}.cpp").write_text(source_content)
    
    # Create Python bindings
    bindings_content = f"""#include <pybind11/pybind11.h>
#include <jarvis/{module_name}.hpp>

namespace py = pybind11;

PYBIND11_MODULE(_{module_name}, m) {{
    m.doc() = "JARVIS {module_camel} C++ Module";
    
    py::class_<jarvis::{module_camel}>(m, "{module_camel}")
        .def(py::init<>())
        .def("process", &jarvis::{module_camel}::process, "Process input string");
}}
"""
    (python_dir / f"{module_name}_bindings.cpp").write_text(bindings_content)
    
    # Create Python module
    create_from_template(
        base_dir / 'cmake' / 'PythonModuleTemplate.py.in',
        module_dir.parent / f"{module_name}_cpp.py",
        template_vars
    )
    
    # Create test file
    test_content = f"""#include <gtest/gtest.h>
#include <jarvis/{module_name}.hpp>

TEST({module_camel}Test, BasicTest) {{
    jarvis::{module_camel} module;
    auto result = module.process("test");
    EXPECT_NE(result.find("Processed by {module_name}:"), std::string::npos);
}}
"""
    (tests_dir / f"test_{module_name}.cpp").write_text(test_content)
    
    # Create tests/CMakeLists.txt
    tests_cmake = """cmake_minimum_required(VERSION 3.14)
project(@MODULE_NAME@_tests)

# Find dependencies
find_package(GTest REQUIRED)
find_package(@MODULE_NAME@ CONFIG REQUIRED)

# Add test executable
add_executable(test_@MODULE_NAME@
    test_@MODULE_NAME@.cpp
)

# Link libraries
target_link_libraries(test_@MODULE_NAME@
    PRIVATE
        @MODULE_NAME@::@MODULE_NAME@
        GTest::GTest
        GTest::Main
)

# Add test
add_test(NAME @MODULE_NAME@_test
    COMMAND test_@MODULE_NAME@
)
"""
    (tests_dir / "CMakeLists.txt").write_text(
        tests_cmake.replace('@MODULE_NAME@', module_name)
    )
    
    print(f"Created {module_name} module at {module_dir}")

def create_from_template(template_path: Path, output_path: Path, vars: Dict[str, Any]) -> None:
    """Create a file from a template.
    
    Args:
        template_path: Path to the template file
        output_path: Path to the output file
        vars: Template variables
    """
    if not template_path.exists():
        print(f"Warning: Template not found: {template_path}")
        return
    
    content = template_path.read_text()
    
    # Replace variables in the template
    for key, value in vars.items():
        content = content.replace(f'@{key}@', str(value))
    
    output_path.write_text(content)

def main():
    parser = argparse.ArgumentParser(description='Create a new C++ module with Python bindings')
    parser.add_argument('module_name', help='Name of the module to create (e.g., nlp, ml)')
    parser.add_argument('--base-dir', default=Path.cwd(), type=Path,
                       help='Base directory of the JARVIS project')
    
    args = parser.parse_args()
    
    create_module(args.module_name, args.base_dir)

if __name__ == "__main__":
    main()
