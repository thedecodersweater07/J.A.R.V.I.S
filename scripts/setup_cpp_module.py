#!/usr/bin/env python3
"""
Script to set up a new JARVIS C++ module with the standard structure.
"""
import os
import sys
import shutil
from pathlib import Path

def create_module(module_name: str, base_dir: Path):
    """Create a new C++ module with the standard structure."""
    module_dir = base_dir / module_name
    cpp_dir = module_dir / 'cpp'
    
    # Create directories
    cpp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create CMakeLists.txt from template
    template = (base_dir / 'cmake' / 'ModuleTemplate.cmake.in').read_text()
    cmake_content = template.replace('@MODULE_NAME@', module_name)
    (cpp_dir / 'CMakeLists.txt').write_text(cmake_content)
    
    # Create basic source files
    header_content = f"""#pragma once

namespace jarvis::{module_name} {{

class {module_name.capitalize()} {{
public:
    {module_name.capitalize()}() = default;
    ~{module_name.capitalize()}() = default;

    // Add your methods here
}};

}} // namespace jarvis::{module_name}
"""
    
    source_content = f"""#include "{module_name}.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_{module_name}(py::module_ &m) {{
    using namespace jarvis::{module_name};
    
    py::class_<{module_name.capitalize()}>(m, "{module_name.capitalize()}")
        .def(py::init<>())
        // Add your Python bindings here
        ;
}}

PYBIND11_MODULE(_{module_name}, m) {{
    m.doc() = "JARVIS {module_name} module";
    init_{module_name}(m);
}}
"""
    
    # Write source files
    (cpp_dir / f"{module_name}.hpp").write_text(header_content)
    (cpp_dir / f"{module_name}.cpp").write_text(source_content)
    
    # Create Python __init__.py
    init_py = f"""# JARVIS {module_name} module

try:
    from ._{module_name} import *
    __all__ = ['{module_name.capitalize()}']
except ImportError as e:
    import warnings
    warnings.warn(f"Failed to import C++ extension for {{__name__}}: {{e}}")
    
    # Fallback Python implementation would go here
    class {module_name.capitalize()}:
        def __init__(self):
            raise RuntimeError("C++ extension not available. Please build the module first.")"""
"""
    
    (module_dir / "__init__.py").write_text(init_py)
    
    # Create a simple test file
    test_dir = base_dir / 'tests' / module_name
    test_dir.mkdir(parents=True, exist_ok=True)
    
    test_content = f"""""Test {module_name} module"""

import unittest

class Test{module_name.capitalize()}(unittest.TestCase):
    def test_import(self):
        try:
            from jarvis.{module_name} import {module_name.capitalize()}
            # Add more tests here
        except ImportError:
            self.skipTest("C++ extension not available")

if __name__ == '__main__':
    unittest.main()
"""
    (test_dir / f"test_{module_name}.py").write_text(test_content)
    
    print(f"Created new module '{module_name}' at {module_dir}")

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <module_name>")
        sys.exit(1)
    
    module_name = sys.argv[1].lower()
    base_dir = Path(__file__).parent.parent
    
    if not (base_dir / 'cmake' / 'JarvisCppConfig.cmake').exists():
        print("Error: Must be run from JARVIS root directory")
        sys.exit(1)
    
    create_module(module_name, base_dir)

if __name__ == '__main__':
    main()
