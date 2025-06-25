from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import os
import sys
import platform
import subprocess
from pathlib import Path

# Check if pybind11 is installed
try:
    import pybind11
    pybind11_include = pybind11.get_include()
except ImportError:
    raise ImportError(
        "pybind11 is required for building C++ extensions. "
        "Install it with: pip install pybind11"
    )

# Get the Python include directory
python_include = sys.exec_prefix + '/include'

# Define the extension module
cpp_extension = Extension(
    name='_models',
    sources=[
        str(Path('cpp/jarvis.cpp')),
        str(Path('cpp/python_bindings.cpp')),
    ],
    include_dirs=[
        str(Path('cpp')),
        pybind11_include,
        python_include,
    ],
    language='c++',
    extra_compile_args=['-std=c++17'] if platform.system() != 'Windows' else ['/std:c++17'],
)

# Custom build command to handle compilation
class BuildExt(build_ext):
    def build_extensions(self):
        # Customize compiler flags based on platform
        if platform.system() == 'Windows':
            for ext in self.extensions:
                ext.extra_compile_args = ['/std:c++17', '/EHsc']
        else:
            for ext in self.extensions:
                ext.extra_compile_args = ['-std=c++17', '-O3']
        
        build_ext.build_extensions(self)

# Setup configuration
setup(
    name="jarvis_models",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=[cpp_extension],
    cmdclass={'build_ext': BuildExt},
    python_requires='>=3.7',
    install_requires=[
        'pybind11>=2.6.0',
    ],
    # Include the compiled extension in the package
    package_data={
        '': ['*.dll', '*.so', '*.pyd'],
    },
    zip_safe=False,
)
