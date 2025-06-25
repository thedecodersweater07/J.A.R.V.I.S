#!/usr/bin/env python3
"""
JARVIS AI - Setup script with C++ extensions support
"""
import os
import sys
import subprocess
import platform
from pathlib import Path
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
from setuptools.dist import Distribution

# Check Python version
if sys.version_info < (3, 8):
    raise RuntimeError("JARVIS requires Python 3.8 or higher")

# Base directory
BASE_DIR = Path(__file__).parent.absolute()

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = str(Path(sourcedir).absolute())

class CMakeBuild(build_ext):
    def run(self):
        try:
            self._check_cmake_installed()
            for ext in self.extensions:
                self.build_extension(ext)
        except Exception as e:
            print(f"Error building extensions: {e}", file=sys.stderr)
            raise
    
    def _check_cmake_installed(self):
        """Check if CMake is installed and available in PATH."""
        try:
            subprocess.run(
                ["cmake", "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            raise RuntimeError(
                "CMake is required to build this extension. "
                "Please install it from https://cmake.org/download/ and ensure it's in your PATH."
            ) from e
    
    def build_extension(self, ext):
        extdir = str(Path(self.get_ext_fullpath(ext.name)).parent.absolute())
        
        # Ensure the extension directory exists
        os.makedirs(extdir, exist_ok=True)
        
        # Configure CMake
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            "-DCMAKE_BUILD_TYPE=Release"
        ]
        
        # Platform-specific settings
        if platform.system() == "Windows":
            cmake_args.extend(["-A", "x64"])
        
        # Create build directory
        build_temp = str(Path(self.build_temp) / ext.name)
        os.makedirs(build_temp, exist_ok=True)
        
        # Run CMake
        self._run_command(['cmake', ext.sourcedir] + cmake_args, cwd=build_temp)
        self._run_command(['cmake', '--build', '.', '--config', 'Release'], cwd=build_temp)
    
    def _run_command(self, cmd, cwd=None):
        """Run a command and raise an error if it fails."""
        try:
            subprocess.check_call(cmd, cwd=cwd)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Command {cmd} failed with error: {e}") from e

# Get requirements
def get_requirements():
    with open('requirements.txt') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Find all Python packages in the project
packages = find_packages(include=['*', 'hyperadvanced_ai', 'hyperadvanced_ai.*', 'llm', 'llm.*', 'nlp', 'nlp.*'])

# Define C++ extensions
cpp_extensions = [
    CMakeExtension('jarvis.nlp._nlp_engine', 'nlp/cpp'),
    # Add other C++ extensions here
]

# Main setup
setup(
    name="jarvis",
    version="0.1.0",
    description="JARVIS AI Assistant",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    author="JARVIS Team",
    author_email="info@jarvis-ai.com",
    url="https://github.com/yourusername/jarvis",
    packages=packages,
    ext_modules=cpp_extensions,
    cmdclass={
        'build_ext': CMakeBuild,
    },
    install_requires=get_requirements(),
    python_requires='>=3.8',
    package_data={
        '': ['*.yaml', '*.json', '*.txt', '*.md', '*.h', '*.hpp', '*.cpp', 'CMakeLists.txt'],
    },
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: C++',
        'Operating System :: OS Independent',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'mypy>=0.910',
            'black>=21.7b0',
            'isort>=5.9.0',
            'cmake>=3.14',
            'pybind11>=2.6.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'jarvis=jarvis.cli:main',
        ],
    },
    setup_requires=[
        'nltk>=3.6.0',
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="JARVIS AI Assistant - A sophisticated AI assistant",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/yourusername/jarvis",
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
