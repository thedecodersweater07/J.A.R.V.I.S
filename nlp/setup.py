"""
Setup script for building the C++ NLP extension.
"""
import os
import sys
import platform
import subprocess
from typing import List, Optional
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from pathlib import Path

class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = '') -> None:
        super().__init__(name, sources=[])
        self.sourcedir = str(Path(sourcedir).absolute())

class CMakeBuild(build_ext):
    def run(self) -> None:
        try:
            self._check_cmake_installed()
            super().run()
        except Exception as e:
            print(f"Error building extensions: {e}", file=sys.stderr)
            raise
    
    def _check_cmake_installed(self) -> None:
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
    
    def build_extension(self, ext: CMakeExtension) -> None:
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
    
    def _run_command(self, cmd: List[str], cwd: Optional[str] = None) -> None:
        """Run a command and raise an error if it fails."""
        try:
            subprocess.check_call(cmd, cwd=cwd)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Command {cmd} failed with error: {e}") from e

# Read the README for the long description
this_dir = Path(__file__).parent
with open(this_dir / "README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="jarvis-nlp",
    version="0.1.0",
    author="JARVIS Team",
    author_email="info@jarvis-ai.com",
    description="NLP module for JARVIS with C++ extensions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/jarvis-nlp",
    packages=find_packages(exclude=["tests"]),
    ext_modules=[CMakeExtension('_nlp_engine', sourcedir=str(this_dir / "cpp"))],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    python_requires=">=3.8",
    install_requires=[
        'numpy>=1.19.0',
        'pybind11>=2.6.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'mypy>=0.910',
            'black>=21.7b0',
            'isort>=5.9.0',
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: C++",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
)
