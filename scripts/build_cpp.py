#!/usr/bin/env python3
"""
Build script for JARVIS C++ extensions.
"""
import os
import sys
import glob
import shutil
import platform
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Any

# Platform-specific settings
IS_WINDOWS = platform.system() == "Windows"
EXT_SUFFIX = ".pyd" if IS_WINDOWS else ".so"

class BuildError(Exception):
    """Custom exception for build errors."""
    pass

def run_command(cmd: List[str], cwd: Optional[Path] = None) -> bool:
    """Run a shell command and return True if successful."""
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.check_call(cmd, cwd=str(cwd) if cwd else None)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error: {e}", file=sys.stderr)
        return False
    except FileNotFoundError as e:
        print(f"Command not found: {e}", file=sys.stderr)
        return False

def find_cmake() -> str:
    """Find the CMake executable."""
    # Try common CMake executable names
    for cmd in ["cmake", "cmake3"]:
        try:
            subprocess.run([cmd, "--version"], capture_output=True, check=True)
            return cmd
        except (subprocess.SubprocessError, FileNotFoundError):
            continue
    raise BuildError("CMake not found. Please install CMake and add it to your PATH.")

def build_module(module_dir: Path, build_dir: Path, install_dir: Path) -> bool:
    """Build a single C++ module.
    
    Args:
        module_dir: Path to the module directory
        build_dir: Build directory
        install_dir: Installation directory
        
    Returns:
        bool: True if build was successful
    """
    module_name = module_dir.parent.name
    module_build_dir = build_dir / module_name
    
    print(f"\n{'='*80}")
    print(f"Building module: {module_name}")
    print(f"Source: {module_dir}")
    print(f"Build: {module_build_dir}")
    print(f"Install: {install_dir}")
    
    # Create build directory
    module_build_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure with CMake
    cmake_cmd = [
        find_cmake(),
        "-S", str(module_dir),
        "-B", str(module_build_dir),
        f"-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_INSTALL_PREFIX={install_dir}",
        f"-DPYTHON_EXECUTABLE={sys.executable}",
    ]
    
    if not run_command(cmake_cmd):
        print(f"Failed to configure {module_name}", file=sys.stderr)
        return False
    
    # Build with CMake
    build_cmd = [
        find_cmake(),
        "--build", str(module_build_dir),
        "--config", "Release"
    ]
    
    if not run_command(build_cmd):
        print(f"Failed to build {module_name}", file=sys.stderr)
        return False
    
    # Install the module
    install_cmd = [
        find_cmake(),
        "--install", str(module_build_dir),
        "--prefix", str(install_dir)
    ]
    
    if not run_command(install_cmd):
        print(f"Failed to install {module_name}", file=sys.stderr)
        return False
    
    # Copy the built module to the package directory
    module_file = find_module_file(module_build_dir, module_name)
    if module_file:
        target_dir = module_dir.parent / "jarvis" / module_name
        target_dir.mkdir(parents=True, exist_ok=True)
        
        target_file = target_dir / f"_{module_name}{EXT_SUFFIX}"
        shutil.copy2(module_file, target_file)
        print(f"Copied {module_file} to {target_file}")
    
    return True

def find_module_file(build_dir: Path, module_name: str) -> Optional[Path]:
    """Find the built module file."""
    patterns = [
        f"**/*{module_name}*{EXT_SUFFIX}",
        f"**/*.so",
        f"**/*.pyd",
        f"**/*.dll"
    ]
    
    for pattern in patterns:
        for file in build_dir.glob(pattern):
            if file.is_file() and module_name in file.name.lower():
                return file
    return None

def find_cpp_modules(base_dir: Path) -> List[Path]:
    """Find all C++ modules in the project."""
    modules = []
    
    # Look for cpp directories in each module
    for module_dir in base_dir.iterdir():
        if not module_dir.is_dir() or module_dir.name.startswith('.'):
            continue
            
        cpp_dir = module_dir / "cpp"
        cmake_file = cpp_dir / "CMakeLists.txt"
        
        if cmake_file.exists():
            modules.append(cpp_dir)
    
    return modules

def build_cpp_extensions() -> bool:
    """Build all C++ extensions."""
    root_dir = Path(__file__).parent.parent
    build_dir = root_dir / "build"
    install_dir = root_dir / "build" / "install"
    
    # Clean previous builds
    if build_dir.exists():
        print(f"Cleaning build directory: {build_dir}")
        shutil.rmtree(build_dir)
    
    # Create build and install directories
    build_dir.mkdir(parents=True, exist_ok=True)
    install_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all C++ modules
    modules = find_cpp_modules(root_dir)
    
    if not modules:
        print("No C++ modules found.")
        return False
    
    print(f"Found {len(modules)} C++ modules:")
    for module in modules:
        print(f"- {module.relative_to(root_dir)}")
    
    # Build each module
    success = True
    for module_dir in modules:
        if not build_module(module_dir, build_dir, install_dir):
            print(f"Failed to build module: {module_dir}", file=sys.stderr)
            success = False
    
    return success

if __name__ == "__main__":
    try:
        if build_cpp_extensions():
            print("\nBuild completed successfully!")
            sys.exit(0)
        else:
            print("\nBuild failed!", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
