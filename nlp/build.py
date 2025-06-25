"""
Build script for the C++ NLP extension.
"""
import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path

def build_extension():
    """Build the C++ extension using CMake."""
    # Get the directory of this file
    current_dir = Path(__file__).parent.absolute()
    cpp_dir = current_dir / "cpp"
    build_dir = cpp_dir / "build"
    
    # Create build directory if it doesn't exist
    build_dir.mkdir(exist_ok=True, parents=True)
    
    # Configure with CMake
    cmake_cmd = [
        "cmake",
        "-S", str(cpp_dir),
        "-B", str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release"
    ]
    
    # Platform-specific settings
    if platform.system() == "Windows":
        cmake_cmd.extend(["-A", "x64"])
    
    print("Configuring with CMake...")
    result = subprocess.run(cmake_cmd, cwd=build_dir)
    if result.returncode != 0:
        print("CMake configuration failed")
        return False
    
    # Build the project
    print("Building with CMake...")
    build_cmd = ["cmake", "--build", str(build_dir), "--config", "Release"]
    result = subprocess.run(build_cmd, cwd=build_dir)
    if result.returncode != 0:
        print("Build failed")
        return False
    
    # Copy the built module to the parent directory
    ext = ".pyd" if platform.system() == "Windows" else ".so"
    built_module = build_dir / "Release" / f"_nlp_engine{ext}"
    
    if not built_module.exists():
        # Try different build directory structure
        built_module = build_dir / f"_nlp_engine{ext}"
    
    if built_module.exists():
        target = current_dir / f"_nlp_engine{ext}"
        shutil.copy2(built_module, target)
        print(f"Successfully built and copied {target}")
        return True
    else:
        print(f"Could not find built module at {built_module}")
        return False

if __name__ == "__main__":
    if build_extension():
        print("Build completed successfully")
    else:
        print("Build failed")
        sys.exit(1)