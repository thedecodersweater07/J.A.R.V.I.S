#!/usr/bin/env python3
"""
Build script for JARVIS C++ extensions.
"""
import os
import sys
import subprocess
from pathlib import Path

def build_extension():
    """Build the C++ extension."""
    print("Building JARVIS C++ extensions...")
    
    # Get the current directory
    current_dir = Path(__file__).parent.absolute()
    
    # Create build directory
    build_dir = current_dir / "build"
    build_dir.mkdir(exist_ok=True)
    
    # Run setup.py build
    cmd = [
        sys.executable,
        "setup.py",
        "build_ext",
        "--inplace"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        subprocess.check_call(cmd, cwd=str(current_dir))
        print("\nBuild completed successfully!")
        print(f"Extension should be available at: {current_dir / '_models.*'}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nBuild failed with error: {e}")
        return False

if __name__ == "__main__":
    if build_extension():
        sys.exit(0)
    else:
        sys.exit(1)
