import subprocess
import sys
import os

def run_command(command):
    print(f"Running: {command}")
    try:
        subprocess.check_call(command, shell=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}")
        print(f"Error: {e}")
        return False

def main():
    print("Updating Python dependencies...")
    
    # First, upgrade pip
    if not run_command(f"{sys.executable} -m pip install --upgrade pip"):
        print("Failed to upgrade pip")
        return
    
    # Install compatible versions
    requirements = [
        "numpy>=1.26.0,<2.0.0",
        "torch==2.2.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118",
        "pandas>=2.1.0,<3.0.0",
        "spacy>=3.7.0,<4.0.0",
        "transformers>=4.35.0,<5.0.0"
    ]
    
    for req in requirements:
        if not run_command(f"{sys.executable} -m pip install --upgrade {req}"):
            print(f"Warning: Failed to install/upgrade {req}")
    
    print("Dependency update completed.")

if __name__ == "__main__":
    main()
