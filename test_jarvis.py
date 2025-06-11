"""
Simple script to test the JARVIS model functionality.
"""

import sys
import os
from pathlib import Path

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

try:
    # Try to import the JarvisModel using relative import
    from models import JarvisModel
    print("✅ Successfully imported JarvisModel")
    
    # Try to create a simple instance
    try:
        model = JarvisModel("jarvis-base")
        print("✅ Successfully created JarvisModel instance")
        print(f"Model device: {model.device}")
    except Exception as e:
        print(f"❌ Failed to create JarvisModel instance: {e}")
        raise
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nTroubleshooting steps:")
    print("1. Make sure you're running this script from the project root directory")
    print("2. Check if all dependencies are installed (pip install -r requirements.txt)")
    print("3. Verify that the 'models' directory exists and contains the jarvis.py file")
    print("\nPython path:")
    for p in sys.path:
        print(f"- {p}")
    raise
