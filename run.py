#!/usr/bin/env python3
"""
Main entry point for running the JARVIS application.
This script starts both the backend FastAPI server and the frontend development server.
"""

import os
import sys
import subprocess
import signal
import time
from pathlib import Path

def run():
    """Run the JARVIS application."""
    # Get the project root directory
    project_root = Path(__file__).parent.absolute()
    server_dir = project_root / 'server'
    web_dir = server_dir / 'web'
    
    # Check if we're in the correct directory
    if not (server_dir.exists() and web_dir.exists()):
        print("Error: Could not find server and web directories. Please run from project root.")
        sys.exit(1)
    
    # Start the backend server
    print("Starting backend server...")
    backend_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", "8080", "--reload"],
        cwd=str(server_dir),
        shell=True
    )
    
    # Give the backend a moment to start
    time.sleep(2)
    
    # Start the frontend development server
    print("Starting frontend development server...")
    frontend_process = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=str(web_dir),
        shell=True
    )
    
    # Handle keyboard interrupt
    def signal_handler(sig, frame):
        print("\nShutting down servers...")
        backend_process.terminate()
        frontend_process.terminate()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    print("\nJARVIS is running!")
    print("  - Backend: http://localhost:8080")
    print("  - Frontend: http://localhost:3000")
    print("\nPress Ctrl+C to stop")
    
    # Keep the script running
    try:
        backend_process.wait()
        frontend_process.wait()
    except KeyboardInterrupt:
        signal_handler(None, None)

if __name__ == "__main__":
    run()
