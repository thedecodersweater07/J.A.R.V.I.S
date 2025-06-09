import os
import sys
import logging
import torch
import psutil
from pathlib import Path
from typing import Dict, Any
import multiprocessing as mp

# Add project root to path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.logging import setup_logging

logger = logging.getLogger("jarvis.diagnostics")

class SystemDiagnostics:
    @staticmethod
    def check_system_requirements() -> Dict[str, Any]:
        return {
            "gpu_available": torch.cuda.is_available(),
            "gpu_info": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
            "memory_available": psutil.virtual_memory().available / (1024 * 1024 * 1024),  # GB
            "cpu_cores": psutil.cpu_count(logical=False),
            "python_version": sys.version.split()[0]
        }
        
    @staticmethod
    def validate_dependencies() -> bool:
        required_packages = [
            "torch", "numpy", "transformers", "fastapi", 
            "uvicorn", "pydantic", "glfw", "imgui"
        ]
        
        missing = []
        for pkg in required_packages:
            try:
                __import__(pkg)
            except ImportError:
                missing.append(pkg)
                
        if missing:
            logger.error(f"Missing required packages: {', '.join(missing)}")
            return False
        return True

def launch_jarvis():
    """Enhanced JARVIS launch with better error handling"""
    try:
        # Setup logging first
        setup_logging(
            level="DEBUG" if "--debug" in sys.argv else "INFO",
            log_dir="logs",
            log_file="jarvis.log"
        )
        
        logger.info("Starting system diagnostics...")
        
        # Run diagnostics
        diag = SystemDiagnostics()
        sys_info = diag.check_system_requirements()
        
        # Validate system state
        if not diag.validate_dependencies():
            logger.error("Critical dependencies missing")
            return 1
            
        # Initialize multiprocessing support for Windows
        if sys.platform == "win32":
            mp.freeze_support()
        
        # Import and start JARVIS
        from main import JARVIS
        
        jarvis = JARVIS()
        jarvis.run()
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
        return 0
    except Exception as e:
        logger.critical(f"Fatal error during launch: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(launch_jarvis())
