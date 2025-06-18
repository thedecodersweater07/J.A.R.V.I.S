"""
Utility functions and helpers for the JARVIS system.
"""

import os
import sys
import logging
import json
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def setup_environment() -> None:
    """
    Set up the environment variables and paths.
    """
    try:
        # Add project root to Python path
        project_root = str(Path(__file__).parent.parent)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        # Set environment variables
        os.environ["PYTHONPATH"] = os.pathsep.join(
            [project_root] + os.environ.get("PYTHONPATH", "").split(os.pathsep)
        )
        
        # Create necessary directories
        data_dir = Path(project_root) / "data"
        data_dir.mkdir(exist_ok=True)
        
        logs_dir = data_dir / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        logger.info("Environment setup complete")
        
    except Exception as e:
        logger.error(f"Error setting up environment: {e}")
        raise

def load_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Load configuration from a YAML or JSON file.
    
    Args:
        config_path: Path to the configuration file. If None, looks for config.yaml or config.json in the project root.
        
    Returns:
        Dict containing the configuration.
    """
    if config_path is None:
        project_root = Path(__file__).parent.parent
        yaml_config = project_root / "config.yaml"
        json_config = project_root / "config.json"
        
        if yaml_config.exists():
            config_path = yaml_config
        elif json_config.exists():
            config_path = json_config
        else:
            logger.warning("No configuration file found. Using default settings.")
            return {}
    
    config_path = Path(config_path)
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.suffix.lower() == '.json':
                return json.load(f)
            else:  # Assume YAML
                return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {e}")
        return {}

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Name of the logger.
        
    Returns:
        Configured logger instance.
    """
    return logging.getLogger(name)

def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        path: Path to the directory.
        
    Returns:
        Path object for the directory.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path object for the project root.
    """
    return Path(__file__).parent.parent

# Initialize environment when module is imported
setup_environment()
