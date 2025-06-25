"""
Utility for dynamic module loading in hyperadvanced_ai.
"""
import importlib
from typing import Any
import logging

def dynamic_import(module_path: str) -> Any:
    """Dynamically import a module by dotted path."""
    return importlib.import_module(module_path)

# Provide a logger for modules that can't use relative imports
logger = logging.getLogger("hyperadvanced_ai.utils.dynamic_loader")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s | %(name)s | %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
