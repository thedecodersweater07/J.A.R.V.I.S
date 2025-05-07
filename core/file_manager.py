import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import logging

logger = logging.getLogger(__name__)

class FileManager:
    def __init__(self, root_dir: str = "/workspaces/J.A.R.V.I.S"):
        self.root = Path(root_dir)
        self.paths = {
            "data": self.root / "data",
            "models": self.root / "models",
            "config": self.root / "config",
            "logs": self.root / "logs",
            "db": self.root / "data/db"
        }
        self._ensure_directories()
        
    def _ensure_directories(self):
        """Ensure all required directories exist"""
        for path in self.paths.values():
            path.mkdir(parents=True, exist_ok=True)
            
    def get_path(self, category: str, subcategory: Optional[str] = None) -> Path:
        """Get path for specific category"""
        base = self.paths.get(category)
        if not base:
            raise ValueError(f"Unknown category: {category}")
        
        if subcategory:
            return base / subcategory
        return base
