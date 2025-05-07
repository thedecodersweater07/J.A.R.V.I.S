from pathlib import Path
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

class ProjectStructure:
    def __init__(self, root: Path):
        self.root = root
        self.required_dirs = {
            "data": ["raw", "processed", "interim", "external", "language"],
            "config": ["defaults", "profiles", "secrets"],
            "models": ["checkpoints", "exports"],
            "logs": [],
            "docs": ["api", "training", "modules"],
            "tests": ["unit", "integration", "e2e"]
        }
        
    def validate(self) -> Dict[str, bool]:
        status = {}
        for dir_name, subdirs in self.required_dirs.items():
            dir_path = self.root / dir_name
            status[dir_name] = dir_path.exists()
            for subdir in subdirs:
                subdir_path = dir_path / subdir
                status[f"{dir_name}/{subdir}"] = subdir_path.exists()
        return status

    def create_missing(self) -> None:
        for dir_name, subdirs in self.required_dirs.items():
            dir_path = self.root / dir_name
            dir_path.mkdir(exist_ok=True)
            for subdir in subdirs:
                (dir_path / subdir).mkdir(exist_ok=True)
