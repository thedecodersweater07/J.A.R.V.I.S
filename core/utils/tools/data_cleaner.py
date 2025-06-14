from pathlib import Path
import shutil
import json
import logging
from typing import Dict, List, Set

logger = logging.getLogger(__name__)

class DataCleaner:
    def __init__(self, data_root: Path):
        self.data_root = data_root
        self.backup_dir = data_root / "backup"
        self.backup_dir.mkdir(exist_ok=True)
        
    def clean_duplicates(self, file_hashes: Dict[str, str]) -> List[Path]:
        """Remove duplicate files based on content hash"""
        removed = []
        hash_map = {}
        
        for file_path, file_hash in file_hashes.items():
            if file_hash in hash_map:
                # Backup and remove duplicate
                self._backup_file(Path(file_path))
                Path(file_path).unlink()
                removed.append(Path(file_path))
            else:
                hash_map[file_hash] = file_path
                
        return removed
        
    def _backup_file(self, file_path: Path) -> None:
        """Backup file before removal"""
        backup_path = self.backup_dir / file_path.name
        shutil.copy2(file_path, backup_path)
