import os
import json
import pickle
import logging
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class StorageBackend(ABC):
    """Abstract base class for storage backends"""
    
    @abstractmethod
    def save(self, key: str, data: Any) -> bool:
        """Save data to storage"""
        pass
        
    @abstractmethod
    def load(self, key: str) -> Any:
        """Load data from storage"""
        pass
        
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if data exists in storage"""
        pass
        
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete data from storage"""
        pass
        
    @abstractmethod
    def list_keys(self, pattern: str = "*") -> List[str]:
        """List available keys in storage"""
        pass


class FileSystemStorage(StorageBackend):
    """File system based storage backend"""
    
    def __init__(self, base_dir: Union[str, Path]):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_path(self, key: str) -> Path:
        """Get full path for a key"""
        return self.base_dir / key
        
    def save(self, key: str, data: Any) -> bool:
        """Save data to file system"""
        try:
            path = self._get_path(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Handle different data types
            if isinstance(data, (dict, list)):
                with open(path, 'w') as f:
                    json.dump(data, f, indent=2)
            elif isinstance(data, str):
                with open(path, 'w') as f:
                    f.write(data)
            else:
                # Use pickle for other data types
                with open(path, 'wb') as f:
                    pickle.dump(data, f)
                    
            logger.debug(f"Saved data to {path}")
            return True
        except Exception as e:
            logger.error(f"Error saving data to {key}: {str(e)}")
            return False
            
    def load(self, key: str) -> Any:
        """Load data from file system"""
        path = self._get_path(key)
        if not path.exists():
            logger.error(f"File not found: {path}")
            raise FileNotFoundError(f"File not found: {path}")
            
        try:
            # Try JSON first
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                # Not JSON, try string
                try:
                    with open(path, 'r') as f:
                        return f.read()
                except UnicodeDecodeError:
                    # Not text, try pickle
                    with open(path, 'rb') as f:
                        return pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading data from {key}: {str(e)}")
            raise
            
    def exists(self, key: str) -> bool:
        """Check if file exists"""
        return self._get_path(key).exists()
        
    def delete(self, key: str) -> bool:
        """Delete file"""
        try:
            path = self._get_path(key)
            if path.exists():
                path.unlink()
                logger.debug(f"Deleted {path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting {key}: {str(e)}")
            return False
            
    def list_keys(self, pattern: str = "*") -> List[str]:
        """List files matching pattern"""
        return [str(p.relative_to(self.base_dir)) for p in self.base_dir.glob(pattern)]


class StorageFactory:
    """Factory for creating storage backends"""
    
    @staticmethod
    def get_storage(storage_type: str, **kwargs) -> StorageBackend:
        """
        Get storage backend by type
        
        Args:
            storage_type: Type of storage backend ('file', 'memory', etc.)
            **kwargs: Additional arguments for the storage backend
            
        Returns:
            StorageBackend instance
        """
        if storage_type == 'file':
            base_dir = kwargs.get('base_dir', 'data')
            return FileSystemStorage(base_dir)
        else:
            raise ValueError(f"Unsupported storage type: {storage_type}")
