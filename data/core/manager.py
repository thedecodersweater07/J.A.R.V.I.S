import pandas as pd
import numpy as np
import json
import os
import logging
from typing import Dict, Any, List, Union, Optional, Callable
from pathlib import Path
from functools import lru_cache

# Setup logging
logger = logging.getLogger(__name__)

class DataManager:
    """
    Enhanced DataManager for handling various dataset formats with improved
    error handling, caching, and performance optimizations.
    """
    def __init__(self, data_dir: str = "ai_training_data"):
        self.data_dir = Path(data_dir)
        self.datasets: Dict[str, Any] = {}
        self.validators: Dict[str, Callable] = {}
        
        # Ensure data directory exists
        if not self.data_dir.exists():
            logger.info(f"Creating data directory: {self.data_dir}")
            self.data_dir.mkdir(parents=True, exist_ok=True)
        
    @lru_cache(maxsize=32)
    def load_dataset(self, name: str) -> Union[pd.DataFrame, dict, list, str]:
        """
        Load a dataset by name with caching for improved performance
        
        Args:
            name: Name of the dataset file
            
        Returns:
            The loaded dataset in appropriate format
            
        Raises:
            FileNotFoundError: If dataset file doesn't exist
            ValueError: If file format is unsupported
        """
        file_path = self.data_dir / name
        
        if not file_path.exists():
            logger.error(f"Dataset {name} not found at {file_path}")
            raise FileNotFoundError(f"Dataset {name} not found")
            
        logger.debug(f"Loading dataset: {name}")
        
        try:
            if name.endswith('.csv'):
                return pd.read_csv(file_path)
            elif name.endswith('.json'):
                with open(file_path) as f:
                    return json.load(f)
            elif name.endswith('.jsonl'):
                data = []
                with open(file_path) as f:
                    for line in f:
                        data.append(json.loads(line))
                return data
            elif name.endswith('.txt'):
                with open(file_path) as f:
                    return f.read()
            elif name.endswith('.npy'):
                return np.load(file_path)
            else:
                logger.error(f"Unsupported file format: {name}")
                raise ValueError(f"Unsupported file format: {name}")
        except Exception as e:
            logger.error(f"Error loading dataset {name}: {str(e)}")
            raise
    
    def save_dataset(self, name: str, data: Any) -> None:
        """
        Save a dataset with improved error handling
        
        Args:
            name: Name to save the dataset as
            data: The dataset to save
            
        Raises:
            ValueError: If data type is unsupported
        """
        file_path = self.data_dir / name
        
        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.debug(f"Saving dataset: {name}")
        
        try:
            if isinstance(data, pd.DataFrame):
                data.to_csv(file_path, index=False)
            elif isinstance(data, (dict, list)):
                if name.endswith('.jsonl'):
                    with open(file_path, 'w') as f:
                        for item in data:
                            f.write(json.dumps(item) + '\n')
                else:
                    with open(file_path, 'w') as f:
                        json.dump(data, f, indent=2)
            elif isinstance(data, str):
                with open(file_path, 'w') as f:
                    f.write(data)
            elif isinstance(data, np.ndarray):
                np.save(file_path, data)
            else:
                logger.error(f"Unsupported data type: {type(data)}")
                raise ValueError(f"Unsupported data type: {type(data)}")
        except Exception as e:
            logger.error(f"Error saving dataset {name}: {str(e)}")
            raise
    
    def list_datasets(self, pattern: str = "*.*") -> List[str]:
        """
        List all available datasets with optional pattern filtering
        
        Args:
            pattern: Glob pattern to filter datasets (default: "*.*")
            
        Returns:
            List of dataset names matching the pattern
        """
        return [f.name for f in self.data_dir.glob(pattern)]
    
    def register_validator(self, dataset_name: str, validator_func: Callable) -> None:
        """
        Register a custom validator function for a specific dataset
        
        Args:
            dataset_name: Name of the dataset to validate
            validator_func: Function that takes dataset as input and returns bool
        """
        self.validators[dataset_name] = validator_func
        logger.debug(f"Registered validator for dataset: {dataset_name}")
    
    def validate_dataset(self, name: str) -> bool:
        """
        Validate dataset structure and content using registered validators
        
        Args:
            name: Name of the dataset to validate
            
        Returns:
            True if validation passes, False otherwise
        """
        try:
            data = self.load_dataset(name)
            
            # Use custom validator if registered
            if name in self.validators:
                return self.validators[name](data)
            
            # Default validators for known datasets
            if name == 'training_data.csv':
                required_cols = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'target']
                return all(col in data.columns for col in required_cols)
            
            return True
        except Exception as e:
            logger.error(f"Validation failed for {name}: {str(e)}")
            return False
    
    def get_dataset_info(self, name: str) -> Dict[str, Any]:
        """
        Get dataset statistics and information with enhanced metrics
        
        Args:
            name: Name of the dataset
            
        Returns:
            Dictionary containing dataset information and statistics
        """
        data = self.load_dataset(name)
        file_path = self.data_dir / name
        
        info = {
            'name': name,
            'size': os.path.getsize(file_path),
            'format': name.split('.')[-1],
            'last_modified': os.path.getmtime(file_path)
        }
        
        if isinstance(data, pd.DataFrame):
            info.update({
                'rows': len(data),
                'columns': list(data.columns),
                'datatypes': {str(k): str(v) for k, v in data.dtypes.to_dict().items()},
                'missing_values': data.isnull().sum().to_dict(),
                'memory_usage': data.memory_usage(deep=True).sum()
            })
        elif isinstance(data, list):
            info['items'] = len(data)
            
        return info
    
    def delete_dataset(self, name: str) -> bool:
        """
        Delete a dataset file
        
        Args:
            name: Name of the dataset to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            file_path = self.data_dir / name
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Deleted dataset: {name}")
                return True
            else:
                logger.warning(f"Dataset {name} not found for deletion")
                return False
        except Exception as e:
            logger.error(f"Error deleting dataset {name}: {str(e)}")
            return False

if __name__ == "__main__":
    # Usage example
    logging.basicConfig(level=logging.INFO)
    dm = DataManager()
    datasets = dm.list_datasets()
    print(f"Available datasets: {datasets}")
