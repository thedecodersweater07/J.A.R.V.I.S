import pandas as pd
import numpy as np
import json
import os
from typing import Dict, Any, List, Union
from pathlib import Path

class DataManager:
    def __init__(self, data_dir: str = "ai_training_data"):
        self.data_dir = Path(data_dir)
        self.datasets: Dict[str, Any] = {}
        
    def load_dataset(self, name: str) -> Union[pd.DataFrame, dict, list, str]:
        """Load a dataset by name"""
        file_path = self.data_dir / name
        
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset {name} not found")
            
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
            raise ValueError(f"Unsupported file format: {name}")
    
    def save_dataset(self, name: str, data: Any) -> None:
        """Save a dataset"""
        file_path = self.data_dir / name
        
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
            raise ValueError(f"Unsupported data type: {type(data)}")
    
    def list_datasets(self) -> List[str]:
        """List all available datasets"""
        return [f.name for f in self.data_dir.glob('*.*')]
    
    def validate_dataset(self, name: str) -> bool:
        """Validate dataset structure and content"""
        try:
            data = self.load_dataset(name)
            
            if name == 'training_data.csv':
                required_cols = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'target']
                return all(col in data.columns for col in required_cols)
            
            return True
        except Exception as e:
            print(f"Validation failed for {name}: {str(e)}")
            return False
    
    def get_dataset_info(self, name: str) -> Dict[str, Any]:
        """Get dataset statistics and information"""
        data = self.load_dataset(name)
        info = {
            'name': name,
            'size': os.path.getsize(self.data_dir / name),
            'format': name.split('.')[-1]
        }
        
        if isinstance(data, pd.DataFrame):
            info.update({
                'rows': len(data),
                'columns': list(data.columns),
                'datatypes': data.dtypes.to_dict()
            })
        elif isinstance(data, list):
            info['items'] = len(data)
            
        return info

if __name__ == "__main__":
    # Usage example
    dm = DataManager()
    datasets = dm.list_datasets()
    print(f"Available datasets: {datasets}")
