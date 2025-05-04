"""Base feature extractor implementation"""
from typing import Dict, Any, List, Optional
import numpy as np
import torch

class FeatureExtractor:
    """Base class for feature extraction"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
    def extract(self, data: Any) -> np.ndarray:
        """Extract features from input data"""
        if isinstance(data, (np.ndarray, torch.Tensor)):
            return self._extract_numeric(data)
        elif isinstance(data, (str, List[str])):
            return self._extract_text(data)
        elif isinstance(data, Dict):
            return self._extract_structured(data)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
            
    def _extract_numeric(self, data: Any) -> np.ndarray:
        """Extract features from numeric data"""
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        return np.asarray(data)
        
    def _extract_text(self, data: Any) -> np.ndarray:
        """Extract features from text data"""
        if isinstance(data, str):
            data = [data]
        unique_words = set(" ".join(data).split())
        word_to_idx = {word: i for i, word in enumerate(unique_words)}
        features = np.zeros((len(data), len(unique_words)))
        for i, text in enumerate(data):
            for word in text.split():
                if word in word_to_idx:
                    features[i, word_to_idx[word]] += 1
        return features
        
    def _extract_structured(self, data: Dict) -> np.ndarray:
        """Extract features from structured data"""
        features = []
        for key in sorted(data.keys()):
            value = data[key]
            if isinstance(value, (int, float)):
                features.append(value)
            elif isinstance(value, str):
                features.append(len(value))
            elif isinstance(value, (list, tuple)):
                features.append(len(value))
        return np.array(features)
