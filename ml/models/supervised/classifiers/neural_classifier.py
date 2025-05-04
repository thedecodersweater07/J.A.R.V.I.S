"""Neural network based classifier"""
import numpy as np
from typing import Dict

class NeuralClassifier:
    def __init__(self, input_size: int, num_classes: int):
        self.input_size = input_size
        self.num_classes = num_classes
        self.weights = None
        self.initialized = False
        
    def initialize(self):
        """Initialize model weights"""
        self.weights = np.random.randn(self.input_size, self.num_classes) * 0.01
        self.initialized = True
    
    def get_params(self) -> Dict:
        """Get model parameters for persistence"""
        return {
            'weights': self.weights,
            'input_size': self.input_size,
            'num_classes': self.num_classes
        }
    
    def load_params(self, params: Dict):
        """Load model parameters from persistence"""
        self.weights = params['weights']
        self.input_size = params['input_size'] 
        self.num_classes = params['num_classes']
        self.initialized = True
