"""
Neural Network - Neuraal netwerk implementatie
Core AI verwerking voor patroonherkenning en gegevensanalyse
"""

import numpy as np
import logging
from typing import Dict, List, Any, Union, Tuple

logger = logging.getLogger(__name__)

class NeuralLayer:
    """Een enkele laag in het neurale netwerk"""
    
    def __init__(self, input_size: int, output_size: int, activation: str = "relu"):
        """
        Initialiseer een neurale netwerklaag
        
        Args:
            input_size: Aantal ingangsneuronen
            output_size: Aantal uitgangsneuronen
            activation: Activatiefunctie ("relu", "sigmoid", "tanh")
        """
        # Xavier/Glorot initialisatie voor gewichten
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2 / (input_size + output_size))
        self.biases = np.zeros(output_size)
        self.activation_type = activation
        self.last_input = None
        self.last_output = None
        
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Voorwaartse passage door de laag"""
        self.last_input = inputs
        result = np.dot(inputs, self.weights) + self.biases
        
        # Activatiefunctie toepassen
        if self.activation_type == "relu":
            output = np.maximum(0, result)
        elif self.activation_type == "sigmoid":
            output = 1 / (1 + np.exp(-result))
        elif self.activation_type == "tanh":
            output = np.tanh(result)
        else:
            output = result  # Lineair/geen activatie
            
        self.last_output = output
        return output

class NeuralNetwork:
    """Neuraal netwerk voor patroonherkenning en gegevensverwerking"""
    
    def __init__(self, hidden_layers: int = 3, neurons_per_layer: int = 128,
                 input_size: int = 64, output_size: int = 32):
        """
        Initialiseer het neurale netwerk
        
        Args:
            hidden_layers: Aantal verborgen lagen
            neurons_per_layer: Neuronen per verborgen laag
            input_size: Grootte van de invoer
            output_size: Grootte van de uitvoer
        """
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.neurons_per_layer = neurons_per_layer
        self.layers = []
        self.initialized = False
        self.learning_rate = 0.001
        
    def initialize(self) -> None:
        """Bouw de netwerkstructuur op"""
        if self.initialized:
            return
            
        logger.info(f"Initialiseren neuraal netwerk met {self.hidden_layers} verborgen lagen")
        
        # Invoerlaag naar eerste verborgen laag
        self.layers.append(NeuralLayer(self.input_size, self.neurons_per_layer))
        
        # Verborgen lagen
        for i in range(self.hidden_layers - 1):
            self.layers.append(NeuralLayer(self.neurons_per_layer, self.neurons_per_layer))
            
        # Laatste verborgen laag naar uitvoerlaag
        self.layers.append(NeuralLayer(self.neurons_per_layer, self.output_size, "sigmoid"))
        
        self.initialized = True
        logger.info("Neuraal netwerk succesvol geïnitialiseerd")
        
    def process(self, input_data: Any) -> np.ndarray:
        """
        Verwerk invoergegevens door het netwerk
        
        Args:
            input_data: Ruwe invoergegevens
            
        Returns:
            Verwerkte gegevens als numpy array
        """
        if not self.initialized:
            self.initialize()
            
        # Converteer invoer naar vector als dat nog niet is gebeurd
        input_vector = self._prepare_input(input_data)
        
        # Voorwaartse propagatie door alle lagen
        current_output = input_vector
        for layer in self.layers:
            current_output = layer.forward(current_output)
            
        return current_output
    
    def _prepare_input(self, input_data: Any) -> np.ndarray:
        """Converteer verschillende invoertypen naar een geschikte vector"""
        if isinstance(input_data, np.ndarray):
            # Zorg ervoor dat de grootte overeenkomt
            if input_data.size != self.input_size:
                # Resize of pad indien nodig
                return np.resize(input_data, self.input_size)
            return input_data
            
        elif isinstance(input_data, (list, tuple)):
            # Converteer lijst naar numpy array
            arr = np.array(input_data, dtype=float)
            # Zorg ervoor dat grootte overeenkomt
            if arr.size != self.input_size:
                return np.resize(arr, self.input_size)
            return arr
            
        elif isinstance(input_data, dict):
            # Plat maken van woordenboek tot vector
            keys = sorted(input_data.keys())
            values = [input_data[k] for k in keys]
            return self._prepare_input(values)
            
        elif isinstance(input_data, str):
            # Eenvoudige tekstcodering (in echte toepassingen zou je embeddings gebruiken)
            char_values = [ord(c) % 256 for c in input_data[:self.input_size]]
            while len(char_values) < self.input_size:
                char_values.append(0)  # Zero-padding
            return np.array(char_values, dtype=float) / 255.0  # Normaliseren
            
        else:
            # Fallback voor andere typen
            logger.warning(f"Onbekend invoertype: {type(input_data)}, gebruik nulvector")
            return np.zeros(self.input_size)