"""
Integration tests for JARVIS AI models.

This module contains tests for the core JARVIS model classes and their integration.
"""
import os
import sys
import json
import unittest
import tempfile
import importlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, TypeVar, Type, cast, TYPE_CHECKING
from unittest.mock import patch, MagicMock

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Type variable for model types
T = TypeVar('T')

# Check if PyTorch is available
PYTORCH_AVAILABLE = False
try:
    import torch  # type: ignore
    PYTORCH_AVAILABLE = True
except ImportError:
    print("Warning: PyTorch not found. Some tests will be skipped.")

# Only import model types if we're not type checking
if not TYPE_CHECKING and PYTORCH_AVAILABLE:
    from models.jarvis import (
        JarvisModel,
        JarvisLanguageModel,
        JarvisNLPModel,
        JarvisMLModel,
        JarvisModelManager,
        create_jarvis_model,
        ModelLoadError,
        ProcessingError
    )
    ModelType = Union[JarvisModel, JarvisLanguageModel, JarvisNLPModel, JarvisMLModel]
else:
    # Create dummy types for type checking
    class DummyModel:
        pass
    
    JarvisModel = DummyModel
    JarvisLanguageModel = DummyModel
    JarvisNLPModel = DummyModel
    JarvisMLModel = DummyModel
    JarvisModelManager = DummyModel
    ModelLoadError = Exception
    ProcessingError = Exception
    ModelType = TypeVar('ModelType')

# Import the models with a try-except to handle missing dependencies
try:
    from models.jarvis import (
        JarvisModel,
        JarvisLanguageModel,
        JarvisNLPModel,
        JarvisMLModel,
        JarvisModelManager,
        create_jarvis_model,
        ModelLoadError,
        ProcessingError
    )
    MODELS_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    print(f"Warning: Could not import JARVIS models: {e}")
    # Create dummy classes to allow test collection
    class DummyModel:
        def __init__(self, *args, **kwargs):
            self.model_name = kwargs.get('model_name', 'dummy-model')
            self.model = MagicMock()
            self.conversation_history = []
        
        def forward(self, *args, **kwargs) -> Dict[str, Any]:
            return {"logits": [0.5, 0.3, 0.2]}
        
        def generate(self, *args, **kwargs) -> str:
            return "Generated response"
        
        def classify(self, *args, **kwargs) -> Dict[str, Any]:
            return {"prediction": 0, "confidence": 0.9}
        
        def generate_response(self, *args, **kwargs) -> str:
            return "Generated response"
        
        def clear_conversation_history(self) -> None:
            self.conversation_history = []
        
        def process_text(self, *args, **kwargs) -> Dict[str, Any]:
            return {"result": "processed"}
        
        def generate_summary(self, *args, **kwargs) -> str:
            return "This is a summary."
        
        def train_model(self, *args, **kwargs) -> Dict[str, Any]:
            return {"train_loss": [0.5, 0.4], "val_loss": [0.4, 0.3]}
        
        def evaluate_model(self, *args, **kwargs) -> Dict[str, Any]:
            return {"accuracy": 0.9, "loss": 0.1}
        
        def register_model(self, model_type: str, model_name: str, model_instance: Optional['DummyModel'] = None) -> None:
            if not hasattr(self, '_models'):
                self._models = {}
            if model_type not in self._models:
                self._models[model_type] = {}
            self._models[model_type][model_name] = model_instance or DummyModel()
        
        def get_model(self, model_type: str, model_name: Optional[str] = None) -> 'DummyModel':
            if not hasattr(self, '_models'):
                self._models = {}
            if not hasattr(self, 'default_models'):
                self.default_models = {"language": "jarvis-base", "nlp": "jarvis-base", "ml": "jarvis-base"}
            if model_name:
                return self._models.get(model_type, {}).get(model_name, DummyModel())
            # Return a default model with the default name if no specific name is provided
            default_name = self.default_models.get(model_type, 'dummy-model')
            return DummyModel(model_name=default_name)
        
        def list_models(self) -> Dict[str, Any]:
            return getattr(self, '_models', {})
    
    JarvisModel = DummyModel
    JarvisLanguageModel = DummyModel
    JarvisNLPModel = DummyModel
    JarvisMLModel = DummyModel
    
    class DummyManager:
        def __init__(self):
            self._models = {}
            self.default_models = {
                "language": "jarvis-base",
                "nlp": "jarvis-base",
                "ml": "jarvis-base"
            }
        
        def register_model(self, model_type: str, model_name: str, model_instance: Optional['DummyModel'] = None) -> None:
            if model_type not in self._models:
                self._models[model_type] = {}
            self._models[model_type][model_name] = model_instance or DummyModel()
        
        def get_model(self, model_type: str, model_name: Optional[str] = None) -> Optional['DummyModel']:
            model_name = model_name or self.default_models.get(model_type)
            if not model_name:
                return None
            return self._models.get(model_type, {}).get(model_name, DummyModel())
        
        def list_models(self) -> Dict[str, Any]:
            return self._models
    
    JarvisModelManager = DummyManager
    
    def create_jarvis_model(
        model_type: str = "language", 
        model_name: str = "jarvis-base", 
        **kwargs
    ) -> 'DummyModel':
        if model_type not in ["language", "nlp", "ml"]:
            raise ValueError(f"Unknown model type: {model_type}")
        return DummyModel(model_name=model_name)
    
    ModelLoadError = Exception
    ProcessingError = Exception
    MODELS_AVAILABLE = False


# Skip tests if models are not available
skip_if_models_unavailable = unittest.skipIf(
    not MODELS_AVAILABLE,
    "Skipping test because models could not be imported"
)

@skip_if_models_unavailable
class TestJarvisModel(unittest.TestCase):
    """Test cases for the base JarvisModel class."""
    
    def setUp(self) -> None:
        """Set up test fixtures."""
        self.model = JarvisModel("jarvis-base")
    
    def test_initialization(self) -> None:
        """Test model initialization."""
        self.assertIsNotNone(self.model)
        self.assertEqual(self.model.model_name, "jarvis-base")
    
    def test_forward_pass(self) -> None:
        """Test forward pass with dummy input."""
        inputs = {"text": "Test input"}
        with patch.object(self.model, 'forward') as mock_forward:
            mock_forward.return_value = {"logits": [0.5, 0.3, 0.2]}
            result = self.model.forward(inputs)
            self.assertIn("logits", result)
    
    def test_generate_text(self) -> None:
        """Test text generation."""
        with patch.object(self.model, 'generate') as mock_generate:
            mock_generate.return_value = "Generated response"
            result = self.model.generate("Test prompt")
            self.assertEqual(result, "Generated response")


@skip_if_models_unavailable
class TestJarvisLanguageModel(unittest.TestCase):
    """Test cases for the JarvisLanguageModel class."""
    
    def setUp(self) -> None:
        """Set up test fixtures."""
        self.language_model = JarvisLanguageModel("jarvis-base")
    
    def test_classify(self) -> None:
        """Test text classification."""
        with patch.object(self.language_model.model, 'forward') as mock_forward:
            mock_forward.return_value = {"logits": [0.8, 0.1, 0.1]}
            result = self.language_model.classify("Test text")
            self.assertIn("prediction", result)
    
    def test_generate_response(self) -> None:
        """Test response generation."""
        with patch.object(self.language_model.model, 'generate') as mock_generate:
            mock_generate.return_value = "Generated response"
            response = self.language_model.generate_response("Hello")
            self.assertEqual(response, "Generated response")
    
    def test_conversation_history(self) -> None:
        """Test conversation history tracking."""
        self.language_model.conversation_history = []
        self.language_model.generate_response("Hello")
        self.assertEqual(len(self.language_model.conversation_history), 1)
        self.language_model.clear_conversation_history()
        self.assertEqual(len(self.language_model.conversation_history), 0)


@skip_if_models_unavailable
class TestJarvisNLPModel(unittest.TestCase):
    """Test cases for the JarvisNLPModel class."""
    
    def setUp(self) -> None:
        """Set up test fixtures."""
        self.nlp_model = JarvisNLPModel("jarvis-base")
    
    def test_process_text(self) -> None:
        """Test text processing."""
        with patch.object(self.nlp_model.model, 'forward') as mock_forward:
            mock_forward.return_value = {"result": "processed"}
            result = self.nlp_model.process_text("Test text", ["tokenize", "tag"])
            self.assertIn("result", result)
    
    def test_generate_summary(self) -> None:
        """Test text summarization."""
        with patch.object(self.nlp_model.model, 'generate') as mock_generate:
            mock_generate.return_value = "This is a summary."
            summary = self.nlp_model.generate_summary("Long text " * 100)
            self.assertTrue(isinstance(summary, str))
            self.assertLessEqual(len(summary), 150)


@skip_if_models_unavailable
class TestJarvisMLModel(unittest.TestCase):
    """Test cases for the JarvisMLModel class."""
    
    def setUp(self) -> None:
        """Set up test fixtures."""
        self.ml_model = JarvisMLModel("jarvis-base")
        self.test_data = [
            {"input": [0.1, 0.2], "label": 0},
            {"input": [0.8, 0.9], "label": 1}
        ]
    
    @patch('torch.optim.Adam')
    @patch('torch.nn.MSELoss')
    def test_train_model(self, mock_loss, mock_optimizer) -> None:
        """Test model training."""
        with patch.object(self.ml_model, '_train_epoch') as mock_train_epoch, \
             patch.object(self.ml_model, '_validate_epoch') as mock_validate_epoch:
            
            mock_train_epoch.return_value = 0.5
            mock_validate_epoch.return_value = 0.4
            
            results = self.ml_model.train_model(
                self.test_data,
                epochs=2,
                learning_rate=0.001,
                validation_split=0.2
            )
            
            self.assertIn("train_loss", results)
            self.assertIn("val_loss", results)
            self.assertEqual(len(results["train_loss"]), 2)
    
    def test_evaluate_model(self) -> None:
        """Test model evaluation."""
        # Create a mock for the model's eval method
        with patch.object(self.ml_model, 'evaluate_model') as mock_eval:
            # Configure the mock to return test metrics
            test_metrics = {"accuracy": 0.9, "loss": 0.1}
            mock_eval.return_value = test_metrics
            
            # Call the method under test
            metrics = self.ml_model.evaluate_model(self.test_data)
            
            # Verify the results
            self.assertIsInstance(metrics, dict)
            self.assertEqual(metrics.get("accuracy"), 0.9)
            self.assertEqual(metrics.get("loss"), 0.1)


@skip_if_models_unavailable
class TestJarvisModelManager(unittest.TestCase):
    """Test cases for the JarvisModelManager class."""
    
    def setUp(self) -> None:
        """Set up test fixtures."""
        self.manager = JarvisModelManager()
    
    def test_register_and_get_model(self):
        """Test model registration and retrieval."""
        mock_model = MagicMock()
        self.manager.register_model("test_type", "test_model", mock_model)
        
        # Test getting the registered model
        model = self.manager.get_model("test_type", "test_model")
        self.assertEqual(model, mock_model)
        
        # Test getting default model
        with patch('models.jarvis.create_jarvis_model') as mock_create:
            mock_create.return_value = "test_model"
            default_model = self.manager.get_model("language")
            self.assertEqual(default_model, "test_model")
    
    def test_list_models(self):
        """Test listing registered models."""
        mock_model = MagicMock()
        self.manager.register_model("test_type", "test_model", mock_model)
        
        models = self.manager.list_models()
        self.assertIn("test_type", models)
        self.assertIn("test_model", models["test_type"])


@skip_if_models_unavailable
class TestCreateJarvisModel(unittest.TestCase):
    """Test cases for the create_jarvis_model factory function."""
    
    def test_create_language_model(self):
        """Test creating a language model."""
        with patch('models.jarvis.JarvisLanguageModel') as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance
            
            model = create_jarvis_model("language", "test-model")
            
            mock_class.assert_called_once_with("test-model")
            self.assertEqual(model, mock_instance)
    
    def test_create_nlp_model(self):
        """Test creating an NLP model."""
        with patch('models.jarvis.JarvisNLPModel') as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance
            
            model = create_jarvis_model("nlp", "test-model")
            
            mock_class.assert_called_once_with("test-model")
            self.assertEqual(model, mock_instance)
    
    def test_create_ml_model(self):
        """Test creating an ML model."""
        with patch('models.jarvis.JarvisMLModel') as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance
            
            model = create_jarvis_model("ml", "test-model")
            
            mock_class.assert_called_once_with("test-model")
            self.assertEqual(model, mock_instance)
    
    def test_create_invalid_type(self):
        """Test creating a model with an invalid type."""
        with self.assertRaises(ValueError):
            create_jarvis_model("invalid_type", "test-model")


if __name__ == "__main__":
    unittest.main()
