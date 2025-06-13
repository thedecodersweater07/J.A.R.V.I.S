"""
Integration tests for J.A.R.V.I.S. components.

These tests verify that different components work together correctly.
"""
import unittest
from unittest.mock import patch, MagicMock
from typing import Dict, Any, Optional
import json

# Import the components we want to test
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
except (ImportError, ModuleNotFoundError):
    MODELS_AVAILABLE = False

# Skip tests if models are not available
skip_if_models_unavailable = unittest.skipIf(
    not MODELS_AVAILABLE,
    "Skipping test because models could not be imported"
)

class TestIntegration(unittest.TestCase):
    """Integration tests for J.A.R.V.I.S. components."""
    
    @skip_if_models_unavailable
    def test_language_model_integration(self):
        """Test integration between language model and base model."""
        # Create a mock for the base model
        with patch('models.jarvis.JarvisModel') as mock_model_class:
            # Configure the mock
            mock_model = MagicMock()
            mock_model.generate.return_value = "Generated response"
            mock_model_class.return_value = mock_model
            
            # Create a language model instance
            language_model = JarvisLanguageModel("test-model")
            
            # Test generate_response
            response = language_model.generate_response("Hello")
            self.assertEqual(response, "Generated response")
            
            # Verify the base model was called correctly
            mock_model.generate.assert_called_once()
    
    @skip_if_models_unavailable
    def test_model_manager_integration(self):
        """Test model manager integration with model creation."""
        # Create a mock for model creation
        with patch('models.jarvis.create_jarvis_model') as mock_create_model:
            # Configure the mock
            mock_model = MagicMock()
            mock_create_model.return_value = mock_model
            
            # Create a model manager instance
            manager = JarvisModelManager()
            
            # Test registering and getting a model
            manager.register_model("test_type", "test_model")
            model = manager.get_model("test_type", "test_model")
            
            # Verify the model was created and registered correctly
            self.assertIsNotNone(model)
            mock_create_model.assert_called_once_with("test_type", "test_model")
    
    @skip_if_models_unavailable
    def test_nlp_model_processing(self):
        """Test NLP model text processing integration."""
        # Create a mock for the base model
        with patch('models.jarvis.JarvisModel') as mock_model_class:
            # Configure the mock
            mock_model = MagicMock()
            mock_model.process_text.return_value = {"tokens": ["test", "text"], "tags": ["NOUN", "NOUN"]}
            mock_model_class.return_value = mock_model
            
            # Create an NLP model instance
            nlp_model = JarvisNLPModel("test-model")
            
            # Test process_text
            result = nlp_model.process_text("test text", ["tokenize", "tag"])
            
            # Verify the results
            self.assertIn("tokens", result)
            self.assertIn("tags", result)
            mock_model.process_text.assert_called_once_with("test text", ["tokenize", "tag"])

if __name__ == "__main__":
    unittest.main()
