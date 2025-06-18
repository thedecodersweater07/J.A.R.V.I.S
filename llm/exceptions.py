"""
Exceptions for the LLM package.

This module contains all custom exceptions used throughout the LLM package.
"""

class LLMError(Exception):
    """Base exception for all LLM-related errors."""
    pass

class ModelLoadError(LLMError):
    """Raised when a model fails to load."""
    pass

class GenerationError(LLMError):
    """Raised when text generation fails."""
    pass

class TokenizationError(LLMError):
    """Raised when tokenization fails."""
    pass

class ConfigurationError(LLMError):
    """Raised when there's an error in the configuration."""
    pass

class ValidationError(LLMError):
    """Raised when input validation fails."""
    pass

class ResourceError(LLMError):
    """Raised when there's an error accessing resources (e.g., files, APIs)."""
    pass

class UnsupportedModelError(LLMError):
    """Raised when an unsupported model is requested."""
    pass

class BatchProcessingError(LLMError):
    """Raised when batch processing fails."""
    def __init__(self, message: str, errors: list = None):
        super().__init__(message)
        self.errors = errors or []

# Export all exceptions
__all__ = [
    'LLMError',
    'ModelLoadError',
    'GenerationError',
    'TokenizationError',
    'ConfigurationError',
    'ValidationError',
    'ResourceError',
    'UnsupportedModelError',
    'BatchProcessingError',
]
