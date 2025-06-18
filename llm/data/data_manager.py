from __future__ import annotations
from typing import List, Dict, Any, Optional, Union, TypeVar, Type, TYPE_CHECKING
from pathlib import Path
import logging

# Type variable for SecureDataHandler
SecureDataHandlerT = TypeVar('SecureDataHandlerT', bound='BaseSecureDataHandler')

# Base SecureDataHandler class
class BaseSecureDataHandler:
    """Base class for secure data handling with fallback implementation."""
    def secure_load(self, *args: Any, **kwargs: Any) -> bytes:
        """Securely load data (to be implemented by subclasses)."""
        raise NotImplementedError("secure_load must be implemented by subclasses")
        
    def secure_save(self, *args: Any, **kwargs: Any) -> None:
        """Securely save data (to be implemented by subclasses)."""
        raise NotImplementedError("secure_save must be implemented by subclasses")

# Try to import PyTorch with fallback
try:
    import torch  # type: ignore[import]
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch is not installed. Some functionality may be limited.")

# Try to import the actual SecureDataHandler implementation if available
SECURITY_AVAILABLE = False
try:
    from security.secure_data_handler import SecureDataHandler as ExternalSecureDataHandler  # type: ignore[import]
    
    class SecureDataHandler(ExternalSecureDataHandler):  # type: ignore[valid-type,misc]
        """Wrapper for external SecureDataHandler implementation."""
        pass
        
    SECURITY_AVAILABLE = True
    
except ImportError:
    logging.warning("External SecureDataHandler is not available. Using fallback implementation.")
    
    # Fallback implementation
    class SecureDataHandler(BaseSecureDataHandler):  # type: ignore[no-redef]
        """Fallback implementation of SecureDataHandler."""
        def secure_load(self, *args: Any, **kwargs: Any) -> bytes:
            """Load data without security features."""
            filepath = args[0] if args else kwargs.get('filepath')
            if not filepath:
                raise ValueError("filepath is required")
            with open(filepath, 'rb') as f:
                return f.read()
                
        def secure_save(self, *args: Any, **kwargs: Any) -> None:
            """Save data without security features."""
            filepath = args[0] if args else kwargs.get('filepath')
            data = args[1] if len(args) > 1 else kwargs.get('data')
            if not filepath or data is None:
                raise ValueError("filepath and data are required")
            mode = 'wb' if isinstance(data, bytes) else 'w'
            with open(filepath, mode) as f:
                f.write(data)

class LLMDataManager:
    """
    Manages data loading and preprocessing for LLM training and inference.
    Handles both secure and regular data loading with optional PyTorch tensor conversion.
    """
    
    def __init__(self, use_secure_loader: bool = True, encoding: str = 'utf-8'):
        """
        Initialize the data manager.
        
        Args:
            use_secure_loader: Whether to use secure data loading if available
            encoding: Text encoding to use for file operations
        """
        self.use_secure_loader = use_secure_loader and SECURITY_AVAILABLE
        self.encoding = encoding
        self.secure_handler = SecureDataHandler()
        
    def load_training_corpus(
        self, 
        texts_path: Union[str, Path], 
        exclude_patterns: Optional[List[str]] = None,
        file_pattern: str = "*.txt"
    ) -> List[str]:
        """
        Load and sanitize training texts from a directory.
        
        Args:
            texts_path: Path to the directory containing text files
            exclude_patterns: List of patterns to exclude from the loaded text
            file_pattern: File pattern to match (e.g., "*.txt", "*.json")
            
        Returns:
            List of sanitized text strings
            
        Raises:
            FileNotFoundError: If the texts directory doesn't exist
        """
        texts_path = Path(texts_path)
        exclude_patterns = exclude_patterns or []
        texts: List[str] = []
        
        if not texts_path.exists():
            raise FileNotFoundError(f"Directory not found: {texts_path}")
        if not texts_path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {texts_path}")
            
        for file in texts_path.glob(file_pattern):
            if file.is_file() and self._is_safe_file(file):
                try:
                    text = self._load_and_sanitize(file, exclude_patterns)
                    texts.extend(text)
                except Exception as e:
                    logging.warning(f"Error processing file {file}: {e}", exc_info=True)
                    continue
                    
        return texts
        
    def _is_safe_file(self, filepath: Path) -> bool:
        """
        Check if a file is safe to load based on its path and content.
        
        Args:
            filepath: Path to the file to check
            
        Returns:
            bool: True if the file is considered safe
        """
        # Check for sensitive patterns in path
        sensitive_terms = ["sensitive", "private", "secret", "password", "token"]
        filepath_str = str(filepath).lower()
        
        # Check file extension
        if not filepath.suffix.lower() in ['.txt', '.json', '.jsonl']:
            logging.debug(f"Skipping non-text file: {filepath}")
            return False
            
        # Check for sensitive patterns
        if any(term in filepath_str for term in sensitive_terms):
            logging.warning(f"Skipping potentially sensitive file: {filepath}")
            return False
            
        # Check file size (skip very large files)
        max_size_mb = 10  # 10MB max file size
        if filepath.stat().st_size > max_size_mb * 1024 * 1024:
            logging.warning(f"File too large, skipping: {filepath}")
            return False
            
        return True
        
    def _load_and_sanitize(self, filepath: Path, exclude_patterns: List[str]) -> List[str]:
        """
        Load and sanitize text data from a file.
        
        Args:
            filepath: Path to the file to load
            exclude_patterns: List of patterns to exclude
            
        Returns:
            List of sanitized text lines
            
        Raises:
            ValueError: If the file cannot be loaded or processed
        """
        try:
            # Read file content
            if self.use_secure_loader:
                content = self.secure_handler.secure_load(filepath)
                if isinstance(content, bytes):
                    content = content.decode(self.encoding, errors='replace')
                lines = content.splitlines()
            else:
                with open(filepath, 'r', encoding=self.encoding, errors='replace') as f:
                    lines = f.readlines()
            
            # Process and filter lines
            filtered_lines = []
            for line in lines:
                try:
                    line = line.strip()
                    if line:  # Skip empty lines
                        # Check for excluded patterns
                        if not any(
                            pattern.lower() in line.lower() 
                            for pattern in exclude_patterns
                        ):
                            filtered_lines.append(line)
                except Exception as e:
                    logging.warning(f"Error processing line in {filepath}: {e}")
                    continue
                    
            return filtered_lines
            
        except Exception as e:
            error_msg = f"Failed to load file {filepath}: {str(e)}"
            logging.error(error_msg, exc_info=True)
            raise ValueError(error_msg) from e
