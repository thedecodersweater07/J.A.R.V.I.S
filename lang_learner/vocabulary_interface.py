"""
Vocabulary Trainer Interface

Defines the common interface for all vocabulary trainer implementations.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Sequence, Protocol, runtime_checkable, Union, TypedDict

# Type aliases for better type safety
class WordEntryDict(TypedDict, total=False):
    """Dictionary representing a word entry with its metadata."""
    word: str
    translation: str
    part_of_speech: str
    examples: List[str]
    difficulty: float
    times_practiced: int
    success_rate: float

WordInput = Union[str, WordEntryDict, Dict[str, Any]]

@runtime_checkable
class VocabularyTrainer(Protocol):
    """Protocol defining the interface for vocabulary trainers."""
    
    def set_language(self, language: str) -> None:
        """Set the language for the vocabulary trainer."""
        ...
        
    def set_level(self, level: str) -> None:
        """Set the difficulty level for the vocabulary trainer."""
        ...
    
    def practice(self, words: Optional[Sequence[WordInput]] = None, **kwargs: Any) -> Dict[str, Any]:
        """Practice a set of words."""
        ...
    
    def load_vocabulary(self, file_path: Optional[str] = None) -> None:
        """Load vocabulary from a file or default source."""
        ...
    
    def save_vocabulary(self, file_path: Optional[str] = None) -> None:
        """Save vocabulary to a file."""
        ...
    
    def add_word(self, word: str, translation: str, part_of_speech: str = "", 
                examples: Optional[List[str]] = None, difficulty: float = 1.0) -> None:
        """Add a new word to the vocabulary."""
        ...
    
    def get_word(self, word: str) -> Optional[Dict[str, Any]]:
        """Get details for a specific word."""
        ...
    
    def remove_word(self, word: str) -> bool:
        """Remove a word from the vocabulary."""
        ...
    
    def get_vocabulary(self) -> Dict[str, Dict[str, Any]]:
        """Get the entire vocabulary."""
        ...
    
    def get_words_by_difficulty(self, min_difficulty: float = 0.0, 
                              max_difficulty: float = 5.0) -> List[Dict[str, Any]]:
        """Get words filtered by difficulty."""
        ...
    
    def get_words_for_practice(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get words for a practice session."""
        ...
    
    # Context manager support
    def __enter__(self) -> 'VocabularyTrainer':
        """Enter the runtime context related to this object."""
        return self
        
    def __exit__(self, exc_type: Optional[type[BaseException]], 
                 exc_val: Optional[BaseException], 
                 exc_tb: Optional[Any]) -> None:
        """Exit the runtime context related to this object."""
        pass
