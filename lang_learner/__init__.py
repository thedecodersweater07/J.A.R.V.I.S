"""
Language Learning Module for J.A.R.V.I.S

This module provides functionality for interactive language learning,
including vocabulary training, grammar checking, and pronunciation analysis.
"""

import sys
import warnings
from pathlib import Path
from typing import Any, Type, Union, Dict, List, Optional, Sequence, TYPE_CHECKING

# Export the main classes
__all__ = [
    'LanguageCoach',
    'LessonProgress',
    'VocabularyTrainer',
    'WordEntryDict',
    'WordInput',
    'GrammarChecker',
    'PronunciationAnalyzer',
    'DummyGrammarChecker',
    'DummyPronunciationAnalyzer',
    'coach'
]

# Import the main components
try:
    from .language_coach import LanguageCoach, LessonProgress
    from .vocabulary_interface import VocabularyTrainer, WordEntryDict, WordInput
    
    # Try to import real implementations, fall back to dummies
    try:
        from .grammar import GrammarChecker
    except ImportError:
        from .dummy_grammar import DummyGrammarChecker as GrammarChecker
        warnings.warn("Using DummyGrammarChecker - real grammar module not available")
    
    try:
        from .pronunciation import PronunciationAnalyzer
    except ImportError:
        from .dummy_pronunciation import DummyPronunciationAnalyzer as PronunciationAnalyzer
        warnings.warn("Using DummyPronunciationAnalyzer - real pronunciation module not available")
    
    # Import dummy versions for direct access
    from .dummy_grammar import DummyGrammarChecker
    from .dummy_pronunciation import DummyPronunciationAnalyzer
    
    # Initialize the global coach instance
    coach = LanguageCoach()
    
except ImportError as e:
    warnings.warn(f"Failed to import main language learning components: {e}")
    
    # Create dummy implementations if imports fail
    class DummyLanguageCoach:
        """Dummy LanguageCoach for when imports fail."""
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.initialized = False
            
        def __getattr__(self, name: str):
            raise RuntimeError("Language learning components failed to initialize. Check your dependencies.")
    
    # Set up dummy classes for type checking
    class DummyGrammarChecker: pass
    class DummyPronunciationAnalyzer: pass
    class VocabularyTrainer: pass
    class GrammarChecker: pass
    class PronunciationAnalyzer: pass
    class WordEntryDict(dict): pass
    
    # Define type aliases
    WordInput = Union[str, Dict[str, Any]]
    
    # Set the default exports
    LanguageCoach = DummyLanguageCoach
    LessonProgress = object
    
    # Initialize the global coach instance
    coach = LanguageCoach()

# Clean up the namespace
del sys, warnings, Path, Any, Type, Union, Dict, List, Optional, Sequence, TYPE_CHECKING

try:
    del RealLanguageCoach, RealLessonProgress
except NameError:
    pass
