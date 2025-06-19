"""
Language Learning Module for J.A.R.V.I.S

This module provides functionality for interactive language learning,
including vocabulary training, grammar checking, and pronunciation analysis.
"""

import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Type, TYPE_CHECKING

# Add the parent directory to the Python path to allow absolute imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Define a dummy LanguageCoach for when imports fail
class DummyLanguageCoach:
    """Dummy LanguageCoach for when imports fail."""
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.initialized = False
    
    def __getattr__(self, name: str) -> Any:
        if not self.initialized:
            raise ImportError("LanguageCoach failed to initialize. Check your imports and dependencies.")
        return super().__getattribute__(name)

# Initialize a global instance of the language coach
LanguageCoach: Type[DummyLanguageCoach]
LessonProgress: Any

# Try to import the real implementations
try:
    from lang_learner.language_coach import LanguageCoach as RealLanguageCoach, LessonProgress as RealLessonProgress  # type: ignore
    LanguageCoach = RealLanguageCoach  # type: ignore
    LessonProgress = RealLessonProgress  # type: ignore
    
    # Create a default instance for convenience
    coach = LanguageCoach()
    __all__ = ['LanguageCoach', 'LessonProgress', 'coach']
    
except (ImportError, ModuleNotFoundError) as e:
    warnings.warn(
        f"Failed to import LanguageCoach: {e}. Using dummy implementation. "
        "Some functionality may be limited.",
        RuntimeWarning,
        stacklevel=2
    )
    LanguageCoach = DummyLanguageCoach  # type: ignore
    LessonProgress = dict  # type: ignore
    coach = DummyLanguageCoach()
    __all__ = ['LanguageCoach', 'LessonProgress', 'coach']

# Clean up the namespace
try:
    del sys, warnings, Path, Any, Dict, Optional, Type, TYPE_CHECKING, DummyLanguageCoach
except NameError:
    pass

try:
    del RealLanguageCoach, RealLessonProgress
except NameError:
    pass
