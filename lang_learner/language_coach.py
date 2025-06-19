"""
Language Coach Module

This module provides a high-level interface for language learning functionality,
coordinating between vocabulary, grammar, and pronunciation components.
"""

import os
import random
import sys
import time
from pathlib import Path

# Add project root to path for absolute imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import Dict, List, Optional, Any, Union, TYPE_CHECKING, Tuple, Type, TypeVar, cast, Protocol, runtime_checkable
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum, auto

# Type variable for generic type hints
T = TypeVar('T')

# Protocol definitions for type checking
class VocabularyTrainerProtocol(Protocol):
    def practice(self, words: Optional[List[Union[str, Dict[str, Any]]]] = None, **kwargs: Any) -> Dict[str, Any]: ...

class GrammarCheckerProtocol(Protocol):
    def check_grammar(self, text: str, **kwargs: Any) -> Dict[str, Any]: ...

class PronunciationAnalyzerProtocol(Protocol):
    def analyze_pronunciation(self, audio_data: Optional[bytes] = None, **kwargs: Any) -> Dict[str, Any]: ...

class LanguageModelProtocol(Protocol):
    def generate_response(self, prompt: str, **kwargs: Any) -> str: ...

class ProgressTrackerProtocol(Protocol):
    def update_vocabulary(self, *args: Any, **kwargs: Any) -> None: ...
    def update_grammar(self, *args: Any, **kwargs: Any) -> None: ...
    def update_pronunciation(self, *args: Any, **kwargs: Any) -> None: ...
    def get_overall_progress(self) -> Dict[str, Any]: ...

# Dummy implementations with proper type hints
class DummyVocabularyTrainer:
    def practice(self, words: Optional[List[Union[str, Dict[str, Any]]]] = None, **kwargs: Any) -> Dict[str, Any]:
        return {"error": "Vocabulary trainer not available"}

class DummyGrammarChecker:
    def check_grammar(self, text: str, **kwargs: Any) -> Dict[str, Any]:
        return {"error": "Grammar checker not available"}

class DummyPronunciationAnalyzer:
    def analyze_pronunciation(self, audio_data: Optional[bytes] = None, **kwargs: Any) -> Dict[str, Any]:
        return {"error": "Pronunciation analyzer not available"}

class DummyLanguageModel:
    def generate_response(self, prompt: str, **kwargs: Any) -> str:
        return "Language model not available"

class DummyProgressTracker:
    def update_vocabulary(self, *args: Any, **kwargs: Any) -> None:
        pass
    def update_grammar(self, *args: Any, **kwargs: Any) -> None:
        pass
    def update_pronunciation(self, *args: Any, **kwargs: Any) -> None:
        pass
    def get_overall_progress(self) -> Dict[str, Any]:
        return {"error": "Progress tracking not available"}

# Type aliases for the actual implementations
VocabularyTrainerType = Union[Type[VocabularyTrainerProtocol], Type[DummyVocabularyTrainer]]
GrammarCheckerType = Union[Type[GrammarCheckerProtocol], Type[DummyGrammarChecker]]
PronunciationAnalyzerType = Union[Type[PronunciationAnalyzerProtocol], Type[DummyPronunciationAnalyzer]]
LanguageModelType = Union[Type[LanguageModelProtocol], Type[DummyLanguageModel]]
ProgressTrackerType = Union[Type[ProgressTrackerProtocol], Type[DummyProgressTracker]]

# Initialize module-level variables with dummy implementations
_vocab_trainer: VocabularyTrainerType = DummyVocabularyTrainer
_grammar_checker: GrammarCheckerType = DummyGrammarChecker
_pronunciation_analyzer: PronunciationAnalyzerType = DummyPronunciationAnalyzer
_language_model: LanguageModelType = DummyLanguageModel
_progress_tracker: ProgressTrackerType = DummyProgressTracker

# Try to import real implementations if available
try:
    from lang_learner.vocabulary import VocabularyTrainer as RealVocabularyTrainer  # type: ignore
    _vocab_trainer = RealVocabularyTrainer
except (ImportError, ModuleNotFoundError) as e:
    logging.warning(f"Could not import VocabularyTrainer: {e}")

try:
    from lang_learner.grammar import GrammarChecker as RealGrammarChecker  # type: ignore
    _grammar_checker = RealGrammarChecker
except (ImportError, ModuleNotFoundError) as e:
    logging.warning(f"Could not import GrammarChecker: {e}")

try:
    from lang_learner.pronunciation import PronunciationAnalyzer as RealPronunciationAnalyzer  # type: ignore
    _pronunciation_analyzer = RealPronunciationAnalyzer
except (ImportError, ModuleNotFoundError) as e:
    logging.warning(f"Could not import PronunciationAnalyzer: {e}")

try:
    from lang_learner.models.language_model import LanguageModel as RealLanguageModel  # type: ignore
    _language_model = RealLanguageModel
except (ImportError, ModuleNotFoundError) as e:
    logging.warning(f"Could not import LanguageModel: {e}")

try:
    from lang_learner.utils.progress_tracker import ProgressTracker as RealProgressTracker  # type: ignore
    _progress_tracker = RealProgressTracker
except (ImportError, ModuleNotFoundError) as e:
    logging.warning(f"Could not import ProgressTracker: {e}")

# Export the actual implementations to be used throughout the module
VocabularyTrainer: Type[VocabularyTrainerProtocol] = _vocab_trainer  # type: ignore
GrammarChecker: Type[GrammarCheckerProtocol] = _grammar_checker  # type: ignore
PronunciationAnalyzer: Type[PronunciationAnalyzerProtocol] = _pronunciation_analyzer  # type: ignore
LanguageModel: Type[LanguageModelProtocol] = _language_model  # type: ignore
ProgressTracker: Type[ProgressTrackerProtocol] = _progress_tracker  # type: ignore

# Add project root to path for absolute imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

@dataclass
class LessonProgress:
    """Tracks progress for a single lesson."""
    lesson_id: str
    vocabulary_learned: int = 0
    grammar_checked: int = 0
    pronunciation_practiced: int = 0
    is_complete: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert lesson progress to dictionary."""
        return asdict(self)
    
    def mark_complete(self) -> None:
        """Mark the lesson as complete."""
        self.is_complete = True

class _DummyVocabularyTrainer:
    def practice(self, words: Optional[List[Union[str, Dict[str, Any]]]] = None, **kwargs: Any) -> Dict[str, Any]:
        return {"error": "Vocabulary trainer not available"}

class _DummyGrammarChecker:
    def check_grammar(self, text: str, **kwargs: Any) -> Dict[str, Any]:
        return {"error": "Grammar checker not available"}

class _DummyPronunciationAnalyzer:
    def analyze_pronunciation(self, audio_data: Optional[bytes] = None, **kwargs: Any) -> Dict[str, Any]:
        return {"error": "Pronunciation analyzer not available"}

class _DummyLanguageModel:
    def generate_response(self, prompt: str, **kwargs: Any) -> str:
        return "Language model not available"

class _DummyProgressTracker:
    def update_vocabulary(self, *args: Any, **kwargs: Any) -> None:
        pass
    def update_grammar(self, *args: Any, **kwargs: Any) -> None:
        pass
    def update_pronunciation(self, *args: Any, **kwargs: Any) -> None:
        pass
    def get_overall_progress(self) -> Dict[str, Any]:
        return {
            "vocabulary_learned": 0,
            "grammar_checked": 0,
            "pronunciation_practiced": 0,
            "overall_score": 0.0
        }

# Initialize with dummy implementations
_vocab_trainer = _DummyVocabularyTrainer
_grammar_checker = _DummyGrammarChecker
_pronunciation_analyzer = _DummyPronunciationAnalyzer
_language_model = _DummyLanguageModel
_progress_tracker = _DummyProgressTracker

# Try to import real implementations if available
try:
    from lang_learner.vocabulary import VocabularyTrainer as RealVocabularyTrainer  # type: ignore
    _vocab_trainer = RealVocabularyTrainer
except (ImportError, ModuleNotFoundError) as e:
    logging.warning(f"Could not import VocabularyTrainer: {e}")

try:
    from lang_learner.grammar import GrammarChecker as RealGrammarChecker  # type: ignore
    _grammar_checker = RealGrammarChecker
except (ImportError, ModuleNotFoundError) as e:
    logging.warning(f"Could not import GrammarChecker: {e}")

try:
    from lang_learner.pronunciation import PronunciationAnalyzer as RealPronunciationAnalyzer  # type: ignore
    _pronunciation_analyzer = RealPronunciationAnalyzer
except (ImportError, ModuleNotFoundError) as e:
    logging.warning(f"Could not import PronunciationAnalyzer: {e}")

try:
    from lang_learner.models.language_model import LanguageModel as RealLanguageModel  # type: ignore
    _language_model = RealLanguageModel
except (ImportError, ModuleNotFoundError) as e:
    logging.warning(f"Could not import LanguageModel: {e}")

try:
    from lang_learner.utils.progress_tracker import ProgressTracker as RealProgressTracker  # type: ignore
    _progress_tracker = RealProgressTracker
except (ImportError, ModuleNotFoundError) as e:
    logging.warning(f"Could not import ProgressTracker: {e}")

# Export the actual implementations to be used throughout the module
VocabularyTrainer: Type[VocabularyTrainerProtocol] = _vocab_trainer  # type: ignore
GrammarChecker: Type[GrammarCheckerProtocol] = _grammar_checker  # type: ignore
PronunciationAnalyzer: Type[PronunciationAnalyzerProtocol] = _pronunciation_analyzer  # type: ignore
LanguageModel: Type[LanguageModelProtocol] = _language_model  # type: ignore
ProgressTracker: Type[ProgressTrackerProtocol] = _progress_tracker  # type: ignore

class LanguageCoach:
    """Main class for language learning functionality."""
    
    def __init__(
        self,
        vocab_trainer: Optional[Any] = None,
        grammar_checker: Optional[Any] = None,
        pron_analyzer: Optional[Any] = None,
        language_model: Optional[Any] = None,
        progress_tracker: Optional[Any] = None,
        data_dir: Union[str, Path, None] = None
    ) -> None:
        """
        Initialize the language coach with optional components.
        
        Args:
            vocab_trainer: Vocabulary trainer instance
            grammar_checker: Grammar checker instance
            pron_analyzer: Pronunciation analyzer instance
            language_model: Language model instance
            progress_tracker: Progress tracker instance
            data_dir: Directory to store data files
        """
        # Initialize components with provided instances or create defaults
        self.vocab_trainer = vocab_trainer or VocabularyTrainer()
        self.grammar_checker = grammar_checker or GrammarChecker()
        self.pron_analyzer = pron_analyzer or PronunciationAnalyzer()
        self.language_model = language_model or LanguageModel()
        self.progress_tracker = progress_tracker or ProgressTracker()
        
        # Initialize language and level
        self.language: str = "english"  # Default language
        self.level: str = "beginner"    # Default level
        
        # Current lesson state
        self.current_lesson: Optional[str] = None
        self.lesson_progress: Dict[str, LessonProgress] = {}
        
        # Set up data directory
        self.data_dir = Path(data_dir) if data_dir else Path.cwd() / "language_data"
    
    def start_lesson(self, topic: str) -> bool:
        """
        Start a new lesson on the given topic.
        
        Args:
            topic: The topic of the lesson to start
            
        Returns:
            bool: True if lesson was started successfully
        """
        try:
            self.current_lesson = topic
            if topic not in self.lesson_progress:
                self.lesson_progress[topic] = LessonProgress(lesson_id=topic)
            return True
        except Exception as e:
            print(f"Error starting lesson: {e}")
            return False
    
    def practice_vocabulary(self, words: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Practice vocabulary words.
        
        Args:
            words: Optional list of words to practice. If None, gets new words.
            
        Returns:
            Dictionary with practice results
        """
        try:
            # Ensure words is in the correct format for the practice method
            practice_words: Optional[List[Union[str, Dict[str, Any]]]] = None
            if words:
                practice_words = [{"word": word} if isinstance(word, str) else word for word in words]
                
            # The practice method in VocabularyTrainer handles getting new words if None
            results = self.vocab_trainer.practice(practice_words)
            
            # Update progress if we have a current lesson
            if hasattr(self, 'current_lesson') and self.current_lesson:
                try:
                    self.progress_tracker.update_vocabulary(
                        words=words or [],
                        results=results,
                        lesson_id=self.current_lesson
                    )
                except Exception as e:
                    print(f"Warning: Failed to update progress: {e}")
            
            return results
        except Exception as e:
            return {"error": f"Failed to practice vocabulary: {str(e)}"}
    
    def check_grammar(self, text: str) -> Dict[str, Any]:
        """
        Check grammar of the given text.
        
        Args:
            text: Text to check
            
        Returns:
            Dictionary with grammar check results
        """
        try:
            # Check if we have a grammar checker available
            if not hasattr(self, 'grammar_checker') or not self.grammar_checker:
                return {"error": "Grammar checking not available"}
                
            # Call the grammar checker with text as a keyword argument
            results = self.grammar_checker.check_grammar(text=text)
            
            # Update progress if we have a current lesson
            if hasattr(self, 'current_lesson') and self.current_lesson:
                try:
                    self.progress_tracker.update_grammar(
                        text=text,
                        results=results,
                        lesson_id=self.current_lesson
                    )
                except Exception as e:
                    print(f"Warning: Failed to update grammar progress: {e}")
            
            return results
        except Exception as e:
            return {"error": f"Failed to check grammar: {str(e)}"}
    
    def practice_pronunciation(self, audio_data: Optional[bytes] = None, 
                             text: Optional[str] = None) -> Dict[str, Any]:
        """
        Practice pronunciation with optional audio data.
        
        Args:
            audio_data: Optional audio data in bytes
            text: Optional text that was spoken
            
        Returns:
            Dictionary with pronunciation analysis
        """
        try:
            # Check if we have a pronunciation analyzer available
            if not hasattr(self, 'pron_analyzer') or not self.pron_analyzer:
                return {"error": "Pronunciation analysis not available"}
                
            if audio_data is None:
                return {"error": "No audio data provided"}
                
            # Analyze pronunciation - only pass audio_data as the method doesn't expect text
            results = self.pron_analyzer.analyze_pronunciation(audio_data=audio_data)
            
            # Update progress if we have a current lesson
            if hasattr(self, 'current_lesson') and self.current_lesson:
                try:
                    self.progress_tracker.update_pronunciation(
                        text=text,
                        results=results,
                        lesson_id=self.current_lesson,
                        audio_data_length=len(audio_data) if audio_data else 0
                    )
                except Exception as e:
                    print(f"Warning: Failed to update pronunciation progress: {e}")
            
            return results
        except Exception as e:
            return {"error": f"Failed to analyze pronunciation: {str(e)}"}
    
    def get_progress(self) -> Dict[str, Any]:
        """
        Get overall learning progress.
        
        Returns:
            Dictionary containing progress information
        """
        try:
            # Initialize progress data with default values
            progress_data = {
                "language": getattr(self, 'language', 'english'),
                "level": getattr(self, 'level', 'beginner'),
                "lessons_completed": 0,
                "total_lessons": 0,
                "vocabulary_learned": 0,
                "grammar_checked": 0,
                "pronunciation_practiced": 0,
                "overall_score": 0.0
            }
            
            # Get progress from tracker if available
            if hasattr(self, 'progress_tracker') and self.progress_tracker:
                try:
                    progress_data.update(self.progress_tracker.get_overall_progress())
                except Exception as e:
                    print(f"Warning: Failed to get progress from tracker: {e}")
            
            # Update lesson progress if available
            if hasattr(self, 'lesson_progress') and isinstance(self.lesson_progress, dict):
                progress_data["total_lessons"] = len(self.lesson_progress)
                progress_data["lessons_completed"] = sum(
                    1 for lesson in self.lesson_progress.values() 
                    if hasattr(lesson, 'is_complete') and lesson.is_complete
                )
            
            return progress_data
            
        except Exception as e:
            return {"error": f"Failed to get progress: {str(e)}"}
    
    def generate_lesson_plan(self, topics: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate a personalized lesson plan.
        
        Args:
            topics: Optional list of topics to include in the lesson plan
            
        Returns:
            Dictionary containing the lesson plan details
        """
        if not topics:
            topics = ["basic_greetings", "common_phrases", "introductions"]
            
        return {
            "language": self.language,
            "level": self.level,
            "topics": topics,
            "estimated_study_time": f"{len(topics) * 30} minutes",
            "recommended_schedule": "Daily practice for best results"
        }
