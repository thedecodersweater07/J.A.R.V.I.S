"""
Language Coach Module - Optimized Version

High-level interface for language learning functionality.
"""

from __future__ import annotations

import json
import os
import sys
import logging
from pathlib import Path
from datetime import datetime

# Import types and protocols
from .types import (
    VocabularyTrainer,
    WordEntryDict,
    WordInput,
    WordList,
    PracticeWords,
    WordEntryType,
    PracticeResult
)

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Type variable for covariant containers
T = TypeVar('T')

# Type aliases
# WordList = Sequence[Union[str, Dict[str, Any]]]
# PracticeWords = Optional[WordList]
# WordEntryType = Dict[str, Any]

# Type for practice results
# class PracticeResult(TypedDict):
#     practiced_words: List[WordEntryDict]
#     correct: List[str]
#     incorrect: List[str]
#     score: float

@dataclass
class LessonProgress:
    """Tracks progress for a single lesson."""
    lesson_id: str
    vocabulary_learned: int = 0
    grammar_checked: int = 0
    pronunciation_practiced: int = 0
    is_complete: bool = False
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def mark_complete(self) -> None:
        self.is_complete = True

# Dummy implementation for when the real one is not available
class DummyVocabularyTrainer:
    """Dummy implementation of VocabularyTrainer for testing and fallback."""
    def __init__(self, language: str = "english", level: str = "beginner"):
        self.language = language
        self.level = level
        self.vocabulary: Dict[str, Dict[str, Any]] = {}
        
    def set_language(self, language: str) -> None:
        self.language = language
        
    def set_level(self, level: str) -> None:
        self.level = level
        
    def practice(self, words: Optional[Sequence[WordInput]] = None, **kwargs: Any) -> Dict[str, Any]:
        """Dummy practice implementation."""
        if words is None:
            words = []
            
        practiced_words = [
            word if isinstance(word, dict) else {"word": word, "translation": f"translation_of_{word}"}
            for word in words[:5]  # Limit to first 5 words for demo
        ]
        
        # Simulate some correct/incorrect answers
        correct = [word["word"] for word in practiced_words[:len(practiced_words)//2]]
        incorrect = [word["word"] for word in practiced_words[len(practiced_words)//2:]]
        
        return {
            "practiced_words": practiced_words,
            "correct": correct,
            "incorrect": incorrect,
            "score": len(correct) / max(1, len(practiced_words))
        }
        
    def load_vocabulary(self, file_path: Optional[str] = None) -> None:
        """Dummy load implementation."""
        self.vocabulary = {
            "hello": {"word": "hello", "translation": "hola", "part_of_speech": "interjection", "difficulty": 1.0},
            "goodbye": {"word": "goodbye", "translation": "adiós", "part_of_speech": "noun", "difficulty": 1.0},
        }
        
    def save_vocabulary(self, file_path: Optional[str] = None) -> None:
        """Dummy save implementation."""
        pass
        
    def add_word(self, word: str, translation: str, part_of_speech: str = "", 
                 examples: Optional[List[str]] = None, difficulty: float = 1.0) -> None:
        """Dummy add_word implementation."""
        if examples is None:
            examples = []
        self.vocabulary[word] = {
            "word": word,
            "translation": translation,
            "part_of_speech": part_of_speech,
            "examples": examples,
            "difficulty": difficulty,
            "times_practiced": 0,
            "success_rate": 0.0
        }
        
    def get_word(self, word: str) -> Optional[Dict[str, Any]]:
        """Dummy get_word implementation."""
        return self.vocabulary.get(word)
        
    def remove_word(self, word: str) -> bool:
        """Dummy remove_word implementation."""
        if word in self.vocabulary:
            del self.vocabulary[word]
            return True
        return False
        
    def get_vocabulary(self) -> Dict[str, Dict[str, Any]]:
        """Dummy get_vocabulary implementation."""
        return self.vocabulary
        
    def get_words_by_difficulty(self, min_difficulty: float = 0.0, 
                               max_difficulty: float = 5.0) -> List[Dict[str, Any]]:
        """Dummy get_words_by_difficulty implementation."""
        return [
            word for word in self.vocabulary.values()
            if min_difficulty <= word.get("difficulty", 0.0) <= max_difficulty
        ]
        
    def get_words_for_practice(self, count: int = 10) -> List[Dict[str, Any]]:
        """Dummy get_words_for_practice implementation."""
        words = list(self.vocabulary.values())
        return words[:min(count, len(words))]
        
    # Context manager support
    def __enter__(self) -> 'DummyVocabularyTrainer':
        return self
        if not word_list:
            word_list = list(self.get_words_for_practice(5))
        
        if not word_list:
            word_list = [
                {"word": "hello"},
                {"word": "world"},
                {"word": "example"}
            ]
        
        # Convert to list of strings for the response
        word_strings = [str(w.get("word", "")) for w in word_list]
        
        # Update practice stats
        for word in word_list:
            word_str = str(word.get("word", ""))
            if word_str in self.vocabulary:
                self.vocabulary[word_str]["times_practiced"] = self.vocabulary[word_str].get("times_practiced", 0) + 1
        
        return {
            "practiced_words": word_list,
            "correct": word_strings,
            "incorrect": [],
            "score": 1.0
        }

class DummyGrammarChecker:
    def __init__(self, language: str = "english", level: str = "beginner"):
        self.language = language
        self.level = level
        
    def check(self, text: str, **kwargs) -> Dict[str, Any]:
        return {
            "original_text": text,
            "is_correct": len(text.split()) < 10,
            "errors": [] if len(text.split()) < 10 else [{"type": "length", "message": "Text is quite long"}],
            "corrected_text": text,
            "suggestions": ["Great job!" if len(text.split()) < 10 else "Consider shorter sentences"]
        }

class DummyPronunciationAnalyzer:
    def __init__(self, language: str = "english"):
        self.language = language
        self.level = "beginner"
        
    def set_language(self, language: str) -> None:
        self.language = language
        
    def set_level(self, level: str) -> None:
        self.level = level
        
    def analyze_pronunciation(self, audio_data: Optional[bytes] = None, **kwargs) -> Dict[str, Any]:
        return {
            "success": True,
            "score": 0.8,
            "feedback": ["Good pronunciation practice!"],
            "suggested_practice": ["Keep practicing regularly"]
        }

class DummyProgressTracker:
    def __init__(self) -> None:
        self.stats = {"vocabulary": 0, "grammar": 0, "pronunciation": 0}
        self.level = "beginner"
    
    def set_level(self, level: str) -> None:
        self.level = level
    
    def update_vocabulary(self, 
                         words: Optional[Sequence[Union[str, Dict[str, Any]]]] = None, 
                         results: Optional[Dict[str, Any]] = None, 
                         lesson_id: Optional[str] = None) -> None:
        self.stats["vocabulary"] += 1
    
    def update_grammar(self, 
                       text: str, 
                       results: Optional[Dict[str, Any]] = None, 
                       lesson_id: Optional[str] = None) -> None:
        self.stats["grammar"] += 1
    
    def update_pronunciation(self, 
                             text: str, 
                             results: Optional[Dict[str, Any]] = None, 
                             lesson_id: Optional[str] = None) -> None:
        self.stats["pronunciation"] += 1
    
    def get_overall_progress(self) -> Dict[str, Any]:
        return {
            "vocabulary_learned": self.stats["vocabulary"],
            "grammar_checked": self.stats["grammar"], 
            "pronunciation_practiced": self.stats["pronunciation"],
            "overall_score": sum(self.stats.values()) / 10.0
        }

# Try to import real implementations
try:
    from lang_learner.vocabulary import VocabularyTrainer
except ImportError:
    VocabularyTrainer = DummyVocabularyTrainer
    logging.warning("Using dummy VocabularyTrainer")

try:
    from lang_learner.grammar import GrammarChecker
except ImportError:
    GrammarChecker = DummyGrammarChecker
    logging.warning("Using dummy GrammarChecker")

try:
    from lang_learner.pronunciation import PronunciationAnalyzer
except ImportError:
    PronunciationAnalyzer = DummyPronunciationAnalyzer
    logging.warning("Using dummy PronunciationAnalyzer")

try:
    from lang_learner.utils.progress_tracker import ProgressTracker
except ImportError:
    ProgressTracker = DummyProgressTracker
    logging.warning("Using dummy ProgressTracker")

class LanguageCoach:
    """Main class for coordinating language learning functionality."""
    
    def __init__(self, language: str = "english", level: str = "beginner", data_dir: Optional[Union[str, Path]] = None):
        """Initialize the language coach."""
        self.language = language
        self.level = level
        self.data_dir = Path(data_dir) if data_dir else Path.cwd() / "language_data"
        
        # Ensure data directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components with proper error handling
        try:
            from .vocabulary import VocabularyTrainer as RealVocabularyTrainer
            self.vocab_trainer: VocabularyTrainer = RealVocabularyTrainer(language=language, level=level)
        except (ImportError, AttributeError) as e:
            print(f"Using dummy vocabulary trainer: {e}")
            self.vocab_trainer = DummyVocabularyTrainer(language=language, level=level)
        
        try:
            from .grammar import GrammarChecker as RealGrammarChecker
            self.grammar_checker = RealGrammarChecker(language=language, level=level)
        except (ImportError, AttributeError) as e:
            print(f"Using dummy grammar checker: {e}")
            from .dummy_grammar import DummyGrammarChecker
            self.grammar_checker = DummyGrammarChecker(language=language, level=level)
        
        try:
            from .pronunciation import PronunciationAnalyzer as RealPronunciationAnalyzer
            self.pronunciation_analyzer = RealPronunciationAnalyzer(language=language)
        except (ImportError, AttributeError) as e:
            print(f"Using dummy pronunciation analyzer: {e}")
            from .dummy_pronunciation import DummyPronunciationAnalyzer
            self.pronunciation_analyzer = DummyPronunciationAnalyzer(language=language)
        
        self.progress_tracker = DummyProgressTracker()
        
        # Load initial vocabulary
        self.vocab_trainer.load_vocabulary()
        
        # Lesson management
        self.current_lesson: Optional[str] = None
        self.lesson_progress: Dict[str, LessonProgress] = {}
    
    def start_lesson(self, topic: str) -> Dict[str, Any]:
        """Start a new lesson on the given topic."""
        try:
            self.current_lesson = topic
            if topic not in self.lesson_progress:
                self.lesson_progress[topic] = LessonProgress(lesson_id=topic)
                
            return {
                "success": True,
                "lesson": topic,
                "message": f"Started lesson: {topic}",
                "progress": self.lesson_progress[topic].to_dict()
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to start lesson"
            }
    
    def practice_vocabulary(self, words: Optional[Sequence[Union[str, Dict[str, Any]]]] = None) -> Dict[str, Any]:
        """Practice vocabulary words."""
        try:
            # Convert words to the format expected by the trainer
            practice_words: List[Union[str, Dict[str, Any]]] = []
            if words:
                # Ensure we have a list of either strings or dicts
                for word in words:
                    if isinstance(word, (str, dict)):
                        practice_words.append(word)
            
            # Pass the words to the trainer
            results = self.vocab_trainer.practice(practice_words if practice_words else None)
            
            # Update progress with the results
            if self.current_lesson:
                self.progress_tracker.update_vocabulary(
                    words=practice_words,
                    results=results,
                    lesson_id=self.current_lesson
                )
                self.lesson_progress[self.current_lesson].vocabulary_learned += 1
            
            return results
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def check_grammar(self, text: str) -> Dict[str, Any]:
        """Check grammar of the given text."""
        try:
            results = self.grammar_checker.check(text)
            
            # Update progress
            if self.current_lesson:
                self.progress_tracker.update_grammar(
                    text=text,
                    results=results,
                    lesson_id=self.current_lesson
                )
                self.lesson_progress[self.current_lesson].grammar_checked += 1
            
            return results
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def practice_pronunciation(self, audio_data: Optional[bytes] = None, 
                             audio_path: Optional[str] = None,
                             text: Optional[str] = None) -> Dict[str, Any]:
        """Practice pronunciation with audio input."""
        try:
            results = self.pronunciation_analyzer.analyze_pronunciation(
                audio_data=audio_data,
                audio_path=audio_path,
                reference_text=text
            )
            
            # Update progress
            if self.current_lesson:
                self.progress_tracker.update_pronunciation(
                    text=text or "",
                    results=results,
                    lesson_id=self.current_lesson
                )
                self.lesson_progress[self.current_lesson].pronunciation_practiced += 1
            
            return results
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_progress(self) -> Dict[str, Any]:
        """Get overall learning progress."""
        try:
            progress_data = {
                "language": self.language,
                "level": self.level,
                "current_lesson": self.current_lesson,
                "total_lessons": len(self.lesson_progress),
                "lessons_completed": sum(1 for lesson in self.lesson_progress.values() 
                                       if lesson.is_complete)
            }
            
            # Add tracker progress
            tracker_progress = self.progress_tracker.get_overall_progress()
            progress_data.update(tracker_progress)
            
            # Add lesson details
            progress_data["lessons"] = {
                lesson_id: lesson.to_dict() 
                for lesson_id, lesson in self.lesson_progress.items()
            }
            
            return progress_data
        except Exception as e:
            return {"error": str(e)}
    
    def complete_lesson(self, lesson_id: Optional[str] = None) -> Dict[str, Any]:
        """Mark a lesson as complete."""
        target_lesson = lesson_id or self.current_lesson
        
        if not target_lesson or target_lesson not in self.lesson_progress:
            return {"success": False, "error": "Lesson not found"}
        
        self.lesson_progress[target_lesson].mark_complete()
        
        return {
            "success": True,
            "lesson": target_lesson,
            "message": f"Lesson '{target_lesson}' completed!",
            "progress": self.lesson_progress[target_lesson].to_dict()
        }
    
    def generate_lesson_plan(self, topics: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate a personalized lesson plan."""
        if not topics:
            topics = self._get_default_topics()
        
        return {
            "language": self.language,
            "level": self.level,
            "topics": topics,
            "estimated_time": f"{len(topics) * 30} minutes",
            "lessons": len(topics),
            "recommended_schedule": "Daily practice recommended",
            "created_at": datetime.now().isoformat()
        }
    
    def _get_default_topics(self) -> List[str]:
        """Get default topics based on language and level."""
        topic_map = {
            "beginner": ["greetings", "introductions", "numbers", "colors", "family"],
            "intermediate": ["shopping", "directions", "hobbies", "food", "travel"],
            "advanced": ["business", "politics", "culture", "literature", "science"]
        }
        return topic_map.get(self.level, topic_map["beginner"])
    
    def set_language(self, language: str) -> None:
        """Set the current language."""
        self.language = language
        # These methods might not exist on all implementations
        if hasattr(self.pronunciation_analyzer, 'set_language'):
            self.pronunciation_analyzer.set_language(language)
        if hasattr(self.grammar_checker, 'set_language'):
            self.grammar_checker.set_language(language)
        if hasattr(self.vocab_trainer, 'set_language'):
            self.vocab_trainer.set_language(language)
    
    def set_level(self, level: str) -> None:
        """Change the proficiency level."""
        self.level = level
        self.grammar_checker = GrammarChecker(language=self.language, level=level)
        self.vocab_trainer.set_level(level)
        self.progress_tracker.set_level(level)
