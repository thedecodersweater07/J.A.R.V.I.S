"""
Progress tracking for language learning.

This module provides functionality to track and manage user progress
in vocabulary, grammar, and pronunciation.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union


class ProgressTracker:
    """Tracks and manages user progress in language learning."""
    
    def __init__(self, user_id: str = 'default', data_dir: Optional[Union[str, Path]] = 'data/progress'):
        """
        Initialize the progress tracker.
        
        Args:
            user_id: Unique identifier for the user
            data_dir: Directory to store progress data
        """
        self.user_id = user_id
        self.data_dir = Path(data_dir) if data_dir is not None else None
        self.vocabulary_progress: Dict[str, Dict[str, Any]] = {}
        self.grammar_progress: Dict[str, Dict[str, Any]] = {}
        self.pronunciation_progress: Dict[str, Dict[str, Any]] = {}
        
        # Create data directory if it doesn't exist
        if self.data_dir is not None and not self.data_dir.exists():
            self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing progress if available
        self.load_progress()
    
    def update_vocabulary(self, word: str, correct: bool, timestamp: Optional[float] = None) -> None:
        """
        Update vocabulary progress for a word.
        
        Args:
            word: The word that was practiced
            correct: Whether the word was answered correctly
            timestamp: Optional timestamp of the practice session
        """
        if timestamp is None:
            timestamp = datetime.now().timestamp()
            
        if word not in self.vocabulary_progress:
            self.vocabulary_progress[word] = {
                'correct': 0,
                'incorrect': 0,
                'last_practiced': timestamp,
                'first_practiced': timestamp
            }
        else:
            self.vocabulary_progress[word]['last_practiced'] = timestamp
        
        if correct:
            self.vocabulary_progress[word]['correct'] += 1
        else:
            self.vocabulary_progress[word]['incorrect'] += 1
        
        self.save_progress()
    
    def update_grammar(self, rule: str, correct: bool, example: str, 
                     timestamp: Optional[float] = None) -> None:
        """
        Update grammar progress for a grammar rule.
        
        Args:
            rule: The grammar rule that was practiced
            correct: Whether the rule was used correctly
            example: Example sentence
            timestamp: Optional timestamp of the practice session
        """
        if timestamp is None:
            timestamp = datetime.now().timestamp()
            
        if rule not in self.grammar_progress:
            self.grammar_progress[rule] = {
                'correct': 0,
                'incorrect': 0,
                'examples': [],
                'last_practiced': timestamp,
                'first_practiced': timestamp
            }
        else:
            self.grammar_progress[rule]['last_practiced'] = timestamp
        
        if correct:
            self.grammar_progress[rule]['correct'] += 1
        else:
            self.grammar_progress[rule]['incorrect'] += 1
            
        # Add example if not already present
        if example and example not in self.grammar_progress[rule]['examples']:
            self.grammar_progress[rule]['examples'].append(example)
        
        self.save_progress()
    
    def update_pronunciation(self, word: str, score: float, feedback: List[str],
                           audio_file: Optional[str] = None, 
                           timestamp: Optional[float] = None) -> None:
        """
        Update pronunciation progress for a word or phrase.
        
        Args:
            word: The word or phrase that was practiced
            score: Pronunciation score (0.0 to 1.0)
            feedback: List of feedback messages
            audio_file: Path to the recorded audio file (optional)
            timestamp: Optional timestamp of the practice session
        """
        if timestamp is None:
            timestamp = datetime.now().timestamp()
            
        if word not in self.pronunciation_progress:
            self.pronunciation_progress[word] = {
                'scores': [],
                'average_score': 0.0,
                'attempts': 0,
                'feedback': [],
                'audio_files': [],
                'last_practiced': timestamp,
                'first_practiced': timestamp
            }
        else:
            self.pronunciation_progress[word]['last_practiced'] = timestamp
        
        # Update scores and feedback
        self.pronunciation_progress[word]['scores'].append(score)
        self.pronunciation_progress[word]['attempts'] += 1
        self.pronunciation_progress[word]['average_score'] = (
            sum(self.pronunciation_progress[word]['scores']) / 
            len(self.pronunciation_progress[word]['scores'])
        )
        
        # Add new feedback
        self.pronunciation_progress[word]['feedback'].extend(feedback)
        
        # Add audio file path if provided
        if audio_file and audio_file not in self.pronunciation_progress[word]['audio_files']:
            self.pronunciation_progress[word]['audio_files'].append(audio_file)
        
        self.save_progress()
    
    def get_overall_progress(self) -> Dict[str, float]:
        """
        Get overall progress across all categories.
        
        Returns:
            Dictionary with progress percentages for each category
        """
        def calculate_average(scores: List[float]) -> float:
            return sum(scores) / len(scores) if scores else 0.0
        
        # Calculate vocabulary progress (percentage of words with good accuracy)
        vocab_scores = []
        for word, data in self.vocabulary_progress.items():
            total = data['correct'] + data['incorrect']
            if total > 0:
                vocab_scores.append(data['correct'] / total)
        
        # Calculate grammar progress
        grammar_scores = []
        for rule, data in self.grammar_progress.items():
            total = data['correct'] + data['incorrect']
            if total > 0:
                grammar_scores.append(data['correct'] / total)
        
        # Calculate pronunciation progress
        pronunciation_scores = [
            data['average_score'] 
            for data in self.pronunciation_progress.values()
        ]
        
        return {
            'vocabulary': calculate_average(vocab_scores) * 100,
            'grammar': calculate_average(grammar_scores) * 100,
            'pronunciation': calculate_average(pronunciation_scores) * 100
        }
    
    def save_progress(self) -> None:
        """Save progress to a JSON file in the data directory."""
        if self.data_dir is None:
            return
            
        progress_data = {
            'user_id': self.user_id,
            'vocabulary': self.vocabulary_progress,
            'grammar': self.grammar_progress,
            'pronunciation': self.pronunciation_progress,
            'last_updated': datetime.now().isoformat()
        }
        
        progress_file = self.data_dir / f"{self.user_id}_progress.json"
        with open(str(progress_file), 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, indent=2, ensure_ascii=False)
    
    def load_progress(self) -> None:
        """Load progress from a JSON file in the data directory."""
        if self.data_dir is None:
            return
            
        progress_file = self.data_dir / f"{self.user_id}_progress.json"
        if not progress_file.exists():
            return
            
        try:
            with open(str(progress_file), 'r', encoding='utf-8') as f:
                progress_data = json.load(f)
            
            self.vocabulary_progress = progress_data.get('vocabulary', {})
            self.grammar_progress = progress_data.get('grammar', {})
            self.pronunciation_progress = progress_data.get('pronunciation', {})
            
        except (json.JSONDecodeError, FileNotFoundError):
            # If there's an error loading the file, start with a fresh progress
            self.vocabulary_progress = {}
            self.grammar_progress = {}
            self.pronunciation_progress = {}
    
    def reset_progress(self) -> None:
        """Reset all progress data."""
        self.vocabulary_progress = {}
        self.grammar_progress = {}
        self.pronunciation_progress = {}
        self.save_progress()
    
    def get_vocabulary_stats(self) -> Dict[str, Any]:
        """Get vocabulary statistics."""
        total_words = len(self.vocabulary_progress)
        known_words = sum(
            1 for data in self.vocabulary_progress.values()
            if data.get('correct', 0) > data.get('incorrect', 0)
        )
        
        return {
            'total_words': total_words,
            'known_words': known_words,
            'mastery_percentage': (known_words / total_words * 100) if total_words > 0 else 0.0,
            'recently_practiced': sorted(
                self.vocabulary_progress.items(),
                key=lambda x: x[1].get('last_practiced', 0),
                reverse=True
            )[:5]
        }
    
    def get_grammar_stats(self) -> Dict[str, Any]:
        """Get grammar statistics."""
        total_rules = len(self.grammar_progress)
        mastered_rules = sum(
            1 for data in self.grammar_progress.values()
            if data.get('correct', 0) >= 3 and data.get('correct', 0) > data.get('incorrect', 0)
        )
        
        return {
            'total_rules': total_rules,
            'mastered_rules': mastered_rules,
            'mastery_percentage': (mastered_rules / total_rules * 100) if total_rules > 0 else 0.0,
            'needs_practice': [
                rule for rule, data in self.grammar_progress.items()
                if data.get('incorrect', 0) > data.get('correct', 0)
            ][:5]
        }
    
    def get_pronunciation_stats(self) -> Dict[str, Any]:
        """Get pronunciation statistics."""
        total_words = len(self.pronunciation_progress)
        if total_words == 0:
            return {
                'total_words': 0,
                'average_score': 0.0,
                'well_pronounced': [],
                'needs_work': []
            }
        
        # Calculate average score across all words
        total_score = sum(data['average_score'] for data in self.pronunciation_progress.values())
        average_score = total_score / total_words
        
        # Categorize words
        well_pronounced = [
            word for word, data in self.pronunciation_progress.items()
            if data['average_score'] >= 0.8
        ]
        
        needs_work = [
            word for word, data in self.pronunciation_progress.items()
            if data['average_score'] < 0.5
        ]
        
        return {
            'total_words': total_words,
            'average_score': average_score * 100,
            'well_pronounced': well_pronounced[:5],
            'needs_work': needs_work[:5]
        }
