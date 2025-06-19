"""
Vocabulary Training Module

Handles vocabulary learning, including word lists, flashcards, and practice sessions.
"""

import random
import time
from typing import List, Dict, Optional, Tuple, Union, Any
from dataclasses import dataclass, asdict
import json
from pathlib import Path

@dataclass
class WordEntry:
    """Represents a vocabulary word with its translations and metadata."""
    word: str
    translation: str
    part_of_speech: str
    examples: List[str]
    difficulty: float = 1.0  # 1.0 (easy) to 5.0 (hard)
    times_practiced: int = 0
    last_practiced: float = 0.0
    success_rate: float = 0.0

class VocabularyTrainer:
    """Manages vocabulary learning and practice sessions."""
    
    def __init__(self, language: str = 'dutch', level: str = 'beginner'):
        """
        Initialize the vocabulary trainer.
        
        Args:
            language: Target language (default: 'dutch')
            level: Proficiency level (beginner, intermediate, advanced)
        """
        self.language = language
        self.level = level
        self.vocabulary: Dict[str, WordEntry] = {}
        self.load_vocabulary()
    
    def load_vocabulary(self, file_path: Optional[str] = None) -> None:
        """Load vocabulary from a JSON file or use default."""
        if file_path is not None and Path(file_path).exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.vocabulary = {
                    word: WordEntry(**word_data) 
                    for word, word_data in data.items()
                }
        else:
            # Default vocabulary if no file is provided
            self.vocabulary = {
                "hallo": WordEntry(
                    word="hallo",
                    translation="hello",
                    part_of_speech="interjection",
                    examples=["Hallo, hoe gaat het?", "Zeg maar hallo tegen je moeder."]
                ),
                "dankjewel": WordEntry(
                    word="dankjewel",
                    translation="thank you",
                    part_of_speech="interjection",
                    examples=["Dankjewel voor je hulp!", "Ja, dankjewel."]
                )
                # More words can be added here
            }
    
    def get_new_words(self, count: int = 5) -> List[WordEntry]:
        """Get a list of new words to learn, based on difficulty and past performance."""
        # Sort words by difficulty and success rate (easier words with lower success rate first)
        words = sorted(
            list(self.vocabulary.values()),
            key=lambda w: (w.difficulty * (1 - w.success_rate), w.times_practiced)
        )
        return words[:count]
    
    def practice(self, words: Optional[List[Union[str, 'WordEntry']]] = None, 
                 user_translations: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Practice vocabulary words and get feedback.
        
        Args:
            words: List of words or WordEntry objects to practice, or None to get new words
            user_translations: Optional list of user-provided translations
            
        Returns:
            Dictionary containing feedback and results
        """
        # Get new words if none provided
        if not words:
            words = self.get_new_words(5) or []  # Ensure we get a list
        
        # Convert WordEntry objects to strings if needed
        word_strings = [word.word if hasattr(word, 'word') else word for word in words]
        
        results = {
            'words': [],
            'score': 0,
            'feedback': []
        }
        
        correct_count = 0
        
        for i, word in enumerate(word_strings):
            if word not in self.vocabulary:
                results['feedback'].append(f"Word '{word}' not found in vocabulary.")
                continue
                
            entry = self.vocabulary[word]
            
            # Update practice stats
            entry.times_practiced += 1
            entry.last_practiced = time.time()
            
            if user_translations and i < len(user_translations):
                user_translation = user_translations[i].lower()
                is_correct = user_translation == entry.translation.lower()
                
                # Update success rate (weighted average)
                entry.success_rate = (
                    (entry.success_rate * (entry.times_practiced - 1) + (1 if is_correct else 0))
                    / entry.times_practiced
                )
                
                if is_correct:
                    correct_count += 1
                    results['feedback'].append(f"Correct! '{word}' means '{entry.translation}'.")
                else:
                    results['feedback'].append(
                        f"Not quite. '{word}' means '{entry.translation}', not '{user_translation}'."
                    )
            
            results['words'].append({
                'word': word,
                'translation': entry.translation,
                'part_of_speech': entry.part_of_speech,
                'examples': entry.examples,
                'times_practiced': entry.times_practiced,
                'success_rate': entry.success_rate
            })
        
        # Calculate overall score
        if words:
            results['score'] = correct_count / len(words)
        
        return results
    
    def add_word(self, word: str, translation: str, part_of_speech: str, 
                 examples: List[str] = None, difficulty: float = 2.5) -> bool:
        """Add a new word to the vocabulary."""
        if not examples:
            examples = []
            
        if word in self.vocabulary:
            return False
            
        self.vocabulary[word] = WordEntry(
            word=word,
            translation=translation,
            part_of_speech=part_of_speech,
            examples=examples,
            difficulty=max(1.0, min(5.0, difficulty))  # Clamp between 1.0 and 5.0
        )
        return True
    
    def get_word_info(self, word: str) -> Optional[Dict]:
        """Get detailed information about a word."""
        if word not in self.vocabulary:
            return None
            
        entry = self.vocabulary[word]
        return {
            'word': entry.word,
            'translation': entry.translation,
            'part_of_speech': entry.part_of_speech,
            'examples': entry.examples,
            'difficulty': entry.difficulty,
            'times_practiced': entry.times_practiced,
            'success_rate': entry.success_rate,
            'last_practiced': entry.last_practiced
        }
    
    def save_vocabulary(self, file_path: Union[str, Path]) -> bool:
        """Save the current vocabulary to a JSON file."""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                # Convert WordEntry objects to dictionaries
                vocab_dict = {}
                for word, entry in self.vocabulary.items():
                    entry_dict = asdict(entry)
                    # Remove the 'word' key as it's the dictionary key
                    entry_dict.pop('word', None)
                    vocab_dict[word] = entry_dict
                json.dump(vocab_dict, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"Error saving vocabulary: {e}")
            return False
