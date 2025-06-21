"""
Pronunciation Analysis Module - Optimized Version

Provides functionality for analyzing and providing feedback on pronunciation.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path

# Optional imports with fallback
try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False
    
    class MockSR:
        class Recognizer:
            def record(self, source): return {}
            def recognize_google(self, *args, **kwargs): 
                raise ImportError("speech_recognition not installed")
        
        class AudioFile:
            def __init__(self, *args): pass
            def __enter__(self): return self
            def __exit__(self, *args): pass
    
    sr = MockSR()

@dataclass
class PronunciationFeedback:
    """Stores feedback about pronunciation analysis."""
    score: float  # 0.0 to 1.0
    feedback: List[str]
    phonemes: List[Dict[str, Any]]
    word_level: List[Dict[str, Any]]
    overall_impression: str
    suggested_practice: List[str]

class PronunciationAnalyzer:
    """Analyzes pronunciation and provides feedback."""
    
    def __init__(self, language: str = 'dutch'):
        self.language = language.lower()
        self.recognizer = sr.Recognizer() if SPEECH_RECOGNITION_AVAILABLE else None
        self.phoneme_map = self._load_phoneme_map()
        self.language_codes = {
            'dutch': 'nl-NL',
            'english': 'en-US',
            'german': 'de-DE',
            'french': 'fr-FR',
            'spanish': 'es-ES'
        }
    
    def _load_phoneme_map(self) -> Dict[str, List[str]]:
        """Load phoneme mappings for the target language."""
        phoneme_maps = {
            'dutch': {
                'vowels': ['a', 'aa', 'e', 'ee', 'i', 'ie', 'o', 'oo', 'u', 'uu'],
                'diphthongs': ['eu', 'ui', 'ij', 'ei'],
                'consonants': ['g', 'ch', 'r', 'ng']
            },
            'english': {
                'vowels': ['a', 'e', 'i', 'o', 'u'],
                'diphthongs': ['ai', 'au', 'oi'],
                'consonants': ['th', 'sh', 'ch', 'ng']
            }
        }
        return phoneme_maps.get(self.language, {})
    
    def analyze_pronunciation(self, audio_data: Optional[bytes] = None, 
                            audio_path: Optional[str] = None,
                            reference_text: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze pronunciation from audio data or file.
        
        Args:
            audio_data: Raw audio bytes
            audio_path: Path to audio file
            reference_text: Expected text that was spoken
            
        Returns:
            Dictionary with analysis results
        """
        if not SPEECH_RECOGNITION_AVAILABLE:
            return {
                'success': False,
                'error': 'Speech recognition not available',
                'score': 0.0
            }
        
        if audio_path and not Path(audio_path).exists():
            return {
                'success': False,
                'error': f'Audio file not found: {audio_path}',
                'score': 0.0
            }
        
        try:
            # Use audio file if provided, otherwise assume audio_data handling
            if audio_path:
                return self._analyze_from_file(audio_path, reference_text or "")
            else:
                return self._analyze_from_data(audio_data, reference_text or "")
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'score': 0.0
            }
    
    def _analyze_from_file(self, audio_path: str, reference_text: str) -> Dict[str, Any]:
        """Analyze pronunciation from audio file."""
        with sr.AudioFile(audio_path) as source:
            audio_data = self.recognizer.record(source)
            recognized_text = self.recognizer.recognize_google(
                audio_data, 
                language=self.language_codes.get(self.language, 'en-US')
            )
        
        return self._compare_texts(recognized_text, reference_text)
    
    def _analyze_from_data(self, audio_data: Optional[bytes], 
                          reference_text: str) -> Dict[str, Any]:
        """Analyze pronunciation from raw audio data."""
        # For now, return a mock analysis since direct audio data processing
        # requires additional audio processing libraries
        return {
            'success': True,
            'score': 0.7,
            'recognized_text': "Mock analysis - audio data processing not implemented",
            'reference_text': reference_text,
            'feedback': ["Audio data analysis requires additional setup"],
            'suggested_practice': ["Use audio file input for detailed analysis"]
        }
    
    def _compare_texts(self, recognized: str, reference: str) -> Dict[str, Any]:
        """Compare recognized text with reference text."""
        ref_words = reference.lower().split()
        rec_words = recognized.lower().split()
        
        # Calculate word accuracy
        correct_words = sum(1 for r, t in zip(rec_words, ref_words[:len(rec_words)]) 
                          if r == t)
        word_accuracy = correct_words / len(ref_words) if ref_words else 0.0
        
        # Phoneme analysis (simplified)
        phoneme_score = self._analyze_phonemes(recognized, reference)
        
        # Overall score
        score = (word_accuracy * 0.6) + (phoneme_score * 0.4)
        
        # Generate feedback
        feedback = self._generate_feedback(word_accuracy, phoneme_score)
        
        return {
            'success': True,
            'score': score,
            'word_accuracy': word_accuracy,
            'phoneme_score': phoneme_score,
            'recognized_text': recognized,
            'reference_text': reference,
            'feedback': feedback,
            'suggested_practice': self._get_practice_suggestions(score)
        }
    
    def _analyze_phonemes(self, recognized: str, reference: str) -> float:
        """Simple phoneme analysis - placeholder implementation."""
        # This would need more sophisticated phonetic analysis
        # For now, return a score based on text similarity
        if not reference:
            return 0.8
        
        similarity = len(set(recognized.lower()) & set(reference.lower())) / len(set(reference.lower()))
        return min(similarity, 1.0)
    
    def _generate_feedback(self, word_accuracy: float, phoneme_score: float) -> List[str]:
        """Generate feedback based on analysis scores."""
        feedback = []
        
        if word_accuracy < 0.5:
            feedback.append("Focus on pronouncing each word clearly.")
        elif word_accuracy < 0.8:
            feedback.append("Good progress! Some words need more practice.")
        else:
            feedback.append("Excellent word recognition!")
        
        if phoneme_score < 0.6:
            feedback.append("Work on individual sound pronunciation.")
        elif phoneme_score < 0.8:
            feedback.append("Your pronunciation is improving!")
        
        return feedback or ["Good job! Keep practicing."]
    
    def _get_practice_suggestions(self, score: float) -> List[str]:
        """Generate practice suggestions based on overall score."""
        if score < 0.5:
            return [
                "Practice speaking slowly and clearly",
                "Record yourself and compare with native speakers",
                "Focus on individual word pronunciation"
            ]
        elif score < 0.8:
            return [
                "Good progress! Keep practicing regularly",
                "Try more complex phrases",
                "Focus on rhythm and intonation"
            ]
        else:
            return [
                "Excellent pronunciation!",
                "Try advanced conversation practice",
                "Work on natural speech patterns"
            ]
    
    def set_language(self, language: str) -> None:
        """Set the target language for analysis."""
        self.language = language.lower()
        self.phoneme_map = self._load_phoneme_map()
    
    def get_phoneme_map(self) -> Dict[str, List[str]]:
        """Get phoneme map for current language."""
        return self.phoneme_map