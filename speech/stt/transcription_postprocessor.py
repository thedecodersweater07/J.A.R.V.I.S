import re
from typing import List, Dict

class TranscriptionPostProcessor:
    def __init__(self):
        self.punctuation_marks = ['.', '!', '?', ',']
    
    def process_transcript(self, transcript: str) -> str:
        """
        Clean and format the transcription text
        """
        # Fix capitalization
        transcript = self._fix_capitalization(transcript)
        
        # Add missing punctuation
        transcript = self._add_punctuation(transcript)
        
        # Remove extra spaces
        transcript = re.sub(r'\s+', ' ', transcript).strip()
        
        return transcript
    
    def _fix_capitalization(self, text: str) -> str:
        sentences = re.split(r'([.!?]+)', text)
        processed = []
        
        for i in range(len(sentences)):
            if i % 2 == 0:  # Text part
                sentence = sentences[i].strip()
                if sentence:
                    sentence = sentence[0].upper() + sentence[1:]
                processed.append(sentence)
            else:  # Punctuation part
                processed.append(sentences[i])
                
        return ''.join(processed)
    
    def _add_punctuation(self, text: str) -> str:
        if not text.strip():
            return text
            
        if text[-1] not in self.punctuation_marks:
            text += '.'
            
        return text
