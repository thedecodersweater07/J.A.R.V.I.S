import torch
import numpy as np

class ProsodyController:
    def __init__(self):
        self.pitch_range = (50, 500)  # Hz
        self.speed_range = (0.5, 2.0)  # relative to normal speed
        
    def adjust_prosody(self, audio_data, emotion="neutral", emphasis_words=None):
        if isinstance(audio_data, np.ndarray):
            audio_data = torch.from_numpy(audio_data)
            
        # Adjust based on emotion
        pitch_shift = self._get_emotion_pitch_shift(emotion)
        speed_factor = self._get_emotion_speed_factor(emotion)
        
        # Apply pitch shift
        pitch_shifted = self._shift_pitch(audio_data, pitch_shift)
        
        # Apply speed adjustment
        speed_adjusted = self._adjust_speed(pitch_shifted, speed_factor)
        
        # Apply emphasis if specified
        if emphasis_words:
            speed_adjusted = self._apply_emphasis(speed_adjusted, emphasis_words)
            
        return speed_adjusted
        
    def _get_emotion_pitch_shift(self, emotion):
        emotion_pitch_map = {
            "happy": 1.1,
            "sad": 0.9,
            "angry": 1.2,
            "neutral": 1.0
        }
        return emotion_pitch_map.get(emotion, 1.0)
        
    def _get_emotion_speed_factor(self, emotion):
        emotion_speed_map = {
            "happy": 1.1,
            "sad": 0.9,
            "angry": 1.2,
            "neutral": 1.0
        }
        return emotion_speed_map.get(emotion, 1.0)
