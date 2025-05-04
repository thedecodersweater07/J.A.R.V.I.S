import torch
import torch.nn as nn
import torchaudio
import numpy as np

class EmotionDetector:
    def __init__(self):
        self.emotions = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]
        self.model = self._build_model()
        
    def _build_model(self):
        model = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, len(self.emotions))
        )
        return model
        
    def detect_emotion(self, audio_data):
        if isinstance(audio_data, np.ndarray):
            audio_data = torch.from_numpy(audio_data)
            
        # Extract features
        mel_spec = torchaudio.transforms.MelSpectrogram()(audio_data)
        mel_spec = mel_spec.unsqueeze(0)  # Add channel dimension
        
        # Predict emotion
        with torch.no_grad():
            predictions = torch.softmax(self.model(mel_spec), dim=-1)
            emotion_idx = torch.argmax(predictions).item()
            
        return {
            'emotion': self.emotions[emotion_idx],
            'confidence': predictions[0][emotion_idx].item()
        }

