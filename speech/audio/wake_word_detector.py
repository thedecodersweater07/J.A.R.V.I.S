import torch
import torch.nn as nn
import torchaudio
import numpy as np

class WakeWordDetector:
    def __init__(self, wake_words=["jarvis", "hey jarvis", "ok jarvis"]):
        self.wake_words = wake_words
        self.sample_rate = 16000
        self.model = self._build_model()
        
    def _build_model(self):
        model = nn.Sequential(
            nn.Conv1d(80, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, len(self.wake_words))
        )
        return model
        
    def detect(self, audio_data, threshold=0.85):
        if isinstance(audio_data, np.ndarray):
            audio_data = torch.from_numpy(audio_data)
            
        # Extract features
        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=80
        )(audio_data)
        
        # Forward pass
        with torch.no_grad():
            predictions = torch.softmax(self.model(mel_spec), dim=-1)
            
        # Check if any wake word was detected
        max_prob, wake_word_idx = torch.max(predictions, dim=-1)
        if max_prob > threshold:
            return self.wake_words[wake_word_idx]
        return None
