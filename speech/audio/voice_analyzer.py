import torch
import torchaudio
import numpy as np

class VoiceAnalyzer:
    def __init__(self):
        self.sample_rate = 16000
        self.feature_extractor = torchaudio.transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=13
        )
        
    def analyze_pitch(self, audio_data):
        # Basic pitch analysis using zero-crossing rate
        zero_crossings = np.where(np.diff(np.signbit(audio_data)))[0]
        pitch = len(zero_crossings) * self.sample_rate / (2 * len(audio_data))
        return pitch
        
    def extract_voice_features(self, audio_data):
        if isinstance(audio_data, np.ndarray):
            audio_data = torch.from_numpy(audio_data)
        
        # Extract MFCC features
        mfcc_features = self.feature_extractor(audio_data)
        
        # Calculate energy
        energy = torch.mean(torch.abs(audio_data))
        
        return {
            'mfcc': mfcc_features,
            'energy': energy
        }

