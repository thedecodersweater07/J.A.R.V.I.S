import torch
import torchaudio
import numpy as np

class AudioPreprocessor:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=400,
            hop_length=160,
            n_mels=80
        )
        
    def preprocess(self, audio_data):
        if isinstance(audio_data, np.ndarray):
            audio_data = torch.from_numpy(audio_data)
        
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = torch.mean(audio_data, dim=0)
            
        # Normalize audio
        audio_data = audio_data / torch.max(torch.abs(audio_data))
        
        # Generate mel spectrogram
        mel_spec = self.mel_transform(audio_data)
        
        # Log mel spectrogram
        mel_spec = torch.log(mel_spec + 1e-9)
        
        return mel_spec