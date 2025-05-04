import torch
import torchaudio
import numpy as np

class AudioRenderer:
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        
    def render(self, audio_features, output_path=None):
        if isinstance(audio_features, np.ndarray):
            audio_features = torch.from_numpy(audio_features)
            
        # Apply inverse mel scaling
        audio_data = self._inverse_mel_scale(audio_features)
        
        # Apply Griffin-Lim algorithm for phase reconstruction
        audio_data = self._griffin_lim(audio_data)
        
        # Normalize audio
        audio_data = audio_data / torch.max(torch.abs(audio_data))
        
        # Save if path provided
        if output_path:
            torchaudio.save(output_path, audio_data, self.sample_rate)
            
        return audio_data
        
    def _inverse_mel_scale(self, mel_spec):
        inverse_mel = torchaudio.transforms.InverseMelScale(
            n_stft=1024,
            n_mels=80,
            sample_rate=self.sample_rate
        )
        return inverse_mel(mel_spec)
        
    def _griffin_lim(self, magnitude_spec, n_iter=32):
        # Griffin-Lim algorithm implementation
        angles = np.angle(np.exp(2j * np.pi * np.random.rand(*magnitude_spec.shape)))
        angles = np.exp(1j * angles)
        spec = magnitude_spec * angles
        
        for _ in range(n_iter):
            audio = torch.istft(torch.from_numpy(spec), n_fft=1024, hop_length=256)
            spec = torch.stft(audio, n_fft=1024, hop_length=256, return_complex=True)
            angles = spec / torch.clamp(torch.abs(spec), min=1e-8)
            spec = magnitude_spec * angles
            
        audio = torch.istft(torch.from_numpy(spec), n_fft=1024, hop_length=256)
        return audio
