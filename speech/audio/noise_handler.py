import torch
import torchaudio
import numpy as np

class NoiseHandler:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        
    def reduce_noise(self, audio_data):
        if isinstance(audio_data, np.ndarray):
            audio_data = torch.from_numpy(audio_data)
            
        # Simple noise reduction using spectral gating
        spec = torch.stft(audio_data, 
                         n_fft=2048, 
                         hop_length=512, 
                         return_complex=True)
        
        magnitude = torch.abs(spec)
        phase = torch.angle(spec)
        
        # Estimate noise floor
        noise_floor = torch.mean(magnitude[:100], dim=0)
        
        # Apply spectral gating
        mask = magnitude > (2 * noise_floor)
        magnitude = magnitude * mask
        
        # Reconstruct signal
        spec_complex = magnitude * torch.exp(1j * phase)
        cleaned = torch.istft(spec_complex, 
                            n_fft=2048, 
                            hop_length=512)
        
        return cleaned

