import torch
import torchaudio
import numpy as np

class SpeechEnhancer:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        
    def enhance(self, audio_data):
        if isinstance(audio_data, np.ndarray):
            audio_data = torch.from_numpy(audio_data)
            
        # Convert to frequency domain
        spec = torch.stft(audio_data, 
                         n_fft=512, 
                         hop_length=256,
                         return_complex=True)
                         
        # Enhance speech frequencies (human voice range: 300-3400 Hz)
        freq_bins = torch.fft.fftfreq(512, d=1/self.sample_rate)
        mask = (freq_bins >= 300) & (freq_bins <= 3400)
        spec[mask] *= 1.2
        
        # Convert back to time domain
        enhanced = torch.istft(spec,
                             n_fft=512,
                             hop_length=256)
                             
        return enhanced
