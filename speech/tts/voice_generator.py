import torch
import torch.nn as nn

class VoiceGenerator(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, text_embedding):
        encoded = self.encoder(text_embedding)
        audio_features = self.decoder(encoded)
        return audio_features

