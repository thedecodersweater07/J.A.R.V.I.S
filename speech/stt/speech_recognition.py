import torch
import torch.nn as nn

class SpeechRecognizer(nn.Module):
    def __init__(self, input_dim=80, hidden_dim=256, num_layers=3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, 29)  # 26 letters + space + blank + apostrophe
        
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x)
        return torch.log_softmax(x, dim=-1)
    
    def decode(self, acoustic_output):
        # Simple greedy decoding
        predictions = torch.argmax(acoustic_output, dim=-1)
        return predictions

