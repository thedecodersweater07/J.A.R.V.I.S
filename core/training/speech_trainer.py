import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import torchaudio
import torchaudio.functional as F
import pandas as pd
import pickle
from tqdm import tqdm

# =====================================================================
# Dataset loader met audio preprocessing
# =====================================================================
class SpeechDataset(Dataset):
    def __init__(self, csv_file, data_dir, idx2label, max_len=16000):
        self.csv_file = csv_file
        self.data_dir = data_dir
        self.max_len = max_len
        self.idx2label = idx2label
        self.label2idx = {v: k for k, v in idx2label.items()}

        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV bestand niet gevonden: {csv_file}")

        self.data = pd.read_csv(csv_file)

        # Detecteer kolom voor audio-bestanden
        file_candidates = [c for c in self.data.columns
                           if self.data[c].apply(lambda x: isinstance(x, str) and x.lower().endswith(('.wav', '.mp3', '.flac'))).any()]
        if not file_candidates:
            raise KeyError("Geen audio-bestand kolom gevonden in CSV")
        self.file_col = file_candidates[0]

        # Detecteer kolom voor labels
        label_candidates = [c for c in self.data.columns if c != self.file_col]
        if not label_candidates:
            raise KeyError("Geen label kolom gevonden in CSV")
        self.label_col = label_candidates[0]

        # Voeg label_idx toe
        self.data['label_idx'] = self.data[self.label_col].map(self.label2idx)

    def __len__(self):
        return len(self.data)

    def preprocess_audio(self, waveform):
        # Mono
        if waveform.ndim > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Pad/truncate
        if waveform.shape[1] < self.max_len:
            pad = self.max_len - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        else:
            waveform = waveform[:, :self.max_len]

        # Normalize volume
        max_val = waveform.abs().max()
        if max_val > 0:
            waveform = waveform / max_val

        # Eenvoudige denoise: high-pass filter 80Hz
        waveform = F.highpass_biquad(waveform, 80, 16000)
        return waveform

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        wav_path = os.path.join(self.data_dir, os.path.basename(row[self.file_col]))

        waveform = torch.zeros(1, self.max_len)
        try:
            torchaudio.set_audio_backend("sox_io")  # beter voor MP3
            wave, sr = torchaudio.load(wav_path)
            waveform = self.preprocess_audio(wave)
        except Exception as e:
            print(f"Fout bij laden van {wav_path}: {e}")

        label_idx = int(row['label_idx'])
        return waveform, torch.tensor(label_idx, dtype=torch.long)

# =====================================================================
# CNN speech model
# =====================================================================
class SpeechModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(1, 16, 5, stride=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, 5, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        return self.network(x)

# =====================================================================
# Trainer
# =====================================================================
class SpeechTrainer:
    def __init__(self, model_pkl, voice_id_pkl, csv_file, data_dir, batch_size=16, lr=1e-3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Laad voice mapping
        if os.path.exists(voice_id_pkl):
            with open(voice_id_pkl, "rb") as f:
                idx2label = pickle.load(f)
        else:
            raise FileNotFoundError(f"voice_id.pkl niet gevonden: {voice_id_pkl}")

        self.dataset = SpeechDataset(csv_file, data_dir, idx2label)
        self.num_classes = len(idx2label)
        self.loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

        # Laad bestaand model of nieuw model
        if os.path.exists(model_pkl):
            with open(model_pkl, "rb") as f:
                self.model = pickle.load(f).to(self.device)
                print(f"Bestaand model geladen: {model_pkl}")
        else:
            self.model = SpeechModel(self.num_classes).to(self.device)
            print("Nieuw model aangemaakt.")

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.model_pkl = model_pkl

    def train(self, epochs=5):
        for epoch in range(epochs):
            running_loss = 0.0
            loop = tqdm(self.loader, desc=f"Epoch {epoch+1}/{epochs}")
            for waveforms, labels in loop:
                waveforms, labels = waveforms.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(waveforms)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                loop.set_postfix(loss=loss.item())
            avg_loss = running_loss / len(self.loader)
            print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")

        # Opslaan
        with open(self.model_pkl, "wb") as f:
            pickle.dump(self.model, f)
        print(f"Model opgeslagen in {self.model_pkl}")

# =====================================================================
# Main
# =====================================================================
if __name__ == "__main__":
    trainer = SpeechTrainer(
        model_pkl="data/models/speech/tts_model.pkl",
        voice_id_pkl="data/models/speech/voice_id.pkl",
        csv_file="data/data_sets/speech_dataset/dataset_manifest.csv",
        data_dir="data/data_sets/speech_dataset/wavs",
        batch_size=16,
        lr=1e-3
    )
    trainer.train(epochs=5)
