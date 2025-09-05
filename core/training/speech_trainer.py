import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torchaudio
import pandas as pd
import pickle
import logging

# =====================================================================
# Logging setup
# =====================================================================
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/speech_trainer.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# =====================================================================
# Dataset loader
# =====================================================================
class SpeechDataset(Dataset):
    def __init__(self, csv_file, data_dir, transform=None, filename_column=None, label_column=None):
        csv_file = os.path.normpath(csv_file)
        if not os.path.exists(csv_file):
            logging.error(f"CSV bestand niet gevonden: {csv_file}")
            raise FileNotFoundError(f"CSV bestand niet gevonden: {csv_file}")

        self.data = pd.read_csv(csv_file)
        self.data_dir = data_dir
        self.transform = transform

        # detecteer filename kolom
        if filename_column and filename_column in self.data.columns:
            self.filename_col = filename_column
        elif "filename" in self.data.columns:
            self.filename_col = "filename"
        else:
            # fallback: eerste kolom met string waarden eindigend op .wav
            candidates = [c for c in self.data.columns if self.data[c].apply(lambda x: isinstance(x, str) and x.lower().endswith(".wav")).any()]
            if candidates:
                self.filename_col = candidates[0]
                logging.warning(f"'filename' kolom niet gevonden. Gebruik {self.filename_col} als filename.")
            else:
                raise KeyError("Geen kolom gevonden voor audiobestanden (.wav) in CSV")

        # detecteer label kolom
        if label_column and label_column in self.data.columns:
            self.label_col = label_column
        elif "label" in self.data.columns:
            self.label_col = "label"
        else:
            # fallback: eerste kolom die niet filename is
            self.label_col = [c for c in self.data.columns if c != self.filename_col][0]
            logging.warning(f"'label' kolom niet gevonden. Gebruik {self.label_col} als label.")

        # sla labels op
        self.labels = self.data[self.label_col].tolist()
        logging.info(f"Dataset geladen: {len(self.data)} samples van {csv_file}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # gebruik alleen basename om paden niet dubbel te maken
        filename = os.path.basename(str(row[self.filename_col]))
        wav_path = os.path.normpath(os.path.join(self.data_dir, filename))

        if not os.path.exists(wav_path):
            logging.warning(f"WAV bestand niet gevonden: {wav_path}")

        waveform, sr = torchaudio.load(wav_path)
        label = row[self.label_col]

        if self.transform:
            waveform = self.transform(waveform)

        return waveform, label

# =====================================================================
# Simpel CNN speech model
# =====================================================================
class SpeechModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        if x.ndim == 2:  # (batch, time)
            x = x.unsqueeze(1)  # (batch, 1, time)
        return self.network(x)

# =====================================================================
# Trainer
# =====================================================================
class SpeechTrainer:
    def __init__(self, dataset, num_classes, batch_size=32, lr=1e-3,
                 log_dir="logs", model_dir="data/models/speech"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SpeechModel(num_classes).to(self.device)
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        self.log_dir = log_dir
        self.model_dir = model_dir
        self.writer = SummaryWriter(log_dir=log_dir)
        self.dataset = dataset

    def train(self, epochs=10):
        logging.info("Training gestart")
        for epoch in range(epochs):
            running_loss = 0.0
            loop = tqdm(self.loader, desc=f"Epoch {epoch+1}/{epochs}")
            for waveforms, labels in loop:
                # convert labels naar tensor indien nodig
                if not torch.is_tensor(labels):
                    labels = torch.tensor(labels, dtype=torch.long)
                waveforms, labels = waveforms.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(waveforms)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                loop.set_postfix(loss=loss.item())

            avg_loss = running_loss / len(self.loader)
            self.writer.add_scalar("Loss/train", avg_loss, epoch)
            logging.info(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")
            print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")

        # Opslaan als .pkl
        tts_model_path = os.path.join(self.model_dir, "tts_model.pkl")
        voice_id_path = os.path.join(self.model_dir, "voice_id.pkl")

        with open(tts_model_path, "wb") as f:
            pickle.dump(self.model, f)
        with open(voice_id_path, "wb") as f:
            pickle.dump(self.dataset.labels, f)

        logging.info(f"Model opgeslagen in: {tts_model_path}")
        logging.info(f"Voice ID labels opgeslagen in: {voice_id_path}")
        print(f"Model opgeslagen in: {tts_model_path}")
        print(f"Voice ID labels opgeslagen in: {voice_id_path}")

        self.writer.close()
        logging.info("Training afgerond")

# =====================================================================
# Main
# =====================================================================
if __name__ == "__main__":
    dataset = SpeechDataset(
        csv_file=r"data/data_sets/speech_dataset/dataset_manifest.csv",
        data_dir=r"data/data_sets/speech_dataset/wavs"
    )
    trainer = SpeechTrainer(dataset, num_classes=10, batch_size=16)
    trainer.train(epochs=5)
    logging.info("Script voltooid")