import os
import pickle
import torch


def load_speech_model(
    model_path="data/models/speech/tts_model.pkl",
    voice_id_path="data/models/speech/voice_id.pkl",
    device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model bestand niet gevonden: {model_path}")
    if not os.path.exists(voice_id_path):
        raise FileNotFoundError(f"Voice ID bestand niet gevonden: {voice_id_path}")

    # Laad model (pickle) - class moet beschikbaar zijn via import
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    model.to(device)
    model.eval()

    # Laad labels
    with open(voice_id_path, "rb") as f:
        idx2label = pickle.load(f)

    return model, idx2label


# =====================================================================
# main
# =====================================================================
if __name__ == "__main__":
    try:
        
        print("Model en labels succesvol geladen.")
        print(f"Labels")
    except Exception as e:
        print(f"Fout bij het laden van het model of de labels: {e}")