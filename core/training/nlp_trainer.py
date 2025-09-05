import os
import pickle
import pandas as pd
from datetime import datetime

# ================================
# Config
# ================================
PROJECT_ROOT = r"C:\J.A.R.V.I.S"
DATA_FILE = os.path.join(PROJECT_ROOT, "data", "data_sets", "text", "text_dataset.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "data", "models", "nlp")
LOG_FILE = os.path.join(PROJECT_ROOT, "nlp_training.log")

os.makedirs(MODEL_DIR, exist_ok=True)

# ================================
# Logging
# ================================
def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")

# ================================
# Load CSV dataset
# ================================
if not os.path.exists(DATA_FILE):
    log(f"ERROR: Dataset not found: {DATA_FILE}")
    raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

df = pd.read_csv(DATA_FILE)
log(f"Loaded {len(df)} rows from dataset")

# ================================
# Helper: load or create pickle model
# ================================
def load_or_create_model(filename, default):
    path = os.path.join(MODEL_DIR, filename)
    if os.path.exists(path):
        log(f"Loading existing model: {filename}")
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            log(f"Failed to load {filename} ({e}), creating new model")
            return default
    else:
        log(f"Model not found, creating new: {filename}")
        return default

# ================================
# Load models
# ================================
chat_model = load_or_create_model("chat_model.pkl", [])
embeddings = load_or_create_model("embeddings.pkl", [])
entity_model = load_or_create_model("entity_model.pkl", [])
intent_model = load_or_create_model("intent_model.pkl", [])

# ================================
# Training / updating models
# ================================
log("Updating models with dataset...")

for idx, row in df.iterrows():
    text = str(row.get("text", "")).strip()
    if not text:
        continue

    # Chat model: append all text
    chat_model.append(text)

    # Embeddings: naive embedding = length of text
    embeddings.append(len(text))

    # Entity model: first word
    first_word = text.split()[0] if text else ""
    entity_model.append(first_word)

    # Intent model: True if sentence is a question
    intent_model.append("?" in text)

    if (idx + 1) % 50 == 0:
        log(f"Processed {idx + 1}/{len(df)} rows")

# ================================
# Save models
# ================================
def save_model(obj, filename):
    path = os.path.join(MODEL_DIR, filename)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    log(f"Saved model: {filename}")

save_model(chat_model, "chat_model.pkl")
save_model(embeddings, "embeddings.pkl")
save_model(entity_model, "entity_model.pkl")
save_model(intent_model, "intent_model.pkl")

log("All models updated successfully!")
