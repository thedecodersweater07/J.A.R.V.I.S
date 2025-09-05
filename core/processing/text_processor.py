import os
import re

RAW_TEXT_DIR = "data/raw/text"
PROCESSED_TEXT_DIR = "data/processed/text"
os.makedirs(PROCESSED_TEXT_DIR, exist_ok=True)

def preprocess_text(text):
    text = text.lower()                     # lowercase
    text = re.sub(r'[^a-z0-9\s]', '', text)  # verwijder punctuatie
    text = re.sub(r'\s+', ' ', text).strip() # normalize whitespace
    return text

# --- Verwerk raw text ---
for filename in os.listdir(RAW_TEXT_DIR):
    if not filename.endswith('.txt'):
        continue
    raw_path = os.path.join(RAW_TEXT_DIR, filename)
    with open(raw_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    processed_text = preprocess_text(raw_text)
    processed_path = os.path.join(PROCESSED_TEXT_DIR, filename)
    with open(processed_path, 'w', encoding='utf-8') as f:
        f.write(processed_text)

print(f"Processed text saved in {PROCESSED_TEXT_DIR}")
