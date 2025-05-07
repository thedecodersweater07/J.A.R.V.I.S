# AI Training Datasets

This directory contains various datasets for training AI models:

## Dataset Structure

- `training_data.csv`: Classification dataset with 1000 samples
  - Features: 5 numerical inputs
  - Target: Binary classification (0/1)
  - Confidence score included

- `nlp_dataset.jsonl`: Conversation pairs for language models
  - Format: JSON Lines with prompt/response pairs
  - Categories: greeting, weather, programming, time, entertainment

- `text_corpus.txt`: Text corpus for language training
  - Contains technical AI-related content
  - ~10,000 words of structured text

- `images/`: Image dataset
  - 10 generated images (PNG format)
  - Labels in image_labels.json

- `speech_dataset/`: Audio samples
  - 5 WAV files (16kHz, mono)
  - Transcripts in transcripts.txt

### Word Lists
- `dutch_words.txt`: List of Dutch vocabulary words (1000 entries)
- `english_words.txt`: List of English vocabulary words (1000 entries)
- `technical_terms.csv`: Technical terminology with descriptions (100 entries)

### JSON Datasets
- `products.json`: Product catalog with 100 items
- `users.json`: User database with 200 records
- `config.json`: Configuration settings

### Matrix Data
- `features_matrix.npy`: Training features (1000x50)
- `labels_matrix.npy`: Training labels (1000)
- `embeddings.npy`: Word embeddings (100x300)

## Usage

Run `generate_datasets.py` to create all datasets in the `ai_training_data` directory.
