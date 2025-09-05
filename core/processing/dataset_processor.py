import os
import pandas as pd

# Directories
PROCESSED_DIR = "data/processed"
DATASET_DIR = "data/data_sets"
os.makedirs(DATASET_DIR, exist_ok=True)

# Folders om te verwerken
folders = ["text", "speech", "image"]

for folder_name in folders:
    folder_path = os.path.join(PROCESSED_DIR, folder_name)
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist, skipping...")
        continue

    files = os.listdir(folder_path)
    if not files:
        print(f"No files in {folder_path}, skipping...")
        continue

    # Zorg dat submap voor dataset bestaat
    folder_dataset_dir = os.path.join(DATASET_DIR, folder_name)
    os.makedirs(folder_dataset_dir, exist_ok=True)

    data = []
    for f in files:
        file_path = os.path.join(folder_path, f)
        try:
            if folder_name == "text":
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
            else:
                content = file_path  # voor speech/image: pad opslaan
            data.append({"filename": f, "content": content})
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    if data:
        df = pd.DataFrame(data)
        dataset_path = os.path.join(folder_dataset_dir, f"{folder_name}_dataset.csv")
        df.to_csv(dataset_path, index=False, encoding='utf-8')
        print(f"Dataset saved: {dataset_path}")
