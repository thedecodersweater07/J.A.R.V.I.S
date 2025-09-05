import os
import requests

SAVE_DIR = "data/raw/images"
os.makedirs(SAVE_DIR, exist_ok=True)

# Voor test: kitten images
image_urls = [
    "https://placekitten.com/400/300",
    "https://placekitten.com/500/400",
    "https://placekitten.com/600/500",
]

# Simuleer echte browser
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/116.0.0.0 Safari/537.36"
}

for i, url in enumerate(image_urls):
    try:
        response = requests.get(url, headers=headers, stream=True)
        if response.status_code == 200:
            file_path = os.path.join(SAVE_DIR, f"image_{i}.jpg")
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            print(f"[OK] {url} opgeslagen als {file_path}")
        else:
            print(f"[FAIL] {url} status code: {response.status_code}")
    except Exception as e:
        print(f"[ERROR] {url} -> {e}")
