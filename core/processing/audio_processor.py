import os
import subprocess
import numpy as np
import soundfile as sf
import noisereduce as nr

INPUT_FOLDER = "data/raw/audio"
OUTPUT_FOLDER = "data/processed/audio"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

for filename in os.listdir(INPUT_FOLDER):
    if not (filename.endswith(".wav") or filename.endswith(".mp3")):
        continue

    input_path = os.path.join(INPUT_FOLDER, filename)
    output_name = os.path.splitext(filename)[0] + ".wav"  # alles als WAV opslaan
    output_path = os.path.join(OUTPUT_FOLDER, output_name)

    # 1) Als het een MP3 is, converteer naar WAV met ffmpeg
    temp_wav = input_path
    if filename.endswith(".mp3"):
        temp_wav = os.path.join(OUTPUT_FOLDER, "temp.wav")
        subprocess.run([
            "ffmpeg", "-y", "-i", input_path, temp_wav
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 2) Laad WAV direct met soundfile
    samples, sr = sf.read(temp_wav)
    if samples.dtype != np.float32:
        samples = samples.astype(np.float32)

    # 3) Noise reduction
    reduced_noise = nr.reduce_noise(y=samples, sr=sr)

    # 4) Opslaan
    sf.write(output_path, reduced_noise, sr)
    print(f"Processed: {filename}")

    # 5) Verwijder tijdelijke file
    if filename.endswith(".mp3"):
        os.remove(temp_wav)

print("All files processed!")
