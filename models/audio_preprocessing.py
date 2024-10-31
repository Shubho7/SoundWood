import os
from pydub import AudioSegment
import noisereduce as nr
import librosa

# Load audio files from Google Drive (e.g., mounted or downloaded files)
audio_dir = '/path_to_your_audio_files'

def preprocess_audio(file_path):
    # Load audio
    audio, sr = librosa.load(file_path, sr=16000)  # Standardize to 16kHz
    # Noise reduction
    reduced_noise = nr.reduce_noise(y=audio, sr=sr)
    return reduced_noise, sr

for filename in os.listdir(audio_dir):
    if filename.endswith('.wav'):
        file_path = os.path.join(audio_dir, filename)
        processed_audio, sr = preprocess_audio(file_path)
        # Save the processed audio if needed for later stages
        librosa.output.write_wav(f'processed_{filename}', processed_audio, sr)