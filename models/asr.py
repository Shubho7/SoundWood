import whisper
from audio_preprocessing import load_and_reduce_noise

# Load Whisper ASR model
model = whisper.load_model("medium")

def transcribe_audio(audio_file):
    # Preprocess and denoise audio
    audio, sample_rate = load_and_reduce_noise(audio_file)
    
    # Run ASR model
    result = model.transcribe(audio)
    return result["text"]
