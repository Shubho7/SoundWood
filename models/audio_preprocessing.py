import librosa

def load_and_reduce_noise(audio_file, noise_reduction_level=0.1):
    audio, sample_rate = librosa.load(audio_file, sr=None)
    reduced_noise_audio = librosa.effects.preemphasis(audio, coef=noise_reduction_level)
    
    return reduced_noise_audio, sample_rate