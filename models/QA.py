import torch
from transformers import BertTokenizer, BertForQuestionAnswering
import whisper
import pandas as pd
import gtts  
import os
import playsound
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav

# Load Whisper Model for Transcription
whisper_model = whisper.load_model("models\whisper-modelv1")

# Load Multilingual BERT Model for Question Answering
qa_model = BertForQuestionAnswering.from_pretrained("bert-base-multilingual-cased")
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

# Function to Record Audio from Microphone
def record_audio(duration=20, sample_rate=16000):
    print("Recording...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()  
    audio_data = np.squeeze(audio_data)  
    wav.write("question.wav", sample_rate, audio_data) 
    print("Recording complete.")
    return "question.wav"

# Function to Transcribe Audio Question to Text
def transcribe_audio(file_path):
    result = whisper_model.transcribe(file_path)
    return result["text"]