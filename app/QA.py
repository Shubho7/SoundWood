import os
import torch
import pandas as pd 
import sounddevice as sd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, pipeline
from notebooks import preprocess_audio

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Initialize TF-IDF vectorizer for text similarity
vectorizer = TfidfVectorizer()
processed_corpus = None
corpus_embeddings = None


def process_corpus(audio_directory):
    transcriptions = []
    timestamps = []
        
    for audio_file in os.listdir(audio_directory):
        if audio_file.endswith('.wav'):
            audio_path = os.path.join(audio_directory, audio_file)
            audio = preprocess_audio(audio_path)
                
            # Transcribe using finetuned model
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
            with torch.no_grad():
                logits = model(inputs.input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.decode(predicted_ids[0])
                
            # Store transcription and metadata
            transcriptions.append(transcription)
            timestamps.append(audio_path)
        
        # Create TF-IDF embeddings
        processed_corpus = transcriptions
        corpus_embeddings = vectorizer.fit_transform(transcriptions)
        return pd.DataFrame({'audio_path': timestamps, 'transcription': transcriptions})

def record_question(duration=10):
    sample_rate = 16000
    recording = sd.rec(int(duration * sample_rate), 
                        amplerate=sample_rate, channels=1)
    sd.wait()
    return recording.squeeze()

def transcribe_question(audio):
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    return processor.decode(predicted_ids[0])

def find_answer(question, threshold=0.3):
    question_embedding = vectorizer.transform([question])
    similarities = cosine_similarity(question_embedding, corpus_embeddings)
        
    best_match_idx = similarities.argmax()
    if similarities[0][best_match_idx] < threshold:
        return None
        
    return processed_corpus[best_match_idx]