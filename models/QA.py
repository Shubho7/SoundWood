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

# Function to answer questions based on corpus transcriptions
def answer_question(question, context):
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = qa_model(**inputs)
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits
        start_idx = torch.argmax(start_scores)
        end_idx = torch.argmax(end_scores) + 1
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_idx:end_idx]))
    return answer

# Function to Search Corpus and Get Answer
def search_answers(question_text, corpus_csv_path):
    # Load the corpus
    corpus = pd.read_csv(corpus_csv_path)

    # Find the most relevant answer from corpus
    answers = []
    for _, row in corpus.iterrows():
        context_text = row["transcription"]
        answer = answer_question(question_text, context_text)
        answers.append((row["file"], answer))

    # Sort by length or relevance and select the top result
        answers = sorted(answers, key=lambda x: len(x[1]), reverse=True)
        return answers[0] 