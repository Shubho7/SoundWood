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