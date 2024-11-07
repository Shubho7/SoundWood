from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline, WhisperProcessor, WhisperForConditionalGeneration
import whisper
import pandas as pd
from gtts import gTTS
import sounddevice as sd
import librosa
import tempfile
import scipy.io.wavfile as wav

# Load Whisper model for transcription
whisper_model = "C:\\Users\\baner\\Documents\\SoundWood\\models\\whisper-modelv1"
whisper_processor = WhisperProcessor.from_pretrained("C:\\Users\\baner\\Documents\\SoundWood\\models\\whisper-processorv1")
whisper_model_QA = WhisperForConditionalGeneration.from_pretrained(whisper_model)

# Load Multilingual BERT model for Question Answering
bert_model_name = "ai4bharat/indic-bert"
tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
bert_model = AutoModelForQuestionAnswering.from_pretrained(bert_model_name)
qa_pipeline = pipeline("question-answering", model=bert_model, tokenizer=tokenizer)

transcriptions_df = pd.read_csv("data\transcriptionsv2.csv")

# Function to transcribe audio question using Whisper model
def transcribe_audio(audio_path):
    audio, rate = librosa.load(audio_path, sr=16000)
    inputs = whisper_processor(audio, sampling_rate=rate, return_tensors="pt")
    generated_ids = whisper_model.generate(inputs["input_features"])
    transcription = whisper_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return transcription

# Function to search audio corpus and get answer
def search_answers(question, transcriptions_df):
    best_answer = ""
    best_score = float("-inf")
    
    for index, row in transcriptions_df.iterrows():
        context = row["transcription"]
        # Perform QA using BERT model
        result = qa_pipeline({
            "question": question,
            "context": context
        })
        # Track best answer based on score
        if result["score"] > best_score:
            best_score = result["score"]
            best_answer = result["answer"]

    return best_answer, best_score

# Function to convert text answer to speech and play
def text_to_speech(text, lang="kn"):
    tts = gTTS(text, lang=lang)
    temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_audio_file.name)
    return temp_audio_file.name