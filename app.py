import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import speech_recognition as sr
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import pyaudio

# @st.cache_resource
# def load_asr_model():
#     model_path = "finetuned_saved\wav2vec_model"
#     processor_path = "finetuned_saved\wav2vec_preprocessor"
#     model = Wav2Vec2ForCTC.from_pretrained(model_path)
#     processor = Wav2Vec2Processor.from_pretrained(processor_path)
#     return model, processor

# Load embedding model for semantic similarity
@st.cache_resource
def load_embedding_model():
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2') 
    return embedding_model

# model, processor = load_asr_model()
embedding_model = load_embedding_model()

# Load transcriptions from CSV
@st.cache_data
def load_transcriptions(csv_file_path):
    df = pd.read_csv(csv_file_path)
    return df

# # Function to process the user's speech query
# def process_user_query():
#     recognizer = sr.Recognizer()
#     with sr.Microphone() as source:
#         st.info("Listening... Speak your question clearly into the microphone.")
#         audio = recognizer.listen(source)
    
#     # Convert speech to text
#     try:
#         query_text = recognizer.recognize_google(audio)
#         st.write(f"Query: {query_text}")
#         return query_text
#     except sr.UnknownValueError:
#         st.error("Sorry, I could not understand your speech. Please try again.")
#         return None
#     except sr.RequestError:
#         st.error("Network error. Please check your connection and try again.")
#         return None

def get_user_query():
    query_text = st.text_input("Enter your question:")
    if query_text:
        st.write(f"Query: {query_text}")
        return query_text
    else:
        return None

# Function to search for relevant information in the transcriptions
def search_transcriptions(query, transcriptions_df):
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    best_match = None
    best_score = -1
    best_segment_text = ""

    for index, row in transcriptions_df.iterrows():
        transcription = row['transcription'] 
        transcription_embedding = embedding_model.encode(transcription, convert_to_tensor=True)
        
        # Compute cosine similarity
        similarity_score = util.pytorch_cos_sim(query_embedding, transcription_embedding).item()
        
        if similarity_score > best_score:
            best_score = similarity_score
            best_match = row['file']
            best_segment_text = transcription

    return best_match, best_segment_text

# Streamlit app
st.title("Enhanced Speech-Based Question Answering System")
st.write("Ask your question using speech, and we'll fetch relevant information from the transcriptions using semantic similarity.")

if st.button("Submit Question"):
    query = get_user_query()
    if query:
        csv_file_path = "path/to/your/transcriptions.csv"
        transcriptions_df = load_transcriptions(csv_file_path)
        best_match, best_segment_text = search_transcriptions(query, transcriptions_df)
        
        if best_match:
            st.success(f"Best Match: {best_match}")
            st.write("Relevant Information: ")
            st.write(best_segment_text)
        else:
            st.warning("No relevant information found in the transcriptions.")
