import streamlit as st
import tempfile
from models.QA import transcribe_audio, search_answers, transcriptions_df

st.title("SoundWood")
st.write("Record your question about sandalwood cultivation:")

# Record audio input
audio_bytes = st.audio("Upload your question here", format="audio/wav")

if audio_bytes and st.button("Get Answer"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
        temp_audio_file.write(audio_bytes)
        temp_audio_path = temp_audio_file.name

    question_text = transcribe_audio(temp_audio_path)
    st.write(f"**Transcribed Question:** {question_text}")

    answer, score = search_answers(question_text, transcriptions_df)
    st.write(f"**Answer:** {answer}")
    st.write(f"**Confidence Score:** {score}")
