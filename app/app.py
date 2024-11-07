import streamlit as st
import tempfile

st.title("SoundWood")
st.write("Record your question about sandalwood cultivation:")

# Record audio input
audio_bytes = st.audio("Upload your question here", format="audio/wav")

if audio_bytes and st.button("Get Answer"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
        temp_audio_file.write(audio_bytes)
        temp_audio_path = temp_audio_file.name