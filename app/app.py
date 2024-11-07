import streamlit as st

st.title("SoundWood")
st.write("Record your question about sandalwood cultivation:")

# Record audio input
audio_bytes = st.audio("Upload your question here", format="audio/wav")
