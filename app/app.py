import streamlit as st

st.title("SoundWood")
st.write("Record your question about sandalwood cultivation:")

# Record audio input
audio_bytes = st.audio("Upload your question here", format="audio/wav")

if audio_bytes and st.button("Get Answer"):
    # Step 1: Save audio input temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
        temp_audio_file.write(audio_bytes)
        temp_audio_path = temp_audio_file.name