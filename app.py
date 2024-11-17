import streamlit as st
import asyncio
from helpers import handle_query

st.set_page_config(page_title="SoundWood", page_icon="assets/wood.png", layout="centered")

st.title("SoundWood")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything related to Sandalwood"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    history = st.session_state.messages[-10:]
    formatted_history = ""
    for entry in history:
        role = entry["role"]
        content = entry["content"]
        formatted_history += f"{role}: {content}\n"

    history_text = formatted_history

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = asyncio.run(handle_query(prompt, formatted_history))
        st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.write("Ask SoundWood anything")