import streamlit as st
import asyncio
from helpers import handle_query

st.set_page_config(page_title="SoundWood", page_icon="assets/wood.png", layout="wide", initial_sidebar_state="expanded")

st.title("SoundWood")

with st.sidebar:
    st.title("ABOUT Santal.AI")
    st.write("Meet Santal.AI — an assistant designed to settle the fight against sandalwood theft, smuggling, and black market crime. Imagine a tool that can instantly provide rich, web-sourced insights on everything sandalwood: from harvesting and conservation to market trends, traditional uses, and real-time updates on theft incidents. This chatbot is made for investigators to effortlessly streamlining inquiries while also educating the public about the hidden world of sandalwood scams and the laws protecting this precious resource. With a mission to make critical information accessible and engaging, it's a powerful ally in safeguarding sandalwood and raising awareness like never before.")
    
    if st.button("CLEAR CHAT"):
        st.session_state.messages = []

    with st.expander("ABOUT THE DATA"):
        st.write("Santal.AI is powered by diverse and reliable dataset curated from multiple trusted sources, including reputable news articles, social question-and-answer platforms, and conversational YouTube videos focused on sandalwood. This multi-faceted approach ensures that the information provided is both comprehensive and grounded in real-world perspectives. The chatbot delivers well-rounded and genuine responses offering an authentic and nuanced understanding of sandalwood-related topics.")
    with st.expander("ABOUT THE MODEL"):
        st.write("The backbone of the Santal.AI is the Llama-3.1-70B-Versatile model, renowned for its exceptional language comprehension and generation capabilities. This powerful model processes vast amounts of data and dynamically retrieves relevant information using Retrieval-Augmented Generation (RAG). By combining the model's deep understanding of language with real-time data retrieval from curated sources, the system ensures accurate, context-aware, and highly specific responses. This enables seamless interaction, delivering reliable insights tailored to each user query.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("How can I help you today?"):
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
    st.write("Ask Santal.AI anything related to Sandalwood!")