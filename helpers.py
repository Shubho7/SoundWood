from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.chains import RetrievalQA
from groq import Groq
from langchain.llms.base import LLM
from pydantic import BaseModel
import dotenv
import os

dotenv.load_dotenv()

GOOGLE_API = os.getenv("GOOGLE_API_KEY")
GROQ_API = os.getenv("GROQ_API_KEY")

with open("data.txt", 'r', encoding='utf-8') as file:
    text = file.read()
doc = Document(page_content=text)
text_splitter = CharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
documents = text_splitter.split_documents([doc])

# Create the Google Generative AI embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API,
)

# Create the FAISS vector store
vector_store = FAISS.from_documents(documents, embeddings)

# Initialize Groq
client = Groq(api_key=GROQ_API)

def get_prompt_template(conversation_history):
    prompt_template = f"""
    You are a knowledgeable expert in the field of sandalwood and red sandalwood, with deep insights into every aspect of these valuable resources. You provide accurate, comprehensive information on topics including harvesting methods, conservation efforts, production processes, traditional and modern uses, market prices, health and environmental benefits, global and local demand, geographic locations where they are found, incidents of theft and illegal smuggling, and associated threats. Additionally, you are well-versed in Indian government regulations and laws surrounding the protection and trade of sandalwood and red sandalwood. Respond in a way that conveys clarity, detail, and expertise, assisting users in understanding both the natural and legal landscapes surrounding these precious resources.

    Instructions for the AI:
    - Carefully analyze the given source documents and context. Use these sources as your primary reference to formulate detailed, expert-level responses that address the question comprehensively.
    - Combine insights from multiple sections of the provided context when necessary to offer a well-rounded and expert response and do provide answers with useful tokens and not rubbish tokens like '\\n'.
    - When responding, use as much relevant information from the "response" section of the source documents as possible, maintaining accuracy and detail but rephrase it in your own helpful comprehensive way.
    - If the context does not provide sufficient information or relevant details, respond with "I don't know."
    - Use the given source documents as your primary reference to answer questions about sandalwood and red sandalwood, ensuring that your responses are accurate, detailed, and expert-level.
    - If specific information is AT ALL not available, minimally use your expertise to provide general guidance or information on the topic.
    - Keep responses concise and focused, providing actionable steps and resources when possible.
    - If the question is a greeting or not related to the context, respond with an appropriate greeting or "I don't know."

    Previous Conversation:
    {conversation_history}

    CONTEXT: {{context}}

    QUESTION: {{question}}
    """

    return PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

async def generate(quest, conversation_history):

    prompt = f"""
    You are SoundWood, an expert in simplifying questions related to sandalwood and red sandalwood. Your task is to transform the given question into a clear, concise, and focused query that will effectively retrieve relevant information from the vector database, without adding any elaboration or explanation.

    QUESTION: {quest}
    CONVERSATION HISTORY: {conversation_history}
    """

    try:
        response =  client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=150
        )
        quest = response.choices[0].message.content.strip()
        return quest
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

async def handle_query(quest, conversation_history):
    question = await generate(quest, conversation_history)
    if not question:
        return "Failed to generate a question."

    # Create the prompt template using the conversation history
    PROMPT = get_prompt_template(conversation_history)

    # Create a retrieval-based QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=client,
        chain_type="stuff",
        retriever=vector_store.as_retriever(score_threshold=0.5),
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    response = qa_chain({"query": question})
    return response