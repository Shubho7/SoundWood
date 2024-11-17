from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.chains import RetrievalQA
from groq import Groq
from langchain.llms.base import LLM
from typing import Any, List, Optional, Dict
from pydantic import BaseModel, Field
import dotenv
import os

dotenv.load_dotenv()

GOOGLE_API = os.getenv("GOOGLE_API_KEY")
GROQ_API = os.getenv("GROQ_API_KEY")

# Custom Groq LLM wrapper
class GroqLLM(LLM):
    client: Groq = Field(default_factory=lambda: Groq(api_key=GROQ_API))
    model_name: str = "llama-3.1-70b-versatile"
    temperature: float = 0.7
    max_tokens: int = 150

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return "groq"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

# Load and process text
with open("data.txt", 'r', encoding='utf-8') as file:
    text = file.read()
doc = Document(page_content=text)
text_splitter = CharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
documents = text_splitter.split_documents([doc])

# Create embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API,
)

# Create vector store
vector_store = FAISS.from_documents(documents, embeddings)

# Initialize GroqLLM
llm = GroqLLM()

def get_prompt_template(conversation_history):
    prompt_template = f"""
    You are SoundWood, an expert having indigenous knowledge about sandalwood, with deep insights into every aspect of valuable resources provided. You can provide accurate, comprehensive information on topics including harvesting methods, conservation efforts, production processes, traditional and modern uses, market prices, health and environmental benefits, global and local demand, geographic locations where they are found, incidents of theft, illegal smuggling and associated threats. Additionally, you are well-versed with the Indian government regulations and laws surrounding the protection and trade of sandalwood and red sandalwood. Respond in a way that conveys clarity, detail, and expertise, assisting users in understanding both the natural and legal landscapes surrounding these precious resources.

    GUIDELINES:
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
        response = llm._call(prompt)
        return response
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

async def handle_query(quest, conversation_history):
    question = await generate(quest, conversation_history)
    if not question:
        return "Failed to generate a question."

    PROMPT = get_prompt_template(conversation_history)

    # Create a retrieval-based QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(score_threshold=0.5),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    response = qa_chain({"query": question})
    return response.get('result', "I don't know.")  