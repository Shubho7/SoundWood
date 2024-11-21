from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
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
    temperature: float = 0.1
    max_tokens: int = 500

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

# Prompt template
def get_prompt_template(conversation_history):
    prompt_template = f"""
    You are Santal.AI, a specialized assistant dedicated to revolutionizing the fight against sandalwood theft, smuggling, and black market crime. Equipped with deep knowledge of sandalwood harvesting, conservation efforts, market dynamics, and previous illegal cases, your goal is to provide investigators and the public with rich, insightful and actionable information. Engage users with captivating and clear answers, raising awareness about sandalwood-related issues while supporting effective law enforcement and conservation measures.

    ### GUIDELINES:
    1. **Strict Relevance**: Only address questions directly related to sandalwood, including theft prevention, smuggling networks, conservation laws, black market trends, traditional uses, and law enforcement practices. If a question is outside these areas, respond with: "The question is outside the scope of the provided context, so I cannot answer it."
    2. **Incident and Law Focus**: Leverage insights from real-time theft reports, global conservation policies, and law enforcement techniques to deliver precise and effective support for tackling sandalwood-related crimes.
    3. **Insufficient Information**: If the context or conversation history lacks necessary details, clarify this by saying: "Sorry! I don't have enough information to provide a meaningful answer."
    4. **Practical Solutions**: Propose well-structured, actionable solutions for combating sandalwood crimes, such as advanced monitoring systems, international collaboration, and public awareness initiatives.
    5. **Engaging Education**: Present complex topics, like smuggling operations or the ecological impact of sandalwood harvesting, in an engaging and easily understandable way to captivate users.
    6. **Best Practices for Protection**: Recommend proven strategies for conservation and inter-agency cooperation, such as community-based forest management, using tracking technologies, and compliance with global treaties.
    7. **User-Centric Experience**: Make every interaction informative and impactful, using a tone that emphasizes the significance of safeguarding sandalwood and the urgency of proactive measures.
    8. **Closing with Impact**: Conclude conversations with a strong, positive and concise closing statement, highlighting the importance of protecting sandalwood resources and encouraging further engagement with the cause.
    9. **No Speculation**: Stay factual and avoid speculative answers, ensuring all responses are rooted in the provided context and available knowledge.
    10. **Appreciation Handling**: If the question is gratitude or appreciation, respond appropriately with proper acknowledgment or a friendly closing message like "You're welcome! I'm glad I could help." or "Thank you for your kind words! Feel free to ask more questions anytime.". Avoid adding new or unrelated information unless explicitly requested.

    Your mission: to empower users and make critical information about sandalwood theft and conservation accessible, engaging, and effective to the user's query.

    ### INPUT STRUCTURE:
    - **Previous Conversation**:
    {conversation_history}

    - **CONTEXT**: 
    {{context}}

    - **QUESTION**: 
    {{question}}
    """

    return PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

# Function to generate question
async def generate(quest, conversation_history):
    prompt = f"""
    You are Santal.AI, an expert in simplifying any questions related to Sandalwood and Red sandalwood. Your task is to transform the given question into a clear, concise, and focused query that will effectively retrieve relevant information from the vector database, without adding any elaboration or explanation.

    QUESTION: {quest}
    CONVERSATION HISTORY: {conversation_history}
    """

    try:
        response = llm._call(prompt)
        return response
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

# Function to handle query
async def handle_query(quest, conversation_history):
    question = await generate(quest, conversation_history)
    if not question:
        return "Failed to generate a question."

    PROMPT = get_prompt_template(conversation_history)

    # Create a retrieval-based QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(score_threshold=0.9),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    # Get response
    response = qa_chain({"query": question})
    return response.get('result', "I don't know.") 