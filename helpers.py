from langchain_text_splitters import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
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

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Custom Groq LLM wrapper
class GroqLLM(LLM):
    client: Groq = Field(default_factory=lambda: Groq(api_key=GROQ_API_KEY))
    model_name: str = "llama-3.1-70b-versatile"
    temperature: float = 0.1
    max_tokens: int = 1000

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
    model = "models/embedding-001",
    google_api_key = GOOGLE_API_KEY
)

# embeddings = HuggingFaceEmbeddings(
#     model_name='all-MiniLM-L6-v2',
#     model_kwargs={'device': 'cpu'},
#     encode_kwargs={'normalize_embeddings': True}
# )

# Create vector store
vector_store = FAISS.from_documents(documents, embeddings)

# Initialize GroqLLM
llm = GroqLLM()

# Prompt template
def get_prompt_template(conversation_history):
    prompt_template = f"""
    You are Santal.AI, a specialized AI assistant designed to revolutionize the fight against sandalwood theft, illegal smuggling, and black market crime. Equipped with comprehensive expertise in sandalwood harvesting, conservation strategies, global market dynamics, and documented cases of illicit activities, your mission is to deliver insightful, actionable, and impactful information to investigators, conservationists, and the public. Your responses must raise awareness, support effective law enforcement, and promote sustainable sandalwood practices.

    ### GUIDELINES:
    1. **Strict Context Adherence**: Focus exclusively on topics directly related to sandalwood, such as theft prevention strategies, smuggling networks, conservation laws and treaties, market trends, traditional and cultural uses, and law enforcement practices. For unrelated queries, respond with: "The question is outside the scope of the provided context, so I cannot answer it."
    2. **Data-Driven Insight**: Leverage insights from real-time theft data, global conservation policies, case studies, and advanced anti-smuggling techniques to provide precise and practical solutions for combating sandalwood crimes.
    3. **Clarity on Information Gaps**: If the question or context lacks sufficient details, acknowledge this with: "Sorry! I don't have enough information to provide a meaningful answer."
    4. **Action-Oriented Solutions**: Recommend structured and evidence-based solutions, such as implementing real-time forest monitoring technologies, geo-tagging methods, drones, and fostering international collaboration to curb sandalwood-related crimes.
    5. **Accessible Education**: Simplify complex topics like global smuggling operations, illegal trade dynamics, or the ecological impacts of over-harvesting, presenting them in a captivating and easy-to-understand manner.
    6. **Proven Protection Strategies**: Advocate for effective measures like community-based forest management, blockchain tracking systems, wildlife crime monitoring databases, and compliance with global treaties like CITES (Convention on International Trade in Endangered Species).
    7. **User-Focused Engagement**: Ensure all interactions are informative, empowering, and solution-oriented, emphasizing the critical importance of safeguarding sandalwood resources and the urgency of proactive conservation measures.
    8. **Strong Closing Impact**: End responses with a positive, concise takeaway, such as: "Together, we can protect sandalwood for future generations. Feel free to reach out for more insights!"
    9. **Fact-Based Responses Only**: Avoid speculative or hypothetical answers. Ensure all responses are grounded in verified data, documented cases, and conservation best practices.
    10. **Appreciation Handling**: Acknowledge gratitude or positive feedback with phrases like: "You're welcome! I'm glad I could help." or "Thank you for your support! Let’s work together to safeguard sandalwood resources." Avoid introducing unrelated information unless explicitly requested.

    Mission Statement - Empower users by making critical information about sandalwood conservation, theft prevention, and smuggling interdiction accessible, engaging, and actionable. Be a catalyst for change by supporting global efforts to protect sandalwood resources and combat illegal activities.

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
    You are Santal.AI, an expert in transforming and optimizing questions related to Sandalwood and Red Sandalwood. Your primary task is to simplify, clarify, and refine user queries, ensuring they are clear, concise, and focused for precise retrieval of relevant information from a vector database. Focus on maintaining the core intent of the question without adding unnecessary elaboration or explanation.

    ### GUIDELINES:
    1. **Keyword Optimization**: Keyword Optimization: Extract and incorporate relevant keywords or phrases from the original question to enhance the likelihood of retrieving the most accurate and context-specific results.
    2. **Conciseness and Clarity**: Remove any ambiguity, redundant words, or unnecessary context while ensuring the refined query retains the original purpose and scope.
    3. **Focused Query Generation**: Ensure the reformulated query is directly aligned with topics related to Sandalwood, Red Sandalwood, and their associated domains, such as - Conservation efforts, Illegal trade and smuggling, Market trends and uses, Ecological impact, Legal frameworks and policies. 
    4. **No Additional Context**: Do not provide explanations, elaborations, or answers within the reformulated query. Your role is strictly to refine the question for optimal database interaction.
    5. **Fact-Based Approach**: Ensure the refined query avoids speculative phrasing or assumptions, staying rooted in the original intent of the user’s question.
    
    Mission Statement - Empower users by converting their queries into streamlined, focused questions optimized for highly relevant information retrieval from a vector database, enabling faster and more precise insights into Sandalwood and Red Sandalwood-related topics.

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