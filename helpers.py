from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import dotenv
import os

dotenv.load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

loader = TextLoader("data.txt")
docs = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=500)
documents = text_splitter.split_documents(docs)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY,
)

# Create the FAISS vector store
vector_store = FAISS.from_documents(documents, embeddings)

# Initialize the OpenAI model
llm = OpenAI(api_key=OPENAI_API_KEY, model_name="gpt-4o")

def get_prompt_template(conversation_history):
    prompt_template = f"""
    You are an experienced coding mentor specializing in various technical niches like web development, machine learning, blockchain, cybersecurity, and more. Your goal is to provide clear, practical, and expert advice on how to start learning coding and advance in these fields.

    Instructions for the AI:
    - Carefully analyze the given source documents and context. Use these sources as your primary reference to formulate detailed, expert-level responses that address the question comprehensively.
    - Combine insights from multiple sections of the provided context when necessary to offer a well-rounded and expert response and do provide answers with useful tokens and not rubbish tokens like '\\n'.
    - When responding, use as much relevant information from the "response" section of the source documents as possible, maintaining accuracy and detail but rephrase it in your own helpful comprehensive way.
    - If the context does not provide sufficient information or relevant details, respond with "I don't know."
    - Use the given source documents as your primary reference to answer questions about starting a career or learning path in these niches.
    - If specific information is AT ALL not available, minimally use your expertise to provide general guidance based on industry standard and best practices.
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
    You are Cody, an expert natural language analyser. Your goal is to rephrase and transform the given question into a single small concise question with proper context by analysing the conversation history, that can be used to query a vector database as well as generate a detailed, expert-level response from another LLM for the questioner. Just answer with the question without unnecessary titles or prompts. Also the question should be related to webdev, blockchain, cybersecurity, or machine learning, not in the context of any other field.

    QUESTION: {quest}
    CONVERSATION HISTORY: {conversation_history}
    """

    try:
        response = await llm.chat_completion(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}], 
            temperature=0.7,
            max_tokens=150
        )
        quest = response['choices'][0]['message']['content'].strip()
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
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(score_threshold=0.5),
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )

    response = qa_chain({"query": question})
    return response