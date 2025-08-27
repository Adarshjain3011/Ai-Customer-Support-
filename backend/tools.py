from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load PDF and prepare retriever
loader = PyPDFLoader("D:\\Projects\\GenAI\\Genai_projects\\SupportCrew\\FAQ.pdf")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(docs)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(texts, embeddings)
retriever = db.as_retriever()

# Create LangChain Retriever Tool
kb_lookup_tool = create_retriever_tool(
    retriever=retriever,
    name="KnowledgeBaseRetriever",
    description="Useful for fetching information from EarnIn's internal FAQ knowledge base."
)
