from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

# Load PDF and prepare retriever
# Use project-relative path to `FAQ.pdf`
pdf_path = os.path.join(os.path.dirname(__file__), "FAQ.pdf")
loader = PyPDFLoader(pdf_path)
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
