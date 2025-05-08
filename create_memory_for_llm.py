import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) MyBot/1.0"

# Load PDFs
def load_pdf_files(data_path):
    loader = DirectoryLoader(data_path, glob='*.pdf', loader_cls=PyPDFLoader)
    return loader.load()

# Load websites
def load_web_pages(url_list):
    loader = WebBaseLoader(url_list)
    return loader.load()

# Create text chunks
def create_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)

# Get embedding model
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Process both PDF and web content
DATA_PATH = "data/"
URL_LIST = [
    "https://www.marketwatch.com/investing/stock/aapl",
    "https://www.moneycontrol.com/india/stockpricequote/auto-carsjeeps/mahindramahindra/MM"
]

# Load data
pdf_docs = load_pdf_files(DATA_PATH)
web_docs = load_web_pages(URL_LIST)
all_docs = pdf_docs + web_docs

# Create chunks
text_chunks = create_chunks(all_docs)

# Embeddings and FAISS
embedding_model = get_embedding_model()
db = FAISS.from_documents(text_chunks, embedding_model)

# Save vector DB
DB_FAISS_PATH = "vectorstore/db_faiss"
db.save_local(DB_FAISS_PATH)
