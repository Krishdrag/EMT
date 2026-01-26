from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

DATA_DIR = "data"
DB_DIR = "vectorstore"

docs = []

for file in os.listdir(DATA_DIR):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(DATA_DIR, file))
        docs.extend(loader.load())

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(chunks, embeddings)

db.save_local(DB_DIR)
print("✅ PDFs indexed into vectorstore!")
