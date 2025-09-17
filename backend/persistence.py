# backend/persistence.py
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

INDEX_DIR = "faiss_index"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def save_faiss(vectorstore, name="default"):
path = Path(INDEX_DIR) / name
path.mkdir(parents=True, exist_ok=True)
vectorstore.save_local(str(path))
return str(path)

def load_faiss(name="default"):
path = Path(INDEX_DIR) / name
if not path.exists():
return None
embeddings = HuggingFaceEmbeddings(model_name=EMB_MODEL)
return FAISS.load_local(str(path), embeddings)
	
