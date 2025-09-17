# backend/utils.py
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

def load_pdf_to_docs(pdf_path):
"""
Load PDF and split into text chunks.
"""
loader = PyPDFLoader(pdf_path)
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(
chunk_size=CHUNK_SIZE,
chunk_overlap=CHUNK_OVERLAP
)
return splitter.split_documents(docs)
	
