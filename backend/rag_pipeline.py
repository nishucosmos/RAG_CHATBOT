# backend/rag_pipeline.py
import os
from dotenv import load_dotenv
load_dotenv()

from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub

EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def build_vectorstore_from_docs(docs):
embeddings = HuggingFaceEmbeddings(model_name=EMB_MODEL)
return FAISS.from_documents(docs, embeddings)

def build_retrieval_qa(vectorstore, k=3):
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
raise ValueError("HUGGINGFACEHUB_API_TOKEN missing in .env")

llm = HuggingFaceHub(
repo_id="google/flan-t5-base",
huggingfacehub_api_token=hf_token,
model_kwargs={"temperature":0.3, "max_length":256}
)
retriever = vectorstore.as_retriever(search_kwargs={"k": k})
return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
	
