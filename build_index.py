# build_index.py
from backend.utils import load_pdf_to_docs
from backend.rag_pipeline import build_vectorstore_from_docs
from backend.persistence import save_faiss

def build_index(pdf_path, index_name="default"):
docs = load_pdf_to_docs(pdf_path)
print(f"[+] Split into {len(docs)} chunks")
vs = build_vectorstore_from_docs(docs)
path = save_faiss(vs, name=index_name)
print(f"[+] Index saved at {path}")

if __name__ == "__main__":
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--pdf", required=True, help="Path to PDF")
parser.add_argument("--name", default="default")
args = parser.parse_args()
build_index(args.pdf, args.name)
	
