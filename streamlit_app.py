# query.py
from backend.persistence import load_faiss
from backend.rag_pipeline import build_retrieval_qa

def query(index_name, question):
vs = load_faiss(index_name)
if vs is None:
raise RuntimeError("Index not found; run build_index.py first")
qa = build_retrieval_qa(vs)
return qa.run(question)

if __name__ == "__main__":
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--index", default="default")
parser.add_argument("--q", required=True)
args = parser.parse_args()
print(query(args.index, args.q))
	
