import json
import faiss
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer, CrossEncoder


# -----------------------------
# Config
# -----------------------------
INDEX_PATH = "D:\\MSc Adv Computer Science\\Transforming Text Into Meaning\\TextMiningRAG\\backend\\faiss_index\\index.faiss"
CHUNKS_PATH = "D:\\MSc Adv Computer Science\\Transforming Text Into Meaning\\TextMiningRAG\\backend\\data\\chunks.json"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# -----------------------------
# Load everything
# -----------------------------
class Retriever:
    def __init__(self):
        print("🔹 Loading FAISS index...")
        self.index = faiss.read_index(INDEX_PATH)

        print("🔹 Loading chunks...")
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        print("🔹 Loading embedding model...")
        self.embed_model = SentenceTransformer(EMBED_MODEL_NAME)

        print("🔹 Loading reranker...")
        self.reranker = CrossEncoder(RERANK_MODEL_NAME)

    # -------------------------
    # Embed query
    # -------------------------
    def embed_query(self, query: str):
        return self.embed_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

    # -------------------------
    # Retrieve top-k
    # -------------------------
    def retrieve(self, query: str, top_k: int = 10) -> List[Dict]:
        query_vec = self.embed_query(query)

        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            chunk = self.chunks[idx]
            chunk["retrieval_score"] = float(score)
            results.append(chunk)

        return results

    # -------------------------
    # Re-rank
    # -------------------------
    def rerank(self, query: str, candidates: List[Dict], top_k: int = 5):
        pairs = [(query, c["text"]) for c in candidates]

        scores = self.reranker.predict(pairs)

        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)

        # sort by rerank score
        candidates = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

        return candidates[:top_k]

    # -------------------------
    # Full pipeline
    # -------------------------
    def get_top_chunks(self, query: str, top_k: int = 5) -> List[Dict]:
        retrieved = self.retrieve(query, top_k=10)
        reranked = self.rerank(query, retrieved, top_k=top_k)
        return reranked


# -----------------------------
# Debug
# -----------------------------
if __name__ == "__main__":
    retriever = Retriever()

    query = "What ingredients are used in carbonara?"

    results = retriever.get_top_chunks(query)

    for i, r in enumerate(results):
        print(f"\n=== Result {i+1} ===")
        print(r["text"])
        print("Score:", r.get("rerank_score"))