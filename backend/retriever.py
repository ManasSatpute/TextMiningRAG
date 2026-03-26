import json
import faiss
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer, CrossEncoder
from Constants import (
    CHUNKS_PATH,
    EMBEDDING_MODEL_NAME,
    RERANK_MODEL_NAME,
    INDEX_PATH,
    TOP_K_RETRIEVE,
    TOP_K_FINAL
)


class Retriever:
    def __init__(self):
        print("Loading FAISS index...")
        self.index = faiss.read_index(str(INDEX_PATH))

        print("Loading chunks...")
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        print("Loading embedding model...")
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        print("Loading reranker...")
        self.reranker = CrossEncoder(RERANK_MODEL_NAME)

        print("Retriever ready.\n")

    def embed_query(self, query: str) -> np.ndarray:
        """Convert query string to a normalised embedding vector."""
        return self.embed_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

    def retrieve(self, query: str, top_k: int = TOP_K_RETRIEVE) -> List[Dict]:
        """
        Stage 1: FAISS similarity search.
        Returns top_k candidate chunks with their cosine similarity scores.
        """
        query_vec = self.embed_query(query)
        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:   # FAISS returns -1 for empty slots
                continue
            chunk = dict(self.chunks[idx])  # copy so we don't mutate original
            chunk["retrieval_score"] = float(score)
            results.append(chunk)

        return results

    def rerank(self, query: str, candidates: List[Dict], top_k: int = TOP_K_FINAL) -> List[Dict]:
        """
        Stage 2: CrossEncoder reranking.
        Scores each (query, chunk) pair more precisely and returns top_k.
        """
        if not candidates:
            return []

        pairs = [(query, c["text"]) for c in candidates]
        scores = self.reranker.predict(pairs)

        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)

        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]

    def get_top_chunks(self, query: str, top_k: int = TOP_K_FINAL) -> List[Dict]:
        """
        Full two-stage pipeline: retrieve then rerank.
        Returns top_k chunks in the required output format:
            {"doc_id": "...", "text": "..."}
        """
        candidates = self.retrieve(query, top_k=TOP_K_RETRIEVE)
        reranked   = self.rerank(query, candidates, top_k=top_k)

    
        return [
            {
                "doc_id": chunk["id"],
                "text":   chunk["text"]
            }
            for chunk in reranked
        ]


if __name__ == "__main__":
    retriever = Retriever()

    test_queries = [
        "What ingredients are used in Sylheti Biryani?",
        "How is Haleem prepared?",
        "What is Dal Bhat?",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 50)
        results = retriever.get_top_chunks(query)
        for i, r in enumerate(results, 1):
            print(f"  [{i}] doc_id: {r['doc_id'][:8]}...")
            print(f"       {r['text'][:120]}...")