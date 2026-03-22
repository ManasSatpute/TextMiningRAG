import os
import json
import faiss
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from Constants import (CORPUS_PATH, INDEX_DIR, CHUNKS_PATH, EMBED_MODEL_NAME)
from chunking import load_and_chunk_corpus

# Method to Load embedding model
def load_model():
    print("🔹 Loading embedding model...")
    model = SentenceTransformer(EMBED_MODEL_NAME)
    return model


# -----------------------------
# Embed chunks
# -----------------------------
def embed_chunks(model, chunks: List[Dict]) -> np.ndarray:
    print("🔹 Generating embeddings...")

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # important for cosine similarity
    )

    return embeddings


# Method to Build FAISS index
def build_faiss_index(embeddings: np.ndarray):
    dim = embeddings.shape[1]

    print(f"🔹 Building FAISS index (dim={dim})...")

    index = faiss.IndexFlatIP(dim)  # cosine similarity (since normalized)
    index.add(embeddings)

    return index


# Method to Save FAISS index and metadata
def save_index(index, chunks: List[Dict]):
    os.makedirs(INDEX_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(CHUNKS_PATH), exist_ok=True)

    index_path = os.path.join(INDEX_DIR, "index.faiss")

    print(f"💾 Saving FAISS index to {index_path}")
    faiss.write_index(index, index_path)

    print(f"💾 Saving chunks metadata to {CHUNKS_PATH}")
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)


# -----------------------------
# Main pipeline
# -----------------------------
def run_ingestion():
    print("🚀 Starting ingestion pipeline...")

    # 1. Chunk data
    chunks = load_and_chunk_corpus(CORPUS_PATH)
    print(f"✅ Total chunks: {len(chunks)}")

    # 2. Load model
    model = load_model()

    # 3. Embed
    embeddings = embed_chunks(model, chunks)

    # 4. Build index
    index = build_faiss_index(embeddings)

    # 5. Save
    save_index(index, chunks)

    print("🎉 Ingestion complete!")


# Method to run the ingestion pipeline
if __name__ == "__main__":
    run_ingestion()