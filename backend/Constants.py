import os

# ── Base path (relative to this file) ─────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── File Paths ─────────────────────────────────────────────────────────────────
CORPUS_PATH  = os.path.join(BASE_DIR, "data", "corpus.json")
CHUNKS_PATH  = os.path.join(BASE_DIR, "data", "chunks.json")
INDEX_PATH   = os.path.join(BASE_DIR, "faiss_index", "index.faiss")
INDEX_DIR    = os.path.join(BASE_DIR, "faiss_index")

# ── Model Names ────────────────────────────────────────────────────────────────
EMBED_MODEL_NAME  = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL_NAME    = "Qwen/Qwen2.5-0.5B-Instruct"

# ── RAG Parameters ─────────────────────────────────────────────────────────────
TOP_K_RETRIEVE = 10   # candidates fetched from FAISS before reranking
TOP_K_FINAL    = 5    # final chunks returned (coursework max = 5)
CHUNK_SIZE     = 300  # words per chunk
CHUNK_OVERLAP  = 50   # word overlap between chunks
MAX_NEW_TOKENS = 300  # max tokens for LLM generation