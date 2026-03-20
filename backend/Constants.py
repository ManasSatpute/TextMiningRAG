from pathlib import Path

# project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Directory Paths
BACKEND_DIR = BASE_DIR / "backend"
DATA_DIR = BACKEND_DIR / "data"
INDEX_DIR = BACKEND_DIR / "faiss_index"

# File Paths
CORPUS_PATH = DATA_DIR / "corpus.json"
INDEX_PATH = INDEX_DIR / "index.faiss"
CHUNKS_PATH = DATA_DIR / "chunks.json"

# Model Names
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TRANSFORMER_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"