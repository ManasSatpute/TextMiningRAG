from pathlib import Path

# Base Path
BASE_DIR    = Path(__file__).resolve().parent
DATA_DIR    = BASE_DIR / "data"
INDEX_DIR   = BASE_DIR / "faiss_index"

# File Paths
CORPUS_PATH = DATA_DIR / "corpus.json"
CHUNKS_PATH = DATA_DIR / "chunks.json"
INDEX_PATH  = INDEX_DIR / "index.faiss"

# Model Names
EMBED_MODEL_NAME      = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL_NAME     = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TRANSFORMER_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME        = "Qwen/Qwen2.5-0.5B-Instruct"

# RAG Parameters
TOP_K_RETRIEVE = 10
TOP_K_FINAL    = 5    
CHUNK_SIZE     = 300  
CHUNK_OVERLAP  = 50   
MAX_NEW_TOKENS = 300  