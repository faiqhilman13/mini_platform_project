"""
Configuration constants for the RAG pipeline.
"""
import os
from pathlib import Path

# Model Names
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL_NAME = "gpt-3.5-turbo"

# Chunking Parameters
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Retrieval Parameters
INITIAL_RETRIEVAL_K = 20
FINAL_RETRIEVAL_K = 5

# Paths
# Assuming this file is in workflows/pipelines/
BASE_DIR = Path(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
DATA_DIR = BASE_DIR / "data"
VECTORSTORE_DIR = DATA_DIR / "vector_store"
DOCUMENTS_DIR = DATA_DIR / "documents"

# Ensure directories exist
def ensure_data_directories_exist():
    DATA_DIR.mkdir(exist_ok=True)
    VECTORSTORE_DIR.mkdir(exist_ok=True)
    DOCUMENTS_DIR.mkdir(exist_ok=True)

ensure_data_directories_exist() 