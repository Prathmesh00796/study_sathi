"""
Configuration module for StudySathi.
Loads environment variables and sets sensible application-wide defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base directories
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

# Load environment variables if .env exists
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    load_dotenv()

# Ollama local settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

# Storage paths
CHROMA_PERSIST_DIRECTORY = str(BASE_DIR / os.getenv("CHROMA_PERSIST_DIRECTORY", "data/chroma_db"))
SQLITE_DB_PATH = str(BASE_DIR / os.getenv("SQLITE_DB_PATH", "data/studysathi.db"))
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# RAG & Chunking settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "4"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.6"))

# Demo fallback mode (ensures app works for interviews/testing even if Ollama is not yet running)
DEMO_MODE_FALLBACK = os.getenv("DEMO_MODE_FALLBACK", "true").lower() in ("true", "1", "yes")

# Available Ollama models list for UI selection
SUPPORTED_MODELS = [
    "llama3.2",
    "llama3.1",
    "llama3",
    "mistral",
    "phi3",
    "gemma2",
    "qwen2.5",
]
