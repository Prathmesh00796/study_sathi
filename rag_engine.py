"""
RAG Engine module for StudySathi.
Manages ChromaDB vector persistence, embeddings (Ollama + local fallback), and semantic retrieval.
"""

import math
import re
import requests
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from config import (
    CHROMA_PERSIST_DIRECTORY,
    OLLAMA_BASE_URL,
    EMBEDDING_MODEL,
    TOP_K_RESULTS,
)

import time

# ChromaDB collection identifier
COLLECTION_NAME = "studysathi_collection"

# Connectivity check cache
_LAST_OLLAMA_CHECK = 0.0
_OLLAMA_ONLINE = None

def is_ollama_online(base_url: str = OLLAMA_BASE_URL) -> bool:
    """Checks Ollama connection status with 10-second caching to prevent timeout overhead."""
    global _LAST_OLLAMA_CHECK, _OLLAMA_ONLINE
    now = time.time()
    if _OLLAMA_ONLINE is not None and (now - _LAST_OLLAMA_CHECK) < 10.0:
        return _OLLAMA_ONLINE
    try:
        res = requests.get(f"{base_url}/api/tags", timeout=0.8)
        _OLLAMA_ONLINE = (res.status_code == 200)
    except Exception:
        _OLLAMA_ONLINE = False
    _LAST_OLLAMA_CHECK = now
    return _OLLAMA_ONLINE


# Generate embedding vector via Ollama REST API with timeout protection
def get_ollama_embedding(text: str, model: str = EMBEDDING_MODEL, base_url: str = OLLAMA_BASE_URL) -> Optional[List[float]]:
    """Requests an embedding vector from Ollama; returns None if Ollama is unreachable."""
    if not is_ollama_online(base_url):
        return None
    try:
        url = f"{base_url}/api/embeddings"
        payload = {"model": model, "prompt": text}
        response = requests.post(url, json=payload, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get("embedding")
    except Exception:
        pass
    return None


# Generate deterministic lightweight dense vector when Ollama is unavailable
def get_fallback_embedding(text: str, dimension: int = 128) -> List[float]:
    """Generates a normalized hash-based bag-of-words vector for fast offline testing."""
    vector = [0.0] * dimension
    tokens = re.findall(r"\w+", text.lower())
    if not tokens:
        return [0.0] * dimension

    for token in tokens:
        # Distribute token character values into vector dimensions
        idx = hash(token) % dimension
        vector[idx] += 1.0

    # L2 normalize the vector
    norm = math.sqrt(sum(v * v for v in vector))
    if norm > 0:
        vector = [v / norm for v in vector]
    return vector


# Generate embedding vector using Ollama if online, falling back to local vectorizer
def get_embedding(text: str) -> List[float]:
    """Tries Ollama first, then falls back to fast local deterministic embedding."""
    vector = get_ollama_embedding(text)
    if vector is not None and len(vector) > 0:
        return vector
    return get_fallback_embedding(text)


from chromadb import EmbeddingFunction

# Custom embedding function wrapper for ChromaDB integration
class StudySathiEmbeddingFunction(EmbeddingFunction):
    """ChromaDB compatible embedding class that handles both Ollama and offline states."""

    def __init__(self):
        pass

    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        return [get_embedding(text) for text in input]

    def name(self) -> str:
        return "studysathi_embedder"


# Initialize and return the persistent ChromaDB collection
def get_collection(persist_dir: str = CHROMA_PERSIST_DIRECTORY) -> chromadb.Collection:
    """Connects to the persistent ChromaDB database and returns the active collection."""
    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False, is_persistent=True),
    )
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "StudySathi document notes collection"},
        embedding_function=StudySathiEmbeddingFunction(),
    )
    return collection


# Index document chunks into the ChromaDB vector database
def add_chunks_to_vectorstore(chunks: List[Dict[str, Any]], persist_dir: str = CHROMA_PERSIST_DIRECTORY) -> int:
    """Adds chunk texts, IDs, and metadata into ChromaDB in batches."""
    if not chunks:
        return 0

    collection = get_collection(persist_dir)

    ids = [chunk["id"] for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = []

    for chunk in chunks:
        # ChromaDB metadata values must be primitive types (str, int, float, bool)
        meta = {}
        for k, v in chunk.get("metadata", {}).items():
            if isinstance(v, (str, int, float, bool)):
                meta[k] = v
            else:
                meta[k] = str(v)
        metadatas.append(meta)

    # Insert in batches of 100 for memory safety
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch_ids = ids[i : i + batch_size]
        batch_docs = documents[i : i + batch_size]
        batch_metas = metadatas[i : i + batch_size]
        collection.upsert(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_metas,
        )

    return len(chunks)


# Search ChromaDB for the most semantically relevant chunks given a user query
def search_relevant_chunks(
    query: str,
    top_k: int = TOP_K_RESULTS,
    persist_dir: str = CHROMA_PERSIST_DIRECTORY,
) -> List[Dict[str, Any]]:
    """Performs semantic vector search and formats retrieved results with source citations."""
    collection = get_collection(persist_dir)
    count = collection.count()

    if count == 0:
        return []

    actual_k = min(top_k, count)
    results = collection.query(
        query_texts=[query],
        n_results=actual_k,
        include=["documents", "metadatas", "distances"],
    )

    matched_chunks = []
    if results and "documents" in results and results["documents"]:
        docs = results["documents"][0]
        metas = results["metadatas"][0] if "metadatas" in results else [{}] * len(docs)
        distances = results["distances"][0] if "distances" in results else [0.0] * len(docs)

        for doc, meta, dist in zip(docs, metas, distances):
            # Compute a similarity score between 0.0 and 1.0 (cosine or l2 distance converted)
            similarity = round(max(0.0, 1.0 - (dist / 2.0)), 3)
            matched_chunks.append({
                "text": doc,
                "metadata": meta,
                "distance": dist,
                "similarity": similarity,
                "source": meta.get("source", "Unknown Document"),
                "page": meta.get("page", 1),
            })

    return matched_chunks


# Return current summary metrics of the vector store
def get_collection_stats(persist_dir: str = CHROMA_PERSIST_DIRECTORY) -> Dict[str, Any]:
    """Returns total chunk count and unique document sources indexed."""
    try:
        collection = get_collection(persist_dir)
        total_chunks = collection.count()
        sources = set()

        if total_chunks > 0:
            sample = collection.get(include=["metadatas"])
            if sample and "metadatas" in sample and sample["metadatas"]:
                for m in sample["metadatas"]:
                    if m and "source" in m:
                        sources.add(m["source"])

        return {
            "total_chunks": total_chunks,
            "total_documents": len(sources),
            "sources": sorted(list(sources)),
        }
    except Exception as e:
        return {
            "total_chunks": 0,
            "total_documents": 0,
            "sources": [],
            "error": str(e),
        }


# Reset and clear all indexed document embeddings from ChromaDB
def clear_vectorstore(persist_dir: str = CHROMA_PERSIST_DIRECTORY) -> bool:
    """Deletes and recreates the ChromaDB collection."""
    try:
        client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False, is_persistent=True),
        )
        client.delete_collection(name=COLLECTION_NAME)
        client.create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "StudySathi document notes collection"},
            embedding_function=StudySathiEmbeddingFunction(),
        )
        return True
    except Exception:
        return False
