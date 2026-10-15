"""
Automated unit tests for rag_engine.py vector store and embeddings.
"""

import os
import shutil
import pytest
from rag_engine import (
    get_fallback_embedding,
    get_embedding,
    add_chunks_to_vectorstore,
    search_relevant_chunks,
    get_collection_stats,
    clear_vectorstore,
)

TEST_CHROMA_DIR = "data/test_chroma_db"

@pytest.fixture(autouse=True)
def clean_test_vectorstore():
    # Setup
    if os.path.exists(TEST_CHROMA_DIR):
        shutil.rmtree(TEST_CHROMA_DIR, ignore_errors=True)
    yield
    # Teardown
    if os.path.exists(TEST_CHROMA_DIR):
        shutil.rmtree(TEST_CHROMA_DIR, ignore_errors=True)

def test_fallback_embedding():
    text = "Machine learning and artificial intelligence"
    vec = get_fallback_embedding(text, dimension=128)
    assert len(vec) == 128
    # Check that vector is L2 normalized (~1.0 magnitude)
    norm = sum(x * x for x in vec) ** 0.5
    assert abs(norm - 1.0) < 1e-4

def test_add_and_search_vectorstore():
    chunks = [
        {
            "id": "c1",
            "text": "Retrieval-Augmented Generation bridges knowledge gaps by querying a vector store.",
            "metadata": {"source": "rag_guide.txt", "page": 1},
        },
        {
            "id": "c2",
            "text": "Stochastic gradient descent updates parameters using single batch items.",
            "metadata": {"source": "ml_guide.txt", "page": 2},
        },
    ]

    added = add_chunks_to_vectorstore(chunks, persist_dir=TEST_CHROMA_DIR)
    assert added == 2

    stats = get_collection_stats(persist_dir=TEST_CHROMA_DIR)
    assert stats["total_chunks"] == 2
    assert stats["total_documents"] == 2

    # Query matching c1
    results = search_relevant_chunks("What is Retrieval-Augmented Generation?", top_k=2, persist_dir=TEST_CHROMA_DIR)
    assert len(results) > 0
    assert results[0]["source"] in ("rag_guide.txt", "ml_guide.txt")
    assert "similarity" in results[0]
    assert 0.0 <= results[0]["similarity"] <= 1.0
