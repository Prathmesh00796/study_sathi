"""
Automated unit tests for document_loader.py parser and chunker.
"""

from pathlib import Path
from document_loader import (
    load_document,
    chunk_documents,
    extract_text_from_txt,
    extract_text_from_docx,
    extract_text_from_pdf,
)

def test_extract_text_from_txt():
    sample_text = b"Hello StudySathi.\nThis is a test document."
    docs = extract_text_from_txt(sample_text, filename="test.txt")
    assert len(docs) == 1
    assert "Hello StudySathi" in docs[0]["text"]
    assert docs[0]["metadata"]["source"] == "test.txt"

def test_load_sample_notes():
    # Test TXT
    txt_path = Path("sample_notes/genai_and_rag_handbook.txt")
    if txt_path.exists():
        docs = load_document(txt_path)
        assert len(docs) > 0
        assert "Retrieval-Augmented Generation" in docs[0]["text"]

    # Test DOCX
    docx_path = Path("sample_notes/machine_learning_foundations.docx")
    if docx_path.exists():
        docs = load_document(docx_path)
        assert len(docs) > 0
        assert "Bias-Variance Tradeoff" in docs[0]["text"]

    # Test PDF
    pdf_path = Path("sample_notes/deep_learning_and_transformers.pdf")
    if pdf_path.exists():
        docs = load_document(pdf_path)
        assert len(docs) > 0
        assert "Transformer" in docs[0]["text"]

def test_chunk_documents():
    long_text = "This is a detailed sentence about vector spaces. " * 40
    raw_docs = [{"text": long_text, "metadata": {"source": "manual.txt", "page": 1}}]
    chunks = chunk_documents(raw_docs, chunk_size=300, chunk_overlap=50)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk["text"]) <= 350
        assert "source" in chunk["metadata"]
        assert "chunk_id" in chunk["metadata"]
        assert chunk["metadata"]["source"] == "manual.txt"
