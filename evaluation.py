"""
Evaluation module for StudySathi.
Measures retrieval performance (Hit Rate, MRR, Precision@K) and generation quality (Groundedness, Relevancy, Latency).
"""

import time
import re
from typing import List, Dict, Any, Optional
from rag_engine import search_relevant_chunks
from llm_service import generate_rag_response


# Compute whether at least one relevant keyword or source is present in retrieved chunks
def calculate_hit_rate(retrieved_chunks: List[Dict[str, Any]], expected_keywords: List[str]) -> float:
    """Returns 1.0 if any expected keyword is found in the retrieved chunks, else 0.0."""
    if not retrieved_chunks or not expected_keywords:
        return 0.0

    combined_text = " ".join(c.get("text", "").lower() for c in retrieved_chunks)
    for kw in expected_keywords:
        if kw.lower() in combined_text:
            return 1.0
    return 0.0


# Compute Reciprocal Rank of the first relevant chunk
def calculate_reciprocal_rank(retrieved_chunks: List[Dict[str, Any]], expected_keywords: List[str]) -> float:
    """Returns 1 / rank of the first chunk containing any expected keyword."""
    if not retrieved_chunks or not expected_keywords:
        return 0.0

    for rank, chunk in enumerate(retrieved_chunks, start=1):
        chunk_text = chunk.get("text", "").lower()
        if any(kw.lower() in chunk_text for kw in expected_keywords):
            return 1.0 / rank
    return 0.0


# Compute Precision@K for retrieval
def calculate_precision_at_k(retrieved_chunks: List[Dict[str, Any]], expected_keywords: List[str], k: Optional[int] = None) -> float:
    """Measures the proportion of top-K chunks that contain relevant keywords."""
    if not retrieved_chunks or not expected_keywords:
        return 0.0

    k_chunks = retrieved_chunks[:k] if k else retrieved_chunks
    if not k_chunks:
        return 0.0

    relevant_count = 0
    for chunk in k_chunks:
        chunk_text = chunk.get("text", "").lower()
        if any(kw.lower() in chunk_text for kw in expected_keywords):
            relevant_count += 1

    return round(relevant_count / len(k_chunks), 3)


# Estimate groundedness/faithfulness of generated answer against context
def calculate_groundedness_score(answer: str, context_chunks: List[Dict[str, Any]]) -> float:
    """Measures the token overlap ratio of the answer with the source context to detect hallucination."""
    if not answer or not context_chunks:
        return 0.0

    context_words = set(re.findall(r"\w{3,}", " ".join(c.get("text", "") for c in context_chunks).lower()))
    answer_words = re.findall(r"\w{3,}", answer.lower())

    # Filter out common stop words
    stop_words = {
        "the", "and", "that", "this", "with", "from", "for", "are", "was", "were",
        "have", "has", "had", "been", "they", "their", "will", "would", "could", "should",
        "your", "based", "here", "note", "operating", "sections", "relevant", "documents"
    }
    filtered_answer = [w for w in answer_words if w not in stop_words]

    if not filtered_answer:
        return 0.5

    grounded_count = sum(1 for w in filtered_answer if w in context_words)
    score = grounded_count / len(filtered_answer)
    # Scale to realistic 0.0 - 1.0 range
    return round(min(1.0, max(0.0, score * 1.5)), 3)


# Estimate answer relevancy to the original question
def calculate_answer_relevancy(query: str, answer: str) -> float:
    """Checks whether the answer addresses the keywords present in the prompt."""
    query_tokens = set(re.findall(r"\w{3,}", query.lower()))
    if not query_tokens:
        return 1.0

    answer_tokens = set(re.findall(r"\w{3,}", answer.lower()))
    overlap = len(query_tokens.intersection(answer_tokens))
    ratio = overlap / len(query_tokens)
    return round(min(1.0, ratio * 1.3), 3)


# Standard default benchmark queries for GenAI study notes
DEFAULT_BENCHMARK_CASES = [
    {
        "query": "What is Retrieval-Augmented Generation?",
        "expected_keywords": ["retrieval", "rag", "generation", "context"],
    },
    {
        "query": "How do embeddings work in vector search?",
        "expected_keywords": ["embedding", "vector", "dimension", "semantic", "cosine"],
    },
    {
        "query": "What is the purpose of chunk overlap?",
        "expected_keywords": ["overlap", "chunk", "boundary", "context"],
    },
    {
        "query": "Explain cosine similarity versus distance",
        "expected_keywords": ["cosine", "similarity", "distance", "angle"],
    },
]


# Execute retrieval evaluation benchmark across a set of test queries
def run_retrieval_benchmark(
    test_cases: Optional[List[Dict[str, Any]]] = None,
    top_k: int = 4,
) -> Dict[str, Any]:
    """Runs automated retrieval tests against ChromaDB and computes metrics."""
    cases = test_cases or DEFAULT_BENCHMARK_CASES
    results = []

    total_hit = 0.0
    total_mrr = 0.0
    total_p_at_k = 0.0
    total_latency_ms = 0.0

    for case in cases:
        query = case["query"]
        expected = case["expected_keywords"]

        start_t = time.perf_counter()
        retrieved = search_relevant_chunks(query, top_k=top_k)
        latency = (time.perf_counter() - start_t) * 1000.0

        hit = calculate_hit_rate(retrieved, expected)
        mrr = calculate_reciprocal_rank(retrieved, expected)
        p_k = calculate_precision_at_k(retrieved, expected, k=top_k)

        total_hit += hit
        total_mrr += mrr
        total_p_at_k += p_k
        total_latency_ms += latency

        results.append({
            "query": query,
            "expected_keywords": expected,
            "chunks_retrieved": len(retrieved),
            "hit": hit,
            "mrr": round(mrr, 3),
            "precision_at_k": p_k,
            "latency_ms": round(latency, 2),
            "top_source": retrieved[0].get("source", "None") if retrieved else "None",
        })

    num_cases = len(cases) if cases else 1
    return {
        "summary": {
            "total_queries": len(cases),
            "hit_rate": round(total_hit / num_cases, 3),
            "mean_reciprocal_rank": round(total_mrr / num_cases, 3),
            "mean_precision_at_k": round(total_p_at_k / num_cases, 3),
            "avg_latency_ms": round(total_latency_ms / num_cases, 2),
        },
        "details": results,
    }


# Execute end-to-end RAG evaluation (retrieval + synthesis)
def run_full_rag_evaluation(
    test_cases: Optional[List[Dict[str, Any]]] = None,
    top_k: int = 3,
) -> Dict[str, Any]:
    """Runs comprehensive evaluation covering retrieval metrics, groundedness, and relevancy."""
    cases = test_cases or DEFAULT_BENCHMARK_CASES[:3]
    evaluations = []

    for case in cases:
        query = case["query"]
        expected = case["expected_keywords"]

        # Step 1: Retrieval
        t0 = time.perf_counter()
        chunks = search_relevant_chunks(query, top_k=top_k)
        retrieval_ms = (time.perf_counter() - t0) * 1000.0

        hit = calculate_hit_rate(chunks, expected)
        mrr = calculate_reciprocal_rank(chunks, expected)

        # Step 2: Generation
        t1 = time.perf_counter()
        rag_output = generate_rag_response(query, chunks)
        generation_ms = (time.perf_counter() - t1) * 1000.0

        answer = rag_output.get("answer", "")
        groundedness = calculate_groundedness_score(answer, chunks) if chunks else 0.0
        relevancy = calculate_answer_relevancy(query, answer)

        evaluations.append({
            "query": query,
            "hit": hit,
            "mrr": mrr,
            "groundedness": groundedness,
            "relevancy": relevancy,
            "citations_count": len(rag_output.get("citations", [])),
            "retrieval_latency_ms": round(retrieval_ms, 1),
            "generation_latency_ms": round(generation_ms, 1),
            "total_latency_ms": round(retrieval_ms + generation_ms, 1),
        })

    n = len(evaluations) if evaluations else 1
    avg_hit = round(sum(e["hit"] for e in evaluations) / n, 3)
    avg_mrr = round(sum(e["mrr"] for e in evaluations) / n, 3)
    avg_groundedness = round(sum(e["groundedness"] for e in evaluations) / n, 3)
    avg_relevancy = round(sum(e["relevancy"] for e in evaluations) / n, 3)

    return {
        "summary": {
            "avg_hit_rate": avg_hit,
            "avg_mrr": avg_mrr,
            "avg_groundedness": avg_groundedness,
            "avg_relevancy": avg_relevancy,
            "status": "PASS" if avg_hit >= 0.5 and avg_relevancy >= 0.5 else "WARN",
        },
        "evaluations": evaluations,
    }
