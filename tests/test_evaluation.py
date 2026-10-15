"""
Automated unit tests for evaluation.py metrics.
"""

from evaluation import (
    calculate_hit_rate,
    calculate_reciprocal_rank,
    calculate_precision_at_k,
    calculate_groundedness_score,
    calculate_answer_relevancy,
    run_retrieval_benchmark,
)

def test_retrieval_metrics():
    retrieved = [
        {"text": "Retrieval Augmented Generation with ChromaDB", "source": "rag.txt"},
        {"text": "Gradient descent optimization in neural networks", "source": "opt.txt"},
    ]
    keywords = ["chromadb", "retrieval"]

    # Hit Rate should be 1.0 (found in first chunk)
    hit = calculate_hit_rate(retrieved, keywords)
    assert hit == 1.0

    # MRR should be 1.0 (rank 1)
    mrr = calculate_reciprocal_rank(retrieved, keywords)
    assert mrr == 1.0

    # Precision@2
    p2 = calculate_precision_at_k(retrieved, keywords, k=2)
    assert p2 == 0.5

    # If keyword is not present
    assert calculate_hit_rate(retrieved, ["unrelated_topic"]) == 0.0
    assert calculate_reciprocal_rank(retrieved, ["unrelated_topic"]) == 0.0

def test_groundedness_score():
    context = [{"text": "Transformers employ multi-head self-attention mechanisms."}]
    answer_grounded = "Transformers use self-attention mechanisms to learn."
    answer_hallucinated = "Quantum teleportation enables subatomic particles to communicate instantly."

    score_grounded = calculate_groundedness_score(answer_grounded, context)
    score_hallucinated = calculate_groundedness_score(answer_hallucinated, context)

    assert score_grounded > score_hallucinated
    assert 0.0 <= score_grounded <= 1.0
    assert 0.0 <= score_hallucinated <= 1.0

def test_answer_relevancy():
    query = "What is Cosine Similarity?"
    relevant_answer = "Cosine similarity measures the angle between vectors."
    irrelevant_answer = "Cooking pasta requires boiling salted water."

    rel_score = calculate_answer_relevancy(query, relevant_answer)
    irrel_score = calculate_answer_relevancy(query, irrelevant_answer)

    assert rel_score > irrel_score

def test_benchmark_runner():
    test_cases = [
        {"query": "Cosine Similarity", "expected_keywords": ["cosine", "similarity"]}
    ]
    report = run_retrieval_benchmark(test_cases=test_cases, top_k=2)
    assert "summary" in report
    assert "hit_rate" in report["summary"]
    assert "mean_reciprocal_rank" in report["summary"]
    assert "avg_latency_ms" in report["summary"]
