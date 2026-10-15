"""
Automated unit tests for llm_service.py prompts and generation routines.
"""

from llm_service import (
    generate_rag_response,
    explain_concept,
    summarize_content,
    generate_flashcards,
    generate_quiz,
    generate_study_plan,
    mock_interview_turn,
)

def test_generate_rag_response():
    chunks = [
        {
            "text": "ChromaDB stores embeddings alongside rich metadata for nearest neighbor search.",
            "source": "chroma_doc.txt",
            "page": 1,
            "similarity": 0.92,
        }
    ]
    res = generate_rag_response("What does ChromaDB store?", chunks)
    assert "answer" in res
    assert len(res["answer"]) > 20
    assert len(res["citations"]) == 1
    assert res["citations"][0]["source"] == "chroma_doc.txt"

def test_explain_concept():
    # Beginner mode
    beg_res = explain_concept("Cosine Similarity", mode="beginner")
    assert "Analogy" in beg_res or "Big Picture" in beg_res or "Similarity" in beg_res

    # Interview mode
    int_res = explain_concept("Cosine Similarity", mode="interview")
    assert "Trade-offs" in int_res or "Architecture" in int_res or "Interview" in int_res

def test_summarize_content():
    text = (
        "Neural networks consist of layers of interconnected artificial neurons. "
        "Each connection has an associated weight adjusted during training. "
        "Activation functions introduce non-linearity into the network."
    )
    for style in ("bullet", "executive", "cheat_sheet"):
        out = summarize_content(text, style=style)
        assert len(out) > 30

def test_generate_flashcards():
    text = "RAG combines vector retrieval with large language model generation."
    cards = generate_flashcards(text, count=3)
    assert isinstance(cards, list)
    assert len(cards) >= 1
    assert "question" in cards[0]
    assert "answer" in cards[0]

def test_generate_quiz():
    text = "Transformers rely entirely on self-attention mechanisms to model relationships."
    quiz = generate_quiz(text, num_questions=2)
    assert isinstance(quiz, list)
    assert len(quiz) >= 1
    assert "options" in quiz[0]
    assert "correct_answer" in quiz[0]
    assert len(quiz[0]["options"]) == 4

def test_generate_study_plan():
    plan = generate_study_plan("GenAI RAG", duration_days=5, hours_per_day=2.0)
    assert "Roadmap" in plan or "Milestone" in plan
    assert "Day" in plan

def test_mock_interview_turn():
    history = [{"speaker": "Interviewer", "text": "What is an embedding?"}]
    user_ans = "An embedding is a dense mathematical vector representing semantic text meaning."
    eval_res = mock_interview_turn("RAG", "GenAI Engineer", history, user_ans)
    assert "feedback" in eval_res
    assert "next_question" in eval_res
    assert len(eval_res["feedback"]) > 10
