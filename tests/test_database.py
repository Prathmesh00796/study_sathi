"""
Automated unit tests for database.py SQLite persistence layer.
"""

import os
import pytest
import database

TEST_DB_PATH = "data/test_studysathi.db"

@pytest.fixture(autouse=True)
def setup_and_teardown():
    # Setup fresh test database
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)
    database.init_db(TEST_DB_PATH)
    yield
    # Cleanup
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)

def test_chat_messages_lifecycle():
    # Save user message
    msg1_id = database.save_chat_message("sess_1", "user", "What is an embedding?", db_path=TEST_DB_PATH)
    assert msg1_id > 0

    # Save assistant message with citation
    citations = [{"source": "note.txt", "page": 1, "similarity": 0.88}]
    msg2_id = database.save_chat_message("sess_1", "assistant", "An embedding is a vector.", sources=citations, db_path=TEST_DB_PATH)
    assert msg2_id > msg1_id

    # Retrieve history
    history = database.get_chat_history("sess_1", db_path=TEST_DB_PATH)
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"
    assert len(history[1]["sources"]) == 1
    assert history[1]["sources"][0]["source"] == "note.txt"

    # Clear chat
    database.clear_chat_history("sess_1", db_path=TEST_DB_PATH)
    cleared = database.get_chat_history("sess_1", db_path=TEST_DB_PATH)
    assert len(cleared) == 0

def test_saved_notes_crud():
    note_id = database.save_note(
        title="Attention Mechanism",
        category="Concept",
        content="Transformer models use self-attention.",
        tags="nlp,transformers",
        db_path=TEST_DB_PATH,
    )
    assert note_id > 0

    # Retrieve all notes
    notes = database.get_saved_notes(db_path=TEST_DB_PATH)
    assert len(notes) == 1
    assert notes[0]["title"] == "Attention Mechanism"

    # Search filter
    filtered = database.get_saved_notes(search_query="nlp", db_path=TEST_DB_PATH)
    assert len(filtered) == 1

    # Delete note
    deleted = database.delete_note(note_id, db_path=TEST_DB_PATH)
    assert deleted is True
    assert len(database.get_saved_notes(db_path=TEST_DB_PATH)) == 0

def test_flashcards_lifecycle():
    card_id = database.save_flashcard(
        deck_name="GenAI",
        question="What is RAG?",
        answer="Retrieval-Augmented Generation",
        difficulty="Easy",
        db_path=TEST_DB_PATH,
    )
    assert card_id > 0

    cards = database.get_flashcards(deck_name="GenAI", db_path=TEST_DB_PATH)
    assert len(cards) == 1
    assert cards[0]["mastered"] is False
    assert cards[0]["review_count"] == 0

    # Update review
    database.update_flashcard_review(card_id, mastered=True, db_path=TEST_DB_PATH)
    updated = database.get_flashcards(deck_name="GenAI", db_path=TEST_DB_PATH)
    assert updated[0]["mastered"] is True
    assert updated[0]["review_count"] == 1

    # Delete card
    deleted = database.delete_flashcard(card_id, db_path=TEST_DB_PATH)
    assert deleted is True
    assert len(database.get_flashcards(db_path=TEST_DB_PATH)) == 0

def test_study_plans_and_quizzes():
    # Study plan
    plan_id = database.save_study_plan("LangChain", 7, "GenAI Engineer", "# Roadmap", db_path=TEST_DB_PATH)
    assert plan_id > 0
    plans = database.get_study_plans(db_path=TEST_DB_PATH)
    assert len(plans) == 1
    assert plans[0]["topic"] == "LangChain"

    # Quiz result
    quiz_id = database.save_quiz_result("RAG Concepts", 4, 5, '["q1", "q2"]', db_path=TEST_DB_PATH)
    assert quiz_id > 0
    history = database.get_quiz_history(db_path=TEST_DB_PATH)
    assert len(history) == 1
    assert history[0]["score"] == 4
