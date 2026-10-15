"""
Database module for StudySathi.
Handles SQLite storage for conversations, notes, flashcards, study plans, and quizzes.
"""

import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from config import SQLITE_DB_PATH


# Initialize the SQLite database and create tables if they do not exist
def init_db(db_path: str = SQLITE_DB_PATH) -> None:
    """Initializes tables for conversations, notes, flashcards, study plans, and quizzes."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. Chat conversations table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            sources TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # 2. Saved notes & study summaries
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS saved_notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            category TEXT NOT NULL,
            content TEXT NOT NULL,
            tags TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # 3. Flashcards table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS flashcards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            deck_name TEXT NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            difficulty TEXT DEFAULT 'Medium',
            mastered INTEGER DEFAULT 0,
            review_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # 4. Study plans table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS study_plans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT NOT NULL,
            duration_days INTEGER NOT NULL,
            target_role TEXT NOT NULL,
            plan_markdown TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # 5. Quizzes table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS quizzes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT NOT NULL,
            score INTEGER NOT NULL,
            total_questions INTEGER NOT NULL,
            quiz_json TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


# Save a single chat message (user or assistant) with optional citations
def save_chat_message(
    session_id: str,
    role: str,
    content: str,
    sources: Optional[List[Dict[str, Any]]] = None,
    db_path: str = SQLITE_DB_PATH,
) -> int:
    """Saves a conversation message and returns the inserted row ID."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    sources_json = json.dumps(sources or [], ensure_ascii=False)
    cursor.execute(
        """
        INSERT INTO conversations (session_id, role, content, sources)
        VALUES (?, ?, ?, ?)
        """,
        (session_id, role, content, sources_json),
    )
    conn.commit()
    msg_id = cursor.lastrowid
    conn.close()
    return msg_id


# Retrieve chat history for a session ordered chronologically
def get_chat_history(
    session_id: Optional[str] = None,
    limit: int = 50,
    db_path: str = SQLITE_DB_PATH,
) -> List[Dict[str, Any]]:
    """Retrieves previous chat messages for the specified session."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    if session_id:
        cursor.execute(
            """
            SELECT id, session_id, role, content, sources, created_at
            FROM conversations
            WHERE session_id = ?
            ORDER BY id ASC
            LIMIT ?
            """,
            (session_id, limit),
        )
    else:
        cursor.execute(
            """
            SELECT id, session_id, role, content, sources, created_at
            FROM conversations
            ORDER BY id ASC
            LIMIT ?
            """,
            (limit,),
        )
    rows = cursor.fetchall()
    conn.close()

    history = []
    for row in rows:
        sources_data = []
        if row[4]:
            try:
                sources_data = json.loads(row[4])
            except Exception:
                sources_data = []
        history.append({
            "id": row[0],
            "session_id": row[1],
            "role": row[2],
            "content": row[3],
            "sources": sources_data,
            "created_at": row[5],
        })
    return history


# Delete chat history for a given session or all sessions
def clear_chat_history(session_id: Optional[str] = None, db_path: str = SQLITE_DB_PATH) -> None:
    """Clears messages for a session, or all messages if session_id is None."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    if session_id:
        cursor.execute("DELETE FROM conversations WHERE session_id = ?", (session_id,))
    else:
        cursor.execute("DELETE FROM conversations")
    conn.commit()
    conn.close()


# Save an AI-generated or user-created study note
def save_note(
    title: str,
    category: str,
    content: str,
    tags: str = "",
    db_path: str = SQLITE_DB_PATH,
) -> int:
    """Saves a study note to SQLite and returns its ID."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO saved_notes (title, category, content, tags)
        VALUES (?, ?, ?, ?)
        """,
        (title.strip(), category.strip(), content.strip(), tags.strip()),
    )
    conn.commit()
    note_id = cursor.lastrowid
    conn.close()
    return note_id


# Retrieve saved notes filtered by category or search keyword
def get_saved_notes(
    category: Optional[str] = None,
    search_query: Optional[str] = None,
    db_path: str = SQLITE_DB_PATH,
) -> List[Dict[str, Any]]:
    """Fetches saved notes with optional filtering by category or keyword."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    query = "SELECT id, title, category, content, tags, created_at FROM saved_notes WHERE 1=1"
    params = []

    if category and category != "All":
        query += " AND category = ?"
        params.append(category)

    if search_query:
        query += " AND (title LIKE ? OR content LIKE ? OR tags LIKE ?)"
        like_term = f"%{search_query}%"
        params.extend([like_term, like_term, like_term])

    query += " ORDER BY id DESC"
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "id": r[0],
            "title": r[1],
            "category": r[2],
            "content": r[3],
            "tags": r[4],
            "created_at": r[5],
        }
        for r in rows
    ]


# Delete a saved note by its unique identifier
def delete_note(note_id: int, db_path: str = SQLITE_DB_PATH) -> bool:
    """Removes a note from the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM saved_notes WHERE id = ?", (note_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted


# Save a single flashcard into a specified study deck
def save_flashcard(
    deck_name: str,
    question: str,
    answer: str,
    difficulty: str = "Medium",
    db_path: str = SQLITE_DB_PATH,
) -> int:
    """Inserts a new flashcard into the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO flashcards (deck_name, question, answer, difficulty)
        VALUES (?, ?, ?, ?)
        """,
        (deck_name.strip(), question.strip(), answer.strip(), difficulty.strip()),
    )
    conn.commit()
    card_id = cursor.lastrowid
    conn.close()
    return card_id


# Retrieve flashcards, optionally filtered by deck name
def get_flashcards(
    deck_name: Optional[str] = None,
    db_path: str = SQLITE_DB_PATH,
) -> List[Dict[str, Any]]:
    """Returns a list of flashcards filtered by deck name if specified."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    if deck_name and deck_name != "All Decks":
        cursor.execute(
            """
            SELECT id, deck_name, question, answer, difficulty, mastered, review_count, created_at
            FROM flashcards
            WHERE deck_name = ?
            ORDER BY id ASC
            """,
            (deck_name,),
        )
    else:
        cursor.execute(
            """
            SELECT id, deck_name, question, answer, difficulty, mastered, review_count, created_at
            FROM flashcards
            ORDER BY id ASC
            """
        )
    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "id": r[0],
            "deck_name": r[1],
            "question": r[2],
            "answer": r[3],
            "difficulty": r[4],
            "mastered": bool(r[5]),
            "review_count": r[6],
            "created_at": r[7],
        }
        for r in rows
    ]


# Update flashcard mastery and review counter
def update_flashcard_review(
    card_id: int,
    mastered: bool,
    db_path: str = SQLITE_DB_PATH,
) -> None:
    """Updates the review counter and mastery flag for a flashcard."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE flashcards
        SET review_count = review_count + 1, mastered = ?
        WHERE id = ?
        """,
        (1 if mastered else 0, card_id),
    )
    conn.commit()
    conn.close()


# Delete a flashcard by ID
def delete_flashcard(card_id: int, db_path: str = SQLITE_DB_PATH) -> bool:
    """Deletes a flashcard from the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM flashcards WHERE id = ?", (card_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted


# Save a personalized study plan roadmap
def save_study_plan(
    topic: str,
    duration_days: int,
    target_role: str,
    plan_markdown: str,
    db_path: str = SQLITE_DB_PATH,
) -> int:
    """Stores a generated study roadmap."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO study_plans (topic, duration_days, target_role, plan_markdown)
        VALUES (?, ?, ?, ?)
        """,
        (topic.strip(), duration_days, target_role.strip(), plan_markdown.strip()),
    )
    conn.commit()
    plan_id = cursor.lastrowid
    conn.close()
    return plan_id


# Retrieve all saved study plans
def get_study_plans(db_path: str = SQLITE_DB_PATH) -> List[Dict[str, Any]]:
    """Fetches all saved study plans ordered by most recent first."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, topic, duration_days, target_role, plan_markdown, created_at
        FROM study_plans
        ORDER BY id DESC
        """
    )
    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "id": r[0],
            "topic": r[1],
            "duration_days": r[2],
            "target_role": r[3],
            "plan_markdown": r[4],
            "created_at": r[5],
        }
        for r in rows
    ]


# Save completed quiz score and answer details
def save_quiz_result(
    topic: str,
    score: int,
    total_questions: int,
    quiz_json: str,
    db_path: str = SQLITE_DB_PATH,
) -> int:
    """Records a completed quiz result with topic and score."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO quizzes (topic, score, total_questions, quiz_json)
        VALUES (?, ?, ?, ?)
        """,
        (topic.strip(), score, total_questions, quiz_json.strip()),
    )
    conn.commit()
    quiz_id = cursor.lastrowid
    conn.close()
    return quiz_id


# Retrieve recent quiz results
def get_quiz_history(limit: int = 20, db_path: str = SQLITE_DB_PATH) -> List[Dict[str, Any]]:
    """Returns past quiz attempts and scores."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, topic, score, total_questions, quiz_json, created_at
        FROM quizzes
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    )
    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "id": r[0],
            "topic": r[1],
            "score": r[2],
            "total_questions": r[3],
            "quiz_json": r[4],
            "created_at": r[5],
        }
        for r in rows
    ]
