"""
StudySathi - GenAI Study Assistant
Main Streamlit Application
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
import streamlit as st

# Configure page metadata and layout
st.set_page_config(
    page_title="StudySathi - GenAI Study Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Import internal modules
import config
import database
import document_loader
import rag_engine
import llm_service
import evaluation

# Initialize database on app startup
database.init_db()

# Custom CSS for rich aesthetics and modern typography
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%);
        border-radius: 16px;
        padding: 24px 32px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(49, 46, 129, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #c7d2fe;
        margin-top: 8px;
        margin-bottom: 0;
    }

    /* Cards */
    .stat-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 16px 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .stat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.06);
    }
    .stat-num {
        font-size: 1.8rem;
        font-weight: 700;
        color: #4338ca;
    }
    .stat-label {
        font-size: 0.85rem;
        color: #64748b;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Citation badge */
    .citation-box {
        background: #f8fafc;
        border-left: 4px solid #6366f1;
        border-radius: 6px;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 0.9rem;
    }
    .citation-tag {
        font-weight: 600;
        color: #4f46e5;
        font-size: 0.8rem;
    }

    /* Flashcard presentation */
    .flashcard {
        background: linear-gradient(145deg, #ffffff, #f8fafc);
        border: 2px solid #e0e7ff;
        border-radius: 16px;
        padding: 32px;
        min-height: 200px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.05);
        margin: 16px 0;
    }
    .flashcard-q {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1e293b;
    }
    .flashcard-a {
        font-size: 1.15rem;
        color: #334155;
        line-height: 1.6;
        margin-top: 16px;
        padding-top: 16px;
        border-top: 1px dashed #cbd5e1;
    }

    /* Badges */
    .badge-online {
        background-color: #dcfce7;
        color: #15803d;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-demo {
        background-color: #fef3c7;
        color: #b45309;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Session state initialization
if "session_id" not in st.session_state:
    st.session_state.session_id = f"session_{int(time.time())}"
if "selected_model" not in st.session_state:
    st.session_state.selected_model = config.LLM_MODEL
if "card_index" not in st.session_state:
    st.session_state.card_index = 0
if "card_flipped" not in st.session_state:
    st.session_state.card_flipped = False
if "quiz_state" not in st.session_state:
    st.session_state.quiz_state = None
if "interview_history" not in st.session_state:
    st.session_state.interview_history = []
if "interview_question" not in st.session_state:
    st.session_state.interview_question = "Tell me about your experience with Large Language Models and how RAG pipelines work in production."

# -------------------------------------------------------------
# SIDEBAR: Status, Controls, and Document Upload
# -------------------------------------------------------------
with st.sidebar:
    st.image("https://api.iconify.design/lucide:graduation-cap.svg?color=%234f46e5", width=42)
    st.markdown("### **StudySathi Control Panel**")

    # Ollama connectivity status
    ollama_ok = llm_service.is_ollama_available(force_refresh=False)
    if ollama_ok:
        st.markdown('<span class="badge-online">🟢 Ollama Local Engine: ONLINE</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-demo">🟡 Mode: Local Heuristic Demo</span>', unsafe_allow_html=True)
        st.caption("💡 Ollama is offline or starting up. StudySathi's local fallback engine is active for instant testing.")

    st.markdown("---")

    # Model selection
    installed_models = llm_service.get_installed_ollama_models()
    model_choices = installed_models if installed_models else config.SUPPORTED_MODELS
    default_idx = 0
    if config.LLM_MODEL in model_choices:
        default_idx = model_choices.index(config.LLM_MODEL)

    selected_model = st.selectbox(
        "🧠 Active LLM Model",
        options=model_choices,
        index=default_idx,
        help="Select which local Ollama model to use for generation.",
    )
    st.session_state.selected_model = selected_model

    # Vector store metrics
    stats = rag_engine.get_collection_stats()
    st.markdown("#### 📚 Knowledge Base Stats")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Indexed Chunks", stats.get("total_chunks", 0))
    with c2:
        st.metric("Doc Sources", stats.get("total_documents", 0))

    if stats.get("sources"):
        with st.expander("📄 Indexed Documents", expanded=False):
            for s in stats["sources"]:
                st.caption(f"• {s}")

    st.markdown("---")
    st.markdown("#### 📥 Upload Notes (PDF, DOCX, TXT)")

    uploaded_files = st.file_uploader(
        "Choose study files",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        help="Upload lecture notes, textbooks, cheat sheets, or documentation.",
    )

    if st.button("🚀 Index Uploaded Files", use_container_width=True, type="primary"):
        if not uploaded_files:
            st.warning("Please upload at least one file to index.")
        else:
            with st.spinner("Parsing documents and generating vector embeddings..."):
                total_chunks_added = 0
                for uploaded_file in uploaded_files:
                    file_bytes = uploaded_file.read()
                    docs = document_loader.load_document(file_bytes, filename=uploaded_file.name)
                    chunks = document_loader.chunk_documents(docs)
                    indexed_count = rag_engine.add_chunks_to_vectorstore(chunks)
                    total_chunks_added += indexed_count

                st.success(f"Successfully indexed {total_chunks_added} chunks across {len(uploaded_files)} file(s)!")
                time.sleep(1)
                st.rerun()

    # Quick demo helper: 1-click index sample notes
    sample_files = list(Path("sample_notes").glob("*.*"))
    sample_valid = [f for f in sample_files if f.suffix in (".txt", ".docx", ".pdf")]
    if sample_valid:
        if st.button("✨ Load Built-in Sample Notes", use_container_width=True):
            with st.spinner("Indexing pre-bundled GenAI & System Design notes..."):
                count = 0
                for sf in sample_valid:
                    docs = document_loader.load_document(sf)
                    chunks = document_loader.chunk_documents(docs)
                    count += rag_engine.add_chunks_to_vectorstore(chunks)
                st.success(f"Indexed {count} chunks from {len(sample_valid)} sample notes!")
                time.sleep(1)
                st.rerun()

    if stats.get("total_chunks", 0) > 0:
        if st.button("🗑️ Reset Vector Database", use_container_width=True):
            rag_engine.clear_vectorstore()
            st.info("ChromaDB vector store cleared.")
            st.rerun()


# -------------------------------------------------------------
# HERO HEADER BANNER
# -------------------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">🎓 StudySathi <span style="font-size: 1.1rem; background: rgba(255,255,255,0.2); padding: 4px 12px; border-radius: 9999px;">AI Study & Interview Mentor</span></div>
    <p class="hero-subtitle">High-performance Retrieval-Augmented Generation (RAG), local semantic embeddings, interactive flashcards, personalized roadmaps, and mock interview coaching.</p>
</div>
""", unsafe_allow_html=True)


# -------------------------------------------------------------
# MAIN TAB NAVIGATION
# -------------------------------------------------------------
tabs = st.tabs([
    "💬 RAG Study Chat",
    "💡 Concept Explainer",
    "📝 Smart Summarizer",
    "🃏 Interactive Flashcards",
    "🎯 Practice Quiz",
    "📅 Study Planner & Mock Coach",
    "💾 Saved Notes & Export",
    "📊 Evaluation & Diagnostics",
])


# =============================================================
# TAB 1: RAG STUDY CHAT
# =============================================================
with tabs[0]:
    st.subheader("💬 Chat with your Uploaded Study Notes")
    st.caption("Ask any question. Answers are synthesized strictly from your indexed documents with citations and confidence scores.")

    # Retrieve existing conversation history from SQLite
    history = database.get_chat_history(session_id=st.session_state.session_id, limit=30)

    # Display chat messages
    chat_container = st.container()
    with chat_container:
        if not history:
            st.info("👋 Welcome to StudySathi! Ask a question about your uploaded notes or click 'Load Built-in Sample Notes' in the sidebar to try it out.")

        for msg in history:
            role = msg["role"]
            avatar = "🧑‍🎓" if role == "user" else "🤖"
            with st.chat_message(role, avatar=avatar):
                st.markdown(msg["content"])

                # Render source citations if available
                sources = msg.get("sources", [])
                if sources:
                    with st.expander(f"📚 Retrieved Context Sources ({len(sources)})", expanded=False):
                        for s in sources:
                            st.markdown(f"**{s.get('tag', 'Citation')}** | Similarity: `{int(s.get('similarity', 0.0) * 100)}%`")
                            st.caption(f"*Source: {s.get('source')} (p. {s.get('page')})*")
                            st.text(s.get("excerpt", ""))

    # Chat input box
    user_query = st.chat_input("Ask a question about your study materials...")
    if user_query:
        # Save user message to database
        database.save_chat_message(st.session_state.session_id, "user", user_query)

        # Retrieve relevant chunks from ChromaDB
        with st.spinner("Searching knowledge base and synthesizing answer..."):
            retrieved = rag_engine.search_relevant_chunks(user_query, top_k=config.TOP_K_RESULTS)
            response_data = llm_service.generate_rag_response(
                query=user_query,
                retrieved_chunks=retrieved,
                chat_history=history,
                model=st.session_state.selected_model,
            )

            answer = response_data["answer"]
            citations = response_data["citations"]

            # Save assistant response to database
            database.save_chat_message(
                st.session_state.session_id,
                "assistant",
                answer,
                sources=citations,
            )

        st.rerun()

    # Chat management controls
    col_c1, col_c2 = st.columns([1, 4])
    with col_c1:
        if st.button("🧹 Clear Chat History"):
            database.clear_chat_history(st.session_state.session_id)
            st.rerun()
    with col_c2:
        if history:
            # Generate markdown download transcript
            transcript = "# StudySathi Chat Session Transcript\n\n"
            transcript += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            for m in history:
                transcript += f"### {m['role'].upper()}:\n{m['content']}\n\n"
            st.download_button(
                "📥 Download Chat Log (Markdown)",
                data=transcript,
                file_name=f"studysathi_chat_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown",
            )


# =============================================================
# TAB 2: CONCEPT EXPLAINER
# =============================================================
with tabs[1]:
    st.subheader("💡 Deep AI Concept Explainer")
    st.caption("Learn foundational concepts from scratch or master architectural trade-offs for technical interviews.")

    col_e1, col_e2 = st.columns([3, 1])
    with col_e1:
        concept_input = st.text_input(
            "Enter Concept / Topic to explain",
            placeholder="e.g. Cosine Similarity, CAP Theorem, Backpropagation, Attention Mechanism",
        )
    with col_e2:
        explanation_mode = st.radio(
            "Explainer Mode",
            options=["beginner", "interview"],
            format_func=lambda x: "🌱 Beginner (ELI5 + Analogy)" if x == "beginner" else "🎯 Technical Interview",
        )

    if st.button("✨ Explain Concept", type="primary"):
        if not concept_input.strip():
            st.warning("Please enter a concept name.")
        else:
            with st.spinner("Synthesizing educational breakdown..."):
                explanation = llm_service.explain_concept(
                    concept=concept_input.strip(),
                    mode=explanation_mode,
                    model=st.session_state.selected_model,
                )
                st.session_state.latest_explanation = {
                    "concept": concept_input.strip(),
                    "mode": explanation_mode,
                    "content": explanation,
                }

    if "latest_explanation" in st.session_state:
        exp_data = st.session_state.latest_explanation
        st.markdown("---")
        st.markdown(exp_data["content"])

        btn_c1, btn_c2 = st.columns([1, 3])
        with btn_c1:
            if st.button("💾 Save to Study Notes"):
                note_id = database.save_note(
                    title=f"Concept: {exp_data['concept']} ({exp_data['mode'].title()})",
                    category="Concept",
                    content=exp_data["content"],
                    tags=f"{exp_data['concept']}, {exp_data['mode']}",
                )
                st.success(f"Saved to Notes (ID #{note_id})!")
        with btn_c2:
            st.download_button(
                "📥 Download Markdown",
                data=exp_data["content"],
                file_name=f"concept_{exp_data['concept'].replace(' ', '_').lower()}.md",
                mime="text/markdown",
            )


# =============================================================
# TAB 3: SMART SUMMARIZER
# =============================================================
with tabs[2]:
    st.subheader("📝 Smart Summarizer & Study Notes Generator")
    st.caption("Transform dense lecture transcripts or textbook passages into structured revision material.")

    source_option = st.radio(
        "Choose Source Input:",
        ["Paste Custom Text", "Use Indexed Document Context"],
        horizontal=True,
    )

    text_to_summarize = ""
    if source_option == "Paste Custom Text":
        text_to_summarize = st.text_area(
            "Paste study text / notes here",
            height=180,
            placeholder="Paste raw lecture notes, article excerpts, or documentation here...",
        )
    else:
        stats = rag_engine.get_collection_stats()
        if stats.get("sources"):
            selected_doc = st.selectbox("Select Indexed Document", options=stats["sources"])
            # Pull chunks for this document
            coll = rag_engine.get_collection()
            docs_match = coll.get(where={"source": selected_doc})
            if docs_match and "documents" in docs_match:
                text_to_summarize = "\n\n".join(docs_match["documents"][:5])
                st.caption(f"Loaded {len(docs_match['documents'])} chunks from `{selected_doc}`.")
        else:
            st.info("No documents indexed yet. Upload notes in the sidebar or paste custom text above.")

    col_s1, col_s2 = st.columns([1, 1])
    with col_s1:
        summary_style = st.selectbox(
            "Summary Style",
            options=["bullet", "executive", "cheat_sheet"],
            format_func=lambda x: {
                "bullet": "📌 Key Bullet Points & Takeaways",
                "executive": "📑 Executive Overview & Findings",
                "cheat_sheet": "⚡ Exam Cheat Sheet & Formulas",
            }.get(x, x),
        )

    if st.button("⚡ Generate Summary", type="primary"):
        if not text_to_summarize.strip():
            st.warning("Please provide study text to summarize.")
        else:
            with st.spinner("Generating structured summary..."):
                summary_out = llm_service.summarize_content(
                    text=text_to_summarize,
                    style=summary_style,
                    model=st.session_state.selected_model,
                )
                st.session_state.latest_summary = {
                    "style": summary_style,
                    "content": summary_out,
                }

    if "latest_summary" in st.session_state:
        sum_data = st.session_state.latest_summary
        st.markdown("---")
        st.markdown(sum_data["content"])

        s_col1, s_col2 = st.columns([1, 3])
        with s_col1:
            if st.button("💾 Save Summary to Notes"):
                nid = database.save_note(
                    title=f"Summary ({sum_data['style'].title()})",
                    category="Summary",
                    content=sum_data["content"],
                    tags="summary,revision",
                )
                st.success(f"Saved to Notes (ID #{nid})!")
        with s_col2:
            st.download_button(
                "📥 Download Summary (Markdown)",
                data=sum_data["content"],
                file_name=f"summary_{sum_data['style']}.md",
                mime="text/markdown",
            )


# =============================================================
# TAB 4: INTERACTIVE FLASHCARDS
# =============================================================
with tabs[3]:
    st.subheader("🃏 Smart Active-Recall Flashcards")
    st.caption("Flip cards to test your recall, track mastery, and generate new flashcard decks with AI.")

    # Deck management & creation
    with st.expander("✨ Generate New Flashcard Deck with AI", expanded=False):
        deck_title = st.text_input("Deck Name", value="GenAI Core")
        card_count = st.slider("Number of Flashcards", min_value=3, max_value=10, value=5)
        raw_study_content = st.text_area(
            "Source Material for Flashcards (or leave blank to use indexed documents)",
            height=100,
            placeholder="Paste definitions, laws, or notes here...",
        )

        if st.button("🚀 Generate and Save Deck"):
            if not raw_study_content.strip():
                # Grab sample text from indexed chunks
                coll = rag_engine.get_collection()
                if coll.count() > 0:
                    sample_data = coll.get(limit=4)
                    raw_study_content = "\n\n".join(sample_data.get("documents", []))
                else:
                    raw_study_content = "RAG, Embeddings, ChromaDB, Cosine Similarity, and Transformers."

            with st.spinner("Generating flashcards..."):
                cards = llm_service.generate_flashcards(
                    text=raw_study_content,
                    count=card_count,
                    model=st.session_state.selected_model,
                )
                saved_count = 0
                for c in cards:
                    database.save_flashcard(
                        deck_name=deck_title.strip(),
                        question=c.get("question", ""),
                        answer=c.get("answer", ""),
                        difficulty=c.get("difficulty", "Medium"),
                    )
                    saved_count += 1
                st.success(f"Generated and saved {saved_count} flashcards to deck '{deck_title}'!")
                st.rerun()

    # Flashcard viewer & study interface
    existing_cards = database.get_flashcards()
    if not existing_cards:
        # Prepopulate with sample cards if empty
        database.save_flashcard("GenAI Foundations", "What is an Embedding?", "A dense vector representation of text capturing semantic relationships.", "Easy")
        database.save_flashcard("GenAI Foundations", "What is Cosine Similarity?", "The cosine of the angle between two vectors, measuring directional alignment regardless of length.", "Medium")
        database.save_flashcard("GenAI Foundations", "Why use Chunk Overlap?", "To ensure boundary context between adjacent chunks is not broken during retrieval.", "Medium")
        existing_cards = database.get_flashcards()

    # Filter by deck
    deck_names = sorted(list(set(c["deck_name"] for c in existing_cards)))
    selected_deck = st.selectbox("Select Study Deck", options=["All Decks"] + deck_names)
    filtered_cards = database.get_flashcards(deck_name=selected_deck)

    if filtered_cards:
        # Bounds check
        if st.session_state.card_index >= len(filtered_cards):
            st.session_state.card_index = 0

        curr_card = filtered_cards[st.session_state.card_index]

        # Flashcard visual box
        st.markdown(f"**Card {st.session_state.card_index + 1} of {len(filtered_cards)}** | Deck: `{curr_card['deck_name']}` | Difficulty: `{curr_card['difficulty']}` | Status: `{'✅ Mastered' if curr_card['mastered'] else '📖 In Review'}`")

        card_html = f"""
        <div class="flashcard">
            <div class="flashcard-q">❓ {curr_card['question']}</div>
        """
        if st.session_state.card_flipped:
            card_html += f"""
            <div class="flashcard-a">💡 <strong>Answer:</strong><br>{curr_card['answer']}</div>
            """
        else:
            card_html += """
            <div style="margin-top: 24px; color: #94a3b8; font-style: italic;">(Click 'Flip Card' below to reveal the answer)</div>
            """
        card_html += "</div>"
        st.markdown(card_html, unsafe_allow_html=True)

        # Controls: Flip, Next, Prev, Mark
        fc_cols = st.columns([1, 1, 1, 1, 1])
        with fc_cols[0]:
            if st.button("⬅️ Previous"):
                st.session_state.card_index = (st.session_state.card_index - 1) % len(filtered_cards)
                st.session_state.card_flipped = False
                st.rerun()
        with fc_cols[1]:
            if st.button("🔄 Flip Card", type="primary"):
                st.session_state.card_flipped = not st.session_state.card_flipped
                st.rerun()
        with fc_cols[2]:
            if st.button("➡️ Next"):
                st.session_state.card_index = (st.session_state.card_index + 1) % len(filtered_cards)
                st.session_state.card_flipped = False
                st.rerun()
        with fc_cols[3]:
            if st.button("✅ Mastered"):
                database.update_flashcard_review(curr_card["id"], mastered=True)
                st.session_state.card_index = (st.session_state.card_index + 1) % len(filtered_cards)
                st.session_state.card_flipped = False
                st.rerun()
        with fc_cols[4]:
            if st.button("🗑️ Delete Card"):
                database.delete_flashcard(curr_card["id"])
                st.session_state.card_index = 0
                st.rerun()


# =============================================================
# TAB 5: PRACTICE QUIZ
# =============================================================
with tabs[4]:
    st.subheader("🎯 Interactive Practice Quiz & Self-Check")
    st.caption("Test your comprehension with automated multiple-choice questions, instant scoring, and explanations.")

    with st.expander("⚙️ Create New Quiz", expanded=(st.session_state.quiz_state is None)):
        q_topic = st.text_input("Quiz Topic", value="Retrieval-Augmented Generation & Vector Search")
        q_diff = st.selectbox("Difficulty Level", ["Easy", "Medium", "Hard"], index=1)
        q_count = st.slider("Questions Count", 2, 8, 4)

        if st.button("🚀 Start Quiz", type="primary"):
            with st.spinner("Generating quiz questions and options..."):
                # Collect study content for questions
                coll = rag_engine.get_collection()
                quiz_source = ""
                if coll.count() > 0:
                    sample = coll.get(limit=5)
                    quiz_source = "\n\n".join(sample.get("documents", []))
                else:
                    quiz_source = q_topic

                questions = llm_service.generate_quiz(
                    text=quiz_source,
                    num_questions=q_count,
                    difficulty=q_diff,
                    model=st.session_state.selected_model,
                )
                st.session_state.quiz_state = {
                    "topic": q_topic,
                    "difficulty": q_diff,
                    "questions": questions,
                    "user_answers": {},
                    "submitted": False,
                    "score": 0,
                }
                st.rerun()

    # Active Quiz Renderer
    if st.session_state.quiz_state and st.session_state.quiz_state.get("questions"):
        q_data = st.session_state.quiz_state
        st.markdown(f"### 📋 Quiz: {q_data['topic']} ({q_data['difficulty']} Mode)")

        for i, q in enumerate(q_data["questions"]):
            st.markdown(f"**Q{i+1}: {q['question']}**")
            options = q.get("options", [])

            # Radio buttons for selection
            choice_key = f"quiz_q_{i}"
            selected = st.radio(
                f"Select answer for Q{i+1}",
                options=options,
                key=choice_key,
                index=None,
                disabled=q_data["submitted"],
                label_visibility="collapsed",
            )
            if selected:
                letter = selected.split(")")[0].strip().upper()
                q_data["user_answers"][i] = letter

            # If submitted, display correctness & explanation
            if q_data["submitted"]:
                correct_letter = q.get("correct_answer", "A").upper()
                user_choice = q_data["user_answers"].get(i, "None")
                is_correct = (user_choice == correct_letter)

                if is_correct:
                    st.success(f"✅ Correct! Answer: {correct_letter}) — {q.get('explanation', '')}")
                else:
                    st.error(f"❌ Your answer: {user_choice} | Correct Answer: {correct_letter}) — {q.get('explanation', '')}")

            st.markdown("---")

        if not q_data["submitted"]:
            if st.button("📝 Submit Quiz & Calculate Score", type="primary"):
                score = 0
                for i, q in enumerate(q_data["questions"]):
                    if q_data["user_answers"].get(i) == q.get("correct_answer", "").upper():
                        score += 1
                q_data["submitted"] = True
                q_data["score"] = score

                # Save score in SQLite
                database.save_quiz_result(
                    topic=q_data["topic"],
                    score=score,
                    total_questions=len(q_data["questions"]),
                    quiz_json=json.dumps(q_data["questions"]),
                )
                st.rerun()
        else:
            pct = int((q_data["score"] / len(q_data["questions"])) * 100)
            st.balloons() if pct >= 70 else None
            st.metric(
                "Final Score",
                f"{q_data['score']} / {len(q_data['questions'])}",
                f"{pct}% Accuracy",
            )
            if st.button("🔄 Retake or Start New Quiz"):
                st.session_state.quiz_state = None
                st.rerun()

    # Past Quiz Results
    quiz_history = database.get_quiz_history(limit=5)
    if quiz_history:
        with st.expander("📈 Recent Quiz Scores", expanded=False):
            for qh in quiz_history:
                st.write(f"• **{qh['topic']}**: `{qh['score']}/{qh['total_questions']}` on {qh['created_at']}")


# =============================================================
# TAB 6: STUDY PLANNER & MOCK INTERVIEW
# =============================================================
with tabs[5]:
    st.subheader("📅 Personalized Study Roadmap & Mock Interview Coach")

    coach_mode = st.radio(
        "Choose Functionality:",
        ["🗺️ Generate Study Roadmap", "🎙️ Mock Interview Simulator"],
        horizontal=True,
    )

    if coach_mode == "🗺️ Generate Study Roadmap":
        st.markdown("#### 🗺️ Custom Learning Schedule Generator")
        plan_col1, plan_col2, plan_col3 = st.columns([2, 1, 1])
        with plan_col1:
            plan_topic = st.text_input("Target Topic", value="GenAI Engineering & RAG Systems")
        with plan_col2:
            plan_days = st.number_input("Duration (Days)", min_value=3, max_value=60, value=7)
        with plan_col3:
            plan_hours = st.number_input("Hours / Day", min_value=1.0, max_value=8.0, value=2.5, step=0.5)

        plan_role = st.selectbox(
            "Target Role / Goal",
            ["Junior GenAI Engineer", "Senior ML Engineer", "Solutions Architect", "University Exam Prep"],
        )

        if st.button("🚀 Generate Personalized Plan", type="primary"):
            with st.spinner("Formulating day-by-day learning roadmap..."):
                plan_md = llm_service.generate_study_plan(
                    topic=plan_topic,
                    duration_days=plan_days,
                    hours_per_day=plan_hours,
                    target_role=plan_role,
                    model=st.session_state.selected_model,
                )
                database.save_study_plan(plan_topic, plan_days, plan_role, plan_md)
                st.session_state.latest_plan = plan_md
                st.success("Study Roadmap Generated & Saved!")

        if "latest_plan" in st.session_state:
            st.markdown("---")
            st.markdown(st.session_state.latest_plan)
            st.download_button(
                "📥 Download Plan (Markdown)",
                data=st.session_state.latest_plan,
                file_name="study_roadmap.md",
                mime="text/markdown",
            )

    else:
        # Mock Interview Simulator
        st.markdown("#### 🎙️ Live Technical Mock Interview Coach")
        st.caption("Practice answering technical interview questions. Receive instant scoring, feedback on trade-offs, and dynamic follow-up questions.")

        interview_role = st.selectbox("Target Position", ["GenAI Engineer", "Machine Learning Engineer", "System Design Lead"])
        interview_topic = st.text_input("Interview Subject", value="RAG Architecture, Embeddings & Vector Search")

        st.markdown(f"""
        <div style="background: #eef2ff; border-left: 4px solid #4f46e5; padding: 16px; border-radius: 8px; margin: 12px 0;">
            <div style="font-weight: 600; color: #3730a3;">👔 Interviewer:</div>
            <div style="font-size: 1.1rem; color: #1e1b4b; margin-top: 6px;">"{st.session_state.interview_question}"</div>
        </div>
        """, unsafe_allow_html=True)

        user_response = st.text_area(
            "Your Answer:",
            height=130,
            placeholder="Structure your answer with architectural details, trade-offs, and examples...",
        )

        if st.button("📤 Submit Answer for Evaluation", type="primary"):
            if not user_response.strip():
                st.warning("Please enter your answer.")
            else:
                with st.spinner("Evaluating technical answer and preparing next question..."):
                    turn_res = llm_service.mock_interview_turn(
                        topic=interview_topic,
                        target_role=interview_role,
                        interview_history=st.session_state.interview_history,
                        user_answer=user_response.strip(),
                        model=st.session_state.selected_model,
                    )

                    # Update history
                    st.session_state.interview_history.append({"speaker": "Interviewer", "text": st.session_state.interview_question})
                    st.session_state.interview_history.append({"speaker": "Candidate", "text": user_response.strip()})
                    st.session_state.latest_interview_feedback = turn_res["feedback"]
                    st.session_state.interview_question = turn_res["next_question"]
                    st.rerun()

        if "latest_interview_feedback" in st.session_state:
            st.markdown("### 📊 Coach Evaluation & Feedback")
            st.markdown(st.session_state.latest_interview_feedback)


# =============================================================
# TAB 7: SAVED NOTES & EXPORT
# =============================================================
with tabs[6]:
    st.subheader("💾 Saved Notes, Cheat Sheets & Export")
    st.caption("All explanations, summaries, and roadmaps saved to your local SQLite database.")

    col_n1, col_n2 = st.columns([1, 2])
    with col_n1:
        cat_filter = st.selectbox("Filter Category", ["All", "Concept", "Summary", "Custom", "Plan"])
    with col_n2:
        search_kw = st.text_input("🔍 Search notes by keyword or tag", placeholder="e.g. RAG, Cosine, Attention")

    notes_list = database.get_saved_notes(category=cat_filter, search_query=search_kw)

    if not notes_list:
        st.info("No saved notes found. Generate and save explanations or summaries in the other tabs!")
    else:
        st.write(f"Showing **{len(notes_list)}** saved notes:")
        for note in notes_list:
            with st.expander(f"📌 {note['title']} ({note['category']}) — {note['created_at']}", expanded=False):
                st.markdown(note["content"])
                if note.get("tags"):
                    st.caption(f"🏷️ Tags: `{note['tags']}`")

                n_col1, n_col2 = st.columns([1, 4])
                with n_col1:
                    if st.button("🗑️ Delete", key=f"del_note_{note['id']}"):
                        database.delete_note(note["id"])
                        st.rerun()
                with n_col2:
                    st.download_button(
                        "📥 Export Markdown",
                        data=note["content"],
                        file_name=f"note_{note['id']}_{note['category'].lower()}.md",
                        mime="text/markdown",
                        key=f"dl_note_{note['id']}",
                    )


# =============================================================
# TAB 8: EVALUATION & DIAGNOSTICS
# =============================================================
with tabs[7]:
    st.subheader("📊 RAG Evaluation & System Health Diagnostics")
    st.caption("Quantitative metrics for retrieval accuracy (Hit Rate, MRR, Precision@K), answer groundedness, and latency.")

    st.markdown("#### 🔬 Run Automated RAG Benchmark")
    st.write("Executes a standardized set of ground-truth test queries against your currently indexed ChromaDB vector store.")

    if st.button("⚡ Run Benchmark Suite Now", type="primary"):
        with st.spinner("Running retrieval and synthesis evaluation tests..."):
            eval_report = evaluation.run_full_rag_evaluation()
            st.session_state.latest_eval_report = eval_report

    if "latest_eval_report" in st.session_state:
        rep = st.session_state.latest_eval_report
        summary = rep["summary"]

        # Top KPI Metrics Cards
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.metric("Hit Rate", f"{int(summary['avg_hit_rate'] * 100)}%", help="Proportion of queries with relevant chunk retrieved.")
        with kpi2:
            st.metric("Mean Reciprocal Rank (MRR)", summary["avg_mrr"], help="Inverse rank of first relevant chunk.")
        with kpi3:
            st.metric("Groundedness Score", f"{int(summary['avg_groundedness'] * 100)}%", help="Token overlap ratio measuring absence of hallucination.")
        with kpi4:
            st.metric("Answer Relevancy", f"{int(summary['avg_relevancy'] * 100)}%", help="Query-to-answer keyword alignment.")

        st.markdown("---")
        st.markdown("#### 📋 Detailed Query-by-Query Test Results")

        eval_table = []
        for row in rep["evaluations"]:
            eval_table.append({
                "Test Query": row["query"],
                "Hit": "✅ Yes" if row["hit"] > 0 else "❌ No",
                "MRR": row["mrr"],
                "Groundedness": f"{int(row['groundedness'] * 100)}%",
                "Relevancy": f"{int(row['relevancy'] * 100)}%",
                "Total Latency": f"{row['total_latency_ms']} ms",
            })
        st.table(eval_table)

    st.markdown("---")
    st.markdown("#### 🩺 Environment & Connection Diagnostics")
    d1, d2 = st.columns(2)
    with d1:
        st.write("**Ollama Server:**")
        st.write(f"- Base URL: `{config.OLLAMA_BASE_URL}`")
        st.write(f"- Reachable: `{'Yes 🟢' if llm_service.is_ollama_available(force_refresh=True) else 'No (Demo Fallback Active) 🟡'}`")
        st.write(f"- Target LLM Model: `{st.session_state.selected_model}`")
        st.write(f"- Target Embedding Model: `{config.EMBEDDING_MODEL}`")
    with d2:
        st.write("**Storage Paths & System:**")
        st.write(f"- SQLite DB: `{config.SQLITE_DB_PATH}`")
        st.write(f"- ChromaDB Directory: `{config.CHROMA_PERSIST_DIRECTORY}`")
        st.write(f"- Chunk Size: `{config.CHUNK_SIZE}` tokens | Overlap: `{config.CHUNK_OVERLAP}`")
