"""
LLM Service module for StudySathi.
Interfaces with local Ollama models with a robust fallback engine for offline demonstrations.
"""

import json
import re
import requests
import time
from typing import List, Dict, Any, Optional
from config import OLLAMA_BASE_URL, LLM_MODEL, DEMO_MODE_FALLBACK

_LAST_STATUS_CHECK = 0.0
_CACHED_STATUS = None

# Verify if the local Ollama server is running and accessible
def is_ollama_available(base_url: str = OLLAMA_BASE_URL, force_refresh: bool = False) -> bool:
    """Pings Ollama root/tag endpoint to check server availability with caching."""
    global _LAST_STATUS_CHECK, _CACHED_STATUS
    now = time.time()
    if not force_refresh and _CACHED_STATUS is not None and (now - _LAST_STATUS_CHECK) < 10.0:
        return _CACHED_STATUS

    try:
        response = requests.get(f"{base_url}/api/tags", timeout=0.8)
        _CACHED_STATUS = (response.status_code == 200)
    except Exception:
        _CACHED_STATUS = False

    _LAST_STATUS_CHECK = now
    return _CACHED_STATUS


# Retrieve list of locally downloaded models in Ollama
def get_installed_ollama_models(base_url: str = OLLAMA_BASE_URL) -> List[str]:
    """Fetches list of available models from Ollama."""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=2)
        if response.status_code == 200:
            data = response.json()
            return [m.get("name", "").split(":")[0] for m in data.get("models", [])]
    except Exception:
        pass
    return []


# Query Ollama generate endpoint with prompt and temperature settings
def call_ollama(
    prompt: str,
    system_prompt: Optional[str] = None,
    model: str = LLM_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    temperature: float = 0.3,
    timeout: int = 60,
) -> Optional[str]:
    """Sends a generation request to the Ollama API."""
    try:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }
        if system_prompt:
            payload["system"] = system_prompt

        response = requests.post(
            f"{base_url}/api/generate",
            json=payload,
            timeout=timeout,
        )
        if response.status_code == 200:
            data = response.json()
            return data.get("response", "").strip()
    except Exception:
        pass
    return None


# Clean and parse JSON data safely from LLM output blocks
def parse_json_from_llm(text: str) -> Any:
    """Extracts JSON object or list from LLM response containing markdown code blocks."""
    # Try finding markdown JSON block ```json ... ```
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if json_match:
        text = json_match.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback: find outer square brackets or curly braces
        bracket_match = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", text)
        if bracket_match:
            try:
                return json.loads(bracket_match.group(1))
            except Exception:
                pass
    return None


# Generate RAG answer grounded in retrieved document chunks
def generate_rag_response(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    chat_history: Optional[List[Dict[str, Any]]] = None,
    model: str = LLM_MODEL,
) -> Dict[str, Any]:
    """Generates an answer using context chunks with citations and confidence scoring."""
    # Format citations and context block
    citations = []
    context_text_blocks = []

    for idx, chunk in enumerate(retrieved_chunks):
        source = chunk.get("source", "Document")
        page = chunk.get("page", 1)
        sim = chunk.get("similarity", 0.0)
        cite_tag = f"[Doc {idx + 1}: {source}, p.{page}]"
        citations.append({
            "tag": cite_tag,
            "source": source,
            "page": page,
            "similarity": sim,
            "excerpt": chunk.get("text", "")[:180] + "...",
        })
        context_text_blocks.append(f"--- Context {idx + 1} ({cite_tag}) ---\n{chunk.get('text', '')}")

    context_prompt = "\n\n".join(context_text_blocks)

    system_prompt = (
        "You are StudySathi, an intelligent, encouraging AI academic mentor and GenAI tutor. "
        "Your task is to answer the student's question accurately using ONLY the provided Study Context. "
        "Cite your sources using the format [Doc N: filename, p.X] whenever stating facts from the context. "
        "If the context does not contain enough information to answer completely, honestly state what is missing "
        "and provide general educational guidance."
    )

    history_str = ""
    if chat_history:
        recent = chat_history[-4:]
        history_str = "Recent Chat History:\n" + "\n".join(
            f"{msg.get('role', 'user').title()}: {msg.get('content', '')}" for msg in recent
        ) + "\n\n"

    full_prompt = (
        f"{history_str}Provided Study Context:\n{context_prompt}\n\n"
        f"Student Question: {query}\n\n"
        "Please provide a structured, clear response with inline citations:"
    )

    response_text = None
    if is_ollama_available():
        response_text = call_ollama(full_prompt, system_prompt=system_prompt, model=model)

    # Offline / Demo Fallback Mode
    if not response_text:
        if retrieved_chunks:
            top_sources = ", ".join(list(set(c["source"] for c in citations)))
            response_text = (
                f"### Analysis based on your indexed notes\n\n"
                f"Based on **{top_sources}**, here is the synthesized answer to **'{query}'**:\n\n"
            )
            # Summarize from the top chunk
            main_chunk = retrieved_chunks[0].get("text", "")
            sentences = [s.strip() for s in main_chunk.split(".") if len(s.strip()) > 15][:4]
            for s in sentences:
                response_text += f"- {s}. {citations[0]['tag']}\n"

            if len(retrieved_chunks) > 1:
                sec_chunk = retrieved_chunks[1].get("text", "")
                sec_sentences = [s.strip() for s in sec_chunk.split(".") if len(s.strip()) > 15][:2]
                for s in sec_sentences:
                    response_text += f"- {s}. {citations[1]['tag']}\n"

            response_text += (
                f"\n\n> 💡 *Note: Operating in local demonstration mode. "
                f"Retrieved {len(retrieved_chunks)} relevant sections from your documents.*"
            )
        else:
            response_text = (
                f"I couldn't find any relevant study context in your uploaded notes matching **'{query}'**. "
                f"Try uploading notes related to this topic or ask another question!"
            )

    return {
        "answer": response_text,
        "citations": citations,
        "chunk_count": len(retrieved_chunks),
    }


# Explain a concept in either Beginner (ELI5) or Interview mode
def explain_concept(concept: str, mode: str = "beginner", model: str = LLM_MODEL) -> str:
    """Explains a topic tailored to foundational learning or technical interview preparation."""
    if mode == "beginner":
        system_prompt = (
            "You are an inspiring educator who explains complex technical concepts simply to a beginner. "
            "Use a creative, memorable real-world analogy, break down how it works step-by-step, "
            "and end with a golden takeaway rule."
        )
        prompt = (
            f"Explain the concept of '{concept}' for a complete beginner:\n"
            "Include:\n"
            "1. 🌟 The Big Picture (Simple definition without jargon)\n"
            "2. 🚗 Real-World Analogy (Relate it to everyday life)\n"
            "3. ⚙️ How it Works (3-4 step chronological breakdown)\n"
            "4. 💡 Golden Takeaway (1 sentence summary to remember)"
        )
    else:  # Interview Mode
        system_prompt = (
            "You are a Senior Staff GenAI & Machine Learning Engineer conducting technical interviews. "
            "Provide rigorous, precise explanations with architectural trade-offs, common interviewer traps, "
            "and a sample high-scoring interview response."
        )
        prompt = (
            f"Explain the concept of '{concept}' for a technical interview:\n"
            "Include:\n"
            "1. 🎯 Executive Definition & Architecture\n"
            "2. ⚖️ Core Trade-offs & Limitations (e.g. latency, memory, cost)\n"
            "3. 🔍 How it Works Under the Hood (Internal mechanics)\n"
            "4. ❓ 2 Typical Interview Questions & Model Answers\n"
            "5. 🚩 Common Pitfalls & Mistakes Candidates Make"
        )

    if is_ollama_available():
        res = call_ollama(prompt, system_prompt=system_prompt, model=model)
        if res:
            return res

    # Offline Demo Fallback
    if mode == "beginner":
        return f"""### 🌟 The Big Picture: {concept.title()}
Think of **{concept}** as an intelligent helper designed to solve a specific problem efficiently without overwhelming you with complex mechanics.

### 🚗 Real-World Analogy
Imagine a busy restaurant kitchen. Instead of the head chef running to the pantry for every single pinch of salt, an organized sous-chef keeps the most frequently used ingredients right on the prep counter. That is exactly what this concept enables in modern computing!

### ⚙️ How it Works
1. **Input & Trigger**: It receives a request or input data.
2. **Processing & Mapping**: It evaluates the requirements against learned patterns or indexed rules.
3. **Execution & Optimization**: It performs the operation with minimal waste of memory or computation.
4. **Output Delivery**: It presents the result in an intuitive format.

### 💡 Golden Takeaway
> **"{concept.title()} transforms a complicated, repetitive chore into an automated, highly reliable routine."**
"""
    else:
        return f"""### 🎯 Executive Definition & Architecture: {concept.title()}
In production systems, **{concept}** represents a fundamental architectural building block that balances algorithmic accuracy, throughput, and state persistence.

### ⚖️ Core Trade-offs & Limitations
| Dimension | Benefit | Potential Bottleneck |
| :--- | :--- | :--- |
| **Latency** | Accelerates lookups and inference | Serialization / network overhead |
| **Cost** | Efficient resource utilization | Memory footprints during peak load |
| **Reliability** | Deterministic error boundaries | Requires graceful fallbacks |

### 🔍 How it Works Under the Hood
1. **State Isolation**: Encapsulates data processing pipelines to avoid race conditions.
2. **Vector / Mathematical Foundations**: Translates raw inputs into vector representations or structured schema.
3. **Optimization Cycles**: Uses caching and pruning strategies to scale under distributed workloads.

### ❓ Typical Interview Questions & Model Answers
**Q1: How do you scale {concept} in high-throughput production?**
> *Model Answer:* Decouple the ingestion pipeline with an asynchronous queue, cache frequent embeddings/results, and implement read-replicas or sharded vector indices.

**Q2: What is the biggest trap when implementing this?**
> *Model Answer:* Failing to monitor data drift and neglecting cold-start latency when indexing large corpora.
"""


# Generate a structured summary of notes in a chosen style
def summarize_content(text: str, style: str = "bullet", model: str = LLM_MODEL) -> str:
    """Summarizes input study notes as bullet points, executive brief, or cheat sheet."""
    style_prompts = {
        "bullet": "Create a clear, structured bullet-point summary highlighting the top 5-7 core concepts and formulas.",
        "executive": "Create a formal Executive Summary: Context, Problem Statement, Key Findings, and Strategic Next Steps.",
        "cheat_sheet": "Create an ultra-compact Exam Cheat Sheet: Key Definitions, Quick Formulas, Syntax/Code Snippets, and Pitfalls to Avoid in Markdown tables.",
    }

    prompt = (
        f"{style_prompts.get(style, style_prompts['bullet'])}\n\n"
        f"Text to summarize:\n{text[:4000]}"
    )

    if is_ollama_available():
        res = call_ollama(prompt, model=model)
        if res:
            return res

    # Offline Demo Fallback
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    preview = paragraphs[0][:200] if paragraphs else "Study content overview"

    if style == "bullet":
        return f"""### 📝 Key Takeaways & Core Concepts
- **Core Theme:** {preview}...
- **Key Principle 1:** Fundamental concepts require structured modularization and continuous revision.
- **Key Principle 2:** Context retrieval and semantic search bridge the gap between static notes and dynamic synthesis.
- **Key Principle 3:** Real-world applications prioritize low-latency inference and reproducible metrics.
- **Crucial Formula / Rule:** Consistent practice and active recall yield significantly higher retention.
"""
    elif style == "executive":
        return f"""### 📑 Executive Brief
**Context:** Overview of the provided learning materials.
**Key Objective:** Understand foundational concepts and architectural implementations.
**Summary of Findings:**
The materials discuss theoretical foundations alongside practical operational techniques. Emphasis is placed on data integrity, semantic retrieval, and production reliability.
**Recommended Next Steps:**
1. Generate flashcards to test recall on critical terminology.
2. Execute diagnostic quiz to measure comprehension.
"""
    else:  # cheat_sheet
        return f"""### ⚡ Exam Cheat Sheet

| Topic / Term | Quick Definition | Key Formula / Rule |
| :--- | :--- | :--- |
| **Embeddings** | Dense vector representation of semantic text | `Cosine Sim = (A · B) / (||A|| ||B||)` |
| **RAG** | Retrieval-Augmented Generation | Retrieve chunks -> Augment prompt -> Synthesize |
| **Vector Store** | Specialized database for similarity search | Top-K Nearest Neighbors |
| **Evaluation** | Measuring retrieval & generation accuracy | Hit Rate, MRR, Faithfulness Score |

#### ⚠️ Exam Traps:
- Don't confuse **Chunk Size** (context breadth) with **Top-K** (number of chunks retrieved).
- Always verify vector store persistence before running batch inference.
"""


# Generate interactive flashcards as structured JSON
def generate_flashcards(text: str, count: int = 5, model: str = LLM_MODEL) -> List[Dict[str, str]]:
    """Generates study flashcards with questions, answers, and difficulty levels."""
    prompt = (
        f"From the following study material, generate exactly {count} high-yield study flashcards. "
        "Return ONLY a valid JSON list of objects. Each object must have these exact keys:\n"
        "- 'question': clear, specific question\n"
        "- 'answer': concise, accurate answer\n"
        "- 'difficulty': 'Easy', 'Medium', or 'Hard'\n\n"
        f"Material:\n{text[:3500]}"
    )

    if is_ollama_available():
        res = call_ollama(prompt, model=model)
        if res:
            parsed = parse_json_from_llm(res)
            if isinstance(parsed, list) and len(parsed) > 0:
                return parsed

    # Offline Demo Fallback
    return [
        {
            "question": "What is the primary role of an Embedding in a RAG pipeline?",
            "answer": "It converts text into a high-dimensional vector capturing semantic meaning so similar concepts sit close together in vector space.",
            "difficulty": "Easy",
        },
        {
            "question": "Why is Chunk Overlap used when splitting documents?",
            "answer": "Overlap prevents contextual information and sentence meaning from being severed across boundary splits.",
            "difficulty": "Medium",
        },
        {
            "question": "What is the difference between Cosine Similarity and Euclidean Distance?",
            "answer": "Cosine similarity measures the angle between vectors (direction/semantic orientation), while Euclidean distance measures the geometric distance between points.",
            "difficulty": "Medium",
        },
        {
            "question": "How does ChromaDB handle persistence locally?",
            "answer": "It uses an embedded SQLite database for metadata/document storage paired with an HNSW index library on the local file system.",
            "difficulty": "Hard",
        },
        {
            "question": "What is 'Hallucination' in Large Language Models?",
            "answer": "When an LLM generates factually incorrect, unverifiable, or fabricated claims with high confidence.",
            "difficulty": "Easy",
        },
    ][:count]


# Generate multiple-choice quiz questions with answer options and explanations
def generate_quiz(text: str, num_questions: int = 5, difficulty: str = "Medium", model: str = LLM_MODEL) -> List[Dict[str, Any]]:
    """Generates a multiple choice quiz in JSON format."""
    prompt = (
        f"Generate {num_questions} multiple-choice quiz questions based on the following content. "
        f"Difficulty: {difficulty}. "
        "Return ONLY a valid JSON list of objects with these exact keys:\n"
        "- 'question': string question\n"
        "- 'options': list of exactly 4 strings prefixed with A), B), C), D)\n"
        "- 'correct_answer': letter string ('A', 'B', 'C', or 'D')\n"
        "- 'explanation': 1-2 sentence explanation of why that answer is correct\n\n"
        f"Content:\n{text[:3500]}"
    )

    if is_ollama_available():
        res = call_ollama(prompt, model=model)
        if res:
            parsed = parse_json_from_llm(res)
            if isinstance(parsed, list) and len(parsed) > 0:
                return parsed

    # Offline Demo Fallback
    return [
        {
            "question": "What does RAG stand for in modern AI engineering?",
            "options": [
                "A) Recursive Attention Generator",
                "B) Retrieval-Augmented Generation",
                "C) Real-time Automated Graphing",
                "D) Recurrent Autoencoder Gradient",
            ],
            "correct_answer": "B",
            "explanation": "RAG stands for Retrieval-Augmented Generation, combining a retrieval vector index with an LLM synthesizer.",
        },
        {
            "question": "Which distance metric is most commonly used for normalized embeddings?",
            "options": [
                "A) Cosine Similarity",
                "B) Manhattan Distance",
                "C) Hamming Distance",
                "D) Levenshtein Distance",
            ],
            "correct_answer": "A",
            "explanation": "Cosine similarity measures the angular orientation between two normalized vectors.",
        },
        {
            "question": "Why is Chunking essential before sending documents to an embedding model?",
            "options": [
                "A) Models can only process one word at a time",
                "B) Embedding models have token context limits and retrieval works best with focused granular context",
                "C) ChromaDB cannot store strings longer than 50 characters",
                "D) To encrypt sensitive student data",
            ],
            "correct_answer": "B",
            "explanation": "Chunking keeps chunks focused on singular concepts within token limits, which vastly improves retrieval precision.",
        },
        {
            "question": "What is the primary benefit of running Ollama locally instead of a cloud API?",
            "options": [
                "A) Requires zero RAM or disk space",
                "B) Data privacy, zero API costs, and full offline execution",
                "C) Works without installing any software",
                "D) Guarantees 100% absence of hallucinations",
            ],
            "correct_answer": "B",
            "explanation": "Local Ollama models keep all proprietary notes and queries private, eliminate per-token charges, and run offline.",
        },
        {
            "question": "Which component is responsible for holding vector embeddings and metadata on disk?",
            "options": [
                "A) ChromaDB",
                "B) Streamlit",
                "C) Requests",
                "D) Jinja2",
            ],
            "correct_answer": "A",
            "explanation": "ChromaDB is the vector database that persists document embeddings and metadata for fast similarity lookups.",
        },
    ][:num_questions]


# Create a personalized day-by-day study roadmap
def generate_study_plan(
    topic: str,
    duration_days: int = 7,
    hours_per_day: float = 2.0,
    target_role: str = "GenAI Engineer",
    model: str = LLM_MODEL,
) -> str:
    """Creates a milestone-based study curriculum tailored to timeframe and goal."""
    prompt = (
        f"Create an intensive, practical {duration_days}-day study plan for mastering '{topic}'. "
        f"Target Goal / Role: {target_role}. "
        f"Available Study Time: {hours_per_day} hours per day.\n"
        "Format as Markdown with:\n"
        "- High-level Strategy & Prerequisites\n"
        "- Day-by-Day schedule (Concepts, Hands-on task, Self-check question)\n"
        "- Final Capstone Project idea\n"
        "- Top recommended free resources / documentation to read"
    )

    if is_ollama_available():
        res = call_ollama(prompt, model=model)
        if res:
            return res

    # Offline Demo Fallback
    plan = f"""# 🚀 {duration_days}-Day Mastery Roadmap: {topic.title()}
**Target Role:** {target_role} | **Commitment:** {hours_per_day} hrs/day

---

### 🎯 Strategic Approach
This curriculum follows the **Learn -> Implement -> Test** loop. Spend 40% of your time studying theory, 50% writing code, and 10% running self-check quizzes.

### 📅 Day-by-Day Learning Schedule

"""
    for day in range(1, duration_days + 1):
        plan += f"""#### 📌 Day {day}: Core Milestone {day}
- **Focus Area:** Foundations & Practical Exercises for Part {day} of {topic.title()}
- **Study Content (1 hr):** Core concepts, architecture, and common pitfalls.
- **Hands-on Lab ({hours_per_day - 1:.1f} hrs):** Build a small script or test case verifying Day {day} concepts.
- **Check Question:** "Can I explain this concept to an interviewer in under 2 minutes?"

"""

    plan += f"""### 🏆 Final Capstone Project
Build a complete end-to-end prototype showcasing **{topic.title()}** integrated with an interactive UI, vector database, and automated evaluation metrics.

### 📚 Recommended Resources
1. Official Framework Documentation
2. Open-source GitHub reference implementations
3. Paper / Architecture Whitepapers
"""
    return plan


# Conduct an interactive mock interview evaluation turn
def mock_interview_turn(
    topic: str,
    target_role: str,
    interview_history: List[Dict[str, str]],
    user_answer: str,
    model: str = LLM_MODEL,
) -> Dict[str, str]:
    """Evaluates the candidate's response and returns constructive feedback plus the next question."""
    history_formatted = "\n".join(
        f"{turn.get('speaker', 'Interviewer')}: {turn.get('text', '')}"
        for turn in interview_history[-6:]
    )

    prompt = (
        f"You are a Lead AI Architect interviewing a candidate for a {target_role} position. "
        f"Topic: {topic}.\n\n"
        f"Interview Conversation so far:\n{history_formatted}\n\n"
        f"Candidate's latest response:\n\"{user_answer}\"\n\n"
        "Provide your evaluation in 2 parts:\n"
        "1. FEEDBACK: Score out of 10, strengths, missing technical details, and how to improve.\n"
        "2. NEXT_QUESTION: A follow-up question digging deeper or challenging an edge case."
    )

    if is_ollama_available():
        res = call_ollama(prompt, model=model)
        if res:
            parts = res.split("NEXT_QUESTION:")
            if len(parts) == 2:
                feedback = parts[0].replace("FEEDBACK:", "").strip()
                next_q = parts[1].strip()
                return {"feedback": feedback, "next_question": next_q}
            return {"feedback": res, "next_question": "Can you elaborate on how you would monitor this system in production?"}

    # Offline Demo Fallback
    word_count = len(user_answer.split())
    score = min(10, max(5, word_count // 5 + 4))

    feedback = f"""**Score: {score}/10**
- **Strengths:** Clear communication and direct addressing of the topic.
- **Areas to Improve:** Try quantifying the architectural impact (e.g., latency reduction, memory trade-offs) and citing real-world constraints like cache invalidation or rate limits.
- **Pro Tip:** Structure interview answers using the **STAR Method** (Situation, Task, Action, Result) or **Problem-Architecture-Tradeoff** framework."""

    next_questions = [
        f"How would you handle cold-start latency and embedding cache misses in a system running {topic}?",
        f"Suppose traffic spikes 10x during an exam period. Where does the bottleneck occur first, and how would you mitigate it?",
        f"How do you evaluate and detect hallucinations in this pipeline before returning answers to students?",
    ]
    next_q = next_questions[len(interview_history) % len(next_questions)]

    return {"feedback": feedback, "next_question": next_q}
