"""
config.py — Centralized configuration for AAX AI Academic Tutor.

All constants, model settings, retrieval parameters, and API key
management live here. No hardcoded values anywhere else in the codebase.
"""

import os
from typing import Dict, List


# ============================================================================
# API Keys — initialized at runtime via init_keys()
# ============================================================================

PINECONE_API_KEY: str = ""
GROQ_API_KEY: str = ""


def init_keys(*, streamlit: bool = True) -> None:
    """Load API keys from Streamlit secrets or environment variables."""
    global PINECONE_API_KEY, GROQ_API_KEY

    # Always load from .env as a fallback for local dev
    from dotenv import load_dotenv
    load_dotenv()

    PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

    if streamlit:
        try:
            import streamlit as st
            if "PINECONE_API_KEY" in st.secrets:
                PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
            if "GROQ_API_KEY" in st.secrets:
                GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
        except (ImportError, FileNotFoundError, KeyError):
            pass

    # LangChain Pinecone integration reads from env
    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY


# ============================================================================
# Embedding Model
# ============================================================================

EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIMENSIONS: int = 768
EMBEDDING_QUERY_PREFIX: str = (
    "Represent this sentence for searching relevant passages: "
)
EMBEDDING_BATCH_SIZE: int = 64

# Legacy — for backward compatibility with existing rag-ai index
LEGACY_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
LEGACY_EMBEDDING_DIMENSIONS: int = 384


# ============================================================================
# Pinecone
# ============================================================================

PINECONE_INDEX: str = "rag-ai-v2"
PINECONE_METRIC: str = "cosine"
LEGACY_PINECONE_INDEX: str = "rag-ai"


# ============================================================================
# LLM (Groq)
# ============================================================================

LLM_MODEL: str = "llama-3.1-8b-instant"
LLM_TEMPERATURE: float = 0.0
LLM_MAX_TOKENS: int = 4096


# ============================================================================
# Retrieval Pipeline
# ============================================================================

SEARCH_TYPE: str = "mmr"
RETRIEVAL_K: int = 8          # Base number of results from vector search
FETCH_K: int = 25             # Candidates pool for MMR diversity sampling
MMR_LAMBDA: float = 0.7       # 0=max diversity, 1=max relevance
SCORE_THRESHOLD: float = 0.35 # Min cosine similarity to keep a result
DEDUP_THRESHOLD: float = 0.92 # Chunks above this similarity are duplicates
RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
ENABLE_RERANKING: bool = True
ENABLE_QUERY_EXPANSION: bool = True

ADAPTIVE_K: Dict[str, int] = {
    "simple": 3,
    "comparison": 5,
    "multi_concept": 6,
    "exam": 8,
    "default": 4,
}


# ============================================================================
# Document Chunking
# ============================================================================

MIN_CHUNK_SIZE: int = 200
MAX_CHUNK_SIZE: int = 2000
DEFAULT_CHUNK_SIZE: int = 1200
CHUNK_OVERLAP: int = 150

# Font-size ratio above median → treat as heading
HEADING_FONT_RATIO: float = 1.2
# Text on >60% pages at same Y-position → boilerplate
BOILERPLATE_THRESHOLD: float = 0.6


# ============================================================================
# Content-Type Detection Markers (used during ingestion)
# ============================================================================

DEFINITION_MARKERS: List[str] = [
    "definition", "is defined as", "refers to", "is called",
    "we define", "formally defined", "def.", "defn.",
]
ALGORITHM_MARKERS: List[str] = [
    "algorithm", "procedure", "pseudocode", "step 1",
    "input:", "output:", "begin", "end procedure",
]
THEOREM_MARKERS: List[str] = [
    "theorem", "lemma", "corollary", "proposition",
    "proof", "q.e.d.", "∎",
]
EXAMPLE_MARKERS: List[str] = [
    "example", "for instance", "consider", "illustration",
    "e.g.", "sample", "case study",
]
EXERCISE_MARKERS: List[str] = [
    "exercise", "problem", "practice", "question",
    "homework", "assignment",
]
CODE_BLOCK_MARKERS: List[str] = [
    "```", "def ", "class ", "import ", "int main",
    "#include", "public static", "void ", "return ",
]


# ============================================================================
# Study Modes
# ============================================================================

DEFAULT_MODE: str = "explain"

AVAILABLE_MODES: List[Dict[str, str]] = [
    {"id": "explain",     "label": " Explain",       "desc": "Full pedagogical breakdown"},
    {"id": "summarize",   "label": " Summarize",     "desc": "Concise bullet-point summary"},
    {"id": "compare",     "label": " Compare",       "desc": "Side-by-side comparison table"},
    {"id": "revise",      "label": " Revise",        "desc": "Quick revision recap"},
    {"id": "quiz",        "label": " Quiz",           "desc": "MCQs with answers"},
    {"id": "flashcards",  "label": " Flashcards",    "desc": "Q&A pairs for spaced repetition"},
    {"id": "interview",   "label": " Interview Prep", "desc": "Technical interview questions"},
    {"id": "practice",    "label": " Practice",       "desc": "Graded exercises with solutions"},
    {"id": "exam",        "label": " Exam Mode",     "desc": "Mixed question types"},
    {"id": "concept_map", "label": " Concept Map",   "desc": "Visual concept relationships"},
    {"id": "notes",       "label": " Notes",          "desc": "Structured study notes"},
    {"id": "student",     "label": " Student Mode",   "desc": "Simpler language, more analogies"},
    {"id": "professor",   "label": " Professor Mode", "desc": "Rigorous, precise, formal"},
    {"id": "research",    "label": " Research Mode",  "desc": "Cross-book synthesis, deep analysis"},
]

MODE_IDS: List[str] = [m["id"] for m in AVAILABLE_MODES]


# ============================================================================
# Paths
# ============================================================================

PDF_DIRECTORY: str = "./study"
LEARNING_STATE_FILE: str = "./learning_state.json"
CONCEPT_GRAPH_FILE: str = "./concept_graph.json"
