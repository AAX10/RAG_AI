"""
rag_ai.py — AAX AI Academic Tutor (Main Entry Point)

Thin orchestrator that wires together all modules:
  - config.py:          API keys and settings
  - chains.py:          RAG pipeline assembly
  - memory.py:          Chat history management
  - study_modes.py:     Mode detection and prompt routing
  - ui.py:              Streamlit components and theme
  - personalization.py: Learning tracking and suggestions

Preserves all original functionality while adding:
  - 14 study modes with auto-detection
  - 10-stage retrieval pipeline
  - Perplexity-style citations with confidence
  - Performance stats
  - Personalized learning tracking
"""

import streamlit as st

from config import init_keys
from chains import build_rag_pipeline
from memory import ChatMemory
from study_modes import detect_mode, format_mode_indicator
from citations import format_citation_details
from personalization import suggest_questions
from ui import (
    setup_page,
    render_sidebar,
    render_chat_history,
    render_response,
    render_welcome_screen,
)
from utils import extract_keywords


# ============================================================================
# Page Setup (must be first Streamlit call)
# ============================================================================

setup_page()


# ============================================================================
# Header
# ============================================================================

st.markdown("###  AAX AI — Your Academic Tutor")
st.markdown("Ask anything. Get precise, professor-quality answers from your textbooks.")


# ============================================================================
# Initialize Session State
# ============================================================================

ChatMemory.init_session(st.session_state)


# ============================================================================
# Sidebar (mode selector, controls, stats)
# ============================================================================

sidebar_config = render_sidebar()


# ============================================================================
# Load RAG Pipeline (cached)
# ============================================================================

@st.cache_resource
def _load_pipeline(provider: str, ollama_model: str):
    init_keys(streamlit=True)
    return build_rag_pipeline(provider, ollama_model)


pipeline, llm = _load_pipeline(sidebar_config.get("provider", "Groq (Cloud)"), sidebar_config.get("ollama_model", "qwen"))


# ============================================================================
# Render Chat History
# ============================================================================

render_chat_history()


# ============================================================================
# Welcome Screen (shown when chat is empty)
# ============================================================================

if not st.session_state.messages:
    render_welcome_screen()


# ============================================================================
# Chat Input & Response
# ============================================================================

# Check for pending query from suggestion buttons
pending = st.session_state.pop("_pending_query", None)
prompt = pending or st.chat_input("Ask a question about your textbooks...")

if prompt:
    # --- Display user message ---
    st.chat_message("user").markdown(prompt)

    # --- Get chat history ---
    chat_history = ChatMemory.to_langchain_history(st.session_state)

    # --- Save user message ---
    ChatMemory.add_message(st.session_state, "user", prompt)

    # --- Generate response ---
    with st.chat_message("assistant"):
        with st.spinner(" Searching textbooks and thinking..."):

            # Invoke the full RAG pipeline
            result = pipeline.invoke(
                query=prompt,
                chat_history=chat_history,
                mode=sidebar_config["mode"],
            )

            answer = result["answer"]
            citations = result["citations"]
            stats = result["stats"]
            mode_used = result["mode"]
            keywords = result["keywords"]

            # Render the response with citations and stats
            mode_label = format_mode_indicator(mode_used)
            render_response(answer, citations, stats, mode_used, mode_label)

    # --- Save assistant message with metadata ---
    citation_dicts = format_citation_details(citations)
    ChatMemory.add_message(
        st.session_state,
        "assistant",
        answer,
        metadata={
            "mode": mode_used,
            "mode_label": mode_label,
            "citations": citation_dicts,
            "stats": stats,
        },
    )

    # --- Update performance stats for sidebar ---
    st.session_state.performance_stats = stats

    # --- Update learning state ---
    learning_state = st.session_state.learning_state
    learning_state.record_query(prompt, mode_used, keywords)
