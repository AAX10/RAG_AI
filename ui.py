"""
ui.py — Streamlit UI components, theme, and layout.

Provides reusable components for the sidebar, chat display,
source viewer, performance stats, and confidence indicators.
Replaces the monolithic inline UI code from the original rag_ai.py.
"""

import streamlit as st
from typing import Any, Dict, List, Optional

from config import AVAILABLE_MODES, DEFAULT_MODE
from citations import SourceCitation, format_citation_details
from utils import clean_source_excerpt


# ============================================================================
# Theme & Custom CSS
# ============================================================================

CUSTOM_CSS = """
<style>
    /* --- Hide Streamlit defaults --- */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* --- Dark academic theme --- */
    .stApp {
        background-color: #0e1117;
    }

    /* --- Chat message styling --- */
    .stChatMessage {
        border-radius: 12px;
        margin-bottom: 8px;
    }

    /* --- Sidebar polish --- */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stRadio label {
        color: #c9d1d9;
    }

    /* --- Mode selector cards --- */
    .mode-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 16px;
        background: linear-gradient(135deg, #1a1b4b 0%, #2d1b69 100%);
        color: #a5b4fc;
        font-size: 0.82em;
        font-weight: 600;
        margin-bottom: 8px;
        border: 1px solid #312e81;
    }

    /* --- Stats panel --- */
    .stats-container {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px;
        margin-top: 8px;
    }
    .stat-item {
        display: flex;
        justify-content: space-between;
        padding: 4px 0;
        font-size: 0.85em;
        color: #8b949e;
        border-bottom: 1px solid #21262d;
    }
    .stat-value {
        color: #58a6ff;
        font-weight: 600;
    }

    /* --- Confidence meter --- */
    .confidence-high { color: #3fb950; }
    .confidence-mid { color: #d29922; }
    .confidence-low { color: #f85149; }

    /* --- Citation footer --- */
    .citation-ref {
        background: #1c2128;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 8px 12px;
        margin: 4px 0;
        font-size: 0.88em;
    }

    /* --- Source excerpt --- */
    .source-excerpt {
        background: #0d1117;
        border-left: 3px solid #58a6ff;
        padding: 10px 14px;
        margin: 8px 0;
        font-size: 0.9em;
        color: #c9d1d9;
        border-radius: 0 6px 6px 0;
    }

    /* --- Performance stats chips --- */
    .perf-chip {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75em;
        margin-right: 6px;
        background: #21262d;
        color: #8b949e;
    }

    /* --- Suggested questions --- */
    .suggestion-btn {
        background: #1c2128;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 8px 14px;
        margin: 4px;
        color: #c9d1d9;
        cursor: pointer;
        transition: all 0.2s;
    }
    .suggestion-btn:hover {
        background: #2d333b;
        border-color: #58a6ff;
        color: #58a6ff;
    }

    /* --- Better code blocks --- */
    .stCodeBlock {
        border-radius: 8px;
    }

    /* --- Smooth transitions --- */
    * {
        transition: background-color 0.2s ease, border-color 0.2s ease;
    }

    /* --- Progress bar --- */
    .progress-bar {
        height: 6px;
        border-radius: 3px;
        background: #21262d;
        overflow: hidden;
    }
    .progress-fill {
        height: 100%;
        border-radius: 3px;
        background: linear-gradient(90deg, #58a6ff, #a5b4fc);
        transition: width 0.3s ease;
    }
</style>
"""


def inject_css() -> None:
    """Inject custom CSS into the Streamlit app."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================================
# Page Configuration
# ============================================================================

def setup_page() -> None:
    """Configure the Streamlit page (must be first Streamlit call)."""
    st.set_page_config(
        page_title="AAX AI — Academic Tutor",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()


# ============================================================================
# Sidebar
# ============================================================================

def render_sidebar() -> Dict[str, Any]:
    """Render the sidebar with mode selector, controls, and stats.

    Returns:
        Dict with user selections: mode, book_filter, etc.
    """
    with st.sidebar:
        st.markdown("##  AAX AI")
        st.caption("Your AI Academic Tutor")
        st.divider()

        # --- LLM Settings ---
        st.markdown("###  LLM Provider")
        provider = st.radio(
            "Select AI Provider",
            ["Groq (Cloud)", "Ollama (Local)"],
            horizontal=True,
            label_visibility="collapsed",
        )
        ollama_model = "qwen"
        if provider == "Ollama (Local)":
            ollama_model = st.text_input("Ollama Model", value="qwen2.5")
            
        st.divider()

        # --- Study Mode Selector ---
        st.markdown("###  Study Mode")
        mode_options = {m["id"]: f'{m["label"]}' for m in AVAILABLE_MODES}
        mode_descs = {m["id"]: m["desc"] for m in AVAILABLE_MODES}

        selected_mode = st.selectbox(
            "Choose a mode",
            options=list(mode_options.keys()),
            format_func=lambda x: mode_options[x],
            index=0,
            key="mode_selector",
            help="Tip: You can also type /quiz, /flashcards, etc. in the chat!",
        )
        st.caption(f"*{mode_descs.get(selected_mode, '')}*")

        st.divider()

        # --- Controls ---
        st.markdown("###  Controls")
        if st.button(" Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        if st.button(" Export Conversation", use_container_width=True):
            _export_conversation()

        st.divider()

        # --- Performance Stats (from last query) ---
        stats = st.session_state.get("performance_stats", {})
        if stats:
            st.markdown("###  Last Query Stats")
            _render_stats_panel(stats)
            st.divider()

        # --- Learning Progress ---
        learning_state = st.session_state.get("learning_state")
        if learning_state and learning_state.total_queries > 0:
            st.markdown("###  Learning Progress")
            _render_progress(learning_state)

    return {
        "mode": selected_mode,
        "provider": provider,
        "ollama_model": ollama_model,
    }


# ============================================================================
# Chat Display
# ============================================================================

def render_chat_history() -> None:
    """Render all messages from the session history."""
    for message in st.session_state.get("messages", []):
        role = message["role"]
        content = message["content"]
        metadata = message.get("metadata", {})

        with st.chat_message(role):
            # Mode indicator for assistant messages
            if role == "assistant" and metadata.get("mode"):
                mode_label = metadata.get("mode_label", metadata["mode"])
                st.markdown(
                    f'<span class="mode-badge">{mode_label}</span>',
                    unsafe_allow_html=True,
                )
            st.markdown(content)

            # Render citations if present
            if role == "assistant" and metadata.get("citations"):
                _render_citations(metadata["citations"])

            # Performance chips
            if role == "assistant" and metadata.get("stats"):
                _render_perf_chips(metadata["stats"])


def render_response(
    answer: str,
    citations: List[SourceCitation],
    stats: Dict[str, Any],
    mode: str,
    mode_label: str,
) -> None:
    """Render a new assistant response with citations and stats."""
    # Mode badge
    st.markdown(
        f'<span class="mode-badge">{mode_label}</span>',
        unsafe_allow_html=True,
    )

    # Answer
    st.markdown(answer)

    # Citations
    if citations:
        _render_citations(citations)

    # Performance chips
    _render_perf_chips(stats)


# ============================================================================
# Citation Display
# ============================================================================

def _render_citations(
    citations: Any,
) -> None:
    """Render expandable citation cards."""
    if isinstance(citations, list) and citations:
        with st.expander(f" Sources ({len(citations)})", expanded=False):
            for citation in citations:
                if isinstance(citation, dict):
                    _render_single_citation_dict(citation)
                elif hasattr(citation, "full_ref"):
                    _render_single_citation(citation)
                else:
                    st.write("Error: Unknown citation format")


def _render_single_citation(citation: SourceCitation) -> None:
    """Render a single SourceCitation object."""
    col1, col2 = st.columns([0.85, 0.15])
    with col1:
        st.markdown(f"**{citation.full_ref}**")
    with col2:
        st.markdown(f"{citation.confidence_emoji} {citation.confidence:.0%}")

    if citation.content_type and citation.content_type != "narrative":
        st.caption(f" Type: {citation.content_type}")

    # Expandable excerpt
    with st.expander("Read excerpt"):
        cleaned = clean_source_excerpt(citation.excerpt)
        st.markdown(
            f'<div class="source-excerpt">{cleaned}</div>',
            unsafe_allow_html=True,
        )

    st.divider()


def _render_single_citation_dict(citation: dict) -> None:
    """Render a citation from a serialized dict (from session state)."""
    col1, col2 = st.columns([0.85, 0.15])
    with col1:
        st.markdown(f'**{citation.get("full_ref", "Unknown")}**')
    with col2:
        emoji = citation.get("confidence_emoji", "")
        conf = citation.get("confidence", 0.5)
        st.markdown(f"{emoji} {conf:.0%}")

    content_type = citation.get("content_type", "")
    if content_type and content_type != "narrative":
        st.caption(f" Type: {content_type}")

    with st.expander("Read excerpt"):
        excerpt = citation.get("excerpt", "")
        cleaned = clean_source_excerpt(excerpt)
        st.markdown(
            f'<div class="source-excerpt">{cleaned}</div>',
            unsafe_allow_html=True,
        )

    st.divider()


# ============================================================================
# Performance Stats
# ============================================================================

def _render_stats_panel(stats: Dict[str, Any]) -> None:
    """Render the performance stats panel in the sidebar."""
    retrieval_time = stats.get("retrieval_time", 0)
    llm_time = stats.get("llm_time", 0)
    total_time = stats.get("total_time", 0)
    retrieval = stats.get("retrieval", {})
    query_type = retrieval.get("query_type", stats.get("query_type", "—"))
    final_count = retrieval.get("final_count", stats.get("final_count", "—"))

    st.markdown(
        f"""
        <div class="stats-container">
            <div class="stat-item">
                <span> Retrieval</span>
                <span class="stat-value">{retrieval_time:.2f}s</span>
            </div>
            <div class="stat-item">
                <span> LLM</span>
                <span class="stat-value">{llm_time:.2f}s</span>
            </div>
            <div class="stat-item">
                <span> Total</span>
                <span class="stat-value">{total_time:.2f}s</span>
            </div>
            <div class="stat-item">
                <span> Query Type</span>
                <span class="stat-value">{query_type}</span>
            </div>
            <div class="stat-item">
                <span> Sources Used</span>
                <span class="stat-value">{final_count}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_perf_chips(stats: Dict[str, Any]) -> None:
    """Render inline performance chips below the answer."""
    retrieval = stats.get("retrieval", {})
    total = stats.get("total_time", 0)
    query_type = retrieval.get("query_type", stats.get("query_type", ""))
    count = retrieval.get("final_count", stats.get("final_count", ""))

    chips = []
    if total:
        chips.append(f" {total:.1f}s")
    if query_type:
        chips.append(f" {query_type}")
    if count:
        chips.append(f" {count} sources")

    if chips:
        html = " ".join(f'<span class="perf-chip">{c}</span>' for c in chips)
        st.markdown(html, unsafe_allow_html=True)


# ============================================================================
# Learning Progress
# ============================================================================

def _render_progress(learning_state: Any) -> None:
    """Render learning progress summary in the sidebar."""
    stats = learning_state.get_stats_summary()

    st.metric("Questions Asked", stats["total_queries"])
    st.metric("Topics Explored", stats["unique_topics"])

    weak = stats.get("weak_topics", [])
    if weak:
        st.markdown("** Needs Review:**")
        for topic in weak[:3]:
            st.caption(f"• {topic}")

    top = stats.get("top_topics", [])
    if top:
        st.markdown("** Most Studied:**")
        for topic, count in top[:3]:
            st.caption(f"• {topic} ({count}×)")


# ============================================================================
# Welcome Screen
# ============================================================================

def render_welcome_screen() -> None:
    """Render a detailed, organized welcome screen for empty chats."""
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 2.5rem;">
            <h2>Welcome to AAX AI Academic Tutor</h2>
            <p style="color: #8b949e; font-size: 1.1em; max-width: 600px; margin: 0 auto;">
                A highly precise, context-aware study companion. Powered by a 10-stage RAG pipeline and semantic chunking to deliver professor-quality answers.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            """
            <div style="background-color: #161b22; padding: 1.5rem; border-radius: 8px; border: 1px solid #30363d; height: 100%; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <h4 style="margin-top: 0; color: #58a6ff;">1. Deep Research</h4>
                <p style="color: #8b949e; font-size: 0.9em; line-height: 1.5;">
                    Query your textbooks and receive heavily-cited, hallucination-free answers. Check the expandable citations for exact excerpts.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
    with col2:
        st.markdown(
            """
            <div style="background-color: #161b22; padding: 1.5rem; border-radius: 8px; border: 1px solid #30363d; height: 100%; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <h4 style="margin-top: 0; color: #3fb950;">2. Active Recall</h4>
                <p style="color: #8b949e; font-size: 0.9em; line-height: 1.5;">
                    Type <code>/quiz me on [topic]</code> to have the AI generate Socratic questions testing your understanding of complex subjects.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
    with col3:
        st.markdown(
            """
            <div style="background-color: #161b22; padding: 1.5rem; border-radius: 8px; border: 1px solid #30363d; height: 100%; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <h4 style="margin-top: 0; color: #d2a8ff;">3. Study Planning</h4>
                <p style="color: #8b949e; font-size: 0.9em; line-height: 1.5;">
                    Use the <b>Learning Progress</b> sidebar to track weak topics. Type <code>/plan</code> for an adaptive, spaced-repetition study schedule.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
    st.markdown("<br><br>", unsafe_allow_html=True)



# ============================================================================
# Export
# ============================================================================

def _export_conversation() -> None:
    """Export conversation as markdown."""
    messages = st.session_state.get("messages", [])
    if not messages:
        st.toast("No conversation to export")
        return

    lines = ["# AAX AI — Conversation Export\n"]
    for msg in messages:
        role = " You" if msg["role"] == "user" else " Tutor"
        lines.append(f"## {role}\n")
        lines.append(msg["content"])
        lines.append("\n---\n")

    content = "\n".join(lines)
    st.download_button(
        " Download Markdown",
        content,
        file_name="aax_conversation.md",
        mime="text/markdown",
        use_container_width=True,
    )
