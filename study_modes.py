"""
study_modes.py — Study mode registry, detection, and routing.

Maps mode IDs to their prompts and provides automatic mode detection
from user queries (e.g., "compare TCP vs UDP" → comparison mode).
"""

import re
from typing import Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate

from prompts import get_prompt_for_mode, MODE_PROMPTS
from config import DEFAULT_MODE, AVAILABLE_MODES


# ============================================================================
# Mode Detection from Query
# ============================================================================

# Patterns: (regex_pattern, mode_id)
# Checked in order — first match wins
_DETECTION_PATTERNS = [
    # Explicit mode commands (highest priority)
    (r"^/quiz\b", "quiz"),
    (r"^/flashcard", "flashcards"),
    (r"^/interview", "interview"),
    (r"^/exam\b", "exam"),
    (r"^/revise", "revise"),
    (r"^/practice", "practice"),
    (r"^/notes?\b", "notes"),
    (r"^/summary|^/summarize", "summarize"),
    (r"^/compare", "compare"),
    (r"^/concept.?map", "concept_map"),
    (r"^/research", "research"),
    (r"^/explain", "explain"),

    # Natural language triggers
    (r"\bquiz\s+me\b|\btest\s+me\b|\bquiz\s+on\b", "quiz"),
    (r"\bflashcard", "flashcards"),
    (r"\binterview\s+question|\binterview\s+prep", "interview"),
    (r"\bexam\s+(?:prep|mode|practice|question)", "exam"),
    (r"\brevise\b|\brevision\b|\breview\b", "revise"),
    (r"\bpractice\s+(?:problem|exercise|question)", "practice"),
    (r"\bnotes?\s+(?:on|for|about)\b|\bstudy\s+notes\b", "notes"),
    (r"\bsummar(?:y|ize|ise)\b", "summarize"),
    (r"\bcompar(?:e|ison)\b.*\b(?:vs|versus|and|with)\b", "compare"),
    (r"\bconcept\s*map\b|\bmind\s*map\b", "concept_map"),
]


def detect_mode(query: str, current_mode: str = DEFAULT_MODE) -> Tuple[str, str]:
    """Detect study mode from user query text.

    Checks for explicit /commands first, then natural language patterns.
    Falls back to the user's currently selected mode.

    Args:
        query: The user's raw input.
        current_mode: The mode currently selected in the UI sidebar.

    Returns:
        Tuple of (mode_id, cleaned_query). The cleaned_query has any
        /command prefix stripped.
    """
    q = query.strip()

    for pattern, mode_id in _DETECTION_PATTERNS:
        if re.search(pattern, q, re.IGNORECASE):
            # Strip /command prefix if present
            cleaned = re.sub(r"^/\w+\s*", "", q).strip()
            return mode_id, cleaned if cleaned else q

    return current_mode, q


# ============================================================================
# Mode Information
# ============================================================================

def get_mode_info(mode_id: str) -> dict:
    """Return display info for a mode (label, description).

    Returns default explain mode info if mode_id is unrecognized.
    """
    for mode in AVAILABLE_MODES:
        if mode["id"] == mode_id:
            return mode
    return AVAILABLE_MODES[0]  # Default to explain


def get_mode_prompt(mode_id: str) -> ChatPromptTemplate:
    """Return the ChatPromptTemplate for a given mode.

    This is the main interface used by chains.py to select the
    correct prompt based on the active study mode.
    """
    return get_prompt_for_mode(mode_id)


def get_all_mode_labels() -> list[dict]:
    """Return all available modes with their labels for UI rendering.

    Used by the sidebar mode selector.
    """
    return [
        {"id": m["id"], "label": m["label"], "desc": m["desc"]}
        for m in AVAILABLE_MODES
    ]


def format_mode_indicator(mode_id: str) -> str:
    """Return a formatted string for display in the chat header.

    Example: " Explain Mode"
    """
    info = get_mode_info(mode_id)
    return f'{info["label"]} Mode'
