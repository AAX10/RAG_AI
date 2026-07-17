"""
utils.py — Utility functions, caching, timing, and text processing.

Provides text cleaning (replacing the expensive LLM-based source
cleaning), embedding caching, query classification, keyword extraction,
and timing utilities used throughout the application.
"""

import hashlib
import re
import time
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# Timing
# ============================================================================

class Timer:
    """Context manager for measuring execution time.

    Usage:
        with Timer("embedding") as t:
            result = embed(query)
        print(t.elapsed)  # seconds as float
    """

    def __init__(self, label: str = ""):
        self.label = label
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self.elapsed = time.perf_counter() - self._start
        if self.label:
            logger.debug(" %s: %.3fs", self.label, self.elapsed)


# ============================================================================
# Text Cleaning
# ============================================================================

def clean_pdf_text(text: str) -> str:
    """Clean raw PDF-extracted text without altering meaning.

    Fixes common PyMuPDF extraction artifacts: broken hyphenation,
    excessive whitespace, stray page numbers, and control characters.
    """
    if not text:
        return ""

    # Fix hyphenated line breaks (e.g., "algo-\nrithm" → "algorithm")
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    # Collapse 3+ newlines into double
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove lone numbers that are likely page numbers
    text = re.sub(r"^\s*\d{1,4}\s*$", "", text, flags=re.MULTILINE)
    # Fix spacing before punctuation
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    # Collapse multiple spaces/tabs
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Remove null bytes and non-printable control characters
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)

    return text.strip()


def clean_source_excerpt(text: str) -> str:
    """Clean a source excerpt for display — no LLM call needed.

    This replaces the previous approach of calling Groq to "edit" each
    source excerpt, saving 2-4 seconds per source (6-12s total).
    """
    text = clean_pdf_text(text)

    # Trim to last complete sentence (avoid mid-sentence cutoff)
    last_period = text.rfind(".")
    if last_period > len(text) * 0.3:
        text = text[: last_period + 1]

    # Join broken lines into flowing text
    text = re.sub(r"\s*\n\s*", " ", text)
    # Fix missing space between sentences
    text = re.sub(r"\.([A-Z])", r". \1", text)

    return text.strip()


def remove_boilerplate_lines(
    text: str,
    page_texts: Dict[int, str],
    threshold: float = 0.6,
) -> str:
    """Remove lines that appear on more than *threshold* fraction of pages.

    Headers, footers, chapter titles repeated on every page, and
    watermarks are detected by their repetition frequency.
    """
    if not page_texts:
        return text

    total_pages = len(page_texts)
    if total_pages < 5:
        return text  # Too few pages for meaningful frequency analysis

    line_counts: Dict[str, int] = {}
    for page_text in page_texts.values():
        seen_on_page: set = set()
        for line in page_text.split("\n"):
            normalized = line.strip().lower()
            if len(normalized) > 3 and normalized not in seen_on_page:
                line_counts[normalized] = line_counts.get(normalized, 0) + 1
                seen_on_page.add(normalized)

    boilerplate = {
        line
        for line, count in line_counts.items()
        if count / total_pages > threshold
    }

    cleaned_lines = [
        line
        for line in text.split("\n")
        if line.strip().lower() not in boilerplate
    ]
    return "\n".join(cleaned_lines)


# ============================================================================
# Hashing & Deduplication
# ============================================================================

def compute_text_hash(text: str) -> str:
    """Compute a short SHA-256 hash for deduplication or cache keys."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def normalize_for_dedup(text: str) -> str:
    """Normalize text for near-duplicate comparison.

    Strips whitespace variations so chunks that differ only in
    formatting are caught as duplicates.
    """
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


# ============================================================================
# Content Analysis
# ============================================================================

def estimate_difficulty(text: str) -> str:
    """Estimate content difficulty from vocabulary complexity and formula density.

    Returns:
        One of "beginner", "intermediate", "advanced".
    """
    words = text.split()
    if not words:
        return "intermediate"

    avg_word_len = sum(len(w) for w in words) / len(words)
    formula_chars = len(
        re.findall(r"[=∀∃∑∫∂≤≥≠∈∉⊂⊃∪∩←→⇒⇐∧∨¬∅∞αβγδεθλμσφψω]", text)
    )
    formula_density = formula_chars / max(len(words), 1)

    if avg_word_len > 6.5 or formula_density > 0.05:
        return "advanced"
    elif avg_word_len > 5.0 or formula_density > 0.02:
        return "intermediate"
    return "beginner"


def detect_content_type(text: str) -> str:
    """Classify chunk content type from textual markers.

    Returns:
        One of "definition", "algorithm", "theorem", "example",
        "exercise", "code", "narrative".
    """
    from config import (
        DEFINITION_MARKERS,
        ALGORITHM_MARKERS,
        THEOREM_MARKERS,
        EXAMPLE_MARKERS,
        EXERCISE_MARKERS,
        CODE_BLOCK_MARKERS,
    )

    text_lower = text.lower()
    first_200 = text_lower[:200]

    # Check markers in priority order (most distinctive first)
    if any(m in first_200 for m in ALGORITHM_MARKERS):
        return "algorithm"
    if any(m in first_200 for m in THEOREM_MARKERS):
        return "theorem"
    if any(m in first_200 for m in DEFINITION_MARKERS):
        return "definition"
    if any(m in first_200 for m in CODE_BLOCK_MARKERS):
        return "code"
    if any(m in first_200 for m in EXERCISE_MARKERS):
        return "exercise"
    if any(m in first_200 for m in EXAMPLE_MARKERS):
        return "example"

    return "narrative"


def extract_keywords(text: str, top_n: int = 10) -> List[str]:
    """Extract key terms using simple term-frequency heuristics.

    Uses a curated CS-aware stopword list to filter out noise.
    """
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "can", "shall",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "above",
        "below", "between", "each", "all", "both", "few", "more",
        "most", "other", "some", "such", "no", "nor", "not", "only",
        "own", "same", "so", "than", "too", "very", "just", "this",
        "that", "these", "those", "it", "its", "if", "or", "and",
        "but", "because", "until", "while", "about", "against",
        "also", "which", "when", "where", "how", "what", "who",
        "we", "they", "their", "them", "our", "your", "there",
        "then", "here", "now", "see", "figure", "table", "chapter",
        "section", "page", "one", "two", "three", "four", "five",
        "used", "using", "use", "called", "given", "shown", "note",
        "following", "many", "new", "first", "second", "number",
    }

    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    word_freq: Dict[str, int] = {}
    for w in words:
        if w not in stopwords:
            word_freq[w] = word_freq.get(w, 0) + 1

    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in sorted_words[:top_n]]


# ============================================================================
# Query Classification
# ============================================================================

def classify_query(query: str) -> str:
    """Classify query complexity for adaptive retrieval (k selection).

    Returns:
        One of "simple", "comparison", "multi_concept", "exam", "default".
    """
    q = query.lower().strip()

    # Comparison queries
    if any(w in q for w in ("compare", "difference", " vs ", "versus", "contrast", "similarities")):
        return "comparison"

    # Multi-concept queries (multiple conjunctions or list-like)
    if q.count(" and ") >= 2 or q.count(",") >= 2:
        return "multi_concept"

    # Exam/interview queries
    if any(w in q for w in ("exam", "interview", "quiz", "test question", "all types", "everything about")):
        return "exam"

    # Simple factual queries (short)
    if len(query.split()) <= 8:
        return "simple"

    return "default"


def detect_book_filter(query: str, available_books: List[str]) -> Optional[str]:
    """Detect if the user is asking about a specific book.

    Checks for patterns like "from Tanenbaum" or "in Silberschatz".

    Returns:
        The matching book filename or None.
    """
    q = query.lower()
    for book in available_books:
        book_lower = book.lower().replace(".pdf", "")
        # Check author name or partial title
        for token in book_lower.split():
            if len(token) > 4 and token in q:
                return book
    return None


# ============================================================================
# Embedding Cache
# ============================================================================

class EmbeddingCache:
    """Simple LRU cache for query embeddings to avoid recomputation.

    Keyed by text hash. Avoids re-embedding identical or near-identical
    queries within the same session.
    """

    def __init__(self, maxsize: int = 256):
        self._cache: Dict[str, List[float]] = {}
        self._order: List[str] = []
        self._maxsize = maxsize

    def get(self, text: str) -> Optional[List[float]]:
        """Retrieve cached embedding, or None if not cached."""
        key = compute_text_hash(text)
        return self._cache.get(key)

    def put(self, text: str, embedding: List[float]) -> None:
        """Store an embedding in the cache."""
        key = compute_text_hash(text)
        if key in self._cache:
            return
        if len(self._cache) >= self._maxsize:
            oldest = self._order.pop(0)
            self._cache.pop(oldest, None)
        self._cache[key] = embedding
        self._order.append(key)

    def __len__(self) -> int:
        return len(self._cache)
