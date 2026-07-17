"""
citations.py — Perplexity-style citation formatting and confidence scoring.

Transforms raw LangChain Document objects into structured, numbered
citations with inline references and confidence indicators.
"""

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class SourceCitation:
    """A single formatted citation with metadata and confidence."""

    index: int
    book_title: str
    author: str
    page: int
    chapter: str
    section: str
    content_type: str
    excerpt: str
    confidence: float  # 0.0 – 1.0 (from re-ranker score)
    raw_metadata: Dict = field(default_factory=dict)

    @property
    def short_ref(self) -> str:
        """Short inline reference like [1]."""
        return f"[{self.index}]"

    @property
    def full_ref(self) -> str:
        """Full citation line for the footer."""
        parts = [f"**[{self.index}]**"]

        if self.book_title and self.book_title != "Unknown":
            parts.append(f"*{self.book_title}*")
        if self.chapter and self.chapter != "Unknown":
            parts.append(f"Ch. {self.chapter}")
        if self.section and self.section != "Unknown":
            parts.append(f"§{self.section}")
        if self.page > 0:
            parts.append(f"p. {self.page}")

        ref = ", ".join(parts[1:])
        return f"{parts[0]} {ref}" if ref else parts[0]

    @property
    def confidence_emoji(self) -> str:
        """Visual confidence indicator."""
        if self.confidence >= 0.8:
            return ""
        elif self.confidence >= 0.5:
            return ""
        return ""


# ============================================================================
# Citation Extraction
# ============================================================================

def extract_citation(doc: Document, index: int, score: float = 0.0) -> SourceCitation:
    """Convert a LangChain Document into a SourceCitation.

    Handles both rich metadata (from new ingestion pipeline) and
    legacy metadata (source filename + page only) gracefully.
    """
    meta = doc.metadata

    # --- Book title ---
    book_title = meta.get("book_title", "")
    if not book_title:
        # Legacy fallback: extract from source filename
        source_file = meta.get("source", "Unknown")
        book_title = _title_from_filename(source_file)

    # --- Author ---
    author = meta.get("author", "")

    # --- Page ---
    page = meta.get("page", 0)
    if isinstance(page, str):
        try:
            page = int(page)
        except ValueError:
            page = 0
    # Legacy offset: PyMuPDF pages are 0-indexed, display as 1-indexed
    if page > 0 and not meta.get("_page_already_offset", False):
        pass  # New ingestion already stores 1-indexed pages

    # --- Chapter / Section ---
    chapter = meta.get("chapter", "Unknown")
    section = meta.get("section", "Unknown")

    # --- Content type ---
    content_type = meta.get("content_type", "narrative")

    # --- Excerpt ---
    # The user explicitly requested the full text of the source to be displayed
    excerpt = doc.page_content

    return SourceCitation(
        index=index,
        book_title=book_title,
        author=author,
        page=page,
        chapter=chapter,
        section=section,
        content_type=content_type,
        excerpt=excerpt,
        confidence=min(max(score, 0.0), 1.0),
        raw_metadata=dict(meta),
    )


def extract_all_citations(
    docs: List[Document],
    scores: Optional[List[float]] = None,
) -> List[SourceCitation]:
    """Convert a list of Documents into numbered citations.

    Args:
        docs: Retrieved documents from the retrieval pipeline.
        scores: Optional relevance scores (from re-ranker), same length as docs.

    Returns:
        List of SourceCitation objects, 1-indexed.
    """
    if scores is None:
        scores = [0.5] * len(docs)

    citations = []
    for i, (doc, score) in enumerate(zip(docs, scores), start=1):
        citations.append(extract_citation(doc, index=i, score=score))
    return citations


# ============================================================================
# Citation Formatting
# ============================================================================

def format_citation_footer(citations: List[SourceCitation]) -> str:
    """Format the full citation footer (displayed below the answer).

    Produces Perplexity-style numbered references.
    """
    if not citations:
        return ""

    lines = ["---", "###  Sources"]
    for c in citations:
        line = f"{c.full_ref} {c.confidence_emoji}"
        lines.append(line)

    return "\n\n".join(lines)


def format_citation_details(citations: List[SourceCitation]) -> List[Dict]:
    """Format citations for expandable display in Streamlit.

    Returns a list of dicts with display-ready information.
    """
    details = []
    for c in citations:
        details.append({
            "index": c.index,
            "title": f"Source {c.index}: {c.book_title}",
            "subtitle": f"Page {c.page}" + (f" · {c.chapter}" if c.chapter != "Unknown" else ""),
            "confidence": c.confidence,
            "confidence_emoji": c.confidence_emoji,
            "content_type": c.content_type,
            "excerpt": c.excerpt,
            "full_ref": c.full_ref,
        })
    return details


def build_context_with_metadata(docs: List[Document]) -> str:
    """Build the context string for the LLM with source metadata embedded.

    Each chunk is wrapped with its metadata so the LLM can cite properly.
    """
    if not docs:
        return "No relevant context found."

    context_parts = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata
        book_title = meta.get("book_title", _title_from_filename(meta.get("source", "Unknown")))
        page = meta.get("page", 0)
        chapter = meta.get("chapter", "")
        section = meta.get("section", "")
        content_type = meta.get("content_type", "")

        header_parts = [f"[Source {i}]", f"Book: {book_title}"]
        if page:
            header_parts.append(f"Page: {page}")
        if chapter:
            header_parts.append(f"Chapter: {chapter}")
        if section:
            header_parts.append(f"Section: {section}")
        if content_type:
            header_parts.append(f"Type: {content_type}")

        header = " | ".join(header_parts)
        context_parts.append(f"{header}\n{doc.page_content}")

    return "\n\n---\n\n".join(context_parts)


# ============================================================================
# Helpers
# ============================================================================

def _title_from_filename(filename: str) -> str:
    """Extract a readable book title from a PDF filename.

    'Computer Networks Tanenbaum .pdf' → 'Computer Networks Tanenbaum'
    """
    name = os.path.basename(filename)
    name = re.sub(r"\.pdf$", "", name, flags=re.IGNORECASE)
    name = name.strip(" .")
    return name if name else "Unknown"
