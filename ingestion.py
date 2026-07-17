"""
ingestion.py — Intelligent document ingestion with semantic chunking.

Replaces the naïve fixed-size splitting in the original upload.py with:
  - Structure-aware extraction via PyMuPDF font analysis
  - Heading / chapter / section detection
  - Atomic unit protection (definitions, algorithms, proofs, code blocks)
  - Dynamic chunk sizing based on content type
  - Rich metadata extraction (book info, content type, difficulty, keywords)
  - Boilerplate removal (frequency-based)
  - Near-duplicate chunk deduplication
  - Batch embedding and upload to Pinecone
"""

import os
import re
import statistics
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    MIN_CHUNK_SIZE,
    MAX_CHUNK_SIZE,
    DEFAULT_CHUNK_SIZE,
    CHUNK_OVERLAP,
    HEADING_FONT_RATIO,
    BOILERPLATE_THRESHOLD,
)
from utils import (
    clean_pdf_text,
    compute_text_hash,
    normalize_for_dedup,
    detect_content_type,
    estimate_difficulty,
    extract_keywords,
    remove_boilerplate_lines,
)


# ============================================================================
# Book Metadata Extraction
# ============================================================================

def extract_book_metadata(doc: fitz.Document, filename: str) -> Dict[str, str]:
    """Extract book-level metadata from the first few pages.

    Attempts to parse title, author, edition, and year from the title
    page and PDF document info. Falls back to filename parsing.
    """
    info = doc.metadata or {}
    meta: Dict[str, str] = {
        "book_title": "",
        "author": "",
        "edition": "",
        "year": "",
        "source": filename,
    }

    # --- From PDF metadata ---
    if info.get("title"):
        meta["book_title"] = info["title"].strip()
    if info.get("author"):
        meta["author"] = info["author"].strip()

    # --- From filename as fallback ---
    if not meta["book_title"]:
        name = os.path.splitext(filename)[0]
        # Remove common patterns
        name = re.sub(r"\s*\(\d+\)\s*$", "", name)
        meta["book_title"] = name.strip()

    # --- Scan first 3 pages for edition/year ---
    for page_idx in range(min(3, len(doc))):
        page = doc[page_idx]
        text = page.get_text()
        if not text:
            continue

        # Edition (e.g., "Third Edition", "5th Ed.")
        if not meta["edition"]:
            edition_match = re.search(
                r"(\d+(?:st|nd|rd|th))?\s*edition", text, re.IGNORECASE
            )
            if edition_match:
                meta["edition"] = edition_match.group(0).strip()

        # Publication year (4-digit year near "" or "published")
        if not meta["year"]:
            year_match = re.search(r"(?:|copyright|published)\s*(\d{4})", text, re.IGNORECASE)
            if year_match:
                meta["year"] = year_match.group(1)
            else:
                # Standalone year on title page
                year_match = re.search(r"\b(19\d{2}|20[0-2]\d)\b", text)
                if year_match:
                    meta["year"] = year_match.group(1)

    return meta


# ============================================================================
# Structural Analysis (Font-Based Heading Detection)
# ============================================================================

def analyze_page_structure(page: fitz.Page) -> List[Dict[str, Any]]:
    """Analyze a page's text blocks with font information.

    Uses PyMuPDF's `get_text("dict")` to access font size and style
    for each text span, enabling heading detection.

    Returns:
        List of blocks, each with text, font_size, is_bold, bbox info.
    """
    blocks = []
    page_dict = page.get_text("dict")

    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:  # Skip images
            continue

        block_text = ""
        font_sizes: List[float] = []
        is_bold = False

        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                if text:
                    block_text += text + " "
                    font_sizes.append(span.get("size", 12.0))
                    if "bold" in span.get("font", "").lower():
                        is_bold = True

        block_text = block_text.strip()
        if not block_text:
            continue

        avg_font = statistics.mean(font_sizes) if font_sizes else 12.0

        blocks.append({
            "text": block_text,
            "font_size": avg_font,
            "is_bold": is_bold,
            "y_top": block.get("bbox", [0, 0, 0, 0])[1],
            "y_bottom": block.get("bbox", [0, 0, 0, 0])[3],
        })

    return blocks


def detect_headings(
    page_blocks: List[Dict[str, Any]],
    median_font: float,
) -> List[Dict[str, Any]]:
    """Mark blocks that are likely headings based on font size.

    A block is a heading if its font size exceeds the median by
    HEADING_FONT_RATIO, or if it's bold + short (< 80 chars).
    """
    heading_threshold = median_font * HEADING_FONT_RATIO

    for block in page_blocks:
        is_heading = False
        if block["font_size"] >= heading_threshold:
            is_heading = True
        elif block["is_bold"] and len(block["text"]) < 80:
            is_heading = True

        block["is_heading"] = is_heading

        # Classify heading level by font size
        if is_heading:
            if block["font_size"] >= median_font * 1.6:
                block["heading_level"] = 1  # Chapter
            elif block["font_size"] >= median_font * 1.3:
                block["heading_level"] = 2  # Section
            else:
                block["heading_level"] = 3  # Subsection
        else:
            block["heading_level"] = 0

    return page_blocks


# ============================================================================
# Semantic Chunking
# ============================================================================

def _is_atomic_content(text: str) -> bool:
    """Check if text is an atomic unit that should not be split.

    Definitions, algorithms, proofs, code blocks, and equations
    are kept intact.
    """
    content_type = detect_content_type(text)
    return content_type in ("definition", "algorithm", "theorem", "code")


def _get_chunk_size_for_type(content_type: str) -> int:
    """Return target chunk size based on content type.

    Algorithms and code get larger chunks to stay intact.
    Definitions get smaller chunks for precision.
    """
    sizes = {
        "algorithm": MAX_CHUNK_SIZE,
        "code": MAX_CHUNK_SIZE,
        "theorem": 1600,
        "definition": 800,
        "example": DEFAULT_CHUNK_SIZE,
        "exercise": DEFAULT_CHUNK_SIZE,
        "narrative": DEFAULT_CHUNK_SIZE,
    }
    return sizes.get(content_type, DEFAULT_CHUNK_SIZE)


def semantic_chunk(
    sections: List[Dict[str, Any]],
    book_meta: Dict[str, str],
) -> List[Document]:
    """Split structured text into semantically coherent chunks.

    Args:
        sections: List of dicts from structural extraction, each with
                  text, page, chapter, section, heading info.
        book_meta: Book-level metadata (title, author, etc.).

    Returns:
        List of LangChain Documents with rich metadata.
    """
    documents: List[Document] = []
    seen_hashes: set = set()

    for sec in sections:
        text = sec["text"]
        if not text.strip():
            continue

        content_type = detect_content_type(text)
        target_size = _get_chunk_size_for_type(content_type)

        # --- Atomic content: keep as one chunk if under max ---
        if _is_atomic_content(text) and len(text) <= MAX_CHUNK_SIZE:
            doc = _make_document(text, sec, book_meta, content_type, len(documents))
            text_hash = compute_text_hash(normalize_for_dedup(text))
            if text_hash not in seen_hashes:
                seen_hashes.add(text_hash)
                documents.append(doc)
            continue

        # --- Long content: split with appropriate splitter ---
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=target_size,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", "; ", ", ", " "],
        )
        chunks = splitter.split_text(text)

        for chunk_text in chunks:
            chunk_text = chunk_text.strip()
            if len(chunk_text) < MIN_CHUNK_SIZE:
                continue  # Will be merged below

            text_hash = compute_text_hash(normalize_for_dedup(chunk_text))
            if text_hash in seen_hashes:
                continue  # Deduplicate
            seen_hashes.add(text_hash)

            doc = _make_document(
                chunk_text, sec, book_meta, content_type, len(documents)
            )
            documents.append(doc)

    # --- Merge orphan chunks (too small to stand alone) ---
    documents = _merge_small_chunks(documents)

    return documents


def _make_document(
    text: str,
    section_info: Dict[str, Any],
    book_meta: Dict[str, str],
    content_type: str,
    chunk_index: int,
) -> Document:
    """Create a Document with rich metadata."""
    keywords = extract_keywords(text, top_n=8)
    difficulty = estimate_difficulty(text)

    metadata = {
        # Book-level
        "source": book_meta.get("source", ""),
        "book_title": book_meta.get("book_title", "Unknown"),
        "author": book_meta.get("author", ""),
        "edition": book_meta.get("edition", ""),
        "year": book_meta.get("year", ""),
        # Structure-level
        "page": section_info.get("page", 0),
        "chapter": section_info.get("chapter", ""),
        "section": section_info.get("section", ""),
        # Content-level
        "content_type": content_type,
        "difficulty": difficulty,
        "concept_tags": ", ".join(keywords),
        "chunk_index": chunk_index,
        "_page_already_offset": True,
    }

    return Document(page_content=text, metadata=metadata)


def _merge_small_chunks(
    documents: List[Document], min_size: int = MIN_CHUNK_SIZE
) -> List[Document]:
    """Merge chunks smaller than min_size into their neighbors."""
    if len(documents) <= 1:
        return documents

    merged: List[Document] = []
    buffer: Optional[Document] = None

    for doc in documents:
        if len(doc.page_content) < min_size:
            if buffer is not None:
                # Append to buffer
                buffer = Document(
                    page_content=buffer.page_content + "\n\n" + doc.page_content,
                    metadata=buffer.metadata,
                )
            else:
                buffer = doc
        else:
            if buffer is not None:
                # Flush buffer by prepending to current doc
                doc = Document(
                    page_content=buffer.page_content + "\n\n" + doc.page_content,
                    metadata=doc.metadata,
                )
                buffer = None
            merged.append(doc)

    # Flush remaining buffer
    if buffer is not None:
        if merged:
            last = merged[-1]
            merged[-1] = Document(
                page_content=last.page_content + "\n\n" + buffer.page_content,
                metadata=last.metadata,
            )
        else:
            merged.append(buffer)

    return merged


# ============================================================================
# Main Ingestion Pipeline
# ============================================================================

def process_pdf(filepath: str) -> List[Document]:
    """Process a single PDF file into richly-annotated Document chunks.

    This is the primary entry point for document ingestion.

    Pipeline:
        1. Extract book-level metadata (title, author, edition, year)
        2. Page-by-page structural analysis (font sizes → headings)
        3. Build chapter/section hierarchy from detected headings
        4. Boilerplate removal (frequency-based)
        5. Semantic chunking with atomic unit protection
        6. Deduplication
        7. Metadata enrichment (content type, difficulty, keywords)

    Args:
        filepath: Absolute or relative path to the PDF file.

    Returns:
        List of LangChain Documents ready for embedding + Pinecone upload.
    """
    filename = os.path.basename(filepath)
    print(f" Processing: {filename}")

    doc = fitz.open(filepath)
    book_meta = extract_book_metadata(doc, filename)
    print(f"    Title: {book_meta['book_title']}")
    print(f"     Author: {book_meta['author'] or 'Unknown'}")

    # --- Pass 1: Collect all font sizes to compute median ---
    all_font_sizes: List[float] = []
    page_texts: Dict[int, str] = {}

    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = analyze_page_structure(page)
        for b in blocks:
            all_font_sizes.append(b["font_size"])
        page_text = page.get_text()
        if page_text.strip():
            page_texts[page_num + 1] = page_text

    median_font = statistics.median(all_font_sizes) if all_font_sizes else 12.0
    print(f"    Median font size: {median_font:.1f}pt")

    # --- Pass 2: Extract structured sections ---
    sections: List[Dict[str, Any]] = []
    current_chapter = ""
    current_section = ""

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_height = page.rect.height

        blocks = analyze_page_structure(page)
        blocks = detect_headings(blocks, median_font)

        for block in blocks:
            # Skip header/footer regions
            if block["y_top"] < page_height * 0.06 or block["y_bottom"] > page_height * 0.94:
                continue

            text = clean_pdf_text(block["text"])
            if not text:
                continue

            # Update chapter/section tracking
            if block["is_heading"]:
                if block["heading_level"] == 1:
                    current_chapter = text[:100]
                    continue  # Don't chunk the heading itself
                elif block["heading_level"] == 2:
                    current_section = text[:100]
                    continue
                elif block["heading_level"] == 3:
                    # Subsection headings become part of content
                    pass

            sections.append({
                "text": text,
                "page": page_num + 1,  # 1-indexed
                "chapter": current_chapter,
                "section": current_section,
            })

    doc.close()
    print(f"    Extracted {len(sections)} text sections from {len(page_texts)} pages")

    # --- Boilerplate removal ---
    for sec in sections:
        sec["text"] = remove_boilerplate_lines(
            sec["text"], page_texts, BOILERPLATE_THRESHOLD
        )

    # --- Semantic chunking ---
    documents = semantic_chunk(sections, book_meta)
    print(f"     Created {len(documents)} semantic chunks")

    # --- Update chunk indices after merging ---
    for i, d in enumerate(documents):
        d.metadata["chunk_index"] = i
    d_meta = documents[0].metadata if documents else {}
    total = len(documents)
    for d in documents:
        d.metadata["total_chunks"] = total

    return documents


def process_directory(directory: str) -> List[Document]:
    """Process all PDF files in a directory.

    Args:
        directory: Path to directory containing PDF files.

    Returns:
        Combined list of Documents from all PDFs.
    """
    all_documents: List[Document] = []

    for filename in sorted(os.listdir(directory)):
        if filename.lower().endswith(".pdf"):
            filepath = os.path.join(directory, filename)
            try:
                docs = process_pdf(filepath)
                all_documents.extend(docs)
                print(f"    {filename}: {len(docs)} chunks")
            except Exception as e:
                print(f"    Failed to process {filename}: {e}")

    print(f"\n Total: {len(all_documents)} chunks from {directory}")
    return all_documents
