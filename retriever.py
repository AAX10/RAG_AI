"""
retriever.py — State-of-the-art academic retrieval pipeline.

Implements a 10-stage retrieval pipeline:
  1. Query Analysis (classify complexity)
  2. Query Expansion (LLM-generated reformulations for complex queries)
  3. MMR Search (diversity-aware vector search)
  4. Score Thresholding (drop low-confidence results)
  5. Semantic Deduplication (remove near-identical chunks)
  6. Cross-Encoder Re-ranking (precision re-scoring)
  7. Adjacent Chunk Expansion (fetch neighboring chunks for context)
  8. Context Compression (trim to query-relevant content)
  9. Lost-in-Middle Reordering (best chunks at start & end)
  10. Adaptive Top-K Selection (dynamic k based on query type)
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq

from config import (
    RETRIEVAL_K,
    FETCH_K,
    MMR_LAMBDA,
    SCORE_THRESHOLD,
    DEDUP_THRESHOLD,
    RERANKER_MODEL,
    ENABLE_RERANKING,
    ENABLE_QUERY_EXPANSION,
    ADAPTIVE_K,
    SEARCH_TYPE,
)
from utils import classify_query, detect_book_filter, Timer

logger = logging.getLogger(__name__)


# ============================================================================
# Cross-Encoder Re-ranker (lazy loaded)
# ============================================================================

_reranker = None


def _get_reranker():
    """Lazy-load the cross-encoder model on first use.

    Saves ~3s startup time by deferring the model download/load.
    """
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
            _reranker = CrossEncoder(RERANKER_MODEL)
            logger.info(" Cross-encoder loaded: %s", RERANKER_MODEL)
        except Exception as e:
            logger.warning(" Cross-encoder unavailable: %s", e)
            _reranker = False  # Sentinel: tried and failed
    return _reranker if _reranker is not False else None


# ============================================================================
# Academic Retriever
# ============================================================================

class AcademicRetriever:
    """Production-grade retrieval pipeline for academic content.

    Wraps a PineconeVectorStore with multi-stage post-processing
    for maximum retrieval quality.
    """

    def __init__(
        self,
        vector_store: PineconeVectorStore,
        embeddings: HuggingFaceEmbeddings,
        llm: Optional[ChatGroq] = None,
    ):
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.llm = llm

        # Base retriever for similarity_search_with_score
        self._retriever = vector_store.as_retriever(
            search_type=SEARCH_TYPE,
            search_kwargs={
                "k": RETRIEVAL_K,
                "fetch_k": FETCH_K,
                "lambda_mult": MMR_LAMBDA,
            },
        )

    # ------------------------------------------------------------------ #
    # Main Entry Point
    # ------------------------------------------------------------------ #

    def retrieve(
        self,
        query: str,
        mode: str = "default",
        book_filter: Optional[str] = None,
        chat_history: Optional[List] = None,
    ) -> Tuple[List[Document], List[float], Dict[str, Any]]:
        """Execute the full retrieval pipeline.

        Args:
            query: User's question (already contextualized if needed).
            mode: Study mode (affects adaptive k).
            book_filter: Optional book filename to filter by.
            chat_history: Optional chat history for context.

        Returns:
            Tuple of:
              - documents: Final ranked list of Documents
              - scores: Confidence scores (0-1) for each document
              - stats: Timing and pipeline statistics
        """
        stats: Dict[str, Any] = {"stages": {}}

        # --- Stage 1: Query Analysis ---
        with Timer("query_analysis") as t:
            query_type = classify_query(query)
            target_k = ADAPTIVE_K.get(query_type, ADAPTIVE_K["default"])
            # Mode overrides
            if mode in ("exam", "research"):
                target_k = max(target_k, ADAPTIVE_K["exam"])
            elif mode in ("quiz", "flashcards"):
                target_k = max(target_k, 5)
        stats["query_type"] = query_type
        stats["target_k"] = target_k
        stats["stages"]["analysis"] = t.elapsed

        # --- Stage 2: Query Expansion ---
        queries = [query]
        if ENABLE_QUERY_EXPANSION and self.llm and query_type != "simple":
            with Timer("query_expansion") as t:
                expanded = self._expand_query(query)
                queries.extend(expanded)
            stats["expanded_queries"] = len(queries)
            stats["stages"]["expansion"] = t.elapsed

        # --- Stage 3: MMR Search ---
        with Timer("vector_search") as t:
            all_docs = self._multi_query_search(queries, book_filter)
        stats["raw_results"] = len(all_docs)
        stats["stages"]["search"] = t.elapsed

        if not all_docs:
            return [], [], stats

        # --- Stage 4: Score Thresholding ---
        with Timer("thresholding") as t:
            docs_with_scores = self._apply_score_threshold(all_docs)
        stats["after_threshold"] = len(docs_with_scores)
        stats["stages"]["threshold"] = t.elapsed

        if not docs_with_scores:
            # Fall back to top results even if below threshold
            docs_with_scores = all_docs[:target_k]

        # --- Stage 5: Semantic Deduplication ---
        with Timer("deduplication") as t:
            docs_with_scores = self._deduplicate(docs_with_scores)
        stats["after_dedup"] = len(docs_with_scores)
        stats["stages"]["dedup"] = t.elapsed

        # --- Stage 6: Cross-Encoder Re-ranking ---
        if ENABLE_RERANKING and len(docs_with_scores) > 1:
            with Timer("reranking") as t:
                docs_with_scores = self._rerank(query, docs_with_scores)
            stats["stages"]["rerank"] = t.elapsed

        # --- Stage 7: Adjacent Chunk Expansion ---
        with Timer("expansion") as t:
            docs_with_scores = self._expand_adjacent(docs_with_scores, top_n=3)
        stats["stages"]["adjacent"] = t.elapsed

        # --- Stage 8: Context Compression ---
        # (Lightweight: just trim obviously irrelevant tail content)
        with Timer("compression") as t:
            docs_with_scores = self._compress_context(query, docs_with_scores)
        stats["stages"]["compress"] = t.elapsed

        # --- Stage 9: Lost-in-Middle Reordering ---
        with Timer("reordering") as t:
            docs_with_scores = self._reorder_lost_in_middle(docs_with_scores)
        stats["stages"]["reorder"] = t.elapsed

        # --- Stage 10: Adaptive Top-K Selection ---
        final_docs = docs_with_scores[:target_k]
        documents = [d for d, _ in final_docs]
        scores = [s for _, s in final_docs]

        stats["final_count"] = len(documents)
        stats["stages"]["total"] = sum(
            v for v in stats["stages"].values() if isinstance(v, float)
        )

        return documents, scores, stats

    # ------------------------------------------------------------------ #
    # Pipeline Stages
    # ------------------------------------------------------------------ #

    def _expand_query(self, query: str) -> List[str]:
        """Use LLM to generate 2-3 alternative query phrasings."""
        if not self.llm:
            return []

        try:
            from prompts import QUERY_EXPANSION_PROMPT
            response = self.llm.invoke(
                QUERY_EXPANSION_PROMPT.format_messages(query=query)
            )
            lines = [
                line.strip()
                for line in response.content.strip().split("\n")
                if line.strip() and not line.strip().startswith("#")
            ]
            return lines[:3]  # Cap at 3 expansions
        except Exception as e:
            logger.warning("Query expansion failed: %s", e)
            return []

    def _multi_query_search(
        self,
        queries: List[str],
        book_filter: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        """Search with multiple queries and merge results."""
        all_results: Dict[str, Tuple[Document, float]] = {}

        search_kwargs: Dict[str, Any] = {
            "k": RETRIEVAL_K,
            "fetch_k": FETCH_K,
            "lambda_mult": MMR_LAMBDA,
        }

        # Metadata filter for specific book
        if book_filter:
            search_kwargs["filter"] = {"source": book_filter}

        for query in queries:
            try:
                results = self.vector_store.similarity_search_with_score(
                    query, k=RETRIEVAL_K, filter=search_kwargs.get("filter")
                )
                for doc, score in results:
                    # Use content hash as dedup key across queries
                    key = doc.page_content[:100]
                    if key not in all_results or score > all_results[key][1]:
                        # For Pinecone, scores are cosine similarity (higher = better)
                        all_results[key] = (doc, float(score))
            except Exception as e:
                logger.warning("Search failed for query variant: %s", e)

        # Sort by score descending
        sorted_results = sorted(all_results.values(), key=lambda x: x[1], reverse=True)
        return sorted_results

    def _apply_score_threshold(
        self, docs_with_scores: List[Tuple[Document, float]]
    ) -> List[Tuple[Document, float]]:
        """Remove results below the score threshold."""
        return [
            (doc, score)
            for doc, score in docs_with_scores
            if score >= SCORE_THRESHOLD
        ]

    def _deduplicate(
        self, docs_with_scores: List[Tuple[Document, float]]
    ) -> List[Tuple[Document, float]]:
        """Remove near-duplicate chunks based on text similarity.

        Uses Jaccard similarity on word sets (fast, no embedding needed).
        """
        if len(docs_with_scores) <= 1:
            return docs_with_scores

        kept: List[Tuple[Document, float]] = []
        kept_texts: List[set] = []

        for doc, score in docs_with_scores:
            words = set(doc.page_content.lower().split())
            is_duplicate = False

            for existing_words in kept_texts:
                if not words or not existing_words:
                    continue
                intersection = words & existing_words
                union = words | existing_words
                jaccard = len(intersection) / len(union) if union else 0
                if jaccard > DEDUP_THRESHOLD:
                    is_duplicate = True
                    break

            if not is_duplicate:
                kept.append((doc, score))
                kept_texts.append(words)

        return kept

    def _rerank(
        self,
        query: str,
        docs_with_scores: List[Tuple[Document, float]],
    ) -> List[Tuple[Document, float]]:
        """Re-score results using a cross-encoder model.

        Cross-encoders jointly encode query + passage for much more
        accurate relevance scoring than bi-encoder cosine similarity.
        """
        reranker = _get_reranker()
        if reranker is None:
            return docs_with_scores

        try:
            pairs = [(query, doc.page_content) for doc, _ in docs_with_scores]
            ce_scores = reranker.predict(pairs)

            # Normalize cross-encoder scores to 0-1 range
            min_s, max_s = min(ce_scores), max(ce_scores)
            score_range = max_s - min_s if max_s > min_s else 1.0
            normalized = [(s - min_s) / score_range for s in ce_scores]

            reranked = sorted(
                zip(docs_with_scores, normalized),
                key=lambda x: x[1],
                reverse=True,
            )
            return [(doc, ce_score) for (doc, _), ce_score in reranked]

        except Exception as e:
            logger.warning("Re-ranking failed: %s", e)
            return docs_with_scores

    def _expand_adjacent(
        self,
        docs_with_scores: List[Tuple[Document, float]],
        top_n: int = 3,
    ) -> List[Tuple[Document, float]]:
        """Expand top results by fetching adjacent chunks.

        Uses chunk_index metadata to find ±1 neighbors from the
        same source document.
        """
        if not docs_with_scores:
            return docs_with_scores

        expanded: List[Tuple[Document, float]] = list(docs_with_scores)
        seen_keys: set = {d.page_content[:80] for d, _ in docs_with_scores}

        for doc, score in docs_with_scores[:top_n]:
            chunk_idx = doc.metadata.get("chunk_index")
            source = doc.metadata.get("source")
            total = doc.metadata.get("total_chunks")

            if chunk_idx is None or source is None or total is None:
                continue

            # Search for adjacent chunks
            for adj_idx in [chunk_idx - 1, chunk_idx + 1]:
                if adj_idx < 0 or adj_idx >= total:
                    continue
                try:
                    # Search by metadata filter
                    adj_results = self.vector_store.similarity_search(
                        doc.page_content[:100],
                        k=3,
                        filter={
                            "source": source,
                            "chunk_index": adj_idx,
                        },
                    )
                    for adj_doc in adj_results:
                        key = adj_doc.page_content[:80]
                        if key not in seen_keys:
                            seen_keys.add(key)
                            # Adjacent chunks get a reduced score
                            expanded.append((adj_doc, score * 0.7))
                except Exception:
                    pass  # Pinecone may not support this filter combo

        return expanded

    def _compress_context(
        self,
        query: str,
        docs_with_scores: List[Tuple[Document, float]],
    ) -> List[Tuple[Document, float]]:
        """Lightweight context compression.
        
        Currently bypassed to preserve the full original textbook excerpt
        as requested for the user interface.
        """
        return docs_with_scores

    def _reorder_lost_in_middle(
        self, docs_with_scores: List[Tuple[Document, float]]
    ) -> List[Tuple[Document, float]]:
        """Reorder to mitigate the 'lost in the middle' effect.

        LLMs attend better to content at the beginning and end of the
        context. Place the highest-scored chunks at positions 1 and N,
        with medium-scored chunks in the middle.

        Reference: Liu et al., "Lost in the Middle" (2023)
        """
        if len(docs_with_scores) <= 2:
            return docs_with_scores

        # Already sorted by score (descending)
        sorted_docs = sorted(docs_with_scores, key=lambda x: x[1], reverse=True)

        reordered = []
        left = True
        front: List[Tuple[Document, float]] = []
        back: List[Tuple[Document, float]] = []

        for item in sorted_docs:
            if left:
                front.append(item)
            else:
                back.append(item)
            left = not left

        # front has positions 0, 2, 4... (highest, 3rd highest, ...)
        # back has positions 1, 3, 5... (2nd highest, 4th highest, ...)
        # Reverse back so highest-of-back is at the end
        back.reverse()
        reordered = front + back

        return reordered

    # ------------------------------------------------------------------ #
    # LangChain Retriever Interface
    # ------------------------------------------------------------------ #

    def as_langchain_retriever(self):
        """Return the base retriever for use with LangChain chains.

        Note: This bypasses the advanced pipeline stages. Use
        retrieve() directly for full pipeline quality.
        """
        return self._retriever
