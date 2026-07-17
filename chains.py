"""
chains.py — LangChain chain orchestration for AAX AI Academic Tutor.

Assembles the RAG chain with mode-aware prompt selection, history-aware
retrieval, and citation-enriched context building.
"""

from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_classic.chains import create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from config import (
    EMBEDDING_MODEL,
    EMBEDDING_QUERY_PREFIX,
    PINECONE_INDEX,
    PINECONE_API_KEY,
    GROQ_API_KEY,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
)
from prompts import CONTEXTUALIZE_PROMPT, get_prompt_for_mode
from retriever import AcademicRetriever
from citations import build_context_with_metadata, extract_all_citations
from study_modes import detect_mode
from utils import Timer, classify_query, extract_keywords


# ============================================================================
# Pipeline Setup
# ============================================================================

def build_embeddings() -> HuggingFaceEmbeddings:
    """Initialize the embedding model.

    Uses BAAI/bge-base-en-v1.5 for improved retrieval accuracy on academic content.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vector_store(embeddings: HuggingFaceEmbeddings) -> PineconeVectorStore:
    """Connect to the existing Pinecone index."""
    return PineconeVectorStore.from_existing_index(
        index_name=PINECONE_INDEX,
        embedding=embeddings,
    )


def build_llm(provider: str = "Groq (Cloud)", ollama_model: str = "qwen") -> BaseChatModel:
    """Initialize the chosen LLM."""
    import os
    if provider == "Ollama (Local)":
        return ChatOllama(
            model=ollama_model,
            temperature=LLM_TEMPERATURE,
        )
    else:
        return ChatGroq(
            model_name=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            groq_api_key=os.environ.get("GROQ_API_KEY", ""),
        )


def build_retriever(
    vector_store: PineconeVectorStore,
    embeddings: HuggingFaceEmbeddings,
    llm: BaseChatModel,
) -> AcademicRetriever:
    """Build the academic retriever with full pipeline."""
    return AcademicRetriever(
        vector_store=vector_store,
        embeddings=embeddings,
        llm=llm,
    )


# ============================================================================
# History-Aware Query Rewriting
# ============================================================================

def rewrite_query_with_history(
    llm: ChatGroq,
    query: str,
    chat_history: List[BaseMessage],
) -> str:
    """Rewrite a query to be standalone using conversation history.

    If there is no chat history, returns the query unchanged.
    Uses the CONTEXTUALIZE_PROMPT to resolve pronouns and references.
    """
    if not chat_history:
        return query

    try:
        messages = CONTEXTUALIZE_PROMPT.format_messages(
            chat_history=chat_history,
            input=query,
        )
        response = llm.invoke(messages)
        rewritten = response.content.strip()
        return rewritten if rewritten else query
    except Exception:
        return query


# ============================================================================
# Main RAG Pipeline
# ============================================================================

class AcademicRAGPipeline:
    """Complete RAG pipeline with mode-aware prompting and citations.

    This is the main interface used by the Streamlit app. It
    orchestrates query rewriting, retrieval, prompt selection,
    LLM generation, and citation formatting.
    """

    def __init__(
        self,
        embeddings: HuggingFaceEmbeddings,
        vector_store: PineconeVectorStore,
        llm: ChatGroq,
        retriever: AcademicRetriever,
    ):
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.llm = llm
        self.retriever = retriever

    def invoke(
        self,
        query: str,
        chat_history: List[BaseMessage],
        mode: str = "explain",
    ) -> Dict[str, Any]:
        """Execute the full RAG pipeline for a user query.

        Args:
            query: The user's question.
            chat_history: LangChain message history.
            mode: Active study mode (explain, quiz, etc.).

        Returns:
            Dict with keys:
              - answer: The LLM's response text
              - sources: List of source Documents
              - citations: Formatted citation objects
              - stats: Pipeline performance statistics
              - mode: The study mode used
              - query_rewritten: The contextualized query
        """
        stats: Dict[str, Any] = {}

        # --- Detect mode from query (overrides sidebar selection) ---
        detected_mode, cleaned_query = detect_mode(query, mode)

        # --- Rewrite query with history context ---
        with Timer("query_rewrite") as t:
            rewritten_query = rewrite_query_with_history(
                self.llm, cleaned_query, chat_history
            )
        stats["rewrite_time"] = t.elapsed

        # --- Retrieve relevant documents ---
        with Timer("retrieval") as t:
            documents, scores, retrieval_stats = self.retriever.retrieve(
                query=rewritten_query,
                mode=detected_mode,
            )
        stats["retrieval_time"] = t.elapsed
        stats["retrieval"] = retrieval_stats

        # --- Build context with metadata for LLM ---
        context_str = build_context_with_metadata(documents)

        # --- Select mode-specific prompt ---
        qa_prompt = get_prompt_for_mode(detected_mode)

        # --- Generate answer ---
        with Timer("llm_generation") as t:
            messages = qa_prompt.format_messages(
                context=context_str,
                chat_history=chat_history,
                input=cleaned_query,
            )
            response = self.llm.invoke(messages)
            answer = response.content
        stats["llm_time"] = t.elapsed

        # --- Build citations ---
        citations = extract_all_citations(documents, scores)

        # --- Total time ---
        stats["total_time"] = (
            stats.get("rewrite_time", 0)
            + stats.get("retrieval_time", 0)
            + stats.get("llm_time", 0)
        )

        # --- Extract keywords for learning tracker ---
        keywords = extract_keywords(cleaned_query, top_n=5)

        return {
            "answer": answer,
            "sources": documents,
            "citations": citations,
            "scores": scores,
            "stats": stats,
            "mode": detected_mode,
            "query_rewritten": rewritten_query,
            "keywords": keywords,
        }


# ============================================================================
# Factory Function
# ============================================================================

def build_rag_pipeline(provider: str = "Groq (Cloud)", ollama_model: str = "qwen") -> Tuple[AcademicRAGPipeline, BaseChatModel]:
    """Assemble the full RAG pipeline and return it with the LLM instance."""
    
    embeddings = build_embeddings()
    vector_store = build_vector_store(embeddings)
    llm = build_llm(provider, ollama_model)
    retriever = build_retriever(vector_store, embeddings, llm)

    pipeline = AcademicRAGPipeline(
        embeddings=embeddings,
        vector_store=vector_store,
        llm=llm,
        retriever=retriever,
    )

    return pipeline, llm
