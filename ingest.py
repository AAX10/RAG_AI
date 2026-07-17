"""
ingest.py — CLI entry point for document ingestion.

Usage:
    python ingest.py                    # Process all PDFs in ./study
    python ingest.py --dir ./my_books   # Custom directory
    python ingest.py --index rag-ai-v2  # Custom Pinecone index name

Replaces the original upload.py with intelligent ingestion.
The original upload.py is preserved as backup.
"""

import argparse
import os
import sys

# Load environment variables before importing modules
from dotenv import load_dotenv
load_dotenv()

from config import (
    init_keys,
    PDF_DIRECTORY,
    PINECONE_INDEX,
    EMBEDDING_MODEL,
    EMBEDDING_QUERY_PREFIX,
    PINECONE_API_KEY,
)
from ingestion import process_directory
from concept_graph import ConceptGraph


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AAX AI — Intelligent Document Ingestion",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--dir",
        default=PDF_DIRECTORY,
        help=f"Directory containing PDF files (default: {PDF_DIRECTORY})",
    )
    parser.add_argument(
        "--index",
        default=PINECONE_INDEX,
        help=f"Pinecone index name (default: {PINECONE_INDEX})",
    )
    parser.add_argument(
        "--build-graph",
        action="store_true",
        default=True,
        help="Build concept graph after ingestion (default: True)",
    )
    args = parser.parse_args()

    # --- Validate ---
    if not os.path.isdir(args.dir):
        print(f" Directory not found: {args.dir}")
        sys.exit(1)

    # --- Initialize keys ---
    init_keys(streamlit=False)
    if not os.environ.get("PINECONE_API_KEY"):
        print(" PINECONE_API_KEY not set. Add it to .env or environment.")
        sys.exit(1)

    # --- Process documents ---
    print(f"\n AAX AI — Intelligent Document Ingestion")
    print(f" Source: {args.dir}")
    print(f" Target: Pinecone index '{args.index}'")
    print(f" Embeddings: {EMBEDDING_MODEL}")
    print("=" * 60)

    documents = process_directory(args.dir)

    if not documents:
        print(" No documents were extracted. Check your PDF files.")
        sys.exit(1)

    # --- Upload to Pinecone ---
    print(f"\n Uploading {len(documents)} chunks to Pinecone...")

    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_pinecone import PineconeVectorStore

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )

    vector_store = PineconeVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        index_name=args.index,
    )

    print(f" Upload complete! {len(documents)} chunks in '{args.index}'")

    # --- Build concept graph ---
    if args.build_graph:
        print("\n Building concept graph...")
        graph = ConceptGraph()
        graph.build_from_documents(documents)
        graph.save()
        concepts = graph.get_all_concepts()
        print(f" Concept graph built: {len(concepts)} unique concepts")
        if concepts[:5]:
            print("   Top concepts:", ", ".join(c for c, _ in concepts[:5]))

    # --- Summary ---
    print("\n" + "=" * 60)
    print(" Ingestion Summary:")

    # Count by book
    books = {}
    for doc in documents:
        title = doc.metadata.get("book_title", "Unknown")
        books[title] = books.get(title, 0) + 1

    for title, count in sorted(books.items()):
        print(f"    {title}: {count} chunks")

    # Count by content type
    types = {}
    for doc in documents:
        ct = doc.metadata.get("content_type", "unknown")
        types[ct] = types.get(ct, 0) + 1

    print(f"\n   Content types:")
    for ct, count in sorted(types.items(), key=lambda x: x[1], reverse=True):
        print(f"      {ct}: {count}")

    print(f"\n   Total chunks: {len(documents)}")
    print(" Done!")


if __name__ == "__main__":
    main()
