"""
concept_graph.py — Lightweight concept knowledge graph.

Builds a graph of concept relationships from chunk metadata
(concept_tags). Enables "Related Topics", "Prerequisites",
and "Next Concepts" features.

The graph is stored as JSON for simplicity (no external graph DB
needed for deployment).
"""

import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from langchain_core.documents import Document

from config import CONCEPT_GRAPH_FILE


# ============================================================================
# Concept Graph
# ============================================================================

class ConceptGraph:
    """A lightweight concept relationship graph.

    Nodes are concepts (keywords extracted during ingestion).
    Edges represent co-occurrence within the same chunk — concepts
    that appear together are related.

    Edge weight = number of chunks where both concepts co-occur.
    """

    def __init__(self) -> None:
        # node → set of connected nodes
        self._adjacency: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        # node → list of (source, page) where it appears
        self._locations: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        # node → total occurrence count
        self._frequency: Dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------ #
    # Building
    # ------------------------------------------------------------------ #

    def add_document(self, doc: Document) -> None:
        """Add a document's concept tags to the graph.

        Reads `concept_tags` from metadata (comma-separated keywords)
        and creates edges between all co-occurring concepts.
        """
        tags_str = doc.metadata.get("concept_tags", "")
        if not tags_str:
            return

        tags = [t.strip().lower() for t in tags_str.split(",") if t.strip()]
        if not tags:
            return

        location = {
            "source": doc.metadata.get("source", ""),
            "book_title": doc.metadata.get("book_title", ""),
            "page": doc.metadata.get("page", 0),
            "chapter": doc.metadata.get("chapter", ""),
        }

        # Record each concept
        for tag in tags:
            self._frequency[tag] += 1
            self._locations[tag].append(location)

        # Create edges between all pairs (co-occurrence)
        for i, tag_a in enumerate(tags):
            for tag_b in tags[i + 1 :]:
                self._adjacency[tag_a][tag_b] += 1
                self._adjacency[tag_b][tag_a] += 1

    def build_from_documents(self, documents: List[Document]) -> None:
        """Build the graph from a list of Documents."""
        for doc in documents:
            self.add_document(doc)

    # ------------------------------------------------------------------ #
    # Querying
    # ------------------------------------------------------------------ #

    def get_related_topics(self, concept: str, top_n: int = 8) -> List[str]:
        """Return the most related concepts (by co-occurrence weight)."""
        concept = concept.lower().strip()
        if concept not in self._adjacency:
            # Fuzzy match: check if concept is a substring of any node
            matches = [n for n in self._adjacency if concept in n or n in concept]
            if matches:
                concept = matches[0]
            else:
                return []

        neighbors = self._adjacency[concept]
        sorted_neighbors = sorted(neighbors.items(), key=lambda x: x[1], reverse=True)
        return [node for node, _ in sorted_neighbors[:top_n]]

    def get_prerequisites(self, concept: str, top_n: int = 5) -> List[str]:
        """Estimate prerequisite concepts.

        Heuristic: prerequisites are related concepts that appear
        earlier in the book (lower page numbers on average) and
        are more frequent (more foundational).
        """
        concept = concept.lower().strip()
        related = self.get_related_topics(concept, top_n=15)

        concept_avg_page = self._avg_page(concept)
        if concept_avg_page == 0:
            return related[:top_n]

        # Concepts appearing earlier and more frequently
        prerequisites = []
        for r in related:
            r_avg_page = self._avg_page(r)
            if r_avg_page > 0 and r_avg_page < concept_avg_page:
                prerequisites.append((r, r_avg_page, self._frequency[r]))

        # Sort by page (earlier = more likely prerequisite)
        prerequisites.sort(key=lambda x: x[1])
        return [p[0] for p in prerequisites[:top_n]]

    def get_advanced_topics(self, concept: str, top_n: int = 5) -> List[str]:
        """Get topics that build on this concept (appear later)."""
        concept = concept.lower().strip()
        related = self.get_related_topics(concept, top_n=15)

        concept_avg_page = self._avg_page(concept)
        if concept_avg_page == 0:
            return related[:top_n]

        advanced = []
        for r in related:
            r_avg_page = self._avg_page(r)
            if r_avg_page > 0 and r_avg_page > concept_avg_page:
                advanced.append((r, r_avg_page))

        advanced.sort(key=lambda x: x[1])
        return [a[0] for a in advanced[:top_n]]

    def get_frequently_confused(self, concept: str, top_n: int = 3) -> List[str]:
        """Get concepts that are often confused with this one.

        Heuristic: strongly co-occurring concepts in the same
        chapter/section that are distinct (different pages).
        """
        concept = concept.lower().strip()
        if concept not in self._adjacency:
            return []

        neighbors = self._adjacency[concept]
        # High co-occurrence + different average page = potentially confused
        candidates = []
        concept_pages = {loc["page"] for loc in self._locations.get(concept, [])}

        for neighbor, weight in neighbors.items():
            neighbor_pages = {loc["page"] for loc in self._locations.get(neighbor, [])}
            # Concepts that appear on overlapping pages but aren't identical
            overlap = concept_pages & neighbor_pages
            if overlap and weight >= 2:
                candidates.append((neighbor, weight))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return [c[0] for c in candidates[:top_n]]

    def get_all_concepts(self) -> List[Tuple[str, int]]:
        """Return all concepts sorted by frequency."""
        return sorted(self._frequency.items(), key=lambda x: x[1], reverse=True)

    def get_concept_locations(self, concept: str) -> List[Dict[str, Any]]:
        """Return all locations where a concept appears."""
        return self._locations.get(concept.lower().strip(), [])

    def _avg_page(self, concept: str) -> float:
        """Average page number where a concept appears."""
        locations = self._locations.get(concept, [])
        pages = [loc["page"] for loc in locations if loc.get("page", 0) > 0]
        return sum(pages) / len(pages) if pages else 0

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, filepath: str = CONCEPT_GRAPH_FILE) -> None:
        """Save graph to JSON file."""
        data = {
            "adjacency": {
                k: dict(v) for k, v in self._adjacency.items()
            },
            "locations": dict(self._locations),
            "frequency": dict(self._frequency),
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    @classmethod
    def load(cls, filepath: str = CONCEPT_GRAPH_FILE) -> "ConceptGraph":
        """Load graph from JSON file."""
        graph = cls()
        if not os.path.exists(filepath):
            return graph
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            for k, v in data.get("adjacency", {}).items():
                graph._adjacency[k] = defaultdict(int, v)
            graph._locations = defaultdict(list, data.get("locations", {}))
            graph._frequency = defaultdict(int, data.get("frequency", {}))
        except (json.JSONDecodeError, KeyError):
            pass
        return graph

    # ------------------------------------------------------------------ #
    # Visualization
    # ------------------------------------------------------------------ #

    def to_mermaid(self, concept: str, depth: int = 1) -> str:
        """Generate a Mermaid graph diagram centered on a concept.

        Args:
            concept: Central concept node.
            depth: How many levels of neighbors to include.

        Returns:
            Mermaid graph markup string.
        """
        concept = concept.lower().strip()
        lines = ["graph TD"]
        visited: Set[str] = set()

        def _add_edges(node: str, current_depth: int) -> None:
            if current_depth > depth or node in visited:
                return
            visited.add(node)

            neighbors = self._adjacency.get(node, {})
            sorted_n = sorted(neighbors.items(), key=lambda x: x[1], reverse=True)

            for neighbor, weight in sorted_n[:6]:
                # Sanitize labels for Mermaid
                n_label = node.replace('"', "'").title()
                nb_label = neighbor.replace('"', "'").title()
                n_id = node.replace(" ", "_")
                nb_id = neighbor.replace(" ", "_")
                lines.append(f'    {n_id}["{n_label}"] --> {nb_id}["{nb_label}"]')

                if current_depth < depth:
                    _add_edges(neighbor, current_depth + 1)

        _add_edges(concept, 0)
        return "\n".join(lines) if len(lines) > 1 else ""
