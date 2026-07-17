"""
memory.py — Chat history and learning state management.

Handles conversation memory (Streamlit session state) and persistent
learning tracking (topics studied, weak areas, quiz scores) for
personalized study recommendations.
"""

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, BaseMessage


# ============================================================================
# Chat History Manager
# ============================================================================

class ChatMemory:
    """Manages chat history within Streamlit session state.

    Converts between the Streamlit message format (list of dicts)
    and LangChain message format (HumanMessage / AIMessage).
    """

    @staticmethod
    def init_session(session_state: Any) -> None:
        """Initialize session state keys if missing."""
        if "messages" not in session_state:
            session_state.messages = []
        if "learning_state" not in session_state:
            session_state.learning_state = LearningState()
        if "performance_stats" not in session_state:
            session_state.performance_stats = {}
        if "bookmarks" not in session_state:
            session_state.bookmarks = []
        if "current_mode" not in session_state:
            session_state.current_mode = "explain"

    @staticmethod
    def get_messages(session_state: Any) -> List[Dict[str, str]]:
        """Return the raw message list from session state."""
        return session_state.get("messages", [])

    @staticmethod
    def add_message(
        session_state: Any, role: str, content: str, metadata: Optional[Dict] = None
    ) -> None:
        """Append a message to session history.

        Args:
            session_state: Streamlit session state.
            role: "user" or "assistant".
            content: Message text.
            metadata: Optional dict with sources, timing, mode, etc.
        """
        msg: Dict[str, Any] = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }
        if metadata:
            msg["metadata"] = metadata
        session_state.messages.append(msg)

    @staticmethod
    def to_langchain_history(session_state: Any) -> List[BaseMessage]:
        """Convert session messages to LangChain message objects.

        Used by the history-aware retriever to understand conversation
        context for query reformulation.
        """
        history: List[BaseMessage] = []
        for msg in session_state.get("messages", []):
            if msg["role"] == "user":
                history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                history.append(AIMessage(content=msg["content"]))
        return history

    @staticmethod
    def clear(session_state: Any) -> None:
        """Clear all chat messages (preserves learning state)."""
        session_state.messages = []

    @staticmethod
    def search_history(
        session_state: Any, query: str, role: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Search conversation history for matching messages.

        Simple substring search — sufficient for session-length histories.
        """
        query_lower = query.lower()
        results = []
        for msg in session_state.get("messages", []):
            if role and msg["role"] != role:
                continue
            if query_lower in msg["content"].lower():
                results.append(msg)
        return results

    @staticmethod
    def get_recent_topics(session_state: Any, n: int = 5) -> List[str]:
        """Extract the last N user questions as recent topics."""
        user_msgs = [
            msg["content"]
            for msg in session_state.get("messages", [])
            if msg["role"] == "user"
        ]
        return user_msgs[-n:]


# ============================================================================
# Learning State Tracker
# ============================================================================

class LearningState:
    """Tracks learning progress across sessions for personalization.

    Records which topics were studied, quiz performance, time spent,
    and identifies weak areas needing revision.
    """

    def __init__(self) -> None:
        self.topic_frequency: Dict[str, int] = {}
        self.topic_last_seen: Dict[str, str] = {}
        self.quiz_scores: Dict[str, List[float]] = {}
        self.session_count: int = 0
        self.total_queries: int = 0
        self.modes_used: Dict[str, int] = {}
        self.weak_topics: List[str] = []
        self.strong_topics: List[str] = []

    def record_query(self, query: str, mode: str, keywords: List[str]) -> None:
        """Record a user query and update topic tracking.

        Args:
            query: The user's question.
            mode: Study mode used (explain, quiz, etc.).
            keywords: Extracted keywords from the query.
        """
        self.total_queries += 1
        self.modes_used[mode] = self.modes_used.get(mode, 0) + 1
        now = datetime.now().isoformat()

        for kw in keywords:
            self.topic_frequency[kw] = self.topic_frequency.get(kw, 0) + 1
            self.topic_last_seen[kw] = now

    def record_quiz_score(self, topic: str, score: float) -> None:
        """Record a quiz score for a topic (0.0 to 1.0).

        Updates weak/strong topic classifications.
        """
        if topic not in self.quiz_scores:
            self.quiz_scores[topic] = []
        self.quiz_scores[topic].append(score)
        self._update_classifications()

    def get_weak_topics(self, threshold: float = 0.6) -> List[str]:
        """Return topics where average quiz score is below threshold."""
        weak = []
        for topic, scores in self.quiz_scores.items():
            if scores and sum(scores) / len(scores) < threshold:
                weak.append(topic)
        # Also include frequently-asked topics (may indicate confusion)
        frequent = sorted(
            self.topic_frequency.items(), key=lambda x: x[1], reverse=True
        )
        for topic, count in frequent[:5]:
            if count >= 3 and topic not in weak:
                weak.append(topic)
        return weak

    def get_revision_suggestions(self, n: int = 5) -> List[str]:
        """Suggest topics that need revision (weak + not recently seen)."""
        weak = set(self.get_weak_topics())
        all_topics = set(self.topic_frequency.keys())

        # Topics seen but not recently
        if self.topic_last_seen:
            sorted_by_time = sorted(self.topic_last_seen.items(), key=lambda x: x[1])
            stale = [t for t, _ in sorted_by_time[:n]]
            candidates = list(weak) + [t for t in stale if t not in weak]
        else:
            candidates = list(weak)

        return candidates[:n]

    def get_stats_summary(self) -> Dict[str, Any]:
        """Return a summary dict for display in the UI."""
        return {
            "total_queries": self.total_queries,
            "unique_topics": len(self.topic_frequency),
            "modes_used": dict(self.modes_used),
            "weak_topics": self.get_weak_topics(),
            "top_topics": sorted(
                self.topic_frequency.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:5],
        }

    def _update_classifications(self) -> None:
        """Refresh weak/strong topic lists from quiz scores."""
        self.weak_topics = []
        self.strong_topics = []
        for topic, scores in self.quiz_scores.items():
            avg = sum(scores) / len(scores) if scores else 0
            if avg < 0.6:
                self.weak_topics.append(topic)
            elif avg >= 0.8:
                self.strong_topics.append(topic)

    # ---- Persistence ----

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "topic_frequency": self.topic_frequency,
            "topic_last_seen": self.topic_last_seen,
            "quiz_scores": self.quiz_scores,
            "session_count": self.session_count,
            "total_queries": self.total_queries,
            "modes_used": self.modes_used,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LearningState":
        """Deserialize from a dictionary."""
        state = cls()
        state.topic_frequency = data.get("topic_frequency", {})
        state.topic_last_seen = data.get("topic_last_seen", {})
        state.quiz_scores = data.get("quiz_scores", {})
        state.session_count = data.get("session_count", 0)
        state.total_queries = data.get("total_queries", 0)
        state.modes_used = data.get("modes_used", {})
        state._update_classifications()
        return state

    def save(self, filepath: str) -> None:
        """Persist learning state to a JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "LearningState":
        """Load learning state from a JSON file."""
        if not os.path.exists(filepath):
            return cls()
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return cls.from_dict(json.load(f))
        except (json.JSONDecodeError, KeyError):
            return cls()
