"""
personalization.py — Personalized learning features.

Provides study plan generation, weak topic practice, revision
scheduling, and adaptive difficulty based on the LearningState
tracked in memory.py.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from memory import LearningState


# ============================================================================
# Study Plan Generator
# ============================================================================

def generate_study_plan(
    learning_state: LearningState,
    available_topics: Optional[List[str]] = None,
    days: int = 7,
) -> List[Dict[str, Any]]:
    """Generate a personalized daily study plan.

    Prioritizes:
      1. Weak topics (low quiz scores)
      2. Stale topics (not reviewed recently)
      3. New topics (never studied)

    Args:
        learning_state: Current learning state.
        available_topics: All topics from the concept graph.
        days: Number of days to plan.

    Returns:
        List of daily plans, each with topics, modes, and activities.
    """
    weak = set(learning_state.get_weak_topics())
    revision = set(learning_state.get_revision_suggestions())
    studied = set(learning_state.topic_frequency.keys())

    # Topics never studied
    new_topics: List[str] = []
    if available_topics:
        new_topics = [t for t in available_topics if t not in studied]

    plan: List[Dict[str, Any]] = []

    for day in range(1, days + 1):
        day_plan: Dict[str, Any] = {
            "day": day,
            "date": (datetime.now() + timedelta(days=day - 1)).strftime("%A, %b %d"),
            "activities": [],
        }

        # Day 1-2: Focus on weak topics
        if day <= 2 and weak:
            topic = weak.pop() if weak else None
            if topic:
                day_plan["activities"].extend([
                    {"topic": topic, "mode": "revise", "duration": "15 min", "priority": " High"},
                    {"topic": topic, "mode": "quiz", "duration": "10 min", "priority": " High"},
                    {"topic": topic, "mode": "practice", "duration": "20 min", "priority": " High"},
                ])

        # Day 3-5: Mix of revision and new material
        elif day <= 5:
            if revision:
                rev_topic = revision.pop()
                day_plan["activities"].append(
                    {"topic": rev_topic, "mode": "revise", "duration": "15 min", "priority": " Medium"}
                )
            if new_topics:
                new_topic = new_topics.pop(0)
                day_plan["activities"].extend([
                    {"topic": new_topic, "mode": "explain", "duration": "20 min", "priority": " Normal"},
                    {"topic": new_topic, "mode": "flashcards", "duration": "10 min", "priority": " Normal"},
                ])

        # Day 6-7: Comprehensive review
        else:
            day_plan["activities"].extend([
                {"topic": "All weak topics", "mode": "quiz", "duration": "20 min", "priority": " Medium"},
                {"topic": "Recent topics", "mode": "exam", "duration": "30 min", "priority": " Medium"},
            ])

        if not day_plan["activities"]:
            day_plan["activities"].append(
                {"topic": "Free study", "mode": "explain", "duration": "30 min", "priority": " Normal"}
            )

        plan.append(day_plan)

    return plan


# ============================================================================
# Adaptive Difficulty
# ============================================================================

def get_recommended_difficulty(learning_state: LearningState, topic: str) -> str:
    """Recommend a difficulty level based on past performance.

    Returns:
        One of "beginner", "intermediate", "advanced".
    """
    scores = learning_state.quiz_scores.get(topic, [])

    if not scores:
        # New topic — start at intermediate
        return "intermediate"

    avg_score = sum(scores) / len(scores)
    recent_trend = scores[-3:] if len(scores) >= 3 else scores
    recent_avg = sum(recent_trend) / len(recent_trend)

    if recent_avg >= 0.85:
        return "advanced"
    elif recent_avg >= 0.5:
        return "intermediate"
    return "beginner"


# ============================================================================
# Progress Dashboard Data
# ============================================================================

def get_progress_data(learning_state: LearningState) -> Dict[str, Any]:
    """Compute progress metrics for the dashboard.

    Returns:
        Dict with computed metrics ready for UI display.
    """
    stats = learning_state.get_stats_summary()

    # Overall mastery score (average quiz scores across all topics)
    all_scores: List[float] = []
    for scores in learning_state.quiz_scores.values():
        all_scores.extend(scores)
    mastery = sum(all_scores) / len(all_scores) if all_scores else 0.0

    # Topics by strength
    topics_by_strength = {
        "strong": learning_state.strong_topics,
        "weak": learning_state.weak_topics,
        "untested": [
            t for t in learning_state.topic_frequency
            if t not in learning_state.quiz_scores
        ],
    }

    # Most used modes
    mode_distribution = dict(
        sorted(
            learning_state.modes_used.items(),
            key=lambda x: x[1],
            reverse=True,
        )
    )

    return {
        "total_queries": stats["total_queries"],
        "unique_topics": stats["unique_topics"],
        "mastery_score": round(mastery * 100, 1),
        "topics_by_strength": topics_by_strength,
        "mode_distribution": mode_distribution,
        "top_topics": stats["top_topics"],
        "weak_topics": stats["weak_topics"],
    }


# ============================================================================
# Suggested Questions
# ============================================================================

def suggest_questions(
    learning_state: LearningState,
    current_topic: str = "",
    concept_graph: Optional[Any] = None,
) -> List[str]:
    """Generate suggested follow-up questions.

    Combines:
      - Related concepts from the concept graph
      - Weak topics that need practice
      - Natural follow-up patterns
    """
    suggestions: List[str] = []

    # Topic-based suggestions
    if current_topic:
        suggestions.extend([
            f"Explain {current_topic} in simple terms",
            f"Quiz me on {current_topic}",
            f"What are common mistakes with {current_topic}?",
        ])

        # Related topics from concept graph
        if concept_graph:
            related = concept_graph.get_related_topics(current_topic, top_n=3)
            for r in related:
                suggestions.append(f"Compare {current_topic} vs {r}")

    # Weak topic suggestions
    weak = learning_state.get_weak_topics()
    for topic in weak[:2]:
        suggestions.append(f"Help me revise {topic}")

    # Revision suggestions
    revision = learning_state.get_revision_suggestions()
    for topic in revision[:2]:
        if f"revise {topic}" not in str(suggestions):
            suggestions.append(f"Flashcards for {topic}")

    return suggestions[:8]  # Cap at 8 suggestions


# ============================================================================
# Session Report
# ============================================================================

def generate_session_report(learning_state: LearningState) -> str:
    """Generate a markdown summary of the current study session.

    Used when the user exports their conversation.
    """
    stats = learning_state.get_stats_summary()

    lines = [
        "#  Study Session Report",
        f"**Date:** {datetime.now().strftime('%B %d, %Y')}",
        f"**Total Questions Asked:** {stats['total_queries']}",
        f"**Unique Topics Covered:** {stats['unique_topics']}",
        "",
    ]

    if stats["top_topics"]:
        lines.append("## Most Studied Topics")
        for topic, count in stats["top_topics"]:
            lines.append(f"- **{topic}**: {count} queries")
        lines.append("")

    if stats["weak_topics"]:
        lines.append("##  Topics Needing Review")
        for topic in stats["weak_topics"]:
            lines.append(f"- {topic}")
        lines.append("")

    if stats["modes_used"]:
        lines.append("## Study Modes Used")
        for mode, count in stats["modes_used"].items():
            lines.append(f"- **{mode}**: {count}×")

    return "\n".join(lines)
