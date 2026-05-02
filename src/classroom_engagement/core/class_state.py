from __future__ import annotations

from .schemas import ClassSnapshot


class ClassStateEngine:
    """Rule-based class state placeholder for the first implementation."""

    def infer_state(self, snapshot: ClassSnapshot) -> str:
        if snapshot.exit_events > 0 and snapshot.attention_ratio < 0.35:
            return "disengaged"
        if snapshot.confusion_ratio >= 0.35 and snapshot.interaction_ratio < 0.30:
            return "confused"
        if snapshot.smile_ratio >= 0.40 and snapshot.interaction_ratio >= 0.35:
            return "energized"
        if snapshot.attention_ratio >= 0.65 and snapshot.engagement_score >= 0.60:
            return "engaged"
        return "neutral"
