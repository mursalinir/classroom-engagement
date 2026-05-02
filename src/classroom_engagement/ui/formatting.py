from __future__ import annotations

from datetime import datetime

from classroom_engagement.core.schemas import PipelineEvent


def format_event_rows(events: list[PipelineEvent]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for event in events:
        rows.append(
            {
                "timestamp": datetime.fromtimestamp(event.timestamp).strftime("%H:%M:%S")
                if event.timestamp
                else "--",
                "type": event.event_type,
                "message": event.message,
            }
        )
    return rows
