"""Render an Answer in the required output format."""
import json

from .operations import Answer


def to_text(query: str, a: Answer) -> str:
    return "\n".join([
        f'Query: "{query}"',
        f"Answer: {a.answer}",
        f"Activity/Event: {a.activity}",
        "Evidence:",
        f"  Timestamp(s): {a.timestamps}",
        f"  Sensor Modality: {a.modality}",
        f"  Sensor Channel(s): {a.channels}",
        f"Explanation: {a.explanation}",
    ])


def to_dict(query: str, a: Answer) -> dict:
    return {
        "query": query,
        "answer": a.answer,
        "activity_event": a.activity,
        "evidence": {"timestamps": a.timestamps, "sensor_modality": a.modality, "sensor_channels": a.channels},
        "explanation": a.explanation,
        "structured": {"op": a.op, "value": a.value, "intervals": [[round(s, 2), round(e, 2)] for s, e in a.intervals]},
    }


def to_json(query: str, a: Answer) -> str:
    return json.dumps(to_dict(query, a), default=str)
