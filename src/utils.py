"""
utils.py — Các hàm tiện ích dùng chung
========================================
"""
import json
import re
from datetime import datetime
from typing import List


def _parse_timestamps(timestamps_json: str) -> List[datetime]:
    """Parse JSON list of ISO 8601 timestamps → list of datetime objects."""
    try:
        timestamps = json.loads(timestamps_json)
    except (json.JSONDecodeError, TypeError):
        return []

    dts = []
    for ts in timestamps:
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            dts.append(dt)
        except ValueError:
            continue

    return dts


def _parse_titles(titles_json: str) -> List[str]:
    """Parse JSON list of titles → list of strings."""
    try:
        titles = json.loads(titles_json)
        return [t.strip() for t in titles if t.strip()]
    except (json.JSONDecodeError, TypeError):
        return []


def _flatten_titles(titles_json: str) -> str:
    """Parse JSON list of titles → single string để regex."""
    titles = _parse_titles(titles_json)
    return " ".join(titles) if titles else ""