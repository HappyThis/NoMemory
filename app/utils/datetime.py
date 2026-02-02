from __future__ import annotations

from datetime import datetime
from typing import Optional


def parse_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    v = value.strip()
    # Support RFC3339 "Z"
    if v.endswith("Z"):
        v = v[:-1] + "+00:00"
    return datetime.fromisoformat(v)

