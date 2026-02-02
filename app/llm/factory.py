from __future__ import annotations

from typing import Any

from app.llm.bigmodel import BigModelClient
from app.settings import settings


def get_chat_client() -> Any:
    """
    Provider router for chat-capable LLM clients.

    - Keep this as the single place that switches on `settings.llm_provider`.
    - Each provider can optionally expose structured-output helpers (e.g. `chat_json`).
    """
    if settings.llm_provider == "bigmodel":
        return BigModelClient()
    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")

