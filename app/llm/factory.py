from __future__ import annotations

from typing import Any

from app.llm.bigmodel import BigModelClient
from app.llm.siliconflow import SiliconFlowClient
from app.settings import settings


def get_chat_model() -> str:
    """
    Return the provider-specific chat model name for the current `LLM_PROVIDER`.

    We keep `LLM_MODEL` as a legacy fallback for older .env files, but new configs
    should use provider-specific model variables (e.g. BIGMODEL_LLM_MODEL).
    """
    if settings.llm_provider == "bigmodel":
        return (settings.bigmodel_llm_model or "").strip() or (settings.llm_model or "").strip()
    if settings.llm_provider == "siliconflow":
        return (settings.siliconflow_llm_model or "").strip() or (settings.llm_model or "").strip()
    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")


def get_chat_client() -> Any:
    """
    Provider router for chat-capable LLM clients.

    - Keep this as the single place that switches on `settings.llm_provider`.
    - Each provider can optionally expose structured-output helpers (e.g. `chat_json`).
    """
    if settings.llm_provider == "bigmodel":
        return BigModelClient()
    if settings.llm_provider == "siliconflow":
        return SiliconFlowClient()
    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
