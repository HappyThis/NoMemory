from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import json
import httpx

from app.settings import settings


class BigModelError(RuntimeError):
    pass


@dataclass(frozen=True)
class BigModelMessage:
    role: str
    content: str


class BigModelClient:
    def __init__(self) -> None:
        if not settings.bigmodel_api_key:
            raise BigModelError("BIGMODEL_API_KEY not configured")
        self._api_key = settings.bigmodel_api_key

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}

    def embeddings(self, *, inputs: list[str], model: str, dimensions: Optional[int] = None) -> list[list[float]]:
        payload: dict[str, Any] = {"model": model, "input": inputs}
        if dimensions is not None:
            payload["dimensions"] = dimensions
        with httpx.Client(timeout=30.0) as client:
            r = client.post(settings.bigmodel_embedding_endpoint, headers=self._headers(), json=payload)
            if r.status_code >= 400:
                raise BigModelError(f"Embeddings API error: {r.status_code} {r.text}")
            data = r.json()
        items = data.get("data") or []
        embeddings: list[list[float]] = []
        for item in items:
            emb = item.get("embedding")
            if not isinstance(emb, list):
                raise BigModelError("Invalid embeddings response")
            embeddings.append([float(x) for x in emb])
        if len(embeddings) != len(inputs):
            raise BigModelError("Embeddings count mismatch")
        return embeddings

    def _chat_raw(self, *, payload: dict[str, Any]) -> dict[str, Any]:
        with httpx.Client(timeout=60.0) as client:
            r = client.post(settings.bigmodel_chat_endpoint, headers=self._headers(), json=payload)
            if r.status_code >= 400:
                raise BigModelError(f"Chat API error: {r.status_code} {r.text}")
            return r.json()

    def chat(self, *, messages: list[BigModelMessage], model: str, temperature: float = 0.2) -> str:
        payload = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "stream": False,
        }
        data = self._chat_raw(payload=payload)
        choices = data.get("choices") or []
        if not choices:
            raise BigModelError("No choices in chat response")
        msg = choices[0].get("message") or {}
        content = msg.get("content")
        if not isinstance(content, str):
            raise BigModelError("Invalid chat response content")
        return content

    def chat_json(self, *, messages: list[BigModelMessage], model: str, temperature: float = 0.2) -> dict[str, Any]:
        """
        Official structured output mode (JSON object).

        Note: This is provider-specific; other vendors may use different fields.
        """
        payload = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "stream": False,
            "response_format": {"type": "json_object"},
        }
        data = self._chat_raw(payload=payload)
        choices = data.get("choices") or []
        if not choices:
            raise BigModelError("No choices in chat response")
        msg = choices[0].get("message") or {}
        content = msg.get("content")
        if not isinstance(content, str):
            raise BigModelError("Invalid chat response content")
        try:
            obj = json.loads(content)
        except Exception as e:
            raise BigModelError("Invalid JSON output in JSON mode") from e
        if not isinstance(obj, dict):
            raise BigModelError("Expected JSON object output in JSON mode")
        return obj
