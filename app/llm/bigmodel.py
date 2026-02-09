from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import json
import httpx

from app.settings import settings


class BigModelError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        response_text: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_text = response_text


@dataclass(frozen=True)
class BigModelMessage:
    role: str
    content: str | None = None
    # Function-calling (tools) support
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class BigModelClient:
    def __init__(self) -> None:
        if not settings.bigmodel_api_key:
            raise BigModelError("BIGMODEL_API_KEY not configured")
        self._api_key = settings.bigmodel_api_key

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}

    def _messages_payload(self, messages: list[BigModelMessage]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in messages:
            d: dict[str, Any] = {"role": m.role}
            # Some messages (e.g. assistant tool_calls) may omit content. Keep the field present as a string.
            d["content"] = m.content if m.content is not None else ""
            if m.tool_call_id is not None:
                d["tool_call_id"] = m.tool_call_id
            if m.tool_calls is not None:
                d["tool_calls"] = m.tool_calls
            out.append(d)
        return out

    def embeddings(self, *, inputs: list[str], model: str, dimensions: Optional[int] = None) -> list[list[float]]:
        payload: dict[str, Any] = {"model": model, "input": inputs}
        if dimensions is not None:
            payload["dimensions"] = dimensions
        try:
            with httpx.Client(timeout=float(settings.bigmodel_embedding_timeout_sec)) as client:
                r = client.post(settings.bigmodel_embedding_endpoint, headers=self._headers(), json=payload)
                if r.status_code >= 400:
                    raise BigModelError(
                        f"Embeddings API error: {r.status_code} {r.text}",
                        status_code=r.status_code,
                        response_text=r.text,
                    )
                data = r.json()
        except httpx.TimeoutException as e:
            raise BigModelError("Embeddings API timeout", status_code=504) from e
        except httpx.RequestError as e:
            raise BigModelError(f"Embeddings API request error: {type(e).__name__}: {e}", status_code=502) from e
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
        try:
            with httpx.Client(timeout=float(settings.bigmodel_chat_timeout_sec)) as client:
                r = client.post(settings.bigmodel_chat_endpoint, headers=self._headers(), json=payload)
                if r.status_code >= 400:
                    raise BigModelError(
                        f"Chat API error: {r.status_code} {r.text}",
                        status_code=r.status_code,
                        response_text=r.text,
                    )
                return r.json()
        except httpx.TimeoutException as e:
            raise BigModelError("Chat API timeout", status_code=504) from e
        except httpx.RequestError as e:
            raise BigModelError(f"Chat API request error: {type(e).__name__}: {e}", status_code=502) from e

    def chat(self, *, messages: list[BigModelMessage], model: str, temperature: float = 0.2) -> str:
        payload = {
            "model": model,
            "messages": self._messages_payload(messages),
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

    def chat_message(
        self,
        *,
        messages: list[BigModelMessage],
        model: str,
        temperature: float = 0.2,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = "auto",
    ) -> dict[str, Any]:
        """
        Chat that returns the raw assistant message object (supports tool_calls).

        BigModel tool-calling docs show the response message may contain `tool_calls`, and tool results should be sent
        back as messages with `role="tool"` and `tool_call_id`.
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": self._messages_payload(messages),
            "temperature": temperature,
            "stream": False,
        }
        if tools is not None:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice
        data = self._chat_raw(payload=payload)
        choices = data.get("choices") or []
        if not choices:
            raise BigModelError("No choices in chat response")
        msg = choices[0].get("message")
        if not isinstance(msg, dict):
            raise BigModelError("Invalid chat response message")
        return msg

    def chat_json(self, *, messages: list[BigModelMessage], model: str, temperature: float = 0.2) -> dict[str, Any]:
        """
        Official structured output mode (JSON object).

        Note: This is provider-specific; other vendors may use different fields.
        """
        payload = {
            "model": model,
            "messages": self._messages_payload(messages),
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
