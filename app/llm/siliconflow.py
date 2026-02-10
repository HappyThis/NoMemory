from __future__ import annotations

import json

from typing import Any

import httpx

from app.llm.bigmodel import BigModelMessage
from app.llm.errors import LLMError
from app.settings import settings


class SiliconFlowError(LLMError):
    pass


class SiliconFlowClient:
    """
    Minimal OpenAI-compatible Chat Completions client for SiliconFlow.

    It intentionally mirrors the BigModelClient chat_message(...) shape used by RecallAgent:
    returns the raw assistant message dict, including optional tool_calls.
    """

    def __init__(self) -> None:
        if not settings.siliconflow_api_key:
            raise SiliconFlowError("SILICONFLOW_API_KEY not configured")
        self._api_key = settings.siliconflow_api_key

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}

    def _messages_payload(self, messages: list[BigModelMessage]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in messages:
            d: dict[str, Any] = {"role": m.role}
            d["content"] = m.content if m.content is not None else ""
            if m.tool_call_id is not None:
                d["tool_call_id"] = m.tool_call_id
            if m.tool_calls is not None:
                d["tool_calls"] = m.tool_calls
            out.append(d)
        return out

    def _chat_raw(self, *, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            with httpx.Client(timeout=float(settings.siliconflow_chat_timeout_sec)) as client:
                r = client.post(settings.siliconflow_chat_endpoint, headers=self._headers(), json=payload)
                if r.status_code >= 400:
                    raise SiliconFlowError(
                        f"Chat API error: {r.status_code} {r.text}",
                        status_code=r.status_code,
                        response_text=r.text,
                    )
                return r.json()
        except httpx.TimeoutException as e:
            raise SiliconFlowError("Chat API timeout", status_code=504) from e
        except httpx.RequestError as e:
            raise SiliconFlowError(f"Chat API request error: {type(e).__name__}: {e}", status_code=502) from e

    def chat_message(
        self,
        *,
        messages: list[BigModelMessage],
        model: str,
        temperature: float = 0.2,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = "auto",
    ) -> dict[str, Any]:
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
            raise SiliconFlowError("No choices in chat response")
        msg = choices[0].get("message")
        if not isinstance(msg, dict):
            raise SiliconFlowError("Invalid chat response message")
        # Some vendors may use "reasoning" instead of "reasoning_content".
        if "reasoning_content" not in msg and isinstance(msg.get("reasoning"), str):
            msg["reasoning_content"] = msg.get("reasoning")
        return msg

    def chat_json(self, *, messages: list[BigModelMessage], model: str, temperature: float = 0.0) -> dict[str, Any]:
        """
        Best-effort JSON mode (OpenAI-compatible).

        Some providers support `response_format={"type":"json_object"}`. If not supported, callers can fall back to
        parsing `chat_message(...).content`.
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": self._messages_payload(messages),
            "temperature": temperature,
            "stream": False,
            "response_format": {"type": "json_object"},
        }
        data = self._chat_raw(payload=payload)
        choices = data.get("choices") or []
        if not choices:
            raise SiliconFlowError("No choices in chat response")
        msg = choices[0].get("message") or {}
        content = msg.get("content")
        if not isinstance(content, str):
            raise SiliconFlowError("Invalid chat response content")
        try:
            obj = json.loads(content)
        except Exception as e:
            raise SiliconFlowError("Invalid JSON output in JSON mode") from e
        if not isinstance(obj, dict):
            raise SiliconFlowError("Expected JSON object output in JSON mode")
        return obj
