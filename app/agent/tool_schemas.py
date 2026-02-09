from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class TimeRange(BaseModel):
    since: datetime | None = Field(default=None, description="ISO8601 datetime or null")
    until: datetime | None = Field(default=None, description="ISO8601 datetime or null")

    @model_validator(mode="after")
    def _require_one_bound(self) -> "TimeRange":
        if self.since is None and self.until is None:
            raise ValueError("time_range requires at least one of since/until")
        return self


class Filter(BaseModel):
    role: Literal["user", "assistant", "any"]
    time_range: TimeRange


class MessagesListArgs(BaseModel):
    since: datetime | None = None
    until: datetime | None = None
    role: Literal["user", "assistant", "any"]
    page_size: int | None = None
    cursor: str | None = None

    @model_validator(mode="after")
    def _require_one_bound(self) -> "MessagesListArgs":
        if self.since is None and self.until is None:
            raise ValueError("messages_list requires at least one of since/until")
        return self


class LexicalSearchArgs(BaseModel):
    query_text: str
    filter: Filter
    page_size: int | None = None
    cursor: str | None = None


class SemanticSearchArgs(BaseModel):
    query_text: str | None = None
    filter: Filter
    top_k: int | None = None
    min_score: float | None = None


class NeighborsArgs(BaseModel):
    message_id: str
    before: int | None = None
    after: int | None = None


def _inline_refs(schema: Any) -> Any:
    """
    Pydantic v2 uses $defs + $ref. Many tool-calling endpoints accept that, but inlining
    keeps the payload simple and vendor-agnostic.
    """
    if not isinstance(schema, dict):
        if isinstance(schema, list):
            return [_inline_refs(x) for x in schema]
        return schema

    defs = schema.get("$defs") if isinstance(schema.get("$defs"), dict) else {}

    def deref(node: Any, *, seen: set[str]) -> Any:
        if isinstance(node, list):
            return [deref(x, seen=seen) for x in node]
        if not isinstance(node, dict):
            return node
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            name = ref.split("#/$defs/", 1)[1]
            if name in seen:
                return node
            target = defs.get(name)
            if not isinstance(target, dict):
                return node
            seen2 = set(seen)
            seen2.add(name)
            # Merge target with node (node overrides), but drop $ref.
            merged = dict(target)
            merged.update({k: v for k, v in node.items() if k != "$ref"})
            return deref(merged, seen=seen2)

        return {k: deref(v, seen=seen) for k, v in node.items()}

    out = deref(schema, seen=set())
    if isinstance(out, dict):
        out.pop("$defs", None)
        out.pop("title", None)
    return out


def _tool_schema(*, name: str, description: str, model: type[BaseModel]) -> dict[str, Any]:
    raw = model.model_json_schema()
    parameters = _inline_refs(raw)
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


def bigmodel_tool_schemas(*, allowed_tools: list[str]) -> list[dict[str, Any]]:
    """
    Return tool schemas in BigModel/OpenAI-style `tools=[...]` format.
    Filtered by runtime `allowed_tools`.
    """
    schemas: dict[str, dict[str, Any]] = {
        "messages_list": _tool_schema(
            name="messages_list",
            description="List messages by time range and role filter. role must be one of: user/assistant/any. Requires at least one of since/until.",
            model=MessagesListArgs,
        ),
        "lexical_search": _tool_schema(
            name="lexical_search",
            description="Lexical search over messages with required filter.time_range and filter.role (user/assistant/any).",
            model=LexicalSearchArgs,
        ),
        "semantic_search": _tool_schema(
            name="semantic_search",
            description="Semantic search over messages with required filter.time_range and filter.role (user/assistant/any).",
            model=SemanticSearchArgs,
        ),
        "neighbors": _tool_schema(
            name="neighbors",
            description="Fetch neighbor messages around a given message_id.",
            model=NeighborsArgs,
        ),
    }
    out: list[dict[str, Any]] = []
    for tool in allowed_tools:
        s = schemas.get(str(tool))
        if s:
            out.append(s)
    return out
