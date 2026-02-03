from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class TimeRange(BaseModel):
    since: str | None = Field(default=None, description="ISO8601 datetime or null")
    until: str | None = Field(default=None, description="ISO8601 datetime or null")


class Filter(BaseModel):
    role: Literal["user", "any"]
    time_range: TimeRange


class MessagesListArgs(BaseModel):
    since: str | None = None
    until: str | None = None
    role: Literal["user", "any"]
    page_size: int | None = None
    cursor: str | None = None


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
            description="List messages by time range and optional role filter. Requires role and at least one of since/until.",
            model=MessagesListArgs,
        ),
        "lexical_search": _tool_schema(
            name="lexical_search",
            description="Lexical search over messages with required filter.time_range and filter.role.",
            model=LexicalSearchArgs,
        ),
        "semantic_search": _tool_schema(
            name="semantic_search",
            description="Semantic search over messages with required filter.time_range and filter.role.",
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

