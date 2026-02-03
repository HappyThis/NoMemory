from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


Role = Literal["user", "assistant", "system"]


class TimeRange(BaseModel):
    since: Optional[datetime] = None
    until: Optional[datetime] = None


class QueryFilter(BaseModel):
    time_range: Optional[TimeRange] = None
    role: Optional[Role] = None


class ChatMessage(BaseModel):
    message_id: str
    ts: datetime
    user_id: str
    role: str
    content: str
    meta: Optional[dict[str, Any]] = None


class PaginatedMessages(BaseModel):
    items: list[ChatMessage]
    next_cursor: Optional[str] = None


class IngestMessage(BaseModel):
    ts: datetime
    role: Role
    content: str
    meta: Optional[dict[str, Any]] = None


class IngestBatchRequest(BaseModel):
    items: list[IngestMessage] = Field(min_length=1, max_length=500)


class IngestBatchResponse(BaseModel):
    inserted: int
    ignored: int
    failed: int = 0
    message_ids: list[str] = Field(default_factory=list)


class LexicalSearchRequest(BaseModel):
    user_id: str
    query_text: str = Field(min_length=1)
    filter: Optional[QueryFilter] = None
    page_size: int = Field(default=50, ge=1, le=200)
    cursor: Optional[str] = None


class SemanticSearchRequest(BaseModel):
    user_id: str
    filter: Optional[QueryFilter] = None
    query_text: Optional[str] = None
    query_embedding: Optional[list[float]] = None
    top_k: int = Field(default=20, ge=1, le=200)
    min_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    provider: Optional[str] = None
    model: Optional[str] = None


class SemanticMessage(ChatMessage):
    semantic_score: float


class SemanticSearchResponse(BaseModel):
    items: list[SemanticMessage]


class NeighborsResponse(BaseModel):
    items: list[ChatMessage]

class RecallRequest(BaseModel):
    question: str = Field(min_length=1)


class MemoryView(BaseModel):
    preferences: list[str] = Field(default_factory=list)
    profile: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)


class RecallLimits(BaseModel):
    time_range: TimeRange
    role: str
    messages_considered: str = "unknown"


class RecallResponse(BaseModel):
    memory_view: MemoryView
    evidence: list[ChatMessage]
    limits: RecallLimits
