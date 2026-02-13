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
    # Optional client-provided id for idempotent ingest.
    message_id: Optional[str] = None
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


class EmbeddingsStatusResponse(BaseModel):
    user_id: str
    provider: str
    model: str
    enabled: bool
    messages: int
    embeddings: int
    candidates: int
    latest_embedding_at: Optional[datetime] = None


class EmbeddingsEnqueueResponse(BaseModel):
    user_id: str
    provider: str
    model: str
    enabled: bool
    missing: int
    scheduled: int


class NeighborsResponse(BaseModel):
    items: list[ChatMessage]

class RecallRequest(BaseModel):
    question: str = Field(min_length=1)


class RecallResponse(BaseModel):
    # A single third-person memory summary derived from evidence (no categories).
    memory_view: str = Field(default="")
    # Evidence messages supporting the summary.
    evidence: list[ChatMessage] = Field(default_factory=list)


class LocomoQARequest(BaseModel):
    question: str = Field(min_length=1)
    # Optional time anchor (ISO8601). If not provided, the service will fall back to the latest message ts.
    now: Optional[datetime] = None


class LocomoQAResponse(BaseModel):
    answer: str = Field(default="")
    evidence: list[ChatMessage] = Field(default_factory=list)


LocomoJudgeLabel = Literal["CORRECT", "WRONG"]


class LocomoJudgeRequest(BaseModel):
    question: str = Field(min_length=1)
    gold_answer: Optional[Any] = None
    pred_answer: str = Field(default="")
    category: Optional[int] = None
    now: Optional[datetime] = None


class LocomoJudgeResponse(BaseModel):
    label: LocomoJudgeLabel
    reason: Optional[str] = None
