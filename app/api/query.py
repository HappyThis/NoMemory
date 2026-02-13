from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from app.api.schemas import (
    ChatMessage,
    EmbeddingsEnqueueResponse,
    EmbeddingsStatusResponse,
    LexicalSearchRequest,
    NeighborsResponse,
    PaginatedMessages,
    SemanticMessage,
    SemanticSearchRequest,
    SemanticSearchResponse,
)
from app.db.models import Message, MessageEmbedding
from app.db.session import get_db
from app.llm.bigmodel import BigModelError
from app.llm.embeddings import embed_query_text, enqueue_embeddings_for_messages
from app.retrieval.lexical import LexicalAnchor, lexical_search
from app.retrieval.messages import list_messages
from app.retrieval.neighbors import get_neighbors
from app.retrieval.semantic import semantic_search
from app.settings import settings
from app.utils.cursor import CursorError, make_seek_cursor, parse_seek_anchor

router = APIRouter(prefix="/v1", tags=["query"])


def _to_msg(m) -> ChatMessage:
    return ChatMessage(
        message_id=m.message_id,
        ts=m.ts,
        user_id=m.user_id,
        role=m.role,
        content=m.content,
        meta=m.meta,
    )


@router.get("/users/{user_id}/messages", response_model=PaginatedMessages)
def messages_list(
    user_id: str,
    since: Optional[datetime] = Query(default=None),
    until: Optional[datetime] = Query(default=None),
    role: Optional[str] = Query(default=None),
    page_size: int = Query(default=50, ge=1, le=200),
    cursor: Optional[str] = Query(default=None),
    db: Session = Depends(get_db),
) -> PaginatedMessages:
    try:
        anchor = parse_seek_anchor(cursor, expected_kind="messages_list")
    except CursorError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    items = list_messages(
        db,
        user_id=user_id,
        since=since,
        until=until,
        role=role,
        page_size=page_size + 1,
        anchor=anchor,
    )
    next_cursor = None
    if len(items) > page_size:
        last = items[page_size - 1]
        next_cursor = make_seek_cursor(kind="messages_list", ts=last.ts, message_id=last.message_id)
        items = items[:page_size]

    return PaginatedMessages(items=[_to_msg(m) for m in items], next_cursor=next_cursor)


@router.get("/users/{user_id}/embeddings/status", response_model=EmbeddingsStatusResponse)
def embeddings_status(
    user_id: str,
    provider: Optional[str] = Query(default=None),
    model: Optional[str] = Query(default=None),
    db: Session = Depends(get_db),
) -> EmbeddingsStatusResponse:
    provider_val = provider or settings.embedding_provider
    model_val = model or settings.bigmodel_embedding_model
    enabled = bool(settings.bigmodel_api_key) and provider_val == "bigmodel"

    messages = db.execute(select(func.count()).select_from(Message).where(Message.user_id == user_id)).scalar_one()
    embeddings = db.execute(
        select(func.count())
        .select_from(MessageEmbedding)
        .where(
            MessageEmbedding.user_id == user_id,
            MessageEmbedding.provider == provider_val,
            MessageEmbedding.model == model_val,
        )
    ).scalar_one()

    candidates = db.execute(
        select(func.count())
        .select_from(MessageEmbedding)
        .join(
            Message,
            and_(
                MessageEmbedding.user_id == Message.user_id,
                MessageEmbedding.message_id == Message.message_id,
            ),
        )
        .where(
            MessageEmbedding.user_id == user_id,
            MessageEmbedding.provider == provider_val,
            MessageEmbedding.model == model_val,
        )
    ).scalar_one()

    latest_row = db.execute(
        select(MessageEmbedding.created_at)
        .where(
            MessageEmbedding.user_id == user_id,
            MessageEmbedding.provider == provider_val,
            MessageEmbedding.model == model_val,
        )
        .order_by(MessageEmbedding.created_at.desc())
        .limit(1)
    ).first()
    latest = latest_row[0] if latest_row else None

    return EmbeddingsStatusResponse(
        user_id=user_id,
        provider=str(provider_val),
        model=str(model_val),
        enabled=bool(enabled),
        messages=int(messages or 0),
        embeddings=int(embeddings or 0),
        candidates=int(candidates or 0),
        latest_embedding_at=latest,
    )


@router.post("/users/{user_id}/embeddings/enqueue", response_model=EmbeddingsEnqueueResponse)
def embeddings_enqueue(
    user_id: str,
    background: BackgroundTasks,
    provider: Optional[str] = Query(default=None),
    model: Optional[str] = Query(default=None),
    limit: int = Query(default=500, ge=1, le=5000),
    db: Session = Depends(get_db),
) -> EmbeddingsEnqueueResponse:
    provider_val = provider or settings.embedding_provider
    model_val = model or settings.bigmodel_embedding_model
    enabled = bool(settings.bigmodel_api_key) and provider_val == "bigmodel"

    if not enabled:
        return EmbeddingsEnqueueResponse(
            user_id=user_id,
            provider=str(provider_val),
            model=str(model_val),
            enabled=False,
            missing=0,
            scheduled=0,
        )

    join_cond = and_(
        MessageEmbedding.user_id == Message.user_id,
        MessageEmbedding.message_id == Message.message_id,
        MessageEmbedding.provider == provider_val,
        MessageEmbedding.model == model_val,
    )

    missing = db.execute(
        select(func.count())
        .select_from(Message)
        .outerjoin(MessageEmbedding, join_cond)
        .where(Message.user_id == user_id, MessageEmbedding.message_id.is_(None))
    ).scalar_one()

    rows = db.execute(
        select(Message.user_id, Message.message_id, Message.content)
        .select_from(Message)
        .outerjoin(MessageEmbedding, join_cond)
        .where(Message.user_id == user_id, MessageEmbedding.message_id.is_(None))
        .order_by(Message.ts.asc())
        .limit(int(limit))
    ).all()

    now = datetime.now(tz=timezone.utc)
    scheduled = len(rows)
    if scheduled:
        background.add_task(
            enqueue_embeddings_for_messages,
            [(r[0], r[1], r[2]) for r in rows],
            provider=str(provider_val),
            model=str(model_val),
            requested_at=now,
        )

    return EmbeddingsEnqueueResponse(
        user_id=user_id,
        provider=str(provider_val),
        model=str(model_val),
        enabled=True,
        missing=int(missing or 0),
        scheduled=int(scheduled),
    )


@router.post("/messages/lexical_search", response_model=PaginatedMessages)
def lexical_search_api(req: LexicalSearchRequest, db: Session = Depends(get_db)) -> PaginatedMessages:
    time_range = req.filter.time_range if req.filter and req.filter.time_range else None
    since = time_range.since if time_range else None
    until = time_range.until if time_range else None
    role = req.filter.role if req.filter else None

    anchor = None
    if req.cursor:
        try:
            from app.utils.cursor import decode_cursor

            payload = decode_cursor(req.cursor)
        except CursorError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        if payload.get("kind") != "lexical_search":
            raise HTTPException(status_code=400, detail="Cursor kind mismatch")
        anchor = LexicalAnchor(
            rank=float(payload["rank"]),
            ts=datetime.fromisoformat(payload["ts"]),
            message_id=str(payload["message_id"]),
        )

    rows = lexical_search(
        db,
        user_id=req.user_id,
        query_text=req.query_text,
        since=since,
        until=until,
        role=role,
        page_size=req.page_size + 1,
        anchor=anchor,
    )
    next_cursor = None
    if len(rows) > req.page_size:
        m, rank = rows[req.page_size - 1]
        next_cursor = make_seek_cursor(
            kind="lexical_search",
            ts=m.ts,
            message_id=m.message_id,
            extra={"rank": round(float(rank), 8)},
        )
        rows = rows[: req.page_size]

    return PaginatedMessages(items=[_to_msg(m) for (m, _rank) in rows], next_cursor=next_cursor)


@router.post("/messages/semantic_search", response_model=SemanticSearchResponse)
def semantic_search_api(req: SemanticSearchRequest, db: Session = Depends(get_db)) -> SemanticSearchResponse:
    time_range = req.filter.time_range if req.filter and req.filter.time_range else None
    since = time_range.since if time_range else None
    until = time_range.until if time_range else None
    role = req.filter.role if req.filter else None

    provider = req.provider or settings.embedding_provider
    model = req.model or settings.bigmodel_embedding_model

    if req.query_embedding is None and req.query_text is None:
        raise HTTPException(status_code=400, detail="One of query_text or query_embedding is required")
    if req.query_embedding is None:
        try:
            req.query_embedding = embed_query_text(req.query_text or "", provider=provider, model=model)
        except (BigModelError, ValueError) as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    rows = semantic_search(
        db,
        user_id=req.user_id,
        query_embedding=req.query_embedding,
        since=since,
        until=until,
        role=role,
        top_k=req.top_k,
        min_score=req.min_score,
        provider=provider,
        model=model,
    )
    items = [
        SemanticMessage(
            message_id=m.message_id,
            ts=m.ts,
            user_id=m.user_id,
            role=m.role,
            content=m.content,
            meta=m.meta,
            semantic_score=score,
        )
        for (m, score) in rows
    ]
    return SemanticSearchResponse(items=items)


@router.get("/users/{user_id}/messages/{message_id}/neighbors", response_model=NeighborsResponse)
def neighbors_api(
    user_id: str,
    message_id: str,
    before: int = Query(default=20, ge=0, le=200),
    after: int = Query(default=0, ge=0, le=200),
    db: Session = Depends(get_db),
) -> NeighborsResponse:
    items = get_neighbors(db, user_id=user_id, message_id=message_id, before=before, after=after)
    return NeighborsResponse(items=[_to_msg(m) for m in items])
