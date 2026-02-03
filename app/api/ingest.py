from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, Depends
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.api.schemas import IngestBatchRequest, IngestBatchResponse
from app.db.models import Message
from app.db.session import get_db
from app.llm.embeddings import enqueue_embeddings_for_messages
from app.settings import settings

router = APIRouter(prefix="/v1", tags=["ingest"])


@router.post("/users/{user_id}/messages:batch", response_model=IngestBatchResponse)
def ingest_messages(
    user_id: str,
    req: IngestBatchRequest,
    background: BackgroundTasks,
    db: Session = Depends(get_db),
) -> IngestBatchResponse:
    now = datetime.now(tz=timezone.utc)
    rows = [
        {
            "user_id": user_id,
            "message_id": uuid.uuid4().hex,
            "ts": item.ts,
            "role": item.role,
            "content": item.content,
            "meta": item.meta,
        }
        for item in req.items
    ]

    stmt = (
        insert(Message)
        .values(rows)
        .returning(Message.user_id, Message.message_id, Message.content)
    )
    inserted = db.execute(stmt).all()
    db.commit()

    if settings.bigmodel_api_key:
        background.add_task(
            enqueue_embeddings_for_messages,
            [(r[0], r[1], r[2]) for r in inserted],
            provider=settings.llm_provider,
            model=settings.bigmodel_embedding_model,
            requested_at=now,
        )

    inserted_count = len(inserted)
    return IngestBatchResponse(inserted=inserted_count, ignored=0, message_ids=[r[1] for r in inserted])
