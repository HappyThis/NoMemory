from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from app.db.models import Message, MessageEmbedding


def semantic_search(
    db: Session,
    *,
    user_id: str,
    query_embedding: list[float],
    since: Optional[datetime],
    until: Optional[datetime],
    role: Optional[str],
    top_k: int,
    min_score: Optional[float],
    provider: str,
    model: str,
) -> list[tuple[Message, float]]:
    distance = MessageEmbedding.embedding.cosine_distance(query_embedding)
    score = (1.0 - distance).label("semantic_score")

    stmt = (
        select(Message, score)
        .join(
            MessageEmbedding,
            and_(
                MessageEmbedding.user_id == Message.user_id,
                MessageEmbedding.message_id == Message.message_id,
            ),
        )
        .where(
            Message.user_id == user_id,
            MessageEmbedding.provider == provider,
            MessageEmbedding.model == model,
        )
    )
    if role:
        stmt = stmt.where(Message.role == role)
    if since:
        stmt = stmt.where(Message.ts >= since)
    if until:
        stmt = stmt.where(Message.ts < until)
    if min_score is not None:
        stmt = stmt.where(score >= float(min_score))

    stmt = stmt.order_by(distance.asc(), Message.ts.desc(), Message.message_id.desc()).limit(top_k)
    rows = db.execute(stmt).all()
    return [(row[0], float(row[1])) for row in rows]
