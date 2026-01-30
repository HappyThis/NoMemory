from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from app.db.models import Message
from app.utils.cursor import SeekAnchor


def list_messages(
    db: Session,
    *,
    user_id: str,
    since: Optional[datetime],
    until: Optional[datetime],
    role: Optional[str],
    page_size: int,
    anchor: Optional[SeekAnchor],
) -> list[Message]:
    stmt = select(Message).where(Message.user_id == user_id)
    if role:
        stmt = stmt.where(Message.role == role)
    if since:
        stmt = stmt.where(Message.ts >= since)
    if until:
        stmt = stmt.where(Message.ts < until)

    if anchor is not None:
        stmt = stmt.where(
            or_(
                Message.ts < anchor.ts,
                and_(Message.ts == anchor.ts, Message.message_id < anchor.message_id),
            )
        )

    stmt = stmt.order_by(Message.ts.desc(), Message.message_id.desc()).limit(page_size)
    return list(db.execute(stmt).scalars().all())
