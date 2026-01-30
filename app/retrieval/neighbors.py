from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Message


def get_neighbors(
    db: Session,
    *,
    user_id: str,
    message_id: str,
    before: int,
    after: int,
) -> list[Message]:
    anchor = db.get(Message, {"user_id": user_id, "message_id": message_id})
    if anchor is None:
        return []

    before_stmt = (
        select(Message)
        .where(Message.user_id == user_id, Message.ts < anchor.ts)
        .order_by(Message.ts.desc(), Message.message_id.desc())
        .limit(before)
    )
    after_stmt = (
        select(Message)
        .where(Message.user_id == user_id, Message.ts > anchor.ts)
        .order_by(Message.ts.asc(), Message.message_id.asc())
        .limit(after)
    )

    before_items = list(db.execute(before_stmt).scalars().all())
    before_items.reverse()
    after_items = list(db.execute(after_stmt).scalars().all())

    return [*before_items, anchor, *after_items]

