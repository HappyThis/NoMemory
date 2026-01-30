from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from sqlalchemy import Float, and_, func, or_, select
from sqlalchemy.orm import Session

from app.db.models import Message


@dataclass(frozen=True)
class LexicalAnchor:
    rank: float
    ts: datetime
    message_id: str


def lexical_search(
    db: Session,
    *,
    user_id: str,
    query_text: str,
    since: Optional[datetime],
    until: Optional[datetime],
    role: Optional[str],
    page_size: int,
    anchor: Optional[LexicalAnchor],
) -> list[tuple[Message, float]]:
    tsq = func.websearch_to_tsquery("simple", query_text)
    rank_expr = func.ts_rank_cd(Message.search_tsv, tsq).cast(Float)

    stmt = select(Message, rank_expr.label("rank")).where(Message.user_id == user_id, Message.search_tsv.op("@@")(tsq))
    if role:
        stmt = stmt.where(Message.role == role)
    if since:
        stmt = stmt.where(Message.ts >= since)
    if until:
        stmt = stmt.where(Message.ts < until)

    if anchor is not None:
        stmt = stmt.where(
            or_(
                rank_expr < anchor.rank,
                and_(
                    rank_expr == anchor.rank,
                    or_(
                        Message.ts < anchor.ts,
                        and_(Message.ts == anchor.ts, Message.message_id < anchor.message_id),
                    ),
                ),
            )
        )

    stmt = stmt.order_by(rank_expr.desc(), Message.ts.desc(), Message.message_id.desc()).limit(page_size)
    rows = db.execute(stmt).all()
    return [(row[0], float(row[1])) for row in rows]
