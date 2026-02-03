from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.db.models import MessageEmbedding
from app.db.session import SessionLocal
from app.log import get_logger
from app.llm.bigmodel import BigModelClient, BigModelError
from app.settings import settings


logger = get_logger(__name__)


def _chunk(items: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def embed_query_text(text: str, *, provider: Optional[str] = None, model: Optional[str] = None) -> list[float]:
    provider = provider or settings.llm_provider
    if provider != "bigmodel":
        raise ValueError(f"Unsupported embedding provider: {provider}")
    bm = BigModelClient()
    model = model or settings.bigmodel_embedding_model
    return bm.embeddings(inputs=[text], model=model)[0]


def enqueue_embeddings_for_messages(
    messages: list[tuple[str, str, str]],
    *,
    provider: str,
    model: str,
    requested_at: Optional[datetime] = None,
) -> None:
    # Best-effort background task: compute embeddings and UPSERT into message_embeddings.
    if not messages:
        return
    if provider != "bigmodel":
        return
    try:
        bm = BigModelClient()
    except BigModelError:
        return

    requested_at = requested_at or datetime.now(tz=timezone.utc)

    # BigModel embeddings API supports batching; keep chunks modest.
    texts = [m[2] for m in messages]
    embeddings: list[list[float]] = []
    for batch in _chunk(texts, 64):
        try:
            embeddings.extend(bm.embeddings(inputs=batch, model=model))
        except BigModelError as e:
            # Do not fail the request after the response has been sent; log and stop.
            logger.warning("embeddings batch failed (%s): %s", type(e).__name__, str(e))
            return
        except Exception as e:
            logger.warning("embeddings batch failed (%s)", type(e).__name__)
            return

    rows = []
    if len(embeddings) != len(messages):
        return

    for (user_id, message_id, _content), emb in zip(messages, embeddings):
        rows.append(
            {
                "user_id": user_id,
                "message_id": message_id,
                "provider": provider,
                "model": model,
                "embedding": emb,
                "created_at": requested_at,
            }
        )

    with SessionLocal() as db:  # separate session for background thread/task
        stmt = (
            insert(MessageEmbedding)
            .values(rows)
            .on_conflict_do_update(
                index_elements=[
                    MessageEmbedding.user_id,
                    MessageEmbedding.message_id,
                    MessageEmbedding.provider,
                    MessageEmbedding.model,
                ],
                set_={"embedding": insert(MessageEmbedding).excluded.embedding, "created_at": requested_at},
            )
        )
        db.execute(stmt)
        db.commit()
