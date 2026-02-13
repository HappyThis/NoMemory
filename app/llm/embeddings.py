from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
import time
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
    provider = provider or settings.embedding_provider
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

    def _retryable(e: BigModelError) -> bool:
        sc = getattr(e, "status_code", None)
        if sc in (429, 500, 502, 503, 504):
            return True
        return False

    def _embed_with_retries(texts: list[str]) -> list[list[float]]:
        backoff = 0.5
        last: Exception | None = None
        for attempt in range(1, 4):  # 3 tries
            try:
                return bm.embeddings(inputs=texts, model=model)
            except BigModelError as e:
                last = e
                if (attempt >= 3) or (not _retryable(e)):
                    raise
                logger.warning(
                    "embeddings batch retrying attempt=%s/3 size=%s (%s): %s",
                    attempt,
                    len(texts),
                    type(e).__name__,
                    str(e),
                )
                time.sleep(backoff)
                backoff = min(5.0, backoff * 2)
        raise BigModelError(f"Embeddings batch failed: {last}")  # should be unreachable

    # BigModel embeddings API supports batching; write per-batch so partial progress is kept.
    with SessionLocal() as db:  # separate session for background thread/task
        for batch_msgs in _chunk(messages, 64):
            texts = [m[2] for m in batch_msgs]
            try:
                batch_embs = _embed_with_retries(texts)
            except BigModelError as e:
                # Do not fail the ingest request after the response has been sent; log and continue.
                logger.warning("embeddings batch failed size=%s (%s): %s", len(texts), type(e).__name__, str(e))
                continue
            except Exception as e:
                logger.warning("embeddings batch failed size=%s (%s)", len(texts), type(e).__name__)
                continue

            if len(batch_embs) != len(batch_msgs):
                logger.warning("embeddings batch failed: count mismatch size=%s", len(texts))
                continue

            rows = []
            for (user_id, message_id, _content), emb in zip(batch_msgs, batch_embs):
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
