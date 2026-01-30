from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, Computed, DateTime, ForeignKeyConstraint, Index, String, Text
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Message(Base):
    __tablename__ = "messages"

    user_id: Mapped[str] = mapped_column(String, primary_key=True)
    message_id: Mapped[str] = mapped_column(String, primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    role: Mapped[str] = mapped_column(String, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    meta: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    search_tsv: Mapped[Any] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('simple', coalesce(content, ''))", persisted=True),
    )

    embeddings: Mapped[list["MessageEmbedding"]] = relationship(
        back_populates="message",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("ix_messages_user_ts_id_desc", "user_id", "ts", "message_id"),
        Index("ix_messages_user_role_ts_id_desc", "user_id", "role", "ts", "message_id"),
    )


class MessageEmbedding(Base):
    __tablename__ = "message_embeddings"

    user_id: Mapped[str] = mapped_column(String, primary_key=True)
    message_id: Mapped[str] = mapped_column(String, primary_key=True)
    provider: Mapped[str] = mapped_column(String, primary_key=True)
    model: Mapped[str] = mapped_column(String, primary_key=True)
    embedding: Mapped[list[float]] = mapped_column(Vector(), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    message: Mapped[Message] = relationship(back_populates="embeddings")

    __table_args__ = (
        ForeignKeyConstraint(
            ["user_id", "message_id"],
            ["messages.user_id", "messages.message_id"],
            ondelete="CASCADE",
        ),
        Index("ix_message_embeddings_user_message", "user_id", "message_id"),
        Index("ix_message_embeddings_user_model", "user_id", "provider", "model"),
    )
