"""init schema

Revision ID: 001_init
Revises:
Create Date: 2026-01-30
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision = "001_init"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    op.create_table(
        "messages",
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("message_id", sa.String(), nullable=False),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("role", sa.String(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("meta", sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint("user_id", "message_id", name="pk_messages"),
    )
    op.create_index(
        "ix_messages_user_ts_id",
        "messages",
        ["user_id", "ts", "message_id"],
    )
    op.create_index(
        "ix_messages_user_role_ts_id",
        "messages",
        ["user_id", "role", "ts", "message_id"],
    )

    # Full-text search: generated tsvector column + GIN index.
    op.execute(
        """
        ALTER TABLE messages
        ADD COLUMN search_tsv tsvector
        GENERATED ALWAYS AS (to_tsvector('simple', coalesce(content, ''))) STORED
        """
    )
    op.execute("CREATE INDEX ix_messages_search_tsv ON messages USING gin (search_tsv)")

    op.create_table(
        "message_embeddings",
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("message_id", sa.String(), nullable=False),
        sa.Column("provider", sa.String(), nullable=False),
        sa.Column("model", sa.String(), nullable=False),
        sa.Column("embedding", Vector(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint(
            "user_id",
            "message_id",
            "provider",
            "model",
            name="pk_message_embeddings",
        ),
        sa.ForeignKeyConstraint(
            ["user_id", "message_id"],
            ["messages.user_id", "messages.message_id"],
            ondelete="CASCADE",
        ),
    )
    op.create_index(
        "ix_message_embeddings_user_message",
        "message_embeddings",
        ["user_id", "message_id"],
    )
    op.create_index(
        "ix_message_embeddings_user_model",
        "message_embeddings",
        ["user_id", "provider", "model"],
    )


def downgrade() -> None:
    op.drop_index("ix_message_embeddings_user_model", table_name="message_embeddings")
    op.drop_index("ix_message_embeddings_user_message", table_name="message_embeddings")
    op.drop_table("message_embeddings")
    op.execute("DROP INDEX IF EXISTS ix_messages_search_tsv")
    op.execute("ALTER TABLE messages DROP COLUMN IF EXISTS search_tsv")
    op.drop_index("ix_messages_user_role_ts_id", table_name="messages")
    op.drop_index("ix_messages_user_ts_id", table_name="messages")
    op.drop_table("messages")
