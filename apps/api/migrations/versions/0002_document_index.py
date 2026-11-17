"""Track indexing lifecycle and document uniqueness.

Revision ID: 0002_document_index
Revises: 0001_conversation_history
"""

from alembic import op
import sqlalchemy as sa

revision = "0002_document_index"
down_revision = "0001_conversation_history"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "document_index",
        sa.Column("document_hash", sa.String(64), primary_key=True),
        sa.Column("document_id", sa.String(36), nullable=False, unique=True),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("content_type", sa.String(255), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("chunks", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint(
            "status IN ('indexing', 'ready', 'failed')",
            name="ck_document_index_status",
        ),
    )
    op.create_index(
        "ix_document_index_status_updated",
        "document_index",
        ["status", "updated_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_document_index_status_updated", table_name="document_index")
    op.drop_table("document_index")
