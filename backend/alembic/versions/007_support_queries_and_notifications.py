"""Add support query, reply, and notification tables

Revision ID: 007_support_queries_and_notifications
Revises: 006_drop_legacy_auth_columns
Create Date: 2026-04-20
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "007_support_queries_and_notifications"
down_revision: Union[str, None] = "006_drop_legacy_auth_columns"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        INSERT INTO system_settings (key, value, description) VALUES
        ('support_email_enabled', 'true', 'Enable/disable support email delivery'),
        ('support_sms_enabled', 'true', 'Enable/disable support SMS delivery'),
        ('support_inbox_email', '', 'Official support mailbox receiver')
        ON CONFLICT (key) DO NOTHING
        """
    )

    op.create_table(
        "support_queries",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("student_name", sa.String(length=120), nullable=False),
        sa.Column("phone", sa.String(length=20), nullable=False),
        sa.Column("email", sa.String(length=255), nullable=True),
        sa.Column("subject", sa.String(length=255), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column(
            "status",
            sa.Enum("pending", "in_progress", "answered", "closed", name="supportquerystatus"),
            nullable=False,
        ),
        sa.Column("assigned_admin_id", sa.Integer(), nullable=True),
        sa.Column("answered_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["assigned_admin_id"], ["users.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_support_queries_user_id", "support_queries", ["user_id"], unique=False)
    op.create_index("ix_support_queries_status", "support_queries", ["status"], unique=False)
    op.create_index("ix_support_queries_assigned_admin_id", "support_queries", ["assigned_admin_id"], unique=False)
    op.create_index("ix_support_queries_created_at", "support_queries", ["created_at"], unique=False)

    op.create_table(
        "support_query_replies",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("query_id", sa.Integer(), nullable=False),
        sa.Column("responder_admin_id", sa.Integer(), nullable=True),
        sa.Column("reply_text", sa.Text(), nullable=False),
        sa.Column("sent_email", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("sent_sms", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["query_id"], ["support_queries.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["responder_admin_id"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_support_query_replies_query_id", "support_query_replies", ["query_id"], unique=False)
    op.create_index("ix_support_query_replies_responder_admin_id", "support_query_replies", ["responder_admin_id"], unique=False)
    op.create_index("ix_support_query_replies_created_at", "support_query_replies", ["created_at"], unique=False)

    op.create_table(
        "user_notifications",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("type", sa.String(length=50), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("related_query_id", sa.Integer(), nullable=True),
        sa.Column("is_read", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("extra_data", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["related_query_id"], ["support_queries.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_user_notifications_user_id", "user_notifications", ["user_id"], unique=False)
    op.create_index("ix_user_notifications_related_query_id", "user_notifications", ["related_query_id"], unique=False)
    op.create_index("ix_user_notifications_is_read", "user_notifications", ["is_read"], unique=False)
    op.create_index("ix_user_notifications_created_at", "user_notifications", ["created_at"], unique=False)


def downgrade() -> None:
    op.drop_table("user_notifications")
    op.drop_table("support_query_replies")
    op.drop_table("support_queries")
    op.execute("DROP TYPE IF EXISTS supportquerystatus")
