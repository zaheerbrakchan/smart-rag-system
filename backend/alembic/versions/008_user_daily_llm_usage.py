"""User daily LLM token usage + quota settings

Revision ID: 008_user_daily_llm_usage
Revises: 007_support_queries_and_notifications
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "008_user_daily_llm_usage"
down_revision: Union[str, None] = "007_support_queries_and_notifications"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "user_daily_llm_usage",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("usage_date", sa.Date(), nullable=False),
        sa.Column("total_tokens", sa.BigInteger(), server_default="0", nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "usage_date", name="uq_user_daily_llm_usage_user_date"),
    )
    op.create_index(
        "ix_user_daily_llm_usage_user_id", "user_daily_llm_usage", ["user_id"], unique=False
    )
    op.create_index(
        "ix_user_daily_llm_usage_usage_date", "user_daily_llm_usage", ["usage_date"], unique=False
    )

    op.execute(
        """
        INSERT INTO system_settings (key, value, description) VALUES
        ('daily_token_limit_enabled', 'false', 'When true, students cannot chat after exceeding daily OpenAI token budget'),
        ('daily_token_limit_per_user', '200000', 'Max OpenAI total_tokens per student per UTC day (admins exempt)')
        ON CONFLICT (key) DO NOTHING
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DELETE FROM system_settings WHERE key IN (
            'daily_token_limit_enabled',
            'daily_token_limit_per_user'
        )
        """
    )
    op.drop_index("ix_user_daily_llm_usage_usage_date", table_name="user_daily_llm_usage")
    op.drop_index("ix_user_daily_llm_usage_user_id", table_name="user_daily_llm_usage")
    op.drop_table("user_daily_llm_usage")
