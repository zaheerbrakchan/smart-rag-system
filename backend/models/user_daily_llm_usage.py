"""Per-user daily aggregate of LLM tokens (OpenAI usage.total_tokens)."""

from datetime import date, datetime
from typing import Optional

from sqlalchemy import BigInteger, Date, DateTime, ForeignKey, Integer, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from database.connection import Base


class UserDailyLlmUsage(Base):
    """
    One row per user per UTC calendar day.
    total_tokens is incremented after each chat completion from reported OpenAI usage.
    """

    __tablename__ = "user_daily_llm_usage"
    __table_args__ = (UniqueConstraint("user_id", "usage_date", name="uq_user_daily_llm_usage_user_date"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    usage_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    total_tokens: Mapped[int] = mapped_column(BigInteger, nullable=False, server_default="0")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
