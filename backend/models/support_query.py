"""
Support query and notification models.
"""

import enum
from datetime import datetime
from typing import Optional, Dict, Any, List
from sqlalchemy import (
    String, Integer, DateTime, Enum, Text, ForeignKey, Boolean, func
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.connection import Base


class SupportQueryStatus(str, enum.Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    ANSWERED = "answered"
    CLOSED = "closed"


class SupportQuery(Base):
    __tablename__ = "support_queries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    student_name: Mapped[str] = mapped_column(String(120), nullable=False)
    phone: Mapped[str] = mapped_column(String(20), nullable=False)
    email: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    subject: Mapped[str] = mapped_column(String(255), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[SupportQueryStatus] = mapped_column(
        Enum(SupportQueryStatus),
        nullable=False,
        default=SupportQueryStatus.PENDING,
        index=True,
    )
    assigned_admin_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    answered_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    user: Mapped["User"] = relationship(
        "User",
        back_populates="support_queries",
        foreign_keys=[user_id],
    )
    assigned_admin: Mapped[Optional["User"]] = relationship(
        "User",
        back_populates="assigned_support_queries",
        foreign_keys=[assigned_admin_id],
    )
    replies: Mapped[List["SupportQueryReply"]] = relationship(
        "SupportQueryReply",
        back_populates="query",
        cascade="all, delete-orphan",
        order_by="SupportQueryReply.created_at",
    )


class SupportQueryReply(Base):
    __tablename__ = "support_query_replies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    query_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("support_queries.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    responder_admin_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    reply_text: Mapped[str] = mapped_column(Text, nullable=False)
    sent_email: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    sent_sms: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )

    query: Mapped["SupportQuery"] = relationship("SupportQuery", back_populates="replies")
    responder_admin: Mapped[Optional["User"]] = relationship(
        "User",
        back_populates="support_query_replies",
        foreign_keys=[responder_admin_id],
    )


class UserNotification(Base):
    __tablename__ = "user_notifications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    type: Mapped[str] = mapped_column(String(50), nullable=False, default="support_update")
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    related_query_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey("support_queries.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    is_read: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)
    extra_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )

    user: Mapped["User"] = relationship("User", back_populates="notifications")
    related_query: Mapped[Optional["SupportQuery"]] = relationship("SupportQuery")


from .user import User
