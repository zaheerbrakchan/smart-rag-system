"""
Conversation & Message Entity Models
Stores chat history with proper relationships
"""

import enum
from datetime import datetime
from typing import Optional, Dict, Any, List
from sqlalchemy import (
    String, Integer, Boolean, DateTime, Enum, Text, ForeignKey, func
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.connection import Base


class MessageRole(str, enum.Enum):
    """Message sender role"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Conversation(Base):
    """
    Conversation entity - groups messages in a chat session
    """
    __tablename__ = "conversations"
    
    # Primary Key
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    
    # Foreign Key to User
    user_id: Mapped[int] = mapped_column(
        Integer, 
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    # Conversation metadata
    title: Mapped[Optional[str]] = mapped_column(
        String(255), 
        nullable=True,
        comment="Auto-generated or user-defined title"
    )
    summary: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="AI-generated conversation summary for context"
    )
    
    # Context and state
    context_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSONB,
        default=dict,
        comment="Conversation context: detected state, exam type, etc."
    )
    
    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )
    
    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="conversations")
    messages: Mapped[List["Message"]] = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.created_at"
    )
    
    def __repr__(self) -> str:
        return f"<Conversation(id={self.id}, user_id={self.user_id}, messages={len(self.messages)})>"
    
    @property
    def message_count(self) -> int:
        return len(self.messages) if self.messages else 0
    
    def to_dict(self, include_messages: bool = False) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = {
            "id": self.id,
            "user_id": self.user_id,
            "title": self.title,
            "summary": self.summary,
            "context_data": self.context_data or {},
            "is_active": self.is_active,
            "message_count": self.message_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
        if include_messages:
            data["messages"] = [msg.to_dict() for msg in self.messages]
        return data


class Message(Base):
    """
    Message entity - individual messages in a conversation
    """
    __tablename__ = "messages"
    
    # Primary Key
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    
    # Foreign Key to Conversation
    conversation_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    # Message content
    role: Mapped[MessageRole] = mapped_column(
        Enum(MessageRole),
        nullable=False
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    
    # RAG metadata
    sources: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(
        JSONB,
        default=list,
        comment="Retrieved document sources used for answer"
    )
    model_used: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        comment="LLM model used: gpt-4o-mini, etc."
    )
    filters_applied: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSONB,
        default=dict,
        comment="Metadata filters applied during retrieval"
    )
    
    # Response metadata
    response_time_ms: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="Response generation time in milliseconds"
    )
    tokens_used: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="Total tokens used (prompt + completion)"
    )
    was_faq_match: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        comment="True if answer came from FAQ, False if RAG"
    )
    faq_confidence: Mapped[Optional[float]] = mapped_column(
        nullable=True,
        comment="FAQ match confidence score"
    )
    
    # Additional data (future-proof)
    extra_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSONB,
        default=dict,
        comment="Additional metadata for future use"
    )
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True
    )
    
    # Relationships
    conversation: Mapped["Conversation"] = relationship(
        "Conversation", 
        back_populates="messages"
    )
    
    def __repr__(self) -> str:
        content_preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"<Message(id={self.id}, role={self.role.value}, content='{content_preview}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "role": self.role.value,
            "content": self.content,
            "sources": self.sources or [],
            "model_used": self.model_used,
            "filters_applied": self.filters_applied or {},
            "response_time_ms": self.response_time_ms,
            "tokens_used": self.tokens_used,
            "was_faq_match": self.was_faq_match,
            "faq_confidence": self.faq_confidence,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# Import to avoid circular imports
from .user import User
