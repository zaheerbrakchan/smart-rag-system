"""
Pending Q&A Entity Model
Auto-learned questions pending admin review
"""

import enum
from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy import (
    String, Integer, Float, DateTime, Enum, Text, ForeignKey, func
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.connection import Base


class QAStatus(str, enum.Enum):
    """Status of pending Q&A entry"""
    PENDING = "pending"          # Awaiting review
    APPROVED = "approved"        # Approved and added to FAQ
    REJECTED = "rejected"        # Rejected by admin
    MODIFIED = "modified"        # Approved with modifications


class PendingQA(Base):
    """
    Pending Q&A entity - stores auto-learned Q&A pairs for admin review
    
    When users ask questions answered via RAG, they can be stored here
    for admin to review and potentially add to FAQ knowledge base.
    """
    __tablename__ = "pending_qa"
    
    # Primary Key
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    
    # Question-Answer pair
    question: Mapped[str] = mapped_column(Text, nullable=False)
    original_answer: Mapped[str] = mapped_column(
        Text, 
        nullable=False,
        comment="Original RAG-generated answer"
    )
    modified_answer: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Admin-modified answer (if approved with changes)"
    )
    
    # Source tracking
    source_conversation_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey("conversations.id", ondelete="SET NULL"),
        nullable=True,
        comment="Conversation where this Q&A originated"
    )
    source_documents: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSONB,
        default=list,
        comment="Documents used to generate original answer"
    )
    
    # Categorization (for FAQ organization)
    detected_state: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        index=True
    )
    detected_exam: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        index=True
    )
    detected_category: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        comment="eligibility, dates, fees, etc."
    )
    
    # Confidence and ranking
    original_confidence: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="RAG retrieval confidence score"
    )
    occurrence_count: Mapped[int] = mapped_column(
        Integer,
        default=1,
        comment="Number of times similar question was asked"
    )
    
    # Review status
    status: Mapped[QAStatus] = mapped_column(
        Enum(QAStatus),
        default=QAStatus.PENDING,
        nullable=False,
        index=True
    )
    reviewed_by: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True
    )
    review_notes: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Admin notes during review"
    )
    
    # FAQ integration
    faq_vector_id: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        comment="Vector store ID if added to FAQ"
    )
    
    # Additional metadata
    extra_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSONB,
        default=dict,
        comment="Additional data for future use"
    )
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    reviewed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )
    
    # Relationships
    reviewed_by_user: Mapped[Optional["User"]] = relationship(
        "User",
        back_populates="reviewed_qas",
        foreign_keys=[reviewed_by]
    )
    
    def __repr__(self) -> str:
        q_preview = self.question[:40] + "..." if len(self.question) > 40 else self.question
        return f"<PendingQA(id={self.id}, status={self.status.value}, q='{q_preview}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "question": self.question,
            "original_answer": self.original_answer,
            "modified_answer": self.modified_answer,
            "detected_state": self.detected_state,
            "detected_exam": self.detected_exam,
            "detected_category": self.detected_category,
            "original_confidence": self.original_confidence,
            "occurrence_count": self.occurrence_count,
            "status": self.status.value,
            "reviewed_by": self.reviewed_by,
            "review_notes": self.review_notes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
        }


# Import to avoid circular imports
from .user import User
