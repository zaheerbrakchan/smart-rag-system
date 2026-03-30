"""
FAQ Entity Model
Frequently Asked Questions with pre-computed embeddings for fast retrieval
"""

from datetime import datetime
from typing import Optional, List
from sqlalchemy import (
    String, Integer, Float, DateTime, Text, Boolean, func
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column

from database.connection import Base


class FAQ(Base):
    """
    FAQ entity - stores frequently asked questions with answers
    
    FAQs are checked BEFORE RAG to provide instant answers for common questions.
    Each FAQ has a pre-computed embedding for semantic similarity search.
    """
    __tablename__ = "faqs"
    
    # Primary Key
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    
    # Question and Answer
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    
    # Category for organization (e.g., "exam_info", "counselling", "eligibility")
    category: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    
    # Keywords for additional matching
    keywords: Mapped[Optional[str]] = mapped_column(
        Text, 
        nullable=True,
        comment="Comma-separated keywords for this FAQ"
    )
    
    # State-specific FAQ (null means applies to all)
    state: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    
    # Embedding vector (stored as array of floats)
    # Note: For production, consider storing embeddings in Pinecone
    embedding: Mapped[Optional[List[float]]] = mapped_column(
        ARRAY(Float),
        nullable=True,
        comment="Pre-computed embedding vector for semantic search"
    )
    
    # Metadata
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    view_count: Mapped[int] = mapped_column(Integer, default=0, comment="Number of times this FAQ was returned")
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now()
    )
    
    def __repr__(self) -> str:
        return f"<FAQ(id={self.id}, question='{self.question[:50]}...')>"
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "question": self.question,
            "answer": self.answer,
            "category": self.category,
            "keywords": self.keywords,
            "state": self.state,
            "is_active": self.is_active,
            "view_count": self.view_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
