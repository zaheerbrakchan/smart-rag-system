"""
Indexed Document Entity Model
Tracks documents that have been indexed in the vector store
"""

from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy import (
    String, Integer, Boolean, DateTime, Text, Float, func
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from database.connection import Base


class IndexedDocument(Base):
    """
    Tracks documents indexed in the vector database
    """
    __tablename__ = "indexed_documents"
    
    # Primary Key
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    
    # Document identification
    file_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    
    # Document metadata
    state: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    document_type: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    category: Mapped[str] = mapped_column(String(100), nullable=False)
    year: Mapped[str] = mapped_column(String(10), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Versioning - auto-increment per (state, document_type, category, year) combination
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    
    # Indexing info
    total_pages: Mapped[int] = mapped_column(Integer, default=0)
    total_vectors: Mapped[int] = mapped_column(Integer, default=0)
    file_size_kb: Mapped[float] = mapped_column(Float, default=0.0)
    
    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    index_status: Mapped[str] = mapped_column(
        String(50), 
        default="indexed",
        comment="Status: indexed, failed, processing, deleted"
    )
    
    # Supabase Storage
    storage_path: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
        comment="Path in Supabase Storage bucket"
    )
    storage_url: Mapped[Optional[str]] = mapped_column(
        String(1000),
        nullable=True,
        comment="Public URL for the file in Supabase Storage"
    )
    
    # Additional metadata (JSONB for flexibility)
    extra_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSONB,
        default=dict,
        comment="Additional metadata like namespace, chunk settings, etc."
    )
    
    # Timestamps
    indexed_at: Mapped[datetime] = mapped_column(
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
    
    # Who uploaded
    uploaded_by: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    def __repr__(self) -> str:
        return f"<IndexedDocument {self.filename} ({self.state})>"
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "file_id": self.file_id,
            "filename": self.filename,
            "original_filename": self.original_filename,
            "state": self.state,
            "document_type": self.document_type,
            "category": self.category,
            "year": self.year,
            "description": self.description,
            "total_pages": self.total_pages,
            "total_vectors": self.total_vectors,
            "file_size_kb": self.file_size_kb,
            "is_active": self.is_active,
            "index_status": self.index_status,
            "storage_path": self.storage_path,
            "storage_url": self.storage_url,
            "indexed_at": self.indexed_at.isoformat() if self.indexed_at else None,
            "uploaded_by": self.uploaded_by
        }
