"""
User Entity Model
Supports both students and admins with flexible JSONB fields
"""

import enum
from datetime import datetime
from typing import Optional, Dict, Any, List
from sqlalchemy import (
    String, Integer, Boolean, DateTime, Enum, Text, func
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.connection import Base


class UserRole(str, enum.Enum):
    """User roles in the system"""
    STUDENT = "student"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"


class User(Base):
    """
    User entity - stores both students and admin users
    
    Similar to @Entity in Spring Boot JPA
    """
    __tablename__ = "users"
    
    # Primary Key
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    
    # Authentication fields
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    
    # Profile information
    full_name: Mapped[str] = mapped_column(String(100), nullable=False)
    phone: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    
    # Role and permissions
    role: Mapped[UserRole] = mapped_column(
        Enum(UserRole), 
        default=UserRole.STUDENT, 
        nullable=False,
        index=True
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    
    # User preferences (JSONB for flexibility)
    preferences: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSONB,
        default=dict,
        comment="User preferences: notification settings, UI preferences, home state, category, etc."
    )
    profile_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSONB,
        default=dict,
        comment="Additional profile data: education, location, etc."
    )
    
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
    last_login_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )
    
    # Relationships
    conversations: Mapped[List["Conversation"]] = relationship(
        "Conversation",
        back_populates="user",
        cascade="all, delete-orphan"
    )
    activity_logs: Mapped[List["ActivityLog"]] = relationship(
        "ActivityLog",
        back_populates="user",
        cascade="all, delete-orphan"
    )
    reviewed_qas: Mapped[List["PendingQA"]] = relationship(
        "PendingQA",
        back_populates="reviewed_by_user",
        foreign_keys="PendingQA.reviewed_by"
    )
    
    def __repr__(self) -> str:
        return f"<User(id={self.id}, username='{self.username}', role={self.role.value})>"
    
    @property
    def is_admin(self) -> bool:
        """Check if user has admin privileges"""
        return self.role in [UserRole.ADMIN, UserRole.SUPER_ADMIN]
    
    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """Convert to dictionary (like DTO)"""
        data = {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "full_name": self.full_name,
            "phone": self.phone,
            "role": self.role.value,
            "is_active": self.is_active,
            "is_verified": self.is_verified,
            "preferences": self.preferences or {},
            "profile_data": self.profile_data or {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login_at": self.last_login_at.isoformat() if self.last_login_at else None,
        }
        if include_sensitive:
            data["password_hash"] = self.password_hash
        return data


# Import at bottom to avoid circular imports
from .conversation import Conversation
from .activity_log import ActivityLog
from .pending_qa import PendingQA
