"""
Activity Log Entity Model
Tracks all admin and user actions for audit and analytics
"""

import enum
from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy import (
    String, Integer, DateTime, Enum, Text, ForeignKey, func
)
from sqlalchemy.dialects.postgresql import JSONB, INET
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.connection import Base


class ActionType(str, enum.Enum):
    """Types of actions that can be logged"""
    # Authentication
    LOGIN = "login"
    LOGOUT = "logout"
    REGISTER = "register"
    PASSWORD_CHANGE = "password_change"
    
    # Chat actions
    CHAT_START = "chat_start"
    CHAT_MESSAGE = "chat_message"
    
    # Document management (Admin)
    DOCUMENT_UPLOAD = "document_upload"
    DOCUMENT_DELETE = "document_delete"
    DOCUMENT_DEACTIVATE = "document_deactivate"
    
    # FAQ management (Admin)
    FAQ_APPROVE = "faq_approve"
    FAQ_REJECT = "faq_reject"
    FAQ_MODIFY = "faq_modify"
    FAQ_BULK_UPLOAD = "faq_bulk_upload"
    
    # User management (Admin)
    USER_CREATE = "user_create"
    USER_UPDATE = "user_update"
    USER_DEACTIVATE = "user_deactivate"
    USER_ROLE_CHANGE = "user_role_change"
    
    # System
    SYSTEM_ERROR = "system_error"
    API_RATE_LIMIT = "api_rate_limit"


class ActivityLog(Base):
    """
    Activity Log entity - audit trail for all actions
    
    Used for:
    - Security auditing
    - Analytics dashboards
    - Debugging user issues
    - Compliance requirements
    """
    __tablename__ = "activity_logs"
    
    # Primary Key
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    
    # Actor (who performed the action)
    user_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,  # Can be null for system actions or anonymous
        index=True
    )
    
    # Action details
    action_type: Mapped[ActionType] = mapped_column(
        Enum(ActionType),
        nullable=False,
        index=True
    )
    description: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Human-readable description of the action"
    )
    
    # Target of action (what was affected)
    target_type: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        comment="Type of entity affected: user, document, faq, etc."
    )
    target_id: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        comment="ID of the affected entity"
    )
    
    # Request context
    ip_address: Mapped[Optional[str]] = mapped_column(
        String(45),  # IPv6 max length
        nullable=True
    )
    user_agent: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True
    )
    
    # Additional data (flexible)
    request_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSONB,
        default=dict,
        comment="Request payload (sanitized, no passwords)"
    )
    response_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSONB,
        default=dict,
        comment="Response summary"
    )
    error_details: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSONB,
        default=dict,
        comment="Error information if action failed"
    )
    
    # Status
    success: Mapped[bool] = mapped_column(
        default=True,
        nullable=False
    )
    
    # Timestamp
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True
    )
    
    # Relationships
    user: Mapped[Optional["User"]] = relationship(
        "User",
        back_populates="activity_logs"
    )
    
    def __repr__(self) -> str:
        return f"<ActivityLog(id={self.id}, action={self.action_type.value}, user_id={self.user_id})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "action_type": self.action_type.value,
            "description": self.description,
            "target_type": self.target_type,
            "target_id": self.target_id,
            "ip_address": self.ip_address,
            "success": self.success,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# Import to avoid circular imports
from .user import User
