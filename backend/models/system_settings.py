"""
System Settings Model
Stores application-wide configuration settings
"""

from datetime import datetime
from typing import Optional
from sqlalchemy import String, Boolean, DateTime, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from database.connection import Base


class SystemSettings(Base):
    """
    System settings - key-value store for application configuration.
    
    Used for:
    - Auto-learning toggle (FAQ auto-capture from RAG responses)
    - Feature flags
    - Runtime configuration
    """
    __tablename__ = "system_settings"
    
    # Primary Key - setting key name
    key: Mapped[str] = mapped_column(String(100), primary_key=True)
    
    # Value (stored as string, parsed by application)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    
    # Description
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Audit fields
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )
    updated_by: Mapped[Optional[int]] = mapped_column(nullable=True)


# Default settings keys
class SettingsKeys:
    """Constants for settings keys"""
    AUTO_LEARNING_ENABLED = "auto_learning_enabled"
    AUTO_LEARNING_MIN_CONFIDENCE = "auto_learning_min_confidence"
    FAQ_SCORE_THRESHOLD = "faq_score_threshold"
    WEB_SEARCH_FALLBACK_ENABLED = "web_search_fallback_enabled"
    CHAT_REFERENCES_ENABLED = "chat_references_enabled"
    CUTOFF_COLLEGE_RESULT_LIMIT = "cutoff_college_result_limit"
    SUPPORT_EMAIL_ENABLED = "support_email_enabled"
    SUPPORT_SMS_ENABLED = "support_sms_enabled"
    SUPPORT_INBOX_EMAIL = "support_inbox_email"
