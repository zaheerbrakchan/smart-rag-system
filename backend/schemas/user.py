"""
User Schemas (DTOs)
Similar to @RequestBody/@ResponseBody DTOs in Spring Boot
"""

from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, EmailStr, Field, field_validator
import re


# ============== REQUEST SCHEMAS ==============

class UserRegister(BaseModel):
    """User registration request"""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=100)
    full_name: str = Field(..., min_length=2, max_length=100)
    phone: Optional[str] = Field(None, max_length=20)
    
    # Student preferences for smart query routing
    preferred_state: Optional[str] = Field(None, description="Home state for counseling")
    category: Optional[str] = Field(None, description="Category: General/OBC/SC/ST/EWS")
    
    # OTP verification token
    verification_token: Optional[str] = Field(None, description="OTP verification token")
    
    @field_validator('username')
    @classmethod
    def validate_username(cls, v: str) -> str:
        if not re.match(r'^[a-zA-Z0-9_]+$', v):
            raise ValueError('Username can only contain letters, numbers, and underscores')
        return v.lower()
    
    @field_validator('phone')
    @classmethod
    def validate_phone(cls, v: Optional[str]) -> Optional[str]:
        if v and not re.match(r'^[\d\s\-\+\(\)]+$', v):
            raise ValueError('Invalid phone number format')
        return v


class UserLogin(BaseModel):
    """User login request"""
    username: str  # Can be username or email
    password: str


class UserCreate(BaseModel):
    """Admin: create user request"""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: str = Field(..., min_length=2, max_length=100)
    phone: Optional[str] = None
    role: str = Field(default="student")
    preferences: Optional[Dict[str, Any]] = Field(default_factory=dict)
    profile_data: Optional[Dict[str, Any]] = Field(default_factory=dict)


class UserUpdate(BaseModel):
    """Admin: update user request"""
    full_name: Optional[str] = Field(None, min_length=2, max_length=100)
    phone: Optional[str] = None
    role: Optional[str] = None
    is_active: Optional[bool] = None
    is_verified: Optional[bool] = None
    preferences: Optional[Dict[str, Any]] = None
    profile_data: Optional[Dict[str, Any]] = None


class UserProfileUpdate(BaseModel):
    """User self-update profile"""
    full_name: Optional[str] = Field(None, min_length=2, max_length=100)
    phone: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = None
    profile_data: Optional[Dict[str, Any]] = None


class PasswordChange(BaseModel):
    """Password change request"""
    current_password: str
    new_password: str = Field(..., min_length=8, max_length=100)
    
    @field_validator('new_password')
    @classmethod
    def validate_password(cls, v: str) -> str:
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not re.search(r'\d', v):
            raise ValueError('Password must contain at least one digit')
        return v


# ============== RESPONSE SCHEMAS ==============

class UserResponse(BaseModel):
    """User response (public data)"""
    id: int
    username: str
    email: str
    full_name: str
    phone: Optional[str] = None
    age: Optional[int] = None
    role: str
    is_active: bool
    is_verified: bool
    target_exams: List[str] = []
    preferences: Dict[str, Any] = {}
    profile_data: Dict[str, Any] = {}
    created_at: datetime
    last_login_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True  # Enables ORM mode (like @JsonProperty in Jackson)


class TokenResponse(BaseModel):
    """JWT token response"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = Field(description="Access token expiry in seconds")
    user: UserResponse


class UserListResponse(BaseModel):
    """Paginated user list response"""
    users: List[UserResponse]
    total: int
    page: int
    page_size: int
    total_pages: int
