"""
Authentication Routes
Handles user registration, login, logout, OTP verification
"""

from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

from database.connection import get_db
from models.user import User, UserRole
from models.activity_log import ActionType
from repositories.user_repository import UserRepository
from repositories.activity_log_repository import ActivityLogRepository
from services.auth_service import AuthService
from services.otp_service import OTPService
from dependencies.auth import get_current_user

router = APIRouter(prefix="/auth", tags=["Authentication"])


# ============== REQUEST SCHEMAS ==============

class SendOTPRequest(BaseModel):
    phone: str = Field(..., min_length=10, max_length=15)
    purpose: str = Field(default="registration")


class VerifyOTPRequest(BaseModel):
    phone: str = Field(..., min_length=10, max_length=15)
    otp: str = Field(..., min_length=4, max_length=8)
    purpose: str = Field(default="registration")


class RegisterRequest(BaseModel):
    full_name: str = Field(..., min_length=2, max_length=100)
    phone: str = Field(..., min_length=10, max_length=15)
    verification_token: str = Field(..., description="Token received after OTP verification")
    email: Optional[str] = Field(None, max_length=255, description="Optional email")
    state_or_ut: Optional[str] = Field(None, description="Home state/UT")
    city: Optional[str] = Field(None, description="City (free text)")


class LoginRequest(BaseModel):
    phone: str = Field(..., min_length=10, max_length=15)
    verification_token: str = Field(..., description="Token received after OTP verification for login")


class RefreshTokenRequest(BaseModel):
    refresh_token: str


# ============== RESPONSE SCHEMAS ==============

class UserResponse(BaseModel):
    id: int
    full_name: str
    phone: Optional[str]
    role: str
    is_active: bool
    is_verified: bool
    preferences: Optional[Dict[str, Any]] = None
    profile_data: Optional[Dict[str, Any]] = None
    created_at: datetime
    
    class Config:
        from_attributes = True


class AuthResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse


# ============== ROUTES ==============

@router.post("/send-otp")
async def send_otp(
    request: SendOTPRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Send OTP to phone number for verification
    """
    user_repo = UserRepository(db)

    # Registration guardrail: don't send signup OTP for an already-registered number.
    if request.purpose == "registration":
        existing = await user_repo.find_by_normalized_phone(request.phone)
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    "An account already exists with this phone number. "
                    "Please sign in, or use a different phone number to create a new account."
                ),
            )
    elif request.purpose == "login":
        existing = await user_repo.find_by_normalized_phone(request.phone)
        if not existing:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    "This phone number is not registered yet. "
                    "Please register first, then sign in."
                ),
            )

    result = OTPService.send_otp(request.phone, request.purpose)
    
    if not result["success"]:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=result["message"]
        )
    
    return result


@router.post("/verify-otp")
async def verify_otp(request: VerifyOTPRequest):
    """
    Verify OTP entered by user.
    Returns a verification_token to use during registration.
    """
    result = OTPService.verify_otp(request.phone, request.otp, request.purpose)
    
    if not result["success"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["message"]
        )
    
    # Generate verification token for use in registration
    # Use formatted phone to ensure consistency with registration
    formatted_phone = OTPService._format_phone(request.phone)
    verification_token = AuthService.create_phone_verification_token(
        phone=formatted_phone,
        purpose=request.purpose
    )
    
    return {
        **result,
        "verification_token": verification_token
    }


@router.post("/register", response_model=AuthResponse)
async def register(
    request: RegisterRequest,
    req: Request,
    db: AsyncSession = Depends(get_db)
):
    """
    Register a new user account
    Requires phone verification via OTP first (verification_token from verify-otp)
    """
    user_repo = UserRepository(db)
    activity_repo = ActivityLogRepository(db)
    
    # Verify phone verification token
    if not AuthService.verify_phone_verification_token(
        token=request.verification_token,
        expected_phone=OTPService._format_phone(request.phone),
        expected_purpose="registration"
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Phone verification expired or invalid. Please verify with OTP again."
        )
    
    normalized_phone = OTPService._format_phone(request.phone)
    existing_phone_user = await user_repo.find_by_normalized_phone(normalized_phone)
    if existing_phone_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this phone number already exists"
        )

    # Keep preferences empty for now; store location fields in profile_data only.
    preferences = {}
    profile_data = {}
    if request.state_or_ut:
        profile_data["state_or_ut"] = request.state_or_ut
    if request.city:
        profile_data["city"] = request.city
    if request.email:
        profile_data["email"] = request.email
    
    user = User(
        full_name=request.full_name,
        phone=normalized_phone,
        role=UserRole.STUDENT,
        is_active=True,
        is_verified=True,  # Verified via OTP
        preferences=preferences if preferences else {},
        profile_data=profile_data if profile_data else {}
    )
    
    user = await user_repo.create(user)
    
    # Clear OTP
    OTPService.clear_otp(request.phone)
    
    # Log activity
    await activity_repo.log_action(
        action_type=ActionType.REGISTER,
        description=f"New user registered: {user.full_name}",
        user_id=user.id,
        target_type="user",
        target_id=str(user.id),
        ip_address=req.client.host if req.client else None
    )
    
    # Generate tokens
    tokens = AuthService.create_token_pair(user.id, user.role.value)
    
    return AuthResponse(
        access_token=tokens["access_token"],
        refresh_token=tokens["refresh_token"],
        token_type=tokens["token_type"],
        expires_in=tokens["expires_in"],
        user=UserResponse(
            id=user.id,
            full_name=user.full_name,
            phone=user.phone,
            role=user.role.value,
            is_active=user.is_active,
            is_verified=user.is_verified,
            preferences=user.preferences or {},
            profile_data=user.profile_data or {},
            created_at=user.created_at
        )
    )


@router.post("/login", response_model=AuthResponse)
async def login(
    request: LoginRequest,
    req: Request,
    db: AsyncSession = Depends(get_db)
):
    """
    Login with phone + OTP verification token
    """
    user_repo = UserRepository(db)
    activity_repo = ActivityLogRepository(db)
    
    normalized_phone = OTPService._format_phone(request.phone)
    if not AuthService.verify_phone_verification_token(
        token=request.verification_token,
        expected_phone=normalized_phone,
        expected_purpose="login"
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Login OTP verification expired or invalid. Please verify OTP again."
        )

    user = await user_repo.find_by_normalized_phone(normalized_phone)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No account found for this phone number. Please register first."
        )
    
    # Check if active
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated"
        )
    
    # Update last login
    await user_repo.update_last_login(user.id)
    OTPService.clear_otp(normalized_phone)
    
    # Log successful login
    await activity_repo.log_action(
        action_type=ActionType.LOGIN,
        description=f"User logged in: {user.full_name}",
        user_id=user.id,
        ip_address=req.client.host if req.client else None
    )
    
    # Generate tokens
    tokens = AuthService.create_token_pair(user.id, user.role.value)
    
    return AuthResponse(
        access_token=tokens["access_token"],
        refresh_token=tokens["refresh_token"],
        token_type=tokens["token_type"],
        expires_in=tokens["expires_in"],
        user=UserResponse(
            id=user.id,
            full_name=user.full_name,
            phone=user.phone,
            role=user.role.value,
            is_active=user.is_active,
            is_verified=user.is_verified,
            preferences=user.preferences or {},
            profile_data=user.profile_data or {},
            created_at=user.created_at
        )
    )


@router.post("/refresh")
async def refresh_token(request: RefreshTokenRequest):
    """
    Get new access token using refresh token
    """
    result = AuthService.refresh_access_token(request.refresh_token)
    
    if not result:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token"
        )
    
    return result


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_user)
):
    """
    Get current authenticated user's information
    """
    return UserResponse(
        id=current_user.id,
        full_name=current_user.full_name,
        phone=current_user.phone,
        role=current_user.role.value,
        is_active=current_user.is_active,
        is_verified=current_user.is_verified,
        preferences=current_user.preferences or {},
        profile_data=current_user.profile_data or {},
        created_at=current_user.created_at
    )


@router.post("/logout")
async def logout(
    req: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Logout current user
    Note: JWT tokens are stateless, so we just log the action.
    For full logout, implement token blacklisting with Redis.
    """
    activity_repo = ActivityLogRepository(db)
    
    await activity_repo.log_action(
        action_type=ActionType.LOGOUT,
        description=f"User logged out: {current_user.full_name}",
        user_id=current_user.id,
        ip_address=req.client.host if req.client else None
    )
    
    return {"message": "Logged out successfully"}


def _mask_phone_hint(phone: str) -> str:
    """Last 4 digits only — avoids leaking full number."""
    p = phone.replace(" ", "").strip()
    if len(p) >= 4:
        return f"******{p[-4:]}"
    return "******"


@router.post("/forgot-password/request")
async def forgot_password_request(
    db: AsyncSession = Depends(get_db),
):
    raise HTTPException(
        status_code=status.HTTP_410_GONE,
        detail="Password reset is disabled in OTP-only authentication mode.",
    )


@router.post("/forgot-password/verify")
async def forgot_password_verify(
    db: AsyncSession = Depends(get_db),
):
    raise HTTPException(
        status_code=status.HTTP_410_GONE,
        detail="Password reset is disabled in OTP-only authentication mode.",
    )


@router.post("/forgot-password/reset")
async def forgot_password_reset(
    db: AsyncSession = Depends(get_db),
):
    raise HTTPException(
        status_code=status.HTTP_410_GONE,
        detail="Password reset is disabled in OTP-only authentication mode.",
    )


@router.post("/change-password")
async def change_password(
    current_user: User = Depends(get_current_user),
):
    raise HTTPException(
        status_code=status.HTTP_410_GONE,
        detail="Password change is disabled in OTP-only authentication mode.",
    )


# ============== ADMIN ROUTES ==============

@router.post("/admin/create-user", response_model=UserResponse)
async def create_user(
    current_user: User = Depends(get_current_user),
):
    raise HTTPException(
        status_code=status.HTTP_410_GONE,
        detail="Admin create-user via password is disabled in OTP-only authentication mode.",
    )
