"""
Authentication Routes
Handles user registration, login, logout, OTP verification
"""

from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

from database.connection import get_db
from models.user import User, UserRole
from models.otp_verification import OTPVerification
from models.activity_log import ActionType
from repositories.user_repository import UserRepository
from repositories.activity_log_repository import ActivityLogRepository
from services.auth_service import AuthService
from services.whatsapp_otp import generate_otp, normalize_phone, send_whatsapp_otp
from dependencies.auth import get_current_user

router = APIRouter(prefix="/auth", tags=["Authentication"])


# ============== REQUEST SCHEMAS ==============

class SendOTPRequest(BaseModel):
    phone: str = Field(..., min_length=10, max_length=15)
    purpose: str = Field(default="registration")


class VerifyOTPRequest(BaseModel):
    phone: str = Field(..., min_length=10, max_length=15)
    otp: str = Field(..., min_length=6, max_length=6)
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

    normalized_phone = normalize_phone(request.phone)
    if len(normalized_phone) < 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid phone number format",
        )

    otp = generate_otp()
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(minutes=10)

    # Invalidate old active OTPs for this phone/purpose.
    existing_query = select(OTPVerification).where(
        OTPVerification.phone == normalized_phone,
        OTPVerification.purpose == request.purpose,
        OTPVerification.is_used == False,  # noqa: E712
    )
    existing = (await db.execute(existing_query)).scalars().all()
    for row in existing:
        row.is_used = True
        row.used_at = now

    record = OTPVerification(
        phone=normalized_phone,
        otp=otp,
        purpose=request.purpose,
        expires_at=expires_at,
        is_used=False,
    )
    db.add(record)
    await db.flush()

    accepted = await send_whatsapp_otp(normalized_phone, otp)
    if not accepted:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send OTP",
        )

    return {"success": True, "message": "OTP sent successfully"}


@router.post("/verify-otp")
async def verify_otp(
    request: VerifyOTPRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Verify OTP entered by user.
    Returns a verification_token to use during registration.
    """
    normalized_phone = normalize_phone(request.phone)
    otp_query = (
        select(OTPVerification)
        .where(
            OTPVerification.phone == normalized_phone,
            OTPVerification.purpose == request.purpose,
            OTPVerification.otp == request.otp,
        )
        .order_by(OTPVerification.created_at.desc())
    )
    otp_row = (await db.execute(otp_query)).scalars().first()

    if not otp_row:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid OTP",
        )

    now = datetime.now(timezone.utc)
    expires_at = otp_row.expires_at
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)

    if otp_row.is_used:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="OTP already used. Please request a new OTP.",
        )
    if now > expires_at:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="OTP expired. Please request a new OTP.",
        )
    otp_row.is_used = True
    otp_row.used_at = now

    verification_token = AuthService.create_phone_verification_token(
        phone=normalized_phone,
        purpose=request.purpose
    )
    
    return {
        "success": True,
        "verified": True,
        "message": "Phone number verified successfully",
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
        expected_phone=normalize_phone(request.phone),
        expected_purpose="registration"
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Phone verification expired or invalid. Please verify with OTP again."
        )
    
    normalized_phone = normalize_phone(request.phone)
    recent_verified_query = (
        select(OTPVerification)
        .where(
            OTPVerification.phone == normalized_phone,
            OTPVerification.purpose == "registration",
            OTPVerification.is_used == True,  # noqa: E712
            OTPVerification.used_at.isnot(None),
            OTPVerification.used_at >= datetime.now(timezone.utc) - timedelta(minutes=30),
        )
        .order_by(OTPVerification.used_at.desc())
    )
    recent_verified = (await db.execute(recent_verified_query)).scalars().first()
    if not recent_verified:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="OTP verification not found or expired. Please verify OTP again.",
        )

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
    
    normalized_phone = normalize_phone(request.phone)
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
    
    recent_verified_query = (
        select(OTPVerification)
        .where(
            OTPVerification.phone == normalized_phone,
            OTPVerification.purpose == "login",
            OTPVerification.is_used == True,  # noqa: E712
            OTPVerification.used_at.isnot(None),
            OTPVerification.used_at >= datetime.now(timezone.utc) - timedelta(minutes=30),
        )
        .order_by(OTPVerification.used_at.desc())
    )
    recent_verified = (await db.execute(recent_verified_query)).scalars().first()
    if not recent_verified:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Login OTP verification expired or invalid. Please verify OTP again."
        )

    # Update last login
    await user_repo.update_last_login(user.id)
    
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
