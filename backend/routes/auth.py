"""
Authentication Routes
Handles user registration, login, logout, OTP verification
"""

from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, Dict, Any

from database.connection import get_db
from models.user import User, UserRole
from models.activity_log import ActionType
from repositories.user_repository import UserRepository
from repositories.activity_log_repository import ActivityLogRepository
from services.auth_service import AuthService, get_password_hash, verify_password
from services.otp_service import OTPService
from dependencies.auth import get_current_user, get_current_admin

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
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: str = Field(..., min_length=2, max_length=100)
    phone: str = Field(..., min_length=10, max_length=15)
    verification_token: str = Field(..., description="Token received after OTP verification")
    
    # Student preferences for personalized experience
    preferred_state: Optional[str] = Field(None, description="Home state for counseling info")
    category: Optional[str] = Field(None, description="Category: General/OBC/SC/ST/EWS")


class LoginRequest(BaseModel):
    username: str  # Can be username, email, or phone
    password: str


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=8)


class ForgotPasswordRequest(BaseModel):
    """Username or email for the account to reset."""
    identifier: str = Field(..., min_length=3, max_length=255)


class ForgotPasswordVerifyRequest(BaseModel):
    phone: str = Field(..., min_length=10, max_length=15)
    otp: str = Field(..., min_length=4, max_length=8)


class ForgotPasswordResetRequest(BaseModel):
    reset_token: str = Field(..., min_length=20)
    new_password: str = Field(..., min_length=8)


# ============== RESPONSE SCHEMAS ==============

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    full_name: str
    phone: Optional[str]
    role: str
    is_active: bool
    is_verified: bool
    preferences: Optional[Dict[str, Any]] = None
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
async def send_otp(request: SendOTPRequest):
    """
    Send OTP to phone number for verification
    """
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
    
    # Check if username exists
    if await user_repo.exists_by_username(request.username):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username already taken"
        )
    
    # Check if email exists
    if await user_repo.exists_by_email(request.email):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered"
        )
    
    # Create user with preferences
    preferences = {}
    if request.preferred_state:
        preferences["preferred_state"] = request.preferred_state
    if request.category:
        preferences["category"] = request.category
    
    user = User(
        username=request.username.lower(),
        email=request.email.lower(),
        password_hash=get_password_hash(request.password),
        full_name=request.full_name,
        phone=request.phone,
        role=UserRole.STUDENT,
        is_active=True,
        is_verified=True,  # Verified via OTP
        preferences=preferences if preferences else {}
    )
    
    user = await user_repo.create(user)
    
    # Clear OTP
    OTPService.clear_otp(request.phone)
    
    # Log activity
    await activity_repo.log_action(
        action_type=ActionType.REGISTER,
        description=f"New user registered: {user.username}",
        user_id=user.id,
        target_type="user",
        target_id=str(user.id),
        ip_address=req.client.host if req.client else None
    )
    
    # Generate tokens
    tokens = AuthService.create_token_pair(user.id, user.username, user.role.value)
    
    return AuthResponse(
        access_token=tokens["access_token"],
        refresh_token=tokens["refresh_token"],
        token_type=tokens["token_type"],
        expires_in=tokens["expires_in"],
        user=UserResponse(
            id=user.id,
            username=user.username,
            email=user.email,
            full_name=user.full_name,
            phone=user.phone,
            role=user.role.value,
            is_active=user.is_active,
            is_verified=user.is_verified,
            preferences=user.preferences or {},
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
    Login with username/email/phone and password
    """
    user_repo = UserRepository(db)
    activity_repo = ActivityLogRepository(db)
    
    # Find user by username, email, or phone
    user = await user_repo.find_by_username_or_email(request.username)
    
    if not user:
        # Try finding by phone
        # Note: You might want to add a find_by_phone method
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    
    # Verify password
    if not verify_password(request.password, user.password_hash):
        # Log failed attempt
        await activity_repo.log_action(
            action_type=ActionType.LOGIN,
            description=f"Failed login attempt for: {request.username}",
            user_id=user.id,
            ip_address=req.client.host if req.client else None,
            success=False
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    
    # Check if active
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated"
        )
    
    # Update last login
    await user_repo.update_last_login(user.id)
    
    # Log successful login
    await activity_repo.log_action(
        action_type=ActionType.LOGIN,
        description=f"User logged in: {user.username}",
        user_id=user.id,
        ip_address=req.client.host if req.client else None
    )
    
    # Generate tokens
    tokens = AuthService.create_token_pair(user.id, user.username, user.role.value)
    
    return AuthResponse(
        access_token=tokens["access_token"],
        refresh_token=tokens["refresh_token"],
        token_type=tokens["token_type"],
        expires_in=tokens["expires_in"],
        user=UserResponse(
            id=user.id,
            username=user.username,
            email=user.email,
            full_name=user.full_name,
            phone=user.phone,
            role=user.role.value,
            is_active=user.is_active,
            is_verified=user.is_verified,
            preferences=user.preferences or {},
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
        username=current_user.username,
        email=current_user.email,
        full_name=current_user.full_name,
        phone=current_user.phone,
        role=current_user.role.value,
        is_active=current_user.is_active,
        is_verified=current_user.is_verified,
        preferences=current_user.preferences or {},
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
        description=f"User logged out: {current_user.username}",
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
    request: ForgotPasswordRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Look up account by username or email and send OTP to the registered mobile number.
    Response is generic if account is missing (no user enumeration).
    """
    user_repo = UserRepository(db)
    user = await user_repo.find_by_username_or_email(request.identifier.strip())

    generic_ok = {
        "success": True,
        "message": "If an account exists for this email or username, an OTP has been sent to the registered mobile number.",
    }

    if not user:
        return generic_ok

    if not user.phone or not str(user.phone).strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No mobile number on file for this account. Please contact support.",
        )

    result = OTPService.send_otp(user.phone, purpose="password_reset")
    if not result["success"]:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=result["message"],
        )

    formatted = OTPService._format_phone(user.phone)
    return {
        "success": True,
        "message": f"OTP sent to registered number ending {_mask_phone_hint(formatted)}",
        "phone_last4": formatted[-4:] if len(formatted) >= 4 else None,
        **({"otp": result["otp"]} if result.get("otp") else {}),
    }


@router.post("/forgot-password/verify")
async def forgot_password_verify(
    request: ForgotPasswordVerifyRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Verify OTP for password reset; returns a short-lived reset_token (JWT).
    """
    result = OTPService.verify_otp(
        request.phone, request.otp, purpose="password_reset"
    )
    if not result["success"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["message"],
        )

    user_repo = UserRepository(db)
    user = await user_repo.find_by_normalized_phone(request.phone)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No account matches this phone number.",
        )
    formatted = OTPService._format_phone(request.phone)
    token = AuthService.create_password_reset_token(user.id, formatted)
    return {"success": True, "reset_token": token}


@router.post("/forgot-password/reset")
async def forgot_password_reset(
    request: ForgotPasswordResetRequest,
    req: Request,
    db: AsyncSession = Depends(get_db),
):
    """Set a new password using reset_token from /forgot-password/verify."""
    payload = AuthService.verify_password_reset_token(request.reset_token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset session. Please start again.",
        )

    user_repo = UserRepository(db)
    activity_repo = ActivityLogRepository(db)
    user = await user_repo.get_by_id(payload["user_id"])
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Account not found or deactivated.",
        )

    expected_phone = OTPService._format_phone(payload["phone"])
    if user.phone:
        actual = OTPService._format_phone(user.phone)
        if actual != expected_phone:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired reset session.",
            )

    user.password_hash = get_password_hash(request.new_password)
    await user_repo.update(user)
    OTPService.clear_otp(expected_phone)

    await activity_repo.log_action(
        action_type=ActionType.PASSWORD_CHANGE,
        description=f"Password reset via OTP for: {user.username}",
        user_id=user.id,
        ip_address=req.client.host if req.client else None,
    )

    return {"success": True, "message": "Password updated. You can sign in now."}


@router.post("/change-password")
async def change_password(
    request: ChangePasswordRequest,
    req: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Change current user's password
    """
    user_repo = UserRepository(db)
    activity_repo = ActivityLogRepository(db)
    
    # Verify current password
    if not verify_password(request.current_password, current_user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect"
        )
    
    # Update password
    current_user.password_hash = get_password_hash(request.new_password)
    await user_repo.update(current_user)
    
    # Log action
    await activity_repo.log_action(
        action_type=ActionType.PASSWORD_CHANGE,
        description=f"Password changed for: {current_user.username}",
        user_id=current_user.id,
        ip_address=req.client.host if req.client else None
    )
    
    return {"message": "Password changed successfully"}


# ============== ADMIN ROUTES ==============

@router.post("/admin/create-user", response_model=UserResponse)
async def create_user(
    request: RegisterRequest,
    role: str = "student",
    current_admin: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Admin: Create a new user without OTP verification
    """
    user_repo = UserRepository(db)
    
    # Validate role
    try:
        user_role = UserRole(role)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role. Must be one of: {[r.value for r in UserRole]}"
        )
    
    # Only super_admin can create admins
    if user_role in [UserRole.ADMIN, UserRole.SUPER_ADMIN]:
        if current_admin.role != UserRole.SUPER_ADMIN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only super admin can create admin users"
            )
    
    # Check if username exists
    if await user_repo.exists_by_username(request.username):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username already taken"
        )
    
    # Check if email exists
    if await user_repo.exists_by_email(request.email):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered"
        )
    
    # Create user
    user = User(
        username=request.username.lower(),
        email=request.email.lower(),
        password_hash=get_password_hash(request.password),
        full_name=request.full_name,
        phone=request.phone,
        age=request.age,
        role=user_role,
        is_active=True,
        is_verified=True,
        target_exams=request.target_exams or []
    )
    
    user = await user_repo.create(user)
    
    return UserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        phone=user.phone,
        age=user.age,
        role=user.role.value,
        is_active=user.is_active,
        is_verified=user.is_verified,
        target_exams=user.target_exams or [],
        created_at=user.created_at
    )
