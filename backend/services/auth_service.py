"""
Authentication Service
Handles password hashing and JWT token management
Similar to Spring Security's AuthenticationService
"""

import os
import bcrypt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from dotenv import load_dotenv

load_dotenv()

# JWT Configuration
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-super-secret-key-change-in-production")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return bcrypt.checkpw(
        plain_password.encode('utf-8'), 
        hashed_password.encode('utf-8')
    )


def get_password_hash(password: str) -> str:
    """Hash a password"""
    # Truncate to 72 bytes (bcrypt limit)
    password_bytes = password.encode('utf-8')[:72]
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password_bytes, salt).decode('utf-8')


class AuthService:
    """
    Authentication service for managing JWT tokens
    Similar to JwtTokenProvider in Spring Security
    """
    
    @staticmethod
    def create_access_token(
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create a JWT access token
        
        Args:
            data: Payload data (typically user_id, username, role)
            expires_delta: Optional custom expiration time
            
        Returns:
            Encoded JWT token
        """
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access"
        })
        
        return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    
    @staticmethod
    def create_refresh_token(
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create a JWT refresh token
        
        Args:
            data: Payload data (typically just user_id)
            expires_delta: Optional custom expiration time
            
        Returns:
            Encoded JWT token
        """
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh"
        })
        
        return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    
    @staticmethod
    def create_token_pair(user_id: int, username: str, role: str) -> Dict[str, Any]:
        """
        Create both access and refresh tokens
        
        Returns:
            Dictionary with access_token, refresh_token, token_type, expires_in
        """
        access_token_data = {
            "sub": str(user_id),
            "username": username,
            "role": role
        }
        refresh_token_data = {
            "sub": str(user_id)
        }
        
        return {
            "access_token": AuthService.create_access_token(access_token_data),
            "refresh_token": AuthService.create_refresh_token(refresh_token_data),
            "token_type": "bearer",
            "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60  # in seconds
        }
    
    @staticmethod
    def verify_token(token: str, token_type: str = "access") -> Optional[Dict[str, Any]]:
        """
        Verify and decode a JWT token
        
        Args:
            token: JWT token string
            token_type: Expected token type ("access" or "refresh")
            
        Returns:
            Decoded payload if valid, None otherwise
        """
        try:
            payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            
            # Check token type
            if payload.get("type") != token_type:
                return None
            
            # Check expiration
            exp = payload.get("exp")
            if exp and datetime.utcfromtimestamp(exp) < datetime.utcnow():
                return None
            
            return payload
            
        except JWTError:
            return None
    
    @staticmethod
    def get_user_id_from_token(token: str) -> Optional[int]:
        """Extract user ID from token"""
        payload = AuthService.verify_token(token)
        if payload:
            try:
                return int(payload.get("sub"))
            except (TypeError, ValueError):
                return None
        return None
    
    @staticmethod
    def refresh_access_token(refresh_token: str) -> Optional[Dict[str, Any]]:
        """
        Create new access token from refresh token
        
        Args:
            refresh_token: Valid refresh token
            
        Returns:
            New token pair if refresh token is valid, None otherwise
        """
        payload = AuthService.verify_token(refresh_token, token_type="refresh")
        
        if not payload:
            return None
        
        user_id = payload.get("sub")
        if not user_id:
            return None
        
        # Note: In production, fetch user from DB to get current role
        # For now, we just create a new access token with minimal data
        access_token_data = {
            "sub": user_id,
            "type": "access"
        }
        
        return {
            "access_token": AuthService.create_access_token(access_token_data),
            "token_type": "bearer",
            "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60
        }
    
    # ============== PHONE VERIFICATION TOKENS ==============
    
    @staticmethod
    def create_phone_verification_token(phone: str, purpose: str = "registration") -> str:
        """
        Create a short-lived token after successful OTP verification.
        This token proves the phone was verified and is used during registration.
        
        Args:
            phone: Verified phone number
            purpose: Purpose of verification (registration, login, reset)
            
        Returns:
            Signed JWT token valid for 30 minutes
        """
        to_encode = {
            "phone": phone,
            "purpose": purpose,
            "type": "phone_verification",
            "exp": datetime.utcnow() + timedelta(minutes=30),
            "iat": datetime.utcnow()
        }
        
        return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    
    @staticmethod
    def verify_phone_verification_token(
        token: str, 
        expected_phone: str, 
        expected_purpose: str = "registration"
    ) -> bool:
        """
        Verify a phone verification token.
        
        Args:
            token: The verification token
            expected_phone: Phone number that should match
            expected_purpose: Expected purpose (registration, login, reset)
            
        Returns:
            True if token is valid and matches phone/purpose, False otherwise
        """
        try:
            payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            
            # Check token type
            if payload.get("type") != "phone_verification":
                return False
            
            # Check phone matches
            if payload.get("phone") != expected_phone:
                return False
            
            # Check purpose matches
            if payload.get("purpose") != expected_purpose:
                return False
            
            return True
            
        except JWTError:
            return False
