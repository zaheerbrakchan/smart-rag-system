"""
Authentication Dependencies
FastAPI dependencies for protected routes
Similar to @PreAuthorize and Spring Security filters
"""

from typing import Optional, List
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession

from database.connection import get_db
from models.user import User, UserRole
from repositories.user_repository import UserRepository
from services.auth_service import AuthService

# HTTP Bearer token scheme
security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
) -> User:
    """
    Get current authenticated user from JWT token.
    Similar to @AuthenticationPrincipal in Spring Security.
    
    Usage:
        @app.get("/protected")
        async def protected_route(current_user: User = Depends(get_current_user)):
            return {"user": current_user.username}
    """
    token = credentials.credentials
    
    payload = AuthService.verify_token(token, token_type="access")
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    user_repo = UserRepository(db)
    user = await user_repo.get_by_id(int(user_id))
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is deactivated"
        )
    
    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """Get current user, ensuring they are active"""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )
    return current_user


async def get_current_admin(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Get current user, ensuring they have admin privileges.
    Similar to @PreAuthorize("hasRole('ADMIN')") in Spring Security.
    
    Usage:
        @app.post("/admin/users")
        async def create_user(admin: User = Depends(get_current_admin)):
            ...
    """
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return current_user


async def get_current_super_admin(
    current_user: User = Depends(get_current_user)
) -> User:
    """Get current user, ensuring they are super admin"""
    if current_user.role != UserRole.SUPER_ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Super admin privileges required"
        )
    return current_user


def require_roles(allowed_roles: List[UserRole]):
    """
    Create a dependency that requires specific roles.
    Similar to @PreAuthorize("hasAnyRole('ADMIN', 'SUPER_ADMIN')").
    
    Usage:
        @app.get("/reports")
        async def get_reports(
            user: User = Depends(require_roles([UserRole.ADMIN, UserRole.SUPER_ADMIN]))
        ):
            ...
    """
    async def role_checker(
        current_user: User = Depends(get_current_user)
    ) -> User:
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Required roles: {[r.value for r in allowed_roles]}"
            )
        return current_user
    
    return role_checker


# Optional: Get user if authenticated, None otherwise
async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(
        HTTPBearer(auto_error=False)
    ),
    db: AsyncSession = Depends(get_db)
) -> Optional[User]:
    """
    Get current user if authenticated, None otherwise.
    Useful for routes that work both authenticated and anonymously.
    """
    if not credentials:
        return None
    
    try:
        return await get_current_user(credentials, db)
    except HTTPException:
        return None
