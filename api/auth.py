"""
Women Safety AI — JWT Authentication Module
=============================================
Implements password hashing, JWT token creation / verification,
and a FastAPI dependency ``get_current_user`` that protects routes.

Default operator: admin / changeme
"""

from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

from config.settings import settings

# ── Password hashing ──
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ── OAuth2 scheme ──
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")

# ── Hardcoded operator (single-user) ──
OPERATOR_USERNAME = "admin"
OPERATOR_PASSWORD_HASH = pwd_context.hash("changeme")


def verify_password(plain: str, hashed: str) -> bool:
    """Check a plain-text password against a bcrypt hash."""
    return pwd_context.verify(plain, hashed)


def create_access_token(
    data: dict, expires_delta: Optional[timedelta] = None
) -> str:
    """Create a signed JWT access token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + (
        expires_delta or timedelta(minutes=settings.jwt_expire_minutes)
    )
    to_encode["exp"] = expire
    return jwt.encode(
        to_encode, settings.jwt_secret_key, algorithm=settings.jwt_algorithm
    )


def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    """
    FastAPI dependency — decode JWT, extract username.

    Raises HTTP 401 if the token is invalid or missing ``sub``.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(
            token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm]
        )
        username: Optional[str] = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    return username
