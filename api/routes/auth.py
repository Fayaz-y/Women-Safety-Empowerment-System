"""
Women Safety AI — Auth Login Route
====================================
POST /login — OAuth2 form-based login returning a JWT.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from api.auth import (
    OPERATOR_PASSWORD_HASH,
    OPERATOR_USERNAME,
    create_access_token,
    verify_password,
)
from api.schemas import Token

router = APIRouter()


@router.post("/login", response_model=Token)
def login(form: OAuth2PasswordRequestForm = Depends()):
    """Authenticate the operator and return a JWT access token."""
    if form.username != OPERATOR_USERNAME or not verify_password(
        form.password, OPERATOR_PASSWORD_HASH
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token({"sub": form.username})
    return {"access_token": token, "token_type": "bearer"}
