"""HTTP Basic auth (opt-in), constant-time, Unicode-safe."""

from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from .config import get_settings

security = HTTPBasic(auto_error=False)


def require_auth(credentials: Annotated[HTTPBasicCredentials | None, Depends(security)]) -> str:
    settings = get_settings()
    if not settings.auth_enabled:
        return ""
    if credentials is None:
        raise _unauthorized()
    try:
        same_user = secrets.compare_digest(credentials.username.encode(), settings.auth_username.encode())
        same_password = secrets.compare_digest(credentials.password.encode(), settings.auth_password.encode())
    except (UnicodeEncodeError, ValueError):
        same_user = same_password = False  # non-ASCII credentials are simply wrong, not a crash
    if not (same_user and same_password):
        raise _unauthorized()
    return credentials.username


def _unauthorized() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required",
        headers={"WWW-Authenticate": "Basic"},
    )
