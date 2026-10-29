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
    if (
        credentials is None
        or not secrets.compare_digest(credentials.username, settings.auth_username)
        or not secrets.compare_digest(credentials.password, settings.auth_password)
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username
