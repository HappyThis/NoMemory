from typing import Optional

from fastapi import Header, HTTPException, status


def get_user_id(x_user_id: Optional[str] = Header(default=None)) -> str:
    # v0 dev-mode auth: callers pass X-User-Id.
    # Replace with real token/session auth in production.
    if not x_user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing X-User-Id")
    return x_user_id
