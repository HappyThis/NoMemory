from typing import Optional

import secrets

from fastapi import Depends, Header, HTTPException, status

from app.settings import settings


def require_ingest_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    if not settings.ingest_api_key:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="INGEST_API_KEY not configured")
    if x_api_key is None or not secrets.compare_digest(x_api_key, settings.ingest_api_key):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


IngestAuthDep = Depends(require_ingest_api_key)
