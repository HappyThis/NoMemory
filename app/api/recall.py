from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.agent.recall_agent import RecallAgent
from app.agent.errors import RecallAgentError
from app.api.schemas import RecallRequest, RecallResponse
from app.auth.user import get_user_id
from app.db.session import get_db
from app.llm.bigmodel import BigModelError
from app.settings import settings

router = APIRouter(prefix="/v1", tags=["recall"])


@router.post("/recall", response_model=RecallResponse)
def recall(req: RecallRequest, user_id: str = Depends(get_user_id), db: Session = Depends(get_db)) -> RecallResponse:
    if not settings.bigmodel_api_key or settings.llm_provider != "bigmodel":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM not configured for skill-driven recall (set BIGMODEL_API_KEY and LLM_PROVIDER=bigmodel).",
        )
    agent = RecallAgent(db, user_id=user_id)
    try:
        return agent.run(question=req.question)
    except RecallAgentError as e:
        # Agent couldn't finish within its internal budgets/constraints.
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail={"code": e.code, "message": str(e)}) from e
    except BigModelError as e:
        code = getattr(e, "status_code", None)
        if code == 429:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(e)) from e
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e)) from e
