from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.agent.recall_agent import RecallAgent
from app.agent.errors import RecallAgentError
from app.api.schemas import RecallRequest, RecallResponse
from app.auth.user import get_user_id
from app.db.session import get_db
from app.llm.errors import LLMError
from app.settings import settings

router = APIRouter(prefix="/v1", tags=["recall"])


@router.post("/recall", response_model=RecallResponse)
def recall(req: RecallRequest, user_id: str = Depends(get_user_id), db: Session = Depends(get_db)) -> RecallResponse:
    llm_ok = False
    if settings.llm_provider == "bigmodel":
        llm_ok = bool(settings.bigmodel_api_key)
    elif settings.llm_provider == "siliconflow":
        llm_ok = bool(settings.siliconflow_api_key)
    if not llm_ok:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM not configured for skill-driven recall (check LLM_PROVIDER and provider API key).",
        )
    agent = RecallAgent(db, user_id=user_id)
    try:
        return agent.run(question=req.question)
    except RecallAgentError as e:
        # Agent couldn't finish within its internal budgets/constraints.
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail={"code": e.code, "message": str(e)}) from e
    except LLMError as e:
        code = getattr(e, "status_code", None)
        if code == 429:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(e)) from e
        if code == 504:
            raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=str(e)) from e
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e)) from e
