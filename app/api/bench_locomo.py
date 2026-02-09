from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.agent.errors import RecallAgentError
from app.agent.recall_agent import RecallAgent
from app.api.schemas import LocomoQARequest, LocomoQAResponse
from app.auth.user import get_user_id
from app.db.session import get_db
from app.llm.bigmodel import BigModelError
from app.settings import settings


router = APIRouter(prefix="/v1/bench/locomo", tags=["bench", "locomo"])


@router.post("/qa", response_model=LocomoQAResponse)
def locomo_qa(req: LocomoQARequest, user_id: str = Depends(get_user_id), db: Session = Depends(get_db)) -> LocomoQAResponse:
    if not settings.bigmodel_api_key or settings.llm_provider != "bigmodel":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM not configured (set BIGMODEL_API_KEY and LLM_PROVIDER=bigmodel).",
        )

    agent = RecallAgent(db, user_id=user_id, skill_name="nomemory-locomo-qa")
    try:
        answer, evidence = agent.run_locomo_qa(question=req.question, now=req.now)
        return LocomoQAResponse(answer=answer, evidence=evidence)
    except RecallAgentError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail={"code": e.code, "message": str(e)}) from e
    except BigModelError as e:
        code = getattr(e, "status_code", None)
        if code == 429:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(e)) from e
        if code == 504:
            raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=str(e)) from e
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e)) from e
