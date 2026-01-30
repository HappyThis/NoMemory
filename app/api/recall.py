from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.agent.recall_agent import RecallAgent
from app.api.schemas import RecallRequest, RecallResponse
from app.auth.user import get_user_id
from app.db.session import get_db

router = APIRouter(prefix="/v1", tags=["recall"])


@router.post("/recall", response_model=RecallResponse)
def recall(req: RecallRequest, user_id: str = Depends(get_user_id), db: Session = Depends(get_db)) -> RecallResponse:
    agent = RecallAgent(db, user_id=user_id)
    ctx = req.context
    return agent.run(
        question=req.question,
        time_range=ctx.time_range if ctx else None,
        role_pref=ctx.role_pref if ctx else None,
    )

