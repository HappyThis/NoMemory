from __future__ import annotations

import json
import re

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.agent.errors import RecallAgentError
from app.agent.recall_agent import RecallAgent
from app.api.schemas import LocomoJudgeRequest, LocomoJudgeResponse, LocomoQARequest, LocomoQAResponse
from app.auth.user import get_user_id
from app.db.session import get_db
from app.llm.errors import LLMError
from app.llm.factory import get_chat_client, get_chat_model
from app.llm.bigmodel import BigModelMessage
from app.settings import settings


router = APIRouter(prefix="/v1/bench/locomo", tags=["bench", "locomo"])


@router.post("/qa", response_model=LocomoQAResponse)
def locomo_qa(req: LocomoQARequest, user_id: str = Depends(get_user_id), db: Session = Depends(get_db)) -> LocomoQAResponse:
    llm_ok = False
    if settings.llm_provider == "bigmodel":
        llm_ok = bool(settings.bigmodel_api_key)
    elif settings.llm_provider == "siliconflow":
        llm_ok = bool(settings.siliconflow_api_key)
    if not llm_ok:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM not configured (check LLM_PROVIDER and provider API key).",
        )

    agent = RecallAgent(db, user_id=user_id, skill_name="nomemory-locomo-qa")
    try:
        answer, evidence = agent.run_locomo_qa(question=req.question, now=req.now)
        return LocomoQAResponse(answer=answer, evidence=evidence)
    except RecallAgentError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail={"code": e.code, "message": str(e)}) from e
    except LLMError as e:
        code = getattr(e, "status_code", None)
        if code == 429:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(e)) from e
        if code == 504:
            raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=str(e)) from e
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e)) from e


def _extract_json_object(text: str) -> dict:
    s = (text or "").strip()
    if not s:
        raise ValueError("empty judge output")
    # Strip common markdown fences.
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    # Try direct parse first.
    try:
        obj = json.loads(s)
    except Exception:
        # Fallback: best-effort locate the outermost JSON object.
        i = s.find("{")
        j = s.rfind("}")
        if i < 0 or j < 0 or j <= i:
            raise
        obj = json.loads(s[i : j + 1])
    if not isinstance(obj, dict):
        raise ValueError("judge output is not a JSON object")
    return obj


@router.post("/judge", response_model=LocomoJudgeResponse)
def locomo_judge(req: LocomoJudgeRequest) -> LocomoJudgeResponse:
    llm_ok = False
    if settings.llm_provider == "bigmodel":
        llm_ok = bool(settings.bigmodel_api_key)
    elif settings.llm_provider == "siliconflow":
        llm_ok = bool(settings.siliconflow_api_key)
    if not llm_ok:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM not configured (check LLM_PROVIDER and provider API key).",
        )

    gold = req.gold_answer
    if gold is None:
        gold_str = "null"
    else:
        try:
            gold_str = json.dumps(gold, ensure_ascii=False)
        except Exception:
            gold_str = str(gold)

    adv_note = ""
    if req.category == 5 and (gold is None or (isinstance(gold, str) and not gold.strip())):
        adv_note = (
            "\nThis is an Adversarial question without a gold answer. The generated answer is CORRECT "
            "ONLY if it is exactly 'unknown' (case-insensitive)."
        )

    now_note = ""
    if req.now is not None:
        now_note = f"\nNow: {req.now.isoformat()}"

    system = (
        "You are a strict but fair judge. You will be given a question, a gold (ground truth) answer, "
        "and a generated answer. Decide if the generated answer is CORRECT or WRONG.\n"
        "Be generous: if the generated answer conveys the same meaning as the gold answer, count it as CORRECT.\n"
        "Return ONLY a JSON object: {\"label\":\"CORRECT\"} or {\"label\":\"WRONG\"}. Do not include any other text."
    )
    user = (
        f"{adv_note}{now_note}\n\n"
        f"Question: {req.question}\n"
        f"Gold answer: {gold_str}\n"
        f"Generated answer: {req.pred_answer}\n"
    ).strip()

    messages = [BigModelMessage(role="system", content=system), BigModelMessage(role="user", content=user)]
    client = get_chat_client()
    model = get_chat_model()

    try:
        obj: dict | None = None
        if hasattr(client, "chat_json"):
            try:
                obj = client.chat_json(messages=messages, model=model, temperature=0.0)  # type: ignore[attr-defined]
            except LLMError as e:
                # Some providers may not support JSON mode; fall back to strict parsing.
                code = getattr(e, "status_code", None)
                if code not in (400, 422):
                    raise
            except Exception:
                obj = None
        if obj is None:
            msg = client.chat_message(messages=messages, model=model, temperature=0.0)  # type: ignore[no-any-return]
            content = msg.get("content") if isinstance(msg, dict) else None
            obj = _extract_json_object(str(content or ""))
        label = obj.get("label")
        if label not in ("CORRECT", "WRONG"):
            raise ValueError(f"invalid judge label: {label!r}")
        reason = obj.get("reason")
        if not isinstance(reason, str):
            reason = None
        return LocomoJudgeResponse(label=label, reason=reason)
    except LLMError as e:
        code = getattr(e, "status_code", None)
        if code == 429:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(e)) from e
        if code == 504:
            raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=str(e)) from e
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"Judge output parse error: {type(e).__name__}: {e}") from e
