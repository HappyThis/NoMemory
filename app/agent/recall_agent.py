from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy.orm import Session

from app.api.schemas import MemoryView, RecallLimits, RecallResponse, TimeRange
from app.llm.bigmodel import BigModelClient, BigModelError, BigModelMessage
from app.retrieval.lexical import lexical_search
from app.retrieval.neighbors import get_neighbors
from app.retrieval.semantic import semantic_search
from app.settings import settings


def _infer_days(question: str) -> int:
    q = question
    if any(k in q for k in ["刚才", "刚刚", "这两天", "今天", "昨日", "昨天"]):
        return 2
    if any(k in q for k in ["最近", "这周", "本周", "这星期"]):
        return 7
    if any(k in q for k in ["上次", "之前", "我们聊过", "前面"]):
        return 30
    if any(k in q for k in ["长期", "一直", "习惯", "从小", "多年"]):
        return 180
    return 30


def _extract_json(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found")
    return json.loads(text[start : end + 1])


@dataclass(frozen=True)
class RecallConfig:
    max_iterations: int = 2
    max_evidence: int = 6
    neighbors_before: int = 8


class RecallAgent:
    def __init__(self, db: Session, *, user_id: str, config: Optional[RecallConfig] = None) -> None:
        self.db = db
        self.user_id = user_id
        self.config = config or RecallConfig()

    def run(
        self,
        *,
        question: str,
        time_range: Optional[TimeRange],
        role_pref: Optional[str],
    ) -> RecallResponse:
        now = datetime.now(tz=timezone.utc)
        if time_range is None:
            days = _infer_days(question)
            time_range = TimeRange(since=now - timedelta(days=days), until=now)
        if time_range.until is None:
            time_range.until = now

        role = None if role_pref == "any" else "user"

        evidence: list[tuple[str, str, str]] = []  # (user_id, message_id, source)
        expanded = False

        for _i in range(self.config.max_iterations):
            lexical = lexical_search(
                self.db,
                user_id=self.user_id,
                query_text=question,
                since=time_range.since,
                until=time_range.until,
                role=role,
                page_size=10,
                anchor=None,
            )
            for m, _rank in lexical:
                if all((uid != m.user_id or mid != m.message_id) for (uid, mid, _src) in evidence):
                    evidence.append((m.user_id, m.message_id, "lexical"))

            # Try semantic only if embedding provider configured and embeddings exist.
            if settings.bigmodel_api_key:
                try:
                    qemb = BigModelClient().embeddings(inputs=[question], model=settings.bigmodel_embedding_model)[0]
                    sem = semantic_search(
                        self.db,
                        user_id=self.user_id,
                        query_embedding=qemb,
                        since=time_range.since,
                        until=time_range.until,
                        role=role,
                        top_k=10,
                        min_score=0.0,
                        provider=settings.llm_provider,
                        model=settings.bigmodel_embedding_model,
                    )
                    for m, _score in sem:
                        if all((uid != m.user_id or mid != m.message_id) for (uid, mid, _src) in evidence):
                            evidence.append((m.user_id, m.message_id, "semantic"))
                except BigModelError:
                    pass

            if evidence or expanded:
                break

            # No evidence: widen window once.
            time_range.since = now - timedelta(days=180)
            expanded = True

        # Materialize evidence messages and add neighbor context for top hits.
        messages = []
        seen = set()
        for _uid, mid, _src in evidence[: self.config.max_evidence]:
            neigh = get_neighbors(
                self.db,
                user_id=self.user_id,
                message_id=mid,
                before=self.config.neighbors_before,
                after=0,
            )
            for m in neigh:
                k = (m.user_id, m.message_id)
                if k in seen:
                    continue
                seen.add(k)
                messages.append(m)

        messages.sort(key=lambda m: (m.ts, m.message_id))
        messages = messages[-self.config.max_evidence :]

        memory_view = self._synthesize(question=question, evidence_messages=messages) if messages else MemoryView()

        return RecallResponse(
            memory_view=memory_view,
            evidence=[
                {
                    "message_id": m.message_id,
                    "ts": m.ts,
                    "user_id": m.user_id,
                    "role": m.role,
                    "content": m.content,
                    "meta": m.meta,
                }
                for m in messages
            ],
            limits=RecallLimits(time_range=time_range, role=role or "any"),
        )

    def _synthesize(self, *, question: str, evidence_messages) -> MemoryView:
        if not settings.bigmodel_api_key or settings.llm_provider != "bigmodel":
            return MemoryView()
        try:
            client = BigModelClient()
        except BigModelError:
            return MemoryView()

        evidence_lines = "\n".join(
            f"- [{m.ts.isoformat()}] ({m.role}) {m.content}" for m in evidence_messages if m.content
        )
        prompt = (
            "你是一个“用户记忆回忆”助手。你只能基于提供的证据提取用户记忆，不要编造。\n"
            "请输出 JSON，对象结构为："
            '{"preferences":[...],"profile":[...],"constraints":[...]}\n'
            "每个数组元素是简短中文字符串。若证据不足，对应数组留空。\n\n"
            f"问题：{question}\n\n"
            f"证据：\n{evidence_lines}\n"
        )
        try:
            text = client.chat(
                model=settings.llm_model,
                messages=[BigModelMessage(role="user", content=prompt)],
                temperature=0.2,
            )
            obj = _extract_json(text)
            return MemoryView(
                preferences=[str(x) for x in obj.get("preferences", [])],
                profile=[str(x) for x in obj.get("profile", [])],
                constraints=[str(x) for x in obj.get("constraints", [])],
            )
        except Exception:  # noqa: BLE001
            return MemoryView()
