from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, ValidationError
from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from app.log import get_logger
from app.agent.errors import RecallAgentError
from app.agent.contracts import LocomoQAOutput, SynthesisOutput, example_json_for_prompt
from app.api.schemas import ChatMessage, RecallResponse
from app.llm.bigmodel import BigModelMessage
from app.llm.embeddings import embed_query_text
from app.llm.factory import get_chat_client, get_chat_model
from app.retrieval.lexical import LexicalAnchor, lexical_search
from app.retrieval.messages import list_messages
from app.retrieval.neighbors import get_neighbors
from app.retrieval.semantic import semantic_search
from app.settings import settings
from app.skills.loader import SkillNotFoundError, load_skill
from app.agent.tool_schemas import (
    LexicalSearchArgs,
    MessagesListArgs,
    NeighborsArgs,
    SemanticSearchArgs,
    bigmodel_tool_schemas,
)
from app.utils.cursor import CursorError, decode_cursor, make_seek_cursor, parse_seek_anchor
from app.utils.datetime import parse_datetime
from app.db.models import Message, MessageEmbedding


logger = get_logger(__name__)


def _make_observation_safe(obs: dict[str, Any]) -> dict[str, Any]:
    def _jsonable_item(item: dict[str, Any]) -> dict[str, Any]:
        out = dict(item)
        ts = out.get("ts")
        if hasattr(ts, "isoformat"):
            out["ts"] = ts.isoformat()
        out.pop("user_id", None)
        return out

    safe: dict[str, Any] = {"tool": obs.get("tool")}
    if "items" in obs:
        safe["items"] = [_jsonable_item(x) for x in (obs.get("items") or [])]
    if "anchors" in obs:
        safe["anchors"] = obs.get("anchors")
    if "next_cursor" in obs:
        safe["next_cursor"] = obs.get("next_cursor")
    if "applied_limits" in obs:
        safe["applied_limits"] = obs.get("applied_limits")
    if "error" in obs:
        safe["error"] = obs.get("error")
    for k in (
        "total_items",
        "returned_items",
        "truncated",
        "truncate_reason",
        "truncate_policy",
        "max_tool_items",
        "sort",
    ):
        if k in obs:
            safe[k] = obs.get(k)
    return safe


_TOOL_ARG_MODELS: dict[str, Any] = {
    "messages_list": MessagesListArgs,
    "lexical_search": LexicalSearchArgs,
    "semantic_search": SemanticSearchArgs,
    "neighbors": NeighborsArgs,
}


def _validate_tool_args(*, tool: str, args: dict[str, Any]) -> Any:
    model = _TOOL_ARG_MODELS.get(tool)
    if model is None:
        raise ValueError(f"No args schema for tool: {tool}")
    return model.model_validate(args)


def _clip_text(text: str, max_len: int) -> str:
    text = " ".join(str(text).split())
    if max_len <= 0:
        return ""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "…"


def _extract_first_json_object(text: str) -> dict[str, Any]:
    """
    Extract the first JSON object from an LLM text response.

    Tolerates:
    - ```json ... ``` code fences
    - Leading/trailing commentary around the JSON
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Empty content")

    s = text.strip()
    if s.startswith("```"):
        first_nl = s.find("\n")
        if first_nl != -1:
            s = s[first_nl + 1 :]
        if s.endswith("```"):
            s = s[: -3]
        s = s.strip()

    # Fast path: content is a single JSON object.
    try:
        v = json.loads(s)
        if isinstance(v, dict):
            return v
    except Exception:
        pass

    start = s.find("{")
    if start == -1:
        raise ValueError("No JSON object found")

    in_string = False
    escape = False
    depth = 0
    end: int | None = None
    for i in range(start, len(s)):
        ch = s[i]
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        raise ValueError("Unclosed JSON object")

    candidate = s[start:end].strip()
    v2 = json.loads(candidate)
    if not isinstance(v2, dict):
        raise ValueError("Expected JSON object")
    return v2


def _normalize_bigmodel_tool_args(args: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize provider-specific argument tagging formats into a plain JSON args dict.

    Observed format:
      {"query_text<arg_value>xxx</arg_value><arg_key>filter": "{\"role\":\"user\",...}"}
    Rewritten to:
      {"query_text": "xxx", "filter": {...}}
    """
    if not isinstance(args, dict):
        return {}

    out: dict[str, Any] = dict(args)

    for raw_key, raw_val in list(args.items()):
        if not isinstance(raw_key, str):
            continue
        if "<arg_value>" not in raw_key or "</arg_value>" not in raw_key or "<arg_key>" not in raw_key:
            continue
        try:
            k1, rest = raw_key.split("<arg_value>", 1)
            v1, rest2 = rest.split("</arg_value>", 1)
            _unused, k2 = rest2.split("<arg_key>", 1)
            k1 = k1.strip()
            k2 = k2.strip()
        except Exception:
            continue

        if k1 and k1 not in out:
            out[k1] = v1

        if k2 and k2 not in out:
            v2: Any = raw_val
            if isinstance(v2, str):
                vv = v2.strip()
                if vv.startswith("{") and vv.endswith("}"):
                    try:
                        v2 = json.loads(vv)
                    except Exception:
                        v2 = raw_val
            out[k2] = v2

        out.pop(raw_key, None)

    if isinstance(out.get("filter"), str):
        s = str(out["filter"]).strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                out["filter"] = json.loads(s)
            except Exception:
                pass

    return out


def _apply_tool_item_budget(obs: dict[str, Any], *, tool: str, max_tool_items: int) -> dict[str, Any]:
    """
    Hard constraint: prevent context explosion by truncating tool-returned `items` after the tool runs.

    - Do NOT clamp tool-call input params (we assume tool performance is OK).
    - When truncating:
      - For ranking-based tools (semantic_search/lexical_search) and neighbors, preserve tool order (take first N).
      - For messages_list, sort by ts desc (latest first) then message_id desc.
    - Report total/returned counts and truncation reason back to the model via role=tool JSON.
    """
    items = obs.get("items")
    if not isinstance(items, list):
        obs["total_items"] = 0
        obs["returned_items"] = 0
        obs["truncated"] = False
        obs["max_tool_items"] = max_tool_items
        return obs

    total = len(items)
    obs["total_items"] = total
    obs["max_tool_items"] = max_tool_items

    if max_tool_items is None or int(max_tool_items) <= 0:
        obs["items"] = []
        obs["returned_items"] = 0
        obs["truncated"] = total > 0
        obs["truncate_reason"] = "max_tool_items_non_positive" if total > 0 else None
        obs["truncate_policy"] = "tool_specific"
        obs["sort"] = "n/a"
        obs["next_cursor"] = None
        obs["anchors"] = []
        return obs

    max_tool_items = int(max_tool_items)
    if total <= max_tool_items:
        obs["returned_items"] = total
        obs["truncated"] = False
        return obs

    if tool in ("semantic_search", "lexical_search", "neighbors"):
        trimmed = items[:max_tool_items]
        obs["truncate_policy"] = "take_first_in_tool_order"
        obs["sort"] = "tool_order"
    else:
        # Default for messages_list (and any other future time-based listing tools).
        def _ts_key(x: Any) -> datetime:
            ts = x.get("ts") if isinstance(x, dict) else None
            if isinstance(ts, datetime):
                return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)
            if isinstance(ts, str):
                dt = parse_datetime(ts)
                if isinstance(dt, datetime):
                    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
            return datetime.min.replace(tzinfo=timezone.utc)

        def _id_key(x: Any) -> str:
            if isinstance(x, dict):
                v = x.get("message_id")
                return "" if v is None else str(v)
            return ""

        sorted_items = sorted(items, key=lambda x: (_ts_key(x), _id_key(x)), reverse=True)
        trimmed = sorted_items[:max_tool_items]
        obs["truncate_policy"] = "sort_by_ts_desc_message_id_desc_then_take_first"
        obs["sort"] = "ts_desc_message_id_desc"

    obs["items"] = trimmed
    obs["returned_items"] = len(trimmed)
    obs["truncated"] = True
    obs["truncate_reason"] = "too_many_items_returned"
    # Cursor ordering may no longer match after truncation; disable it to avoid misuse.
    obs["next_cursor"] = None
    why = "first_after_truncation" if tool in ("semantic_search", "lexical_search", "neighbors") else "latest_after_truncation"
    obs["anchors"] = [(str(it.get("message_id")), why) for it in trimmed[:3] if isinstance(it, dict) and it.get("message_id")]
    return obs


def _summarize_tool_args(tool: str, args: dict[str, Any]) -> dict[str, Any]:
    if tool == "messages_list":
        return {
            "since": args.get("since"),
            "until": args.get("until"),
            "role": args.get("role"),
            "page_size": args.get("page_size"),
            "cursor_present": bool(args.get("cursor")),
        }

    if tool in ("lexical_search", "semantic_search"):
        filt = args.get("filter") if isinstance(args.get("filter"), dict) else {}
        tr = filt.get("time_range") if isinstance(filt, dict) else {}
        base = {
            "query_text": _clip_text(str(args.get("query_text") or ""), 120),
            "time_range": {"since": (tr or {}).get("since"), "until": (tr or {}).get("until")},
            "role": (filt or {}).get("role"),
            "cursor_present": bool(args.get("cursor")),
        }
        if tool == "lexical_search":
            base["page_size"] = args.get("page_size")
        if tool == "semantic_search":
            base["top_k"] = args.get("top_k")
            base["min_score"] = args.get("min_score")
        return base

    if tool == "neighbors":
        return {
            "message_id": args.get("message_id"),
            "before": args.get("before"),
            "after": args.get("after"),
        }

    return {"args_keys": sorted([str(k) for k in (args or {}).keys()])}


def _format_tool_call_human(tool: str, args: dict[str, Any]) -> str:
    if tool == "messages_list":
        return (
            f"role={args.get('role')} since={args.get('since')!r} until={args.get('until')!r} "
            f"page_size={args.get('page_size')} cursor={'yes' if args.get('cursor') else 'no'}"
        )

    if tool in ("lexical_search", "semantic_search"):
        query_text = _clip_text(str(args.get("query_text") or ""), 160)
        filt = args.get("filter") if isinstance(args.get("filter"), dict) else {}
        tr = filt.get("time_range") if isinstance(filt, dict) else {}
        since = (tr or {}).get("since")
        until = (tr or {}).get("until")
        role = (filt or {}).get("role")
        if tool == "lexical_search":
            return (
                f"query={query_text!r} role={role} since={since!r} until={until!r} "
                f"page_size={args.get('page_size')} cursor={'yes' if args.get('cursor') else 'no'}"
            )
        return f"query={query_text!r} role={role} since={since!r} until={until!r} top_k={args.get('top_k')} min_score={args.get('min_score')}"

    if tool == "neighbors":
        return f"message_id={args.get('message_id')} before={args.get('before')} after={args.get('after')}"

    return f"args={_summarize_tool_args(tool, args)}"


def _log_tool_observation(*, iteration: int, tool: str, safe_obs: dict[str, Any]) -> None:
    items = safe_obs.get("items") or []
    anchors = safe_obs.get("anchors") or []
    next_cursor = safe_obs.get("next_cursor")
    applied = safe_obs.get("applied_limits") or {}
    total_items = safe_obs.get("total_items")
    returned_items = safe_obs.get("returned_items")
    truncated = safe_obs.get("truncated")
    max_tool_items = safe_obs.get("max_tool_items")
    err = safe_obs.get("error")

    role = None
    since = None
    until = None
    if isinstance(applied, dict):
        role = applied.get("role")
        tr = applied.get("time_range") or {}
        if isinstance(tr, dict):
            since = tr.get("since")
            until = tr.get("until")

    logger.info(
        "recall.obs iter=%s tool=%s returned_items=%s total_items=%s truncated=%s max_tool_items=%s anchors=%s next_cursor=%s role=%s since=%s until=%s",
        iteration,
        tool,
        returned_items if returned_items is not None else (len(items) if isinstance(items, list) else 0),
        total_items if total_items is not None else (len(items) if isinstance(items, list) else 0),
        bool(truncated) if truncated is not None else False,
        max_tool_items,
        len(anchors) if isinstance(anchors, list) else 0,
        bool(next_cursor),
        role,
        since,
        until,
    )
    if isinstance(err, dict):
        et = err.get("type")
        details = err.get("details")
        first = None
        if isinstance(details, list) and details:
            first = details[0]
        logger.info("recall.obs_error iter=%s tool=%s type=%s first=%s", iteration, tool, et, _clip_text(str(first), 500))

    if isinstance(anchors, list):
        for idx, a in enumerate(anchors[:3], start=1):
            if isinstance(a, (list, tuple)) and len(a) >= 2:
                logger.info("recall.obs_anchor iter=%s idx=%s message_id=%s why=%s", iteration, idx, a[0], a[1])

    if isinstance(items, list):
        for idx, it in enumerate(items[:3], start=1):
            if not isinstance(it, dict):
                continue
            logger.info(
                "recall.obs_item iter=%s idx=%s ts=%s role=%s id=%s content=%s",
                iteration,
                idx,
                it.get("ts"),
                it.get("role"),
                it.get("message_id"),
                _clip_text(str(it.get("content") or ""), 200),
            )


def _bigmodel_tool_schemas(allowed_tools: list[str]) -> list[dict[str, Any]]:
    return bigmodel_tool_schemas(allowed_tools=allowed_tools)


@dataclass(frozen=True)
class RecallAgentDefaults:
    neighbors_before: int = 8
    neighbors_after: int = 0
    messages_page_size: int = 100
    lexical_page_size: int = 50
    semantic_top_k: int = 20
    semantic_min_score: float = 0.0
    fallback_query_text: str = "偏好 背景 约束"
    state_top_items: int = 3
    allowed_tools: list[str] = None  # type: ignore[assignment]
    allowed_role_values: list[str] = None  # type: ignore[assignment]
    embedding_provider: str = "bigmodel"
    embedding_model: str = "embedding-3"


def _msg_to_dict(m) -> dict[str, Any]:
    return {
        "message_id": m.message_id,
        "ts": m.ts,
        "user_id": m.user_id,
        "role": m.role,
        "content": m.content,
        "meta": m.meta,
    }


class RecallAgent:
    """
    Retrieval agent that loads exactly one retrieval skill from ./skills and follows it.

    - The user only provides `query` (question).
    - The agent loads exactly one Skill and uses the LLM to plan multi-step tool calls (Observe→Reflect→Decide).
    - Search strategy (filters, budgets, tool choice) must live in the Skill.
    """

    def __init__(self, db: Session, *, user_id: str, skill_name: Optional[str] = None) -> None:
        self.db = db
        self.user_id = user_id
        self._semantic_preflight_logged = False
        # Optional override to pin a specific skill; otherwise the LLM will select one
        # from the displayed skill list for this task.
        self.pinned_skill_id = skill_name
        self.skill = None
        self.defaults = RecallAgentDefaults(
            allowed_tools=["messages_list", "lexical_search", "semantic_search", "neighbors"],
            allowed_role_values=["user", "assistant", "any"],
        )

    def run(
        self,
        *,
        question: str,
    ) -> RecallResponse:
        # Skill selection is configuration-driven (no LLM selection).
        skill_id = self.pinned_skill_id or settings.recall_skill
        model = get_chat_model()
        logger.info(
            "recall.start user_id=%s skill_id=%s question_len=%s model=%s max_iterations=%s",
            self.user_id,
            skill_id,
            len(question or ""),
            model,
            settings.recall_max_iterations,
        )
        client = get_chat_client()
        try:
            t1 = time.monotonic()
            self.skill = load_skill(skill_id)
            logger.debug("recall.skill_load done skill_id=%s elapsed_ms=%s", skill_id, int((time.monotonic() - t1) * 1000))
        except SkillNotFoundError as e:
            raise e

        tool_rules = (
            "你是一个记忆检索工具编排 Agent。你已经加载了当前任务的 skill。\n"
            "接下来你要做的是：反复选择合适的工具进行检索，直到你认为证据已足够。\n"
            "工具调用必须使用 tool_calls（函数调用）方式；工具结果会以 role=tool 的消息追加到对话里。\n"
            "运行时会在每次工具调用后做硬约束：若返回 items 过多，会按时间倒序（最新在前）截断到 max_tool_items，并告知 total_items 与截断原因。\n"
            "硬性要求：messages_list/lexical_search/semantic_search 必须显式提供 time_range 与 role（role 可为 any）。若 since/until 均为空，则表示全时间范围。\n"
            "role 取值：user / assistant / any（any=不按 role 过滤）。\n"
            "硬性要求：拿不准、且允许不填的参数就不要填写（例如 until/cursor/min_score/page_size/top_k/before/after 等），让运行时默认值接管。\n"
            "建议：每次最多调用 1 个工具；尽量先收窄再扩展；找到锚点后优先 neighbors 补上下文。\n"
            "重要：请预留至少 1 次迭代用于最终合成输出（输出最终 JSON）；不要把最后一次迭代用在工具调用上。\n"
            "当你不再需要调用工具时：不要再发起 tool_calls，按 skill 规定的最终输出格式，直接输出最终 JSON（只输出 JSON，不要 Markdown/代码块）。\n"
        )

        # Rebuild a clean planning conversation:
        # skill + tool rules are provided once; then the user task; then iterative tool results.
        conversation = [
            BigModelMessage(role="system", content=self.skill.text),
            BigModelMessage(role="system", content=tool_rules),
            BigModelMessage(role="user", content=f"TASK_QUERY:\n{question}"),
        ]

        logger.debug(
            "recall.context initialized messages=%s skill_chars=%s",
            len(conversation),
            len(self.skill.text or ""),
        )

        memory_view, evidence_message_ids = self._run_llm_loop(
            client=client,
            conversation=conversation,
            final_output_model=SynthesisOutput,
            final_primary_field="memory_view",
        )
        evidence_items = self._fetch_messages_by_ids(evidence_message_ids)
        logger.info("recall.done evidence_items=%s memory_view_len=%s", len(evidence_items), len(memory_view or ""))

        return RecallResponse(
            memory_view=memory_view,
            evidence=evidence_items,
        )

    def run_locomo_qa(
        self,
        *,
        question: str,
        now: datetime | None = None,
    ) -> tuple[str, list[ChatMessage]]:
        skill_id = self.pinned_skill_id or "nomemory-locomo-qa"
        model = get_chat_model()
        logger.info(
            "locomo.start user_id=%s skill_id=%s question_len=%s model=%s max_iterations=%s",
            self.user_id,
            skill_id,
            len(question or ""),
            model,
            settings.recall_max_iterations,
        )
        client = get_chat_client()
        self.skill = load_skill(skill_id)

        now_dt = now or self._latest_message_ts() or datetime.now(tz=timezone.utc)
        tool_rules = (
            "你是一个记忆检索工具编排 Agent。你已经加载了当前任务的 skill。\n"
            "接下来你要做的是：反复选择合适的工具进行检索，直到你认为证据已足够。\n"
            "工具调用必须使用 tool_calls（函数调用）方式；工具结果会以 role=tool 的消息追加到对话里。\n"
            "运行时会在每次工具调用后做硬约束：若返回 items 过多，会按时间倒序（最新在前）截断到 max_tool_items，并告知 total_items 与截断原因。\n"
            "硬性要求：messages_list/lexical_search/semantic_search 必须显式提供 time_range 与 role（role 可为 any）。若 since/until 均为空，则表示全时间范围。\n"
            "注意：在 JSON 里空值必须用 null（或直接省略字段）；不要写字符串 \"None\" / \"null\" 作为占位。\n"
            "role 取值：user / assistant / any（any=不按 role 过滤）。\n"
            "硬性要求：拿不准、且允许不填的参数就不要填写（例如 until/cursor/min_score/page_size/top_k/before/after 等），让运行时默认值接管。\n"
            "重要：请预留至少 1 次迭代用于最终合成输出（输出最终 JSON）；不要把最后一次迭代用在工具调用上。\n"
            "当你不再需要调用工具时：不要再发起 tool_calls，按 skill 规定的最终输出格式，直接输出最终 JSON（只输出 JSON，不要 Markdown/代码块）。\n"
        )

        conversation = [
            BigModelMessage(role="system", content=self.skill.text),
            BigModelMessage(role="system", content=tool_rules),
            BigModelMessage(role="user", content=f"TASK_QUERY:\n{question}\nNOW:\n{now_dt.isoformat()}"),
        ]

        answer, evidence_message_ids = self._run_llm_loop(
            client=client,
            conversation=conversation,
            final_output_model=LocomoQAOutput,
            final_primary_field="answer",
        )
        evidence_items = self._fetch_messages_by_ids(evidence_message_ids)
        logger.info("locomo.done evidence_items=%s answer_len=%s", len(evidence_items), len(answer or ""))
        return (answer, evidence_items)

    def _to_chat_message(self, m: Message) -> ChatMessage:
        return ChatMessage(
            message_id=m.message_id,
            ts=m.ts,
            user_id=m.user_id,
            role=m.role,
            content=m.content,
            meta=m.meta,
        )

    def _fetch_messages_by_ids(self, message_ids: list[str]) -> list[ChatMessage]:
        ids = [str(x) for x in (message_ids or []) if str(x).strip()]
        if not ids:
            return []
        stmt = select(Message).where(Message.user_id == self.user_id, Message.message_id.in_(ids))
        rows = list(self.db.execute(stmt).scalars().all())
        by_id = {m.message_id: m for m in rows}
        out: list[ChatMessage] = []
        for mid in ids:
            m = by_id.get(mid)
            if m is None:
                continue
            out.append(self._to_chat_message(m))
        return out

    def _latest_message_ts(self) -> datetime | None:
        stmt = (
            select(Message.ts)
            .where(Message.user_id == self.user_id)
            .order_by(Message.ts.desc(), Message.message_id.desc())
            .limit(1)
        )
        row = self.db.execute(stmt).first()
        if not row:
            return None
        ts = row[0]
        return ts if isinstance(ts, datetime) else None

    def _run_llm_loop(
        self,
        *,
        client: Any,
        conversation: list[BigModelMessage],
        final_output_model: type[BaseModel],
        final_primary_field: str,
    ) -> tuple[str, list[str]]:
        max_format_retries = 2
        format_retries = 0
        for i in range(settings.recall_max_iterations):
            iteration = i + 1
            remaining = settings.recall_max_iterations - iteration
            logger.info("recall.iter iter=%s/%s remaining=%s", iteration, settings.recall_max_iterations, remaining)
            logger.debug("recall.loop state conversation_msgs=%s", len(conversation))

            allow_tools = iteration < settings.recall_max_iterations
            # The model cannot otherwise "sense" the remaining iteration budget. Provide a lightweight
            # runtime hint without persisting it in `conversation` (to avoid context bloat).
            budget_hint = BigModelMessage(
                role="system",
                content=(
                    f"RUNTIME_BUDGET: iter={iteration}/{settings.recall_max_iterations} "
                    f"remaining={remaining} allow_tools={str(bool(allow_tools)).lower()}. "
                    "If allow_tools=false, you MUST NOT return tool_calls and MUST output the final JSON now."
                ),
            )
            prompt_messages = [*conversation, budget_hint]
            if allow_tools:
                msg = client.chat_message(
                    model=get_chat_model(),
                    messages=prompt_messages,
                    temperature=0.2,
                    tools=_bigmodel_tool_schemas(self.defaults.allowed_tools),
                    tool_choice="auto",
                )
            else:
                msg = client.chat_message(
                    model=get_chat_model(),
                    messages=prompt_messages,
                    temperature=0.2,
                )
            content = msg.get("content")
            if isinstance(content, str) and content:
                logger.info("recall.think iter=%s content=%s", iteration, _clip_text(content, 500))
            reasoning = msg.get("reasoning_content")
            if isinstance(reasoning, str) and reasoning:
                logger.info("recall.think iter=%s reasoning=%s", iteration, _clip_text(reasoning, 1500))
            tool_calls = msg.get("tool_calls") or []
            if not isinstance(tool_calls, list):
                tool_calls = []

            # Persist the assistant tool_calls message in the conversation so tool_call_id references remain valid.
            conversation.append(
                BigModelMessage(
                    role="assistant",
                    content=msg.get("content"),
                    tool_calls=tool_calls or None,
                )
            )

            if not tool_calls:
                try:
                    out = final_output_model.model_validate(_extract_first_json_object(str(content or "")))
                    primary = getattr(out, final_primary_field, "")
                    evidence = getattr(out, "evidence_message_ids", [])
                    primary_text = str(primary or "").strip()
                    evidence_message_ids = [str(x) for x in (evidence or []) if str(x).strip()]
                    logger.info("recall.stop iter=%s reason=synthesis_ok", iteration)
                    return (primary_text, evidence_message_ids)
                except Exception as e:
                    format_retries += 1
                    logger.info(
                        "recall.stop iter=%s reason=synthesis_invalid err=%s",
                        iteration,
                        _clip_text(f"{type(e).__name__}: {e}", 300),
                    )
                    if format_retries > max_format_retries:
                        raise RecallAgentError(
                            "Recall synthesis output was invalid JSON after retries",
                            code="synthesis_invalid",
                        ) from e

                    # Ask the model to re-format in-place (same conversation) without tool calls.
                    example = example_json_for_prompt(final_output_model)
                    fixup = BigModelMessage(
                        role="user",
                        content=(
                            "上一步你已经停止调用工具了。现在请直接给出最终结果。\n"
                            "要求：只输出 1 个 JSON 对象；不要 Markdown/代码块；不要多余解释文字。\n"
                            f"EXAMPLE:{example}\n"
                        ),
                    )

                    if remaining > 0:
                        conversation.append(fixup)
                        continue

                    # Last iteration: do a couple in-place retries (no tools) before failing.
                    last_err: Exception = e
                    for attempt in range(1, max_format_retries + 1):
                        logger.info("recall.synth_retry iter=%s attempt=%s/%s", iteration, attempt, max_format_retries)
                        conversation.append(fixup)
                        msg2 = client.chat_message(
                            model=get_chat_model(),
                            messages=conversation,
                            temperature=0.2,
                        )
                        content2 = msg2.get("content")
                        tool_calls2 = msg2.get("tool_calls") or []
                        if not isinstance(tool_calls2, list):
                            tool_calls2 = []
                        conversation.append(
                            BigModelMessage(
                                role="assistant",
                                content=content2,
                                tool_calls=tool_calls2 or None,
                            )
                        )
                        if tool_calls2:
                            last_err = RuntimeError("Synthesis retry unexpectedly returned tool_calls")
                            break
                        try:
                            out2 = final_output_model.model_validate(_extract_first_json_object(str(content2 or "")))
                            primary2 = getattr(out2, final_primary_field, "")
                            evidence2 = getattr(out2, "evidence_message_ids", [])
                            primary_text2 = str(primary2 or "").strip()
                            evidence_message_ids2 = [str(x) for x in (evidence2 or []) if str(x).strip()]
                            logger.info("recall.stop iter=%s reason=synthesis_ok_retry", iteration)
                            return (primary_text2, evidence_message_ids2)
                        except Exception as e2:
                            last_err = e2
                            continue

                    raise RecallAgentError(
                        "Recall synthesis output was invalid JSON after retries",
                        code="synthesis_invalid",
                    ) from last_err

            # Execute each tool call and feed results back as role=tool messages.
            for j, tc in enumerate(tool_calls):
                if not isinstance(tc, dict):
                    continue
                tc_id = str(tc.get("id") or f"call-{i+1}-{j+1}")
                fn = tc.get("function") or {}
                if not isinstance(fn, dict):
                    continue
                tool = str(fn.get("name") or "").strip()
                arg_text = fn.get("arguments")
                if not isinstance(arg_text, str):
                    arg_text = "{}"
                try:
                    args = json.loads(arg_text) if arg_text.strip() else {}
                except Exception:
                    args = {}
                if isinstance(args, dict):
                    args = _normalize_bigmodel_tool_args(args)

                if tool not in set(self.defaults.allowed_tools):
                    logger.warning("recall.tool_call tool_not_allowed tool=%s", tool)
                    raise ValueError(f"Tool not allowed: {tool}")

                logger.info("recall.act iter=%s tool=%s tool_call_id=%s %s", iteration, tool, tc_id, _format_tool_call_human(tool, args))
                t1 = time.monotonic()
                try:
                    validated_args = _validate_tool_args(tool=tool, args=args)
                    obs = self._execute_tool(tool=tool, args=validated_args)
                except ValidationError as e:
                    obs = {
                        "tool": tool,
                        "items": [],
                        "error": {"type": "validation_error", "details": e.errors(include_url=False)[:3]},
                    }
                obs = _apply_tool_item_budget(obs, tool=tool, max_tool_items=settings.recall_max_tool_items)
                logger.debug("recall.tool_result applied_limits=%s", obs.get("applied_limits"))
                safe_obs = _make_observation_safe(obs)
                logger.debug("recall.tool_exec_elapsed iter=%s tool=%s elapsed_ms=%s", iteration, tool, int((time.monotonic() - t1) * 1000))
                _log_tool_observation(iteration=iteration, tool=tool, safe_obs=safe_obs)
                conversation.append(
                    BigModelMessage(
                        role="tool",
                        tool_call_id=tc_id,
                        content=json.dumps(safe_obs, ensure_ascii=False, default=str),
                    )
                )
        else:
            # If the agent couldn't converge within the iteration budget, return a safe "unknown"/empty
            # output instead of failing the whole request. This is especially useful for benchmarks.
            logger.warning("recall.loop max_iterations_exceeded max_iterations=%s", settings.recall_max_iterations)
            fallback_primary = "unknown" if final_primary_field == "answer" else ""
            return (fallback_primary, [])

    def _execute_tool(
        self,
        *,
        tool: str,
        args: Any,
    ) -> dict[str, Any]:
        # Note: args are already validated via Pydantic (see _validate_tool_args in _run_llm_loop).
        logger.debug("recall.tool_exec tool=%s", tool)
        # Enforce user binding. The model must explicitly provide filters (no default fallbacks).
        if tool == "messages_list":
            a: MessagesListArgs = args
            since_dt = a.since
            until_dt = a.until
            role_val = str(a.role)
            page_size = int(a.page_size or self.defaults.messages_page_size)
            cursor = a.cursor
            try:
                anchor = parse_seek_anchor(cursor, expected_kind="messages_list") if cursor else None
            except CursorError:
                anchor = None
            role_dt = None if role_val == "any" else role_val
            if role_val and role_val not in set(self.defaults.allowed_role_values or []):
                raise ValueError(f"messages_list role not allowed: {role_val}")

            fetch_size = page_size + 1 if page_size > 0 else 1
            items = list_messages(
                self.db,
                user_id=self.user_id,
                since=since_dt,
                until=until_dt,
                role=role_dt,
                page_size=fetch_size,
                anchor=anchor,
            )
            out_items = [_msg_to_dict(m) for m in items]
            if out_items:
                logger.debug(
                    "recall.messages_list result_count=%s first_ts=%s last_ts=%s",
                    len(out_items),
                    out_items[0].get("ts"),
                    out_items[-1].get("ts"),
                )
            next_cursor = None
            if len(items) > page_size and page_size > 0:
                last = items[page_size - 1]
                next_cursor = make_seek_cursor(kind="messages_list", ts=last.ts, message_id=last.message_id)
                items = items[:page_size]
                out_items = out_items[:page_size]
            applied_limits = {
                "time_range": {"since": since_dt.isoformat() if since_dt else None, "until": until_dt.isoformat() if until_dt else None},
                "role": role_val or "any",
            }
            return {"tool": tool, "items": out_items, "next_cursor": next_cursor, "applied_limits": applied_limits}

        if tool == "lexical_search":
            a: LexicalSearchArgs = args
            query_text = str(a.query_text or "").strip()
            since_dt = a.filter.time_range.since
            until_dt = a.filter.time_range.until
            role_val = str(a.filter.role)
            role_dt = None if role_val == "any" else role_val
            if role_val and role_val not in set(self.defaults.allowed_role_values or []):
                raise ValueError(f"lexical_search role not allowed: {role_val}")
            page_size = int(a.page_size or self.defaults.lexical_page_size)
            cursor = a.cursor
            anchor = None
            if cursor:
                try:
                    payload = decode_cursor(str(cursor))
                    if payload.get("kind") == "lexical_search":
                        anchor = LexicalAnchor(
                            rank=float(payload["rank"]),
                            ts=parse_datetime(payload["ts"]) or since_dt,
                            message_id=str(payload["message_id"]),
                        )
                except Exception:
                    anchor = None
            rows = lexical_search(
                self.db,
                user_id=self.user_id,
                query_text=query_text or self.defaults.fallback_query_text,
                since=since_dt,
                until=until_dt,
                role=role_dt,
                page_size=(page_size + 1 if page_size > 0 else 1),
                anchor=anchor,
            )
            logger.debug("recall.lexical_search rows=%s", len(rows))
            next_cursor = None
            if len(rows) > page_size and page_size > 0:
                m, rank = rows[page_size - 1]
                next_cursor = make_seek_cursor(kind="lexical_search", ts=m.ts, message_id=m.message_id, extra={"rank": round(float(rank), 8)})
                rows = rows[:page_size]

            items = [_msg_to_dict(m) for (m, _rank) in rows]
            anchors = [(m.message_id, "lexical_top") for (m, _rank) in rows[:3]]
            applied_limits = {
                "time_range": {"since": since_dt.isoformat() if since_dt else None, "until": until_dt.isoformat() if until_dt else None},
                "role": role_val or "any",
            }
            return {"tool": tool, "items": items, "anchors": anchors, "next_cursor": next_cursor, "applied_limits": applied_limits}

        if tool == "semantic_search":
            a: SemanticSearchArgs = args
            query_text = str(a.query_text or "").strip()
            since_dt = a.filter.time_range.since
            until_dt = a.filter.time_range.until
            role_val = str(a.filter.role)
            role_dt = None if role_val == "any" else role_val
            if role_val and role_val not in set(self.defaults.allowed_role_values or []):
                raise ValueError(f"semantic_search role not allowed: {role_val}")
            top_k = int(a.top_k or self.defaults.semantic_top_k)
            min_score = a.min_score
            if min_score is None:
                min_score = self.defaults.semantic_min_score
            provider = self.defaults.embedding_provider
            model = self.defaults.embedding_model
            try:
                qemb = embed_query_text(query_text or self.defaults.fallback_query_text, provider=provider, model=model)
            except Exception as e:
                logger.warning("recall.semantic embed_failed (%s): %s", type(e).__name__, str(e))
                applied_limits = {
                    "time_range": {"since": since_dt.isoformat() if since_dt else None, "until": until_dt.isoformat() if until_dt else None},
                    "role": role_val or "any",
                }
                return {"tool": tool, "items": [], "applied_limits": applied_limits}

            # Preflight: verify embeddings exist for this user/provider/model before querying.
            # This makes "returned_items=0" immediately explainable (embeddings not ready vs. score/filter).
            try:
                msg_where = [Message.user_id == self.user_id]
                if role_dt:
                    msg_where.append(Message.role == role_dt)
                if since_dt:
                    msg_where.append(Message.ts >= since_dt)
                if until_dt:
                    msg_where.append(Message.ts < until_dt)
                msg_cnt = self.db.execute(select(func.count()).select_from(Message).where(*msg_where)).scalar_one()

                emb_where = [
                    MessageEmbedding.user_id == self.user_id,
                    MessageEmbedding.provider == provider,
                    MessageEmbedding.model == model,
                ]
                emb_cnt = self.db.execute(select(func.count()).select_from(MessageEmbedding).where(*emb_where)).scalar_one()

                cand_where = list(emb_where)
                if role_dt:
                    cand_where.append(Message.role == role_dt)
                if since_dt:
                    cand_where.append(Message.ts >= since_dt)
                if until_dt:
                    cand_where.append(Message.ts < until_dt)
                cand_cnt = self.db.execute(
                    select(func.count())
                    .select_from(MessageEmbedding)
                    .join(
                        Message,
                        and_(
                            MessageEmbedding.user_id == Message.user_id,
                            MessageEmbedding.message_id == Message.message_id,
                        ),
                    )
                    .where(*cand_where)
                ).scalar_one()

                if (not self._semantic_preflight_logged) or int(cand_cnt or 0) == 0:
                    logger.info(
                        "recall.semantic_preflight user_id=%s provider=%s model=%s messages=%s embeddings=%s candidates=%s role=%s since=%s until=%s",
                        self.user_id,
                        provider,
                        model,
                        int(msg_cnt or 0),
                        int(emb_cnt or 0),
                        int(cand_cnt or 0),
                        role_val or "any",
                        since_dt.isoformat() if since_dt else None,
                        until_dt.isoformat() if until_dt else None,
                    )
                    self._semantic_preflight_logged = True
            except Exception:
                pass

            rows = semantic_search(
                self.db,
                user_id=self.user_id,
                query_embedding=qemb,
                since=since_dt,
                until=until_dt,
                role=role_dt,
                top_k=top_k,
                min_score=float(min_score) if min_score is not None else None,
                provider=provider,
                model=model,
            )
            logger.debug("recall.semantic_search rows=%s provider=%s model=%s", len(rows), provider, model)
            items = [_msg_to_dict(m) for (m, _score) in rows]
            anchors = [(m.message_id, "semantic_top") for (m, _score) in rows[:3]]
            applied_limits = {
                "time_range": {"since": since_dt.isoformat() if since_dt else None, "until": until_dt.isoformat() if until_dt else None},
                "role": role_val or "any",
            }
            return {"tool": tool, "items": items, "anchors": anchors, "applied_limits": applied_limits}

        if tool == "neighbors":
            a: NeighborsArgs = args
            message_id = str(a.message_id or "")
            before = int(a.before or self.defaults.neighbors_before)
            after = int(a.after or self.defaults.neighbors_after)
            items = get_neighbors(
                self.db,
                user_id=self.user_id,
                message_id=message_id,
                before=before,
                after=after,
            )
            logger.debug("recall.neighbors items=%s message_id=%s", len(items), message_id)
            applied_limits = {
                "time_range": {"since": None, "until": None},
                "role": "any",
            }
            return {"tool": tool, "items": [_msg_to_dict(m) for m in items], "anchors": [(message_id, "neighbors")], "applied_limits": applied_limits}

        return {"tool": tool, "items": []}

    # Synthesis is handled inside _run_llm_loop (same retrieval context).
