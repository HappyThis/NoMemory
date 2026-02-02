from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from sqlalchemy.orm import Session

from app.agent.contracts import (
    PlannerOutput,
    SkillSelectionOutput,
    SynthesisOutput,
    example_json_for_prompt,
    schema_json_for_prompt,
)
from app.api.schemas import MemoryView, RecallLimits, RecallResponse, TimeRange
from app.llm.bigmodel import BigModelMessage
from app.llm.embeddings import embed_query_text
from app.llm.factory import get_chat_client
from app.retrieval.lexical import LexicalAnchor, lexical_search
from app.retrieval.messages import list_messages
from app.retrieval.neighbors import get_neighbors
from app.retrieval.semantic import semantic_search
from app.settings import settings
from app.skills.loader import SkillNotFoundError, load_skill, list_skill_metadata
from app.utils.cursor import CursorError, decode_cursor, make_seek_cursor, parse_seek_anchor
from app.utils.datetime import parse_datetime


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RecallAgentDefaults:
    neighbors_before: int = 8
    neighbors_after: int = 0
    messages_page_size: int = 100

    max_page_size: int = 200
    max_neighbors: int = 200
    max_top_k: int = 200

    max_iterations: int = 3
    max_evidence: int = 6
    lexical_page_size: int = 50
    semantic_top_k: int = 20
    semantic_min_score: float = 0.0
    fallback_query_text: str = "偏好 背景 约束"
    state_top_items: int = 3
    allowed_tools: list[str] = None  # type: ignore[assignment]
    allowed_role_values: list[str] = None  # type: ignore[assignment]
    embedding_provider: str = "bigmodel"
    embedding_model: str = "embedding-3"


def _load_recall_agent_config() -> dict[str, Any]:
    path = Path(settings.recall_agent_config_path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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
        # Optional override to pin a specific skill; otherwise the LLM will select one
        # from the displayed skill list for this task.
        self.pinned_skill_id = skill_name
        self.skill = None
        self.defaults = self._defaults_from_config(_load_recall_agent_config())

    def _defaults_from_config(self, config: dict[str, Any]) -> RecallAgentDefaults:
        d = (config or {}).get("defaults") or {}
        try:
            allowed = d.get("allowed_tools") or ["messages_list", "lexical_search", "semantic_search", "neighbors"]
            allowed_roles = d.get("allowed_role_values") or ["user", "any"]
            return RecallAgentDefaults(
                messages_page_size=int(d.get("messages_page_size", 100)),
                neighbors_before=int(d.get("neighbors_before", 8)),
                neighbors_after=int(d.get("neighbors_after", 0)),
                max_page_size=int(d.get("max_page_size", 200)),
                max_neighbors=int(d.get("max_neighbors", 200)),
                max_top_k=int(d.get("max_top_k", 200)),
                max_iterations=int(d.get("max_iterations", 3)),
                max_evidence=int(d.get("max_evidence", 6)),
                lexical_page_size=int(d.get("lexical_page_size", 50)),
                semantic_top_k=int(d.get("semantic_top_k", 20)),
                semantic_min_score=float(d.get("semantic_min_score", 0.0)),
                fallback_query_text=str(d.get("fallback_query_text", "偏好 背景 约束")),
                state_top_items=int(d.get("state_top_items", 3)),
                allowed_tools=[str(x) for x in allowed],
                allowed_role_values=[str(x) for x in allowed_roles],
                embedding_provider=str(d.get("embedding_provider", "bigmodel")),
                embedding_model=str(d.get("embedding_model", "embedding-3")),
            )
        except Exception:
            return RecallAgentDefaults(
                allowed_tools=["messages_list", "lexical_search", "semantic_search", "neighbors"],
                allowed_role_values=["user", "any"],
            )

    def run(
        self,
        *,
        question: str,
    ) -> RecallResponse:
        client = get_chat_client()
        conversation, skill_id = self._select_skill_conversation(client=client, question=question)
        try:
            self.skill = load_skill(skill_id)
        except SkillNotFoundError as e:
            raise e

        # Progressive loading: only after selecting a skill do we load its SKILL.md into context.
        conversation.append(BigModelMessage(role="system", content=self.skill.text))

        evidence_msgs, final_limits = self._run_llm_loop(client=client, conversation=conversation)
        memory_view = self._synthesize(client=client, conversation=conversation)

        return RecallResponse(
            memory_view=memory_view,
            evidence=evidence_msgs,
            limits=RecallLimits(
                time_range=TimeRange(
                    since=parse_datetime((final_limits.get("time_range") or {}).get("since")),
                    until=parse_datetime((final_limits.get("time_range") or {}).get("until")),
                ),
                role=str(final_limits.get("role") or "any"),
            ),
        )

    # --- Skill-driven (LLM) loop ---

    def _select_skill_conversation(self, *, client: Any, question: str) -> tuple[list[BigModelMessage], str]:
        skills = list_skill_metadata()
        skill_rows = [{"id": s.skill_id, "name": s.name, "description": s.description} for s in skills]
        skill_ids = {s.skill_id for s in skills}

        intro = (
            "你是一个记忆检索与合成 Agent。\n"
            "你将看到可用的 skill 列表（仅包含名称/描述），以及用户的任务（query）。\n"
            "你必须先选择一个最合适的 skill，然后再按该 skill 的规则完成检索与 memory_view 合成。\n"
            "注意：工具调用与合成输出都必须严格输出 JSON（不输出额外文本）。\n"
            "可用工具（全局）：messages_list / lexical_search / semantic_search / neighbors。\n"
            f"SKILLS:{json.dumps(skill_rows, ensure_ascii=False)}"
        )

        conversation: list[BigModelMessage] = [
            BigModelMessage(role="system", content=intro),
            BigModelMessage(role="user", content=f"TASK_QUERY:\n{question}"),
        ]

        if self.pinned_skill_id:
            if self.pinned_skill_id not in skill_ids:
                raise SkillNotFoundError(f"Skill not found: {self.pinned_skill_id}")
            selection = SkillSelectionOutput(skill_id=self.pinned_skill_id)
            conversation.append(BigModelMessage(role="assistant", content=selection.model_dump_json(ensure_ascii=False)))
            logger.info("recall.skill pinned skill_id=%s", selection.skill_id)
            return (conversation, selection.skill_id)

        prompt = (
            "请选择一个 skill 来完成该任务。\n"
            "只输出一个 JSON 对象，必须匹配以下 schema。\n"
            f"SCHEMA:{schema_json_for_prompt(SkillSelectionOutput)}\n"
            f"EXAMPLE:{example_json_for_prompt(SkillSelectionOutput)}"
        )

        last_err: Optional[Exception] = None
        selected: SkillSelectionOutput | None = None
        for attempt in range(1, 4):
            try:
                selected = SkillSelectionOutput.model_validate(
                    client.chat_json(
                        model=settings.llm_model,
                        messages=conversation + [BigModelMessage(role="user", content=prompt)],
                        temperature=0.2,
                    )
                )
                if selected.skill_id not in skill_ids:
                    raise SkillNotFoundError(f"Skill not found: {selected.skill_id}")
                conversation.append(BigModelMessage(role="assistant", content=selected.model_dump_json(ensure_ascii=False)))
                logger.info("recall.skill selected skill_id=%s", selected.skill_id)
                return (conversation, selected.skill_id)
            except Exception as e:
                last_err = e
                logger.warning("recall.skill_select attempt=%s/3 failed (%s)", attempt, type(e).__name__)
                continue
        logger.exception("recall.skill_select failed after 3 attempts")
        raise last_err or RuntimeError("Failed to select a skill")

    def _run_llm_loop(
        self,
        *,
        client: Any,
        conversation: list[BigModelMessage],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        max_structured_retries = 3

        evidence: list[dict[str, Any]] = []
        anchors: list[tuple[str, str]] = []  # (message_id, why)

        # The user only provides a natural-language `question`; do NOT pre-impose time_range/role.
        # The skill-driven planner must infer and apply filters via tool args when needed.
        current_since: Optional[datetime] = None
        current_until: Optional[datetime] = None
        current_role: Optional[str] = None

        for i in range(self.defaults.max_iterations):
            logger.info(
                "recall.loop iteration=%s/%s evidence=%s",
                i + 1,
                self.defaults.max_iterations,
                len(evidence),
            )

            plan_prompt = (
                "请规划下一步检索工具调用（或 stop=true 结束检索）。严格遵守当前已加载的 skill。\n"
                "你将看到历史对话中追加的 OBSERVATION_JSON（工具调用结果），请据此决定下一步。\n"
                f"迭代预算：当前 {i + 1}/{self.defaults.max_iterations}（最多 {self.defaults.max_iterations}）。\n"
                f"允许工具：{json.dumps(self.defaults.allowed_tools, ensure_ascii=False)}\n"
                "硬性要求：messages_list/lexical_search/semantic_search 必须显式提供 time_range 与 role（role 可为 any）。\n"
                "只输出一个 JSON 对象，必须匹配以下 schema。\n"
                f"SCHEMA:{schema_json_for_prompt(PlannerOutput)}\n"
                f"EXAMPLE:{example_json_for_prompt(PlannerOutput)}\n"
            )
            last_err: Optional[Exception] = None
            plan_obj: PlannerOutput | None = None
            for attempt in range(1, max_structured_retries + 1):
                try:
                    plan_obj = PlannerOutput.model_validate(
                        client.chat_json(
                            model=settings.llm_model,
                            messages=conversation + [BigModelMessage(role="user", content=plan_prompt)],
                            temperature=0.2,
                        )
                    )
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    plan_obj = None
                    logger.warning(
                        "recall.planner attempt=%s/%s failed (%s)",
                        attempt,
                        max_structured_retries,
                        type(e).__name__,
                    )
                    continue
            if last_err is not None or plan_obj is None:
                logger.exception("recall.planner failed after %s attempts", max_structured_retries)
                raise last_err or RuntimeError("Failed to obtain valid planner output")
            conversation.append(BigModelMessage(role="assistant", content=plan_obj.model_dump_json(ensure_ascii=False)))
            if plan_obj.stop is True:
                logger.info("recall.planner stop=true")
                break

            tool = str(plan_obj.tool or "")
            args = plan_obj.args or {}
            if tool not in set(self.defaults.allowed_tools):
                logger.warning("recall.planner tool_not_allowed tool=%s", tool)
                raise ValueError(f"Tool not allowed: {tool}")
            logger.debug("recall.planner tool=%s args_keys=%s", tool, sorted([str(k) for k in args.keys()]))

            obs = self._execute_tool(
                tool=tool,
                args=args,
            )
            conversation.append(BigModelMessage(role="user", content=self._format_observation(obs)))

            # Update anchors/evidence.
            for item in obs.get("items") or []:
                if item.get("role") == "system":
                    continue
                key = (item["user_id"], item["message_id"])
                if any((e["user_id"], e["message_id"]) == key for e in evidence):
                    continue
                evidence.append(item)

            if obs.get("anchors"):
                anchors.extend(obs["anchors"])

            applied = obs.get("applied_limits") or {}
            if isinstance(applied, dict):
                tr = applied.get("time_range") or {}
                if isinstance(tr, dict):
                    current_since = parse_datetime(tr.get("since")) or current_since
                    current_until = parse_datetime(tr.get("until")) or current_until
                role_val = applied.get("role")
                current_role = None if role_val in (None, "any") else ("user" if role_val == "user" else current_role)

            if len(evidence) >= self.defaults.max_evidence:
                logger.info("recall.loop evidence_cap_reached cap=%s", self.defaults.max_evidence)
                break

        final_limits = {
            "time_range": {
                "since": current_since.isoformat() if current_since else None,
                "until": current_until.isoformat() if current_until else None,
            },
            "role": current_role or "any",
        }
        return (evidence[: self.defaults.max_evidence], final_limits)

    def _format_observation(self, obs: dict[str, Any]) -> str:
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
        return "OBSERVATION_JSON:\n" + json.dumps(safe, ensure_ascii=False)

    def _execute_tool(
        self,
        *,
        tool: str,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        # Enforce user binding. The model must explicitly provide filters (no default fallbacks).
        if tool == "messages_list":
            if "since" not in args and "until" not in args:
                raise ValueError("messages_list requires args.since and/or args.until")
            if "role" not in args:
                raise ValueError("messages_list requires args.role (use 'any' to disable role filtering)")

            since = args.get("since")
            until = args.get("until")
            role_arg = args.get("role")
            page_size = int(args.get("page_size") or self.defaults.messages_page_size)
            cursor = args.get("cursor")
            try:
                anchor = parse_seek_anchor(cursor, expected_kind="messages_list") if cursor else None
            except CursorError:
                anchor = None

            since_dt = parse_datetime(since)
            until_dt = parse_datetime(until)
            role_val = str(role_arg) if role_arg is not None else ""
            role_dt = None if role_val == "any" else role_val
            if role_val and role_val not in set(self.defaults.allowed_role_values or []):
                raise ValueError(f"messages_list role not allowed: {role_val}")

            items = list_messages(
                self.db,
                user_id=self.user_id,
                since=since_dt,
                until=until_dt,
                role=role_dt,
                page_size=min(page_size, self.defaults.max_page_size),
                anchor=anchor,
            )
            out_items = [_msg_to_dict(m) for m in items]
            next_cursor = None
            if items:
                last = items[-1]
                next_cursor = make_seek_cursor(kind="messages_list", ts=last.ts, message_id=last.message_id)
            applied_limits = {
                "time_range": {"since": since_dt.isoformat() if since_dt else None, "until": until_dt.isoformat() if until_dt else None},
                "role": role_val or "any",
            }
            return {"tool": tool, "items": out_items, "next_cursor": next_cursor, "applied_limits": applied_limits}

        if tool == "lexical_search":
            if "query_text" not in args:
                raise ValueError("lexical_search requires args.query_text")
            query_text = str(args.get("query_text") or "").strip()
            filt = args.get("filter")
            if not isinstance(filt, dict):
                raise ValueError("lexical_search requires args.filter (object)")
            tr = (filt.get("time_range") or {}) if isinstance(filt, dict) else {}
            if not isinstance(tr, dict) or ("since" not in tr and "until" not in tr):
                raise ValueError("lexical_search requires filter.time_range.since and/or filter.time_range.until")
            if "role" not in filt:
                raise ValueError("lexical_search requires filter.role (use 'any' to disable role filtering)")

            since_dt = parse_datetime(tr.get("since"))
            until_dt = parse_datetime(tr.get("until"))
            role_arg = filt.get("role")
            role_val = str(role_arg) if role_arg is not None else ""
            role_dt = None if role_val == "any" else role_val
            if role_val and role_val not in set(self.defaults.allowed_role_values or []):
                raise ValueError(f"lexical_search role not allowed: {role_val}")
            page_size = int(args.get("page_size") or self.defaults.lexical_page_size)
            cursor = args.get("cursor")
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
                page_size=min(page_size, self.defaults.max_page_size),
                anchor=anchor,
            )
            items = [_msg_to_dict(m) for (m, _rank) in rows]
            anchors = [(m.message_id, "lexical_top") for (m, _rank) in rows[:3]]
            next_cursor = None
            if rows:
                m, rank = rows[-1]
                next_cursor = make_seek_cursor(
                    kind="lexical_search",
                    ts=m.ts,
                    message_id=m.message_id,
                    extra={"rank": round(float(rank), 8)},
                )
            applied_limits = {
                "time_range": {"since": since_dt.isoformat() if since_dt else None, "until": until_dt.isoformat() if until_dt else None},
                "role": role_val or "any",
            }
            return {"tool": tool, "items": items, "anchors": anchors, "next_cursor": next_cursor, "applied_limits": applied_limits}

        if tool == "semantic_search":
            query_text = str(args.get("query_text") or "").strip()
            filt = args.get("filter")
            if not isinstance(filt, dict):
                raise ValueError("semantic_search requires args.filter (object)")
            tr = (filt.get("time_range") or {}) if isinstance(filt, dict) else {}
            if not isinstance(tr, dict) or ("since" not in tr and "until" not in tr):
                raise ValueError("semantic_search requires filter.time_range.since and/or filter.time_range.until")
            if "role" not in filt:
                raise ValueError("semantic_search requires filter.role (use 'any' to disable role filtering)")

            since_dt = parse_datetime(tr.get("since"))
            until_dt = parse_datetime(tr.get("until"))
            role_arg = filt.get("role")
            role_val = str(role_arg) if role_arg is not None else ""
            role_dt = None if role_val == "any" else role_val
            if role_val and role_val not in set(self.defaults.allowed_role_values or []):
                raise ValueError(f"semantic_search role not allowed: {role_val}")
            top_k = int(args.get("top_k") or self.defaults.semantic_top_k)
            min_score = args.get("min_score")
            if min_score is None:
                min_score = self.defaults.semantic_min_score
            provider = self.defaults.embedding_provider
            model = self.defaults.embedding_model
            try:
                qemb = embed_query_text(query_text or self.defaults.fallback_query_text, provider=provider, model=model)
            except Exception:
                applied_limits = {
                    "time_range": {"since": since_dt.isoformat() if since_dt else None, "until": until_dt.isoformat() if until_dt else None},
                    "role": role_val or "any",
                }
                return {"tool": tool, "items": [], "applied_limits": applied_limits}

            rows = semantic_search(
                self.db,
                user_id=self.user_id,
                query_embedding=qemb,
                since=since_dt,
                until=until_dt,
                role=role_dt,
                top_k=min(top_k, self.defaults.max_top_k),
                min_score=float(min_score) if min_score is not None else None,
                provider=provider,
                model=model,
            )
            items = [_msg_to_dict(m) for (m, _score) in rows]
            anchors = [(m.message_id, "semantic_top") for (m, _score) in rows[:3]]
            applied_limits = {
                "time_range": {"since": since_dt.isoformat() if since_dt else None, "until": until_dt.isoformat() if until_dt else None},
                "role": role_val or "any",
            }
            return {"tool": tool, "items": items, "anchors": anchors, "applied_limits": applied_limits}

        if tool == "neighbors":
            message_id = str(args.get("message_id") or "")
            before = int(args.get("before") or self.defaults.neighbors_before)
            after = int(args.get("after") or self.defaults.neighbors_after)
            items = get_neighbors(
                self.db,
                user_id=self.user_id,
                message_id=message_id,
                before=min(before, self.defaults.max_neighbors),
                after=min(after, self.defaults.max_neighbors),
            )
            applied_limits = {
                "time_range": {"since": None, "until": None},
                "role": "any",
            }
            return {"tool": tool, "items": [_msg_to_dict(m) for m in items], "anchors": [(message_id, "neighbors")], "applied_limits": applied_limits}

        return {"tool": tool, "items": []}

    # --- Synthesis ---

    def _synthesize(self, *, client: Any, conversation: list[BigModelMessage]) -> MemoryView:
        max_structured_retries = 3

        synth_prompt = (
            "现在请基于历史对话中的 OBSERVATION_JSON（工具调用结果）合成 memory_view。\n"
            "只允许输出被证据支持的条目；证据不足则对应数组保持为空。\n"
            "只输出一个 JSON 对象，必须匹配以下 schema。\n"
            f"SCHEMA:{schema_json_for_prompt(SynthesisOutput)}\n"
            f"EXAMPLE:{example_json_for_prompt(SynthesisOutput)}\n"
        )
        last_err: Optional[Exception] = None
        out: SynthesisOutput | None = None
        for attempt in range(1, max_structured_retries + 1):
            try:
                out = SynthesisOutput.model_validate(
                    client.chat_json(
                        model=settings.llm_model,
                        messages=conversation + [BigModelMessage(role="user", content=synth_prompt)],
                        temperature=0.2,
                    )
                )
                last_err = None
                break
            except Exception as e:
                last_err = e
                out = None
                logger.warning(
                    "recall.synthesis attempt=%s/%s failed (%s)",
                    attempt,
                    max_structured_retries,
                    type(e).__name__,
                )
                continue
        if last_err is not None or out is None:
            logger.exception("recall.synthesis failed after %s attempts", max_structured_retries)
            raise last_err or RuntimeError("Failed to obtain valid synthesis output")

        return MemoryView(
            preferences=[str(x) for x in out.preferences],
            profile=[str(x) for x in out.profile],
            constraints=[str(x) for x in out.constraints],
        )
