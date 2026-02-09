from __future__ import annotations

import json
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PlannerOutput(BaseModel):
    """
    Contract for the agent's "next-step" tool plan.

    - When stop=true, tool/args can be omitted.
    - When stop=false, tool must be present.
    """

    model_config = ConfigDict(extra="forbid")

    stop: bool = Field(
        default=False,
        description="是否停止检索循环；true 表示本轮不再调用工具，直接结束 loop。",
        examples=[False],
    )
    tool: Optional[str] = Field(
        default=None,
        description="下一步要调用的工具名；当 stop=false 时必填，且应为运行时允许的 tool（allowed_tools）之一（例如 lexical_search / semantic_search / neighbors / messages_list）。",
        examples=["lexical_search"],
    )
    args: dict[str, Any] = Field(
        default_factory=dict,
        description="工具参数对象；具体字段由 tool 决定（例如 query_text/filter/top_k/min_score/message_id/cursor 等）。",
        examples=[{"query_text": "偏好 背景 约束"}],
    )

    @model_validator(mode="after")
    def _check_required_fields(self) -> "PlannerOutput":
        if self.stop:
            return self
        if not self.tool or not str(self.tool).strip():
            raise ValueError("tool is required when stop=false")
        if self.args is None:
            raise ValueError("args must be an object when stop=false")
        return self


class SynthesisOutput(BaseModel):
    """Contract for the final memory_view synthesis output."""

    model_config = ConfigDict(extra="forbid")

    memory_view: str = Field(
        default="",
        description="基于证据的第三人称记忆描述（单段文本）。不得编造无证据事实。",
        examples=["该用户曾提及多次不喜欢吃辣。"],
    )
    evidence_message_ids: list[str] = Field(
        default_factory=list,
        description="本次回忆用于支撑结论的证据 message_id 列表（按重要性排序）。",
        examples=[[]],
    )


class LocomoQAOutput(BaseModel):
    """Contract for LoCoMo-style QA final answer output (for benchmark scoring)."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(
        default="",
        description="对问题的最终答案（尽量复用对话中的原词；简短；不编造）。未知则输出字符串 'unknown'。",
        examples=["unknown"],
    )
    evidence_message_ids: list[str] = Field(
        default_factory=list,
        description="支撑该答案的证据 message_id 列表（按重要性排序）。",
        examples=[[]],
    )

class SkillSelectionOutput(BaseModel):
    """Contract for selecting which skill to use for the current task."""

    model_config = ConfigDict(extra="forbid")

    skill_id: str = Field(
        description="要加载的 skill 目录名（位于 ./skills/<skill_id>/SKILL.md）。必须来自可用技能列表。",
        examples=["nomemory-recall-default"],
        min_length=1,
    )


def schema_json_for_prompt(model: type[BaseModel]) -> str:
    """
    JSON Schema string for embedding into prompts.
    Keep it compact (no pretty printing) to reduce context usage.
    """

    schema = model.model_json_schema()
    return json.dumps(schema, ensure_ascii=False, separators=(",", ":"))


def example_json_for_prompt(model: type[BaseModel]) -> str:
    """
    Minimal example JSON for embedding into prompts.
    """

    if model is PlannerOutput:
        ex = PlannerOutput(stop=False, tool="lexical_search", args={"query_text": "偏好 背景 约束"})
        return ex.model_dump_json(ensure_ascii=False)
    if model is SynthesisOutput:
        # Use an "empty lists" example to avoid biasing the model into hallucinating specific facts.
        ex = SynthesisOutput(memory_view="", evidence_message_ids=[])
        return ex.model_dump_json(ensure_ascii=False)
    if model is LocomoQAOutput:
        # For LoCoMo QA, we want a stable "unknown" token for unanswerable questions.
        ex = LocomoQAOutput(answer="unknown", evidence_message_ids=[])
        return ex.model_dump_json(ensure_ascii=False)
    if model is SkillSelectionOutput:
        ex = SkillSelectionOutput(skill_id="nomemory-recall-default")
        return ex.model_dump_json(ensure_ascii=False)
    return "{}"
