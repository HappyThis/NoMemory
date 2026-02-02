---
name: retrieval-skill-creator
description: 根据用户场景生成 NoMemory 的回忆/检索 Skill 目录（SKILL.md + 可选 scripts/references）。适用于用户要求创建/定制“回忆/检索 skill”或检索策略（证据契约、playbook）。
---

# 回忆检索 Skill 生成器（Creator）

以 `references/spec.md` 作为生成规范（本 skill 必须自包含，不依赖仓库其它文档）。

## 工作流（摘要）

- 最多问 3 个澄清问题，优先给 A/B/C 选项。
- 信息不全也要先产出一份可用的默认 skill，再提供增量优化点。
- 生成物必须只依赖 `references/spec.md` 中列出的 4 个 Query API 工具。

## 输出

在 `skills/` 下创建一个新目录：

```
<skill-name>/
  SKILL.md
  references/        # optional
  scripts/           # optional
```

生成的 `SKILL.md` 必须包含 YAML frontmatter（`name`、`description`），并给出清晰可执行的 Observe → Reflect → Decide 检索步骤说明。

同时，为了让 Skill 更“纯粹”（SKILL.md 只放核心规则），建议把“工具调用示例/格式”放在 `scripts/` 中，并在 `SKILL.md` 里引用，例如：

- `scripts/tool_call_examples.md`：存放 Planner JSON 的工具调用方式与示例（messages_list / lexical_search / semantic_search / neighbors）

另外，启发式参数（时间窗、top_k、预算等）建议也放在 `scripts/` 中，便于运行时读取与调整：
  
注意：本项目将“预算/阈值/页大小/允许工具列表”等运行时硬约束视为**系统配置**的一部分，而不是 skill 的一部分；skill 只负责说明“如何检索/如何决策/如何停止”。
