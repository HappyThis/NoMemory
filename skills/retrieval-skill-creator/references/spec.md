# 回忆检索 Skill 生成规范（NoMemory v0）

本文件是 `retrieval-skill-creator` 在生成新的回忆/检索 skill 时必须遵循的**自包含规范**。

## 术语

- **creator skill**：用于生成其它 skills 的 skill（本目录）
- **回忆/检索 skill**：给回忆/检索 Agent 运行时加载并遵循的 skill（由 creator 产出）
- 两者都是标准 Agent Skill：目录包含 `SKILL.md`（以及可选的 `references/`、`scripts/` 等资源）

## 目标

- **默认可用**：用户只给一句场景描述，也能产出可用的检索 skill（zero-config）。
- **可定制**：用户补充信息后，应能增强 query 改写、收窄策略与证据契约。
- **只使用既有 Query API 能力**：生成物最终只能映射到以下 4 个工具组合：
  - `messages_list`
  - `lexical_search`
  - `semantic_search`
  - `neighbors`

## 非目标

- 不负责执行回忆（不直接调用 Query API）。
- 不规定检索后端实现细节（ES/向量库等不在本文范围内）。

## 交互规则（强约束）

- 除非用户明确愿意继续补充，否则最多 **3 个澄清问题**。
- 优先给 **A/B/C 选项**降低输入成本。
- **先产出再迭代**：信息不全也先给默认 skill；再提示可选增强点。

## 最小必需信息（缺失则用默认值）

1) 场景描述（1 句话即可）
2) 记忆用途（输出拿来做什么）
3) 证据契约（至少确认 role 默认值与是否需要 `neighbors`）

## 默认策略（zero-config）

- `time_range_default`：最近 30 天
- `role_default`：偏好/画像默认 `user`（减少 assistant 复述噪声）
- 检索优先级：
  - 偏好/背景/约束：优先 `semantic_search`
  - 明确关键词/短语：优先 `lexical_search`
- `neighbors`：对 top 锚点默认开启（例如 `before=10`）避免断章取义

## 生成的 SKILL.md 必须包含的结构

生成的回忆/检索 skill `SKILL.md` 应当“可读、可执行、可追溯”，建议包含以下章节：

- Frontmatter：`name`、`description`
- Scenario：场景描述原文
- Defaults：时间窗 / role / top_k 等默认值
- Lexicon（可选）：领域词典（canonical → 别名列表）
- Query Templates：按意图列模板（preferences/profile/constraints/schedule）
- Playbook：命中太多/太少如何调参、何时切换 lexical/semantic、停止条件、冲突处理
- Evidence Contract：证据字段要求 + 覆盖范围声明（limits）

## 工具调用示例的放置位置（推荐）

为了让生成的 skill 既自包含又保持 `SKILL.md` 精炼，建议将“工具调用方式/示例（Planner JSON）”放到：

- `scripts/tool_call_examples.md`

并在生成的 `SKILL.md` 中用一小节引用该文件（不要把长示例全部内联到 `SKILL.md`）。

## 启发式参数的放置位置（推荐）

为了避免把启发式参数写死在代码中，推荐将“预算/阈值/页大小/允许工具列表”等**硬约束**放到宿主系统的配置中（而不是放在 skill 目录里）。

Skill 只描述“如何检索/如何决策/如何停止”的策略；运行时由系统配置提供资源预算与安全边界，避免不同 skill 自带配置导致不可移植或加载耦合。

## 输出前必须自检

- 生成物是否只使用 4 个 Query API 工具？
- 是否避免要求用户提供除 `query` 之外的额外参数（时间/role 等应由 agent 自行推断并在工具参数中体现）？
- Evidence Contract 是否要求可核对字段（`message_id/ts/role/content`）以及 `limits`？
