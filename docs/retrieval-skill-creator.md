# v0 组件：retrieval-skill-creator（Agent Skill 设计）

本文定义一个“creator skill”（按 Claude Agent Skills 的标准组织为一个 Skill 目录 + `SKILL.md`）的设计：它通过多轮对话引导用户描述自己的业务场景，并基于 NoMemory v0（chat）的既有 Query API（见 `docs/query-api.md`）生成一份**回忆 skill**（同样是一个标准 Agent Skill），供回忆 Agent 直接加载执行。

术语澄清（避免歧义）：
- **creator skill**：生成其它 skills 的 skill（本文件描述的对象）
- **回忆 skill**：给回忆 Agent 用的 skill（由 creator skill 产出）
- 两者都是“标准 Agent Skill”（目录包含 `SKILL.md`，可选包含 `references/`、`scripts/` 等资源）

## 1. 目标与边界

### 1.1 目标

- **默认可用**：用户只给一句场景描述，也能生成一份可用的检索策略（zero-config）。
- **可定制**：用户补充业务信息后，creator skill 能自动优化回忆 skill 的内容（更稳的 query 改写、更合适的收窄策略、更清晰的证据契约）。
- **只用现有接口**：回忆 skill 的运行时动作必须只依赖 `docs/query-api.md` 中的接口能力（即你为回忆 Agent 提供的 Query API 工具）。

### 1.2 非目标

- 不负责执行回忆（不直接调用 Query API 做检索）。
- 不规定存储/索引实现细节（ES、向量库等不在本文范围内）。

## 2. Creator Skill 的“多轮交互”设计

creator skill 的核心能力不是“填表”，而是：**发现缺信息 → 用最少问题补齐 → 产出可加载的回忆 skill**。

### 2.1 触发方式（建议）

当用户表达类似意图时触发：
- “为我的业务生成/定制记忆检索策略”
- “帮我做一份回忆 skill”
- “我想让回忆更适合 XXX 场景”

### 2.2 交互策略（强约束）

- **最多 3 个澄清问题**：除非用户明确愿意继续补充，否则不超过 3 个问题。
- **优先给选项**：用 A/B/C 选项降低用户输入成本；允许用户只回复 “A”。
- **先产出再迭代**：信息不全也先给一个默认版本；再提供“你补充 X，我能改进 Y”的增量路径。

### 2.3 必问信息（最小集）

1) **场景描述**（用户一句话即可）
2) **记忆用途**（输出拿来干什么：个性化建议/写邮件/提醒/风格适配/总结）
3) **证据契约**（至少确认 `role` 与是否需要 `neighbors`）

若用户不愿回答，则采用默认值（见 3.2）。

### 2.4 可选信息（用于增强）

- 领域词典（术语/缩写/别名）
- 时间分布偏好（更偏“最近”还是“长期”）
- 噪声来源（线索是否常出现在 assistant 总结里）

## 3. 生成物（回忆 skill）规范

### 3.1 生成物用途

回忆 Agent 加载生成物后，按其中的策略把用户问题改写为 Query API 请求，并在召回证据后合成 `memory_view`。

### 3.1.1 生成物的“标准 Skill 目录”形态（建议）

回忆 skill 建议输出为一个目录（可被打包/安装为 Agent Skill），例如：

```
nomemory-recall-<your-domain>/
  SKILL.md
  references/
    query-api.md        # 可选：摘录项目 Query API 要点（不是实现细节）
    lexicon.md          # 可选：领域词典与同义扩展说明
```

其中 `SKILL.md` 的 YAML frontmatter 只包含：
- `name`
- `description`

### 3.2 默认策略（zero-config）

当用户只给一句场景描述时，采用以下默认：
- `time_range_default`: 最近 30 天（运行时换算为 `since/until`）
- `role_default`: `user`（偏好/画像类默认只看用户输入）
- 检索优先级：偏好/背景/约束类优先 `semantic_search`；关键词很明确时优先 `lexical_search`
- `neighbors`: 对 top 命中默认开启（例如 `before=10`），避免断章取义

### 3.3 回忆 skill（`SKILL.md`）内容结构（建议）

回忆 skill 不是“机器解析的配置文件”，而是一份会被 Agent 直接阅读并遵循的技能说明。建议在 `SKILL.md` 里固定如下章节，保证每个业务定制 skill 都“可读、可执行、可追溯”：

- **Frontmatter**
  - `name`：例如 `nomemory-recall-crm`
  - `description`：描述该回忆 skill 适用的业务场景与目标
- **Scenario**
  - 场景描述原文（用户给的一句话）
- **Defaults**
  - `time_range_default`（示例：`P30D`，运行时换算成 `since/until`）
  - `role_default`（示例：`user`，或留空表示不限定）
  - `top_k_default`（示例：`20`）
- **Lexicon（可选）**
  - 领域词典（canonical → 别名列表），用于 query 改写/扩展
- **Query Templates**
  - 按意图列模板：`preferences/profile/constraints/schedule`
  - 每个模板写清：首选工具（`lexical_search`/`semantic_search`/范围读取）、query_text 改写规则、何时需要 `neighbors`
- **Playbook**
  - 命中太多/太少时如何扩/缩时间窗、是否切换 lexical/semantic、何时停止
- **Evidence Contract**
  - 证据必须包含字段：`message_id/ts/role/content`
  - 输出必须声明覆盖范围：`time_range/role`

## 4. 生成过程（自然语言工作流）

creator skill 的生成流程建议固定为：

1) **理解场景**：从场景描述里识别记忆用途与重点信息（偏好/背景/约束/日程）。
2) **最少澄清**：缺少“证据契约/时间分布/领域术语”就提问；能给选项就给选项。
3) **产出回忆 skill**：填充默认策略，保证可执行。
4) **自检**（强制）：
   - 是否只使用 `docs/query-api.md` 的接口能力？
   - `role_default` 是否为 `user/assistant/system` 或为空？
   - `time_range_default` 是否可换算成 `since/until`？
   - `output_contract` 是否包含可核对证据字段？
5) **给出可选增强点**：提示用户补充领域词典/时间分布偏好能提升效果。

## 5. 例子：用户只给一句话

用户输入：
> “我是 CRM 销售助手，想记住客户信息、沟通偏好与跟进事项，用于写跟进邮件。”

creator skill（可选澄清，最多 3 问）：
1) “证据默认只看用户输入吗？A 只看 user；B user+assistant；C 不限定”
2) “默认时间窗？A 7 天；B 30 天；C 180 天”
3) “要不要提供几个你们常用术语/缩写（可选）？”

若用户只回复：
> “B”

则把 `role_default` 设置为不限定（或显式允许 `user+assistant`），并生成一份可加载的回忆 skill。

## 6. 与 Query API 的对应关系（约束）

生成物最终只能映射到以下接口调用组合（来自 `docs/query-api.md`）：
- `GET /v1/users/{user_id}/messages`：范围读取（按分页游标遍历）
- `POST /v1/messages/lexical_search`：关键词检索
- `POST /v1/messages/semantic_search`：语义检索
- `GET /v1/users/{user_id}/messages/{message_id}/neighbors`：邻域上下文补齐

creator skill 生成的任何回忆 skill，都必须能落到上述调用上，不引入“解释/估算/聚合/批量获取”等额外能力。
