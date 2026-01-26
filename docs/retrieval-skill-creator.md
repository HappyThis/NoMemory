# retrieval-skill-creator（文档草案）

`retrieval-skill-creator` 是一个“生成 Skills 的 Skill”：它在**生成阶段**读取租户的词表/配置（vocabulary），把可用的枚举值、别名映射、推荐过滤器与默认检索策略“编译”进检索（retrieval）相关的 Skills 中，从而让运行时的搜索/回忆 Agent **不需要额外拉词表**也能稳定工作。

> 边界澄清：它不负责配置或生成写入侧的索引字段（例如 `search_text` 的抽取规则）。`search_text` 属于 NoMemory 写入/索引管道在入库时生成的内部字段；retrieval skills 关注的是“如何使用查询接口组织检索”（query_text/过滤器/别名归一化/预算等）。

## 1. 背景与目标

### 背景
- NoMemory 以“事件（Event）”为唯一权威来源，查询层负责召回证据，回忆 Agent 在上层合成记忆视图。
- `actor_type / event_type / source / tags` 在实现上是 `string`，但在语义上需要“受控词表”，便于一致检索与治理。
- 如果运行时 Agent 主动拉词表，会引入额外调用与延迟。

### 目标
- 让不同场景（agent 自进化 / chat 用户理解 / 其他业务场景）的检索策略只需更换 Skills。
- 把词表与策略的“学习成本”前移到生成阶段：运行时只使用生成好的 Skills。
- 生成结果可审计、可版本化、可回滚。

## 2. 核心思路（Compile vocab into skills）

- 系统维护两层词表：
  - **系统默认词表（只读）**
  - **租户覆盖词表（可写）**：新增/覆盖/禁用/废弃（deprecate）某些取值
- `retrieval-skill-creator` 在生成阶段读取“合并视图”，产出面向某个场景的 retrieval skills：
  - 可用的 `actor_type/event_type/source/tags` 子集（只注入检索所需最小集合）
  - 别名/旧名 → 规范名（canonical）的映射
  - 默认检索参数（时间窗、top_k、预算、过滤器优先级）
  - 输出约束（例如必须引用 `event_id`）
- 运行时的搜索/回忆 Agent 只加载 retrieval skills，不再主动请求词表。

## 3. 输入（Input）

`retrieval-skill-creator` 的输入是“场景 + 词表快照 + 策略模板”。

### 3.1 场景（Scenario）
用于决定生成哪一类 retrieval skills，例如：
- `chat.user_memory`
- `agent.self_evolve`
- `biz.<domain>.<use_case>`

### 3.2 词表快照（Vocabulary Snapshot）
词表是逻辑概念，不限定存储方式（表/文件/API）。

最低要求：
- 有系统默认词表与租户覆盖词表的合并结果
- 每个词条至少包含：
  - `type`: `actor_type` / `event_type` / `source` / `tag`
  - `key`: 实际写入事件的字符串值（canonical）
  - `status`: `active` / `deprecated` / `disabled`
  - `aliases`: 可选（旧名/同义词）
  - `description/examples`: 可选（给生成器理解语义）

### 3.3 策略模板（Template）
模板决定生成结果的形态与约束（强烈建议结构化模板，而不是自由文本）：
- 默认过滤器组合（scope + filter）
- 召回阶段划分（关键词/语义/混合）
- 输出 schema（例如 JSON 结构、引用字段、置信度字段）

## 4. 输出（Output）

输出是一份可直接被运行时加载的 retrieval skill（或 skill bundle）。

建议包含：
- `skill_id`: 例如 `retrieval.chat.user_memory.v1`
- `scenario`: 与输入一致
- `vocab_version`（或 `vocab_hash`）：用于一致性校验与再生成触发
- `canonicalization`：
  - `event_type_aliases`: `alias -> canonical`
  - `tag_aliases`: `alias -> canonical`
  - （可选）`actor_type_aliases` / `source_aliases`
- `defaults`：
  - `time_range_default`（例如最近 30 天）
  - `top_k_default`
  - `budget`（例如最大扫描量/最大 token/最大轮数）
- `recommended_filters`：
  - 场景常用的 `event_types/tags` 子集
  - 不建议/禁用的词条（例如 `disabled`）
- `query_playbook`：
  - 多轮检索步骤（先粗后细）
  - 何时用 `search`、何时用 `semantic_search`
- `output_contract`：
  - 必须引用 `event_id`
  - 必须声明不确定性（如 `confidence`）
  - 不输出敏感字段（best-effort，可与调用方策略叠加）

> 注：本仓库暂不规定 skills 的具体文件格式（YAML/JSON/代码），这里只定义应包含的语义信息。

## 5. 版本化与再生成（Versioning）

### 5.1 为什么需要版本
- 词表变化会影响检索策略与别名归一化；不版本化会导致运行时行为漂移。

### 5.2 建议的版本策略
- retrieval skills 内写入 `vocab_version/hash`
- 当词表变更（新增/废弃/禁用/别名变化）时触发再生成
- 支持回滚到旧 skills（与旧词表版本兼容时）

### 5.3 运行时校验（不额外调用的前提下）
运行时不拉词表，但可以：
- 将 `vocab_version` 作为观测字段上报/日志打印
- 当发现服务端返回 `warnings: vocab_mismatch`（可选机制）时，由调用方后台触发再生成并更新 skills

## 6. 使用方式（How it fits the system）

推荐流程：
1. 租户配置/更新词表（覆盖表）
2. 触发 `retrieval-skill-creator` 生成对应场景的 retrieval skills
3. 上游服务把生成结果作为运行时配置下发（或打包发布）
4. 搜索/回忆 Agent 运行时只依赖 retrieval skills + 查询层 API

## 7. 边界与非目标（Non-goals）

- 不在 NoMemory 内部持久化“现成记忆结论”（仍遵循 NoMemory 的 evidence-first 原则）
- 不承担最终内容安全裁决：skills 可以尽力约束/标注风险，但是否采纳由调用方决定
- 不强制规定词表存储实现与 skills 文件格式（这属于后续工程化决策）

## 8. 一个直观例子（概念层）

当租户把 `event_type` 规范成命名空间（例如 `chat.message`、`tool.call`、`tool.result`），并把旧值 `message/tool_call/tool_result` 作为 `aliases`：

- `retrieval-skill-creator` 会把这些映射写进 retrieval skills
- 运行时 Agent 即使仍用旧词（或用户自然语言里出现旧词），也能通过 skills 指导把查询重写为 canonical 值，从而稳定命中索引与过滤条件
