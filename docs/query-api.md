# Query API 文档（事件查询层）

本文件定义 NoMemory 的“查询层”接口：**只返回事件证据（events）**，不返回“现成记忆结论”。回忆（recall）发生在上层回忆 Agent/调用方中。

## 1. 设计目标

- 支持多维检索：时间、结构化字段、关键词、语义向量、混合检索。
- 支持回放与溯源：按会话/链路追踪取回上下文，输出可引用的 `event_id`。
- 硬隔离：保证 **right data, right principal**（不串租户/不串用户/不越权）。

## 2. 通用约定

### 2.0 示例值（Examples）

本文中所有“示例值”会复用下面这些约定，便于你在不同接口间对照理解：

- `tenant_id`: `t_acme`
- `user_id`: `u_12345`
- `session_id`: `sess_20260126_0001`
- `trace_id`: `tr_20260126_abcd`
- `actor_id`（agent 节点）: `agent_planner`
- `event_id`: `evt_01HTZ2K8J9Q1ZQ4QYQ8M9J7P6A`
- `ts`: `2026-01-26T10:47:00Z`
- `cursor`: `c_ts:2026-01-26T10:40:00Z|evt_01HTZ2...`（示意字符串）

### 2.1 鉴权与隔离（Scope）

所有查询都必须在服务端强制绑定隔离范围（最少 `tenant_id`）：

- `tenant_id`：租户/组织边界（强隔离）。建议从鉴权上下文推导（JWT claim、mTLS 证书、API key 映射），而不是完全信任请求体传入。
- `user_id`：租户内的最终用户（chat 常用）。
- `session_id`：一次对话/一次 agent run 的会话边界（常用于缩小召回范围、降低成本）。

> 文档里会把它们写在输入里作为“逻辑字段”，实现上可以由服务端从鉴权上下文注入/校验。

**字段说明与示例值**

| 字段 | 含义 | 示例值 |
|---|---|---|
| `tenant_id` | 硬隔离的租户/组织 ID；任何查询都不能跨 `tenant_id`。 | `t_acme` |
| `user_id` | 租户内用户 ID；常用于 chat 记忆归属与授权。 | `u_12345` |
| `session_id` | 一次对话/一次 agent run 的会话边界；用于缩小检索范围与回放。 | `sess_20260126_0001` |

### 2.2 事件对象（Event）

**输出字段（建议最小集合）**

| 字段 | 含义 | 示例值 |
|---|---|---|
| `event_id` | 事件唯一标识；用于引用、回查、邻域检索的锚点。 | `evt_01HTZ2K8J9Q1ZQ4QYQ8M9J7P6A` |
| `ts` | 事件发生时间戳（RFC3339/ISO8601）。 | `2026-01-26T10:47:00Z` |
| `tenant_id` | 事件归属的租户/组织。 | `t_acme` |
| `user_id` | 事件归属的用户（可选，chat 常用）。 | `u_12345` |
| `session_id` | 事件归属的会话/run（可选）。 | `sess_20260126_0001` |
| `actor_type` | 发生者类型（可选；实现为 string，推荐来自词表；见下表）。 | `user` |
| `actor_id` | 发生者 ID（可选：用户/agent 节点/工具名等）。 | `agent_planner` |
| `source` | 事件来源域。 | `chat` |
| `event_type` | 事件类型（可扩展）。 | `message` |
| `tags` | 轻量标签（用于过滤/聚合）。 | `["topic:food","privacy:sensitive"]` |
| `payload` | 原始内容（JSON 或字符串）；可按 `return_fields` 裁剪/脱敏/截断。 | `{"text":"我不吃辣","role":"user"}` |
| `refs` | 引用/关联对象（可选：线程、trace 等）。 | `{"trace_id":"tr_20260126_abcd","parent_id":"evt_01HTZ2K8J9Q1ZQ4QYQ8M9J7P5Z"}` |
| `refs.parent_id` | 父事件 ID（可选：线程/树结构）。 | `evt_01HTZ2K8J9Q1ZQ4QYQ8M9J7P5Z` |
| `refs.trace_id` | 一次复杂执行链的 trace 标识（可选）。 | `tr_20260126_abcd` |

**`actor_type` 默认词表（可扩展）**

说明：
- `actor_type` 在实现上是 `string`，但建议有一份“系统默认词表”，并允许租户通过词表覆盖/扩展（见 2.6）。
- 为了跨场景互操作，建议至少保留下面这些默认值（即使租户扩展了，也不建议移除/改义）。

| 取值 | 含义 | `actor_id` 示例值 |
|---|---|---|
| `user` | 最终用户（人）。 | `u_12345` |
| `assistant` | 助手人格（对话侧的“助手”角色）。 | `assistant_default` |
| `agent` | 系统内部 agent 节点/子代理。 | `agent_planner` |
| `tool` | 工具/插件/函数调用主体。 | `search` |
| `env` | 外部环境/系统反馈（奖励、监控、webhook 等）。 | `reward_model` |

**Event 示例（响应片段）**

```json
{
  "event_id": "evt_01HTZ2K8J9Q1ZQ4QYQ8M9J7P6A",
  "ts": "2026-01-26T10:47:00Z",
  "tenant_id": "t_acme",
  "user_id": "u_12345",
  "session_id": "sess_20260126_0001",
  "actor_type": "user",
  "actor_id": "u_12345",
  "source": "chat",
  "event_type": "message",
  "tags": ["topic:food"],
  "payload": { "text": "我不吃辣", "role": "user" },
  "refs": { "trace_id": "tr_20260126_abcd", "parent_id": "evt_01HTZ2K8J9Q1ZQ4QYQ8M9J7P5Z" }
}
```

### 2.3 分页（Pagination）

**输入字段说明与示例值**

| 字段 | 含义 | 示例值 |
|---|---|---|
| `page_size` | 本页返回的最大条数；建议设上限（如 200）。 | `50` |
| `cursor` | 游标（seek key）；用于获取下一页。 | `c_ts:2026-01-26T10:40:00Z|evt_01HTZ2...` |

**输出字段说明与示例值**

| 字段 | 含义 | 示例值 |
|---|---|---|
| `items` | 事件列表。 | `[{"event_id":"evt_01HTZ2...","ts":"2026-01-26T10:47:00Z", "...":"..."}]` |
| `next_cursor` | 下一页游标；为空表示没有更多。 | `c_ts:2026-01-26T10:35:00Z|evt_01HTZ1...` |

**关于顺序**
- 本文档不提供通用的“自定义排序”字段；每个接口会在其说明中定义固定的默认顺序（例如按 `ts`、或按相关性）。
- `cursor` 与该接口的默认顺序强绑定；更换顺序会导致游标失效（因此不暴露排序字段）。

### 2.4 过滤器（Filter）

**输入字段说明与示例值（建议）**

| 字段 | 含义 | 示例值 |
|---|---|---|
| `time_range` | 时间范围对象；用于缩小候选集合、控制成本。 | `{"since":"2026-01-01T00:00:00Z","until":"2026-02-01T00:00:00Z"}` |
| `time_range.since` | 起始时间（含）。 | `2026-01-01T00:00:00Z` |
| `time_range.until` | 结束时间（不含或含：以实现约定为准）。 | `2026-02-01T00:00:00Z` |
| `sources` | 限定事件来源域。 | `["chat","tool"]` |
| `event_types` | 限定事件类型。 | `["message","tool_call","error"]` |
| `tags_any` | 任意匹配这些 tag 即可命中。 | `["topic:food","privacy:sensitive"]` |
| `tags_all` | 必须同时包含这些 tag 才命中。 | `["topic:food","lang:zh"]` |
| `actor_id` | 限定发生者 ID（如某个 agent 节点/工具名）。 | `agent_planner` |
| `user_id` | 限定用户（与 scope 的 user 授权协同）。 | `u_12345` |
| `session_id` | 限定会话/run。 | `sess_20260126_0001` |
| `payload_predicates` | 结构化谓词对象（JSONPath/DSL 任选）；用于过滤 payload 内的字段。 | `{"path":"$.tool.name","op":"==","value":"search"}` |
| `payload_predicates.path` | 要匹配的 payload 字段路径（例如 JSONPath）。 | `$.tool.name` |
| `payload_predicates.op` | 比较运算符（实现自定义枚举）。 | `==` |
| `payload_predicates.value` | 比较目标值（字符串/数值/布尔等）。 | `search` |

### 2.5 请求/响应包裹字段（Envelope）

很多接口会复用一些“顶层字段”（例如 `scope/filter/items`）。下面统一说明其含义与示例值，避免在每个接口重复展开。

**通用请求字段**

| 字段 | 含义 | 示例值 |
|---|---|---|
| `scope` | 隔离范围对象（硬边界）；字段含义见 2.1。 | `{"tenant_id":"t_acme","user_id":"u_12345"}` |
| `filter` | 过滤条件对象（软条件）；字段含义见 2.4。 | `{"event_types":["message"],"time_range":{"since":"2026-01-01T00:00:00Z"}}` |
| `weights` | 混合检索融合权重对象（仅 hybrid，可选）。 | `{"lexical":0.6,"semantic":0.4}` |
| `top_examples` | 聚合接口的示例返回控制对象（仅 aggregate，可选）。 | `{"per_bucket":2}` |

**通用响应字段**

| 字段 | 含义 | 示例值 |
|---|---|---|
| `items` | 列表型响应的结果集合（通常是事件列表）。 | `[{"event_id":"evt_01HTZ2...","ts":"2026-01-26T10:47:00Z","event_type":"message","payload":{"text":"我不吃辣"}}]` |
| `event` | 单对象响应的事件实体。 | `{"event_id":"evt_01HTZ2...","ts":"2026-01-26T10:47:00Z","event_type":"message","payload":{"text":"我不吃辣"}}` |
| `highlights` | 高亮片段集合（可选）。 | `[{"event_id":"evt_01HTZ2...","snippets":["…我不吃辣…"]}]` |
| `scores` | 分数集合（可选）。 | `[{"event_id":"evt_01HTZ2...","score":12.34}]` |
| `buckets` | 聚合桶集合（仅 aggregate）。 | `[{"key":{"event_type":"error"},"metrics":{"count":17}}]` |
| `plan` | explain 的执行计划摘要对象。 | `{"steps":["route by tenant_id"],"indexes":["idx_tenant_ts"]}` |
| `error` | 错误对象（失败时返回）。 | `{"code":"FORBIDDEN","message":"tenant scope mismatch","retryable":false}` |

### 2.6 词表（Vocabulary）与“可扩展枚举”

本系统中，`actor_type / event_type / source / tags` 在语义上接近“枚举”，但实现为 `string`。为了让检索/聚合/策略更稳定，建议引入可配置词表（vocabulary）：

- 系统提供**默认词表**（只读）
- 租户可维护**覆盖词表**（可写）：新增/扩展/废弃/禁用某些取值
- 查询/写入时以“合并视图（默认 + 覆盖）”为准

> 运行时搜索 Agent 不一定要主动拉词表：可以用 `retrieval-skill-creator` 在生成阶段读词表并把结果写入 retrieval skills（见 `docs/retrieval-skill-creator.md`）。

**词表条目（VocabEntry）字段说明与示例值（概念层）**

| 字段 | 含义 | 示例值 |
|---|---|---|
| `tenant_id` | 词表归属租户；系统默认词表可用保留值表示（例如 `_system`）。 | `_system` |
| `vocab_type` | 词表类型。 | `event_type` |
| `key` | 规范值（canonical），写入事件与查询过滤都建议使用该值。 | `tool.result` |
| `status` | 词条状态：`active`/`deprecated`/`disabled`。 | `active` |
| `aliases` | 别名/旧名列表（用于归一化）。 | `["tool_result","toolResult"]` |
| `description` | 人可读描述（帮助生成器/审计理解）。 | `Tool execution result event` |
| `examples` | 示例（可选：给出典型事件片段/文本）。 | `["{\"tool\":\"search\",\"ok\":true}"]` |
| `version` | 词表版本（或 hash）；用于触发再生成与一致性校验。 | `v_20260126_01` |

**关于变更的建议**
- 不建议硬删除已被历史事件使用的 `key`；推荐改为 `deprecated/disabled`，并用 `aliases` 保持兼容。
- 建议为 `event_type/tags` 使用命名空间（例如 `chat.message`、`tool.call`、`tool.result`），减少不同团队的命名冲突。

### 2.7 字段所有权（Who sets what）

本节用于减少歧义：哪些字段由系统/写入管道控制，哪些字段允许事件生产者（上游）提供。仓库当前尚未定义“写入 API”，因此这里以**推荐约定**的形式给出（后续实现写入 API 时可对齐）。

| 字段 | 建议由谁设置 | 说明 |
|---|---|---|
| `tenant_id` | 服务端（从鉴权上下文绑定） | 硬隔离边界；不建议信任客户端自报。 |
| `event_id` | 服务端 | 用于去重/引用/游标 tie-breaker；如需幂等写入，可另设 `idempotency_key`。 |
| `ts` | 事件生产者或服务端 | 可允许客户端提供“事件发生时间”；服务端记录 `ingested_at` 以便审计与重放。 |
| `user_id` | 事件生产者（服务端校验） | chat 场景常见；应校验属于当前 `tenant_id`。 |
| `session_id` | 事件生产者或运行时 | 用于把一次对话/一次 run 的事件聚合在一起；可由上游 runtime 生成。 |
| `actor_type` / `actor_id` | 事件生产者 | 表达“谁做的/谁说的”；建议来自词表（见 2.6）。 |
| `source` | 写入通道/采集端（服务端覆盖） | 表达“从哪个子系统/管道写入”；避免被伪造影响检索与治理。 |
| `event_type` | 事件生产者 | 表达“发生了什么”；建议来自词表（见 2.6）。 |
| `tags` | 事件生产者 | 可自由扩展；建议命名空间化；可由 skills/策略补充。 |
| `payload` | 事件生产者 | 原始证据内容；可结构化（JSON）或字符串；敏感字段建议在写入侧脱敏或分级。 |
| `refs` | 事件生产者或运行时 | 用于线程/trace 关系；`refs.parent_id` 建议尽量指向同一 `tenant_id` 下的事件。 |
| `embedding` | 写入管道（可选） | 属于索引/派生物；可在写入后异步生成并回填。 |
| `search_text`（内部） | 写入/索引管道（内部字段） | 用于倒排索引；由 `payload` 抽取生成（见 3.1 的说明）。 |

## 3. 接口清单

下面以 HTTP/JSON 形式描述（同样可映射到 gRPC/SDK）。

---

### 3.1 搜索事件（结构化 + 关键词）

**接口**：`POST /v1/events/search`

**作用**
- 用时间窗 + 结构化过滤 + 关键词全文检索（倒排/BM25）找事件证据。

**输入**
- `scope`（逻辑字段）：`tenant_id`（强制绑定）、可选 `user_id/session_id`
- `filter`：见 2.4
- `query_text?: string`：关键词/短语（可选）
- `return_fields?: string[]`：裁剪字段（减少 payload 下发）
- 分页：`page_size/cursor`

**字段说明与示例值（本接口新增字段）**

| 字段 | 含义 | 示例值 |
|---|---|---|
| `query_text` | 关键词/短语查询（全文检索）；语法可按实现支持（短语、AND/OR、字段限定等）。 | `"不吃辣" AND 火锅` |
| `return_fields` | 控制返回字段/裁剪 payload；字段路径格式由实现定义。 | `["event_id","ts","payload.text","event_type"]` |

**输出**
- `items: Event[]`
- `next_cursor?: string`
- `highlights?: { event_id: string, snippets: string[] }[]`（可选）
- `scores?: { event_id: string, score: number }[]`（可选）

**字段说明与示例值（本接口新增字段）**

| 字段 | 含义 | 示例值 |
|---|---|---|
| `highlights[].event_id` | 高亮片段对应的事件 ID。 | `evt_01HTZ2K8J9Q1ZQ4QYQ8M9J7P6A` |
| `highlights[].snippets` | 命中关键词的文本片段（用于 UI 展示）。 | `["…我不吃辣…"]` |
| `scores[].event_id` | 分数对应的事件 ID。 | `evt_01HTZ2K8J9Q1ZQ4QYQ8M9J7P6A` |
| `scores[].score` | 检索相关性分数（BM25 等），仅用于排序/调试。 | `12.34` |

**请求示例**

```json
{
  "scope": { "tenant_id": "t_acme", "user_id": "u_12345" },
  "filter": { "time_range": { "since": "2026-01-01T00:00:00Z", "until": "2026-02-01T00:00:00Z" }, "event_types": ["message"] },
  "query_text": "\"不吃辣\"",
  "return_fields": ["event_id", "ts", "payload.text", "event_type"],
  "page_size": 50
}
```

**使用场景**
- chat：查“我之前说过/提到过 XX 吗”
- agent：按工具名/错误码/标签快速定位问题段落

**检索语义（如何理解 `search`）**

- `search` 是“关键词/全文检索（lexical search）”：对事件的可检索文本建立倒排索引，用 `query_text` 做匹配（再叠加 `filter/time_range` 缩小范围），返回事件证据。
- 当 `query_text` 为空时，本接口退化为“按过滤条件列出事件”：
  - 默认顺序：按 `ts` 由新到旧（`ts desc`）
  - 仍然支持分页：`page_size/cursor`

**被索引的文本（`search_text`，内部概念）**

为避免要求所有 `payload` 全局统一结构，推荐在写入时把事件的可检索文本抽取为内部字段 `search_text`（仅用于检索，不一定对外暴露），并按 `event_type` 配置抽取规则。

> 重要：`search_text` 的抽取规则属于**写入/索引配置**（server-controlled），在事件入库时生成并写入倒排索引；它不是 retrieval skills 的“必需配置”。retrieval skills 只需要指导检索 Agent 如何组织 `query_text`、如何收窄范围（`event_type/tags/time_range` 等）、以及必要的别名归一化即可。

常见默认抽取示例（概念层）：

| `event_type` 示例 | 从 `payload` 抽取到 `search_text` 的字段示例 | `payload` 示例值 |
|---|---|---|
| `message` | `payload.text` / `payload.content` | `{"text":"我不吃辣"}` |
| `tool_call` | `payload.tool` + `payload.input` | `{"tool":"search","input":"NoMemory 设计"}` |
| `tool_result` | `payload.tool` + `payload.output`（或摘要字段） | `{"tool":"search","output":"..."} ` |
| `error` | `payload.code` + `payload.message` | `{"code":"TIMEOUT","message":"upstream timeout"}` |

> 结构化过滤（例如按 `payload.tool == "search"`）建议使用 `payload_predicates`；不要依赖 `query_text` 的“字符串包含”去模拟结构化过滤。

**`query_text` 语法（建议最小集合）**

| 语法能力 | 含义 | 示例值 |
|---|---|---|
| 词项（terms） | 空格分隔多个词项；具体分词策略由实现决定（中文可用 n-gram/分词）。 | `不吃辣 火锅` |
| 短语（phrase） | 引号包裹的短语尽量按整体匹配。 | `"不吃辣"` |
| 布尔组合（boolean） | 支持 `AND/OR` 组合（可选）。 | `"不吃辣" AND 火锅` |
| 排除（negation） | 支持排除某个词项（可选）。 | `火锅 -辣椒` |

**排序与游标**
- 默认按相关性从高到低返回（例如 BM25 分数）；若分数相同，使用 `ts desc`、再用 `event_id` 作为稳定 tie-breaker。
- `cursor` 与该默认顺序强绑定。

**实现方式（自然语言）**
- 以 `tenant_id` 分区路由；优先用时间窗缩小候选集合。
- 对 `search_text` 建倒排索引（由 `payload` 抽取生成）；对 `ts/event_type/tags/session_id` 建二级索引。
- 倒排召回后按相关性/时间排序；大 payload 可按 `return_fields` 裁剪与截断。

---

### 3.2 语义检索（向量相似度）

**接口**：`POST /v1/events/semantic_search`

**作用**
- 用 embedding 相似度召回语义相关事件证据。

**输入**
- `scope`：`tenant_id`（强制绑定）、可选 `user_id/session_id`
- `filter`：见 2.4
- 二选一：
  - `query_text: string`（服务端生成 embedding），或
  - `query_embedding: number[]`
- `top_k?: number`：默认 20
- `min_score?: number`（可选）
- `return_fields?: string[]`

**字段说明与示例值（本接口新增字段）**

| 字段 | 含义 | 示例值 |
|---|---|---|
| `query_text` | 用于生成 embedding 的查询文本。 | `我不吃辣` |
| `query_embedding` | 直接传入查询向量（当调用方已生成 embedding）。 | `[0.012, -0.334, 0.981]` |
| `top_k` | 返回的候选条数上限。 | `20` |
| `min_score` | 相似度阈值（低于阈值过滤）。 | `0.35` |

**输出**
- `items: (Event & { semantic_score: number })[]`

**字段说明与示例值（本接口新增字段）**

| 字段 | 含义 | 示例值 |
|---|---|---|
| `semantic_score` | 向量相似度分数（范围由实现决定）。 | `0.82` |

**请求示例**

```json
{
  "scope": { "tenant_id": "t_acme", "user_id": "u_12345" },
  "filter": { "event_types": ["message"], "time_range": { "since": "2026-01-01T00:00:00Z" } },
  "query_text": "不吃辣",
  "top_k": 20,
  "min_score": 0.35,
  "return_fields": ["event_id", "ts", "payload.text"]
}
```

**使用场景**
- chat：用户偏好/背景“换说法”也能召回（同义改写）
- agent：查相似错误、相似环境反馈、相似决策理由

**实现方式（自然语言）**
- 对事件保存 embedding（属于“索引/派生物”，允许持久化），在 `tenant_id` 分区内建向量索引（如 HNSW/IVF）。
- 能预过滤则预过滤（例如先按 `session_id/event_type` 选子索引），否则向量召回后再按 filter 过滤并补召回。

---

### 3.3 混合检索（关键词 + 语义）

**接口**：`POST /v1/events/hybrid_search`

**作用**
- 同时做关键词召回（更精确）与语义召回（更覆盖），合并候选并重排。

**输入**
- `scope`
- `filter`
- `query_text: string`
- `top_k?: number`：默认 20
- `weights?: { lexical: number, semantic: number }`（可选）
- `rerank?: "none"|"cross_encoder"|"llm"`（可选，默认 none）
- `return_fields?: string[]`

**字段说明与示例值（本接口新增字段）**

| 字段 | 含义 | 示例值 |
|---|---|---|
| `top_k` | 返回条数上限。 | `20` |
| `weights.lexical` | 关键词相关性权重。 | `0.6` |
| `weights.semantic` | 语义相关性权重。 | `0.4` |
| `rerank` | 二阶段重排方式：不重排/交叉编码器/LLM 重排。 | `cross_encoder` |

**输出**
- `items: (Event & { lexical_score?: number, semantic_score?: number, final_score: number })[]`

**字段说明与示例值（本接口新增字段）**

| 字段 | 含义 | 示例值 |
|---|---|---|
| `lexical_score` | 关键词检索分数（可选）。 | `8.90` |
| `semantic_score` | 语义检索分数（可选）。 | `0.77` |
| `final_score` | 融合/重排后的最终分数。 | `0.91` |

**请求示例**

```json
{
  "scope": { "tenant_id": "t_acme", "user_id": "u_12345" },
  "filter": { "time_range": { "since": "2026-01-01T00:00:00Z" } },
  "query_text": "不吃辣 火锅",
  "top_k": 20,
  "weights": { "lexical": 0.6, "semantic": 0.4 },
  "rerank": "none",
  "return_fields": ["event_id", "ts", "payload.text"]
}
```

**使用场景**
- 既要命中关键 token（如工具名/错误码），又要覆盖语义近似描述的 recall。

**实现方式（自然语言）**
- 并行跑倒排与向量检索得到两份候选；做 union 去重。
- 用融合策略（如 RRF/线性加权）得到 `final_score`；可选二阶段重排（cross-encoder 或 LLM）。

---

### 3.4 按 ID 精确读取

**接口**：`GET /v1/events/{event_id}`

**作用**
- 取回单条事件，用于溯源/审计/引用校验。

**输入**
- 路径：`event_id`
- `scope`：服务端强制绑定 `tenant_id`

**字段说明与示例值**

| 字段 | 含义 | 示例值 |
|---|---|---|
| `event_id` | 要读取的事件 ID（路径参数）。 | `evt_01HTZ2K8J9Q1ZQ4QYQ8M9J7P6A` |
| `tenant_id` | 由服务端从鉴权上下文绑定的租户范围。 | `t_acme` |

**输出**
- `event: Event`

**使用场景**
- 回忆 Agent 输出引用 `event_id` 后，调用方回查原文确认

**实现方式（自然语言）**
- 主键索引直达（建议 `tenant_id + event_id`）；强制校验事件归属租户。

---

### 3.5 批量按 ID 读取

**接口**：`POST /v1/events/batch_get`

**作用**
- 批量回查事件，避免 N+1 调用。

**输入**
- `event_ids: string[]`
- `scope`：服务端强制绑定 `tenant_id`

**字段说明与示例值**

| 字段 | 含义 | 示例值 |
|---|---|---|
| `event_ids` | 要批量读取的事件 ID 列表。 | `["evt_01HTZ2...P6A","evt_01HTZ2...P6B"]` |
| `tenant_id` | 由服务端从鉴权上下文绑定的租户范围。 | `t_acme` |

**输出**
- `items: Event[]`
- `misses?: string[]`

**字段说明与示例值（本接口新增字段）**

| 字段 | 含义 | 示例值 |
|---|---|---|
| `misses` | 未命中的 event_id（不存在或因权限不可见）；策略可选“返回 misses”或“直接拒绝请求”。 | `["evt_01HTZ2...P6C"]` |

**使用场景**
- 回忆 Agent 先召回 event_id 列表，再批量取全文

**实现方式（自然语言）**
- 按分区聚合后批量主键查询；对不存在或越权的 ID 返回 `misses` 或拒绝（取决于安全策略）。

---

### 3.6 邻域上下文（Neighbors）

**接口**：`GET /v1/events/{event_id}/neighbors`

**作用**
- 以某事件为锚点，取同一上下文里的前后 N 条事件（帮助补齐上下文）。

**输入**
- `event_id`
- `before?: number`（默认 20）
- `after?: number`（默认 0）
- `mode?: "session"|"trace"`（默认 session）
- `scope`：服务端强制绑定 `tenant_id`

**字段说明与示例值**

| 字段 | 含义 | 示例值 |
|---|---|---|
| `event_id` | 锚点事件 ID。 | `evt_01HTZ2K8J9Q1ZQ4QYQ8M9J7P6A` |
| `before` | 向前取多少条（同一 session/trace）。 | `20` |
| `after` | 向后取多少条（同一 session/trace）。 | `10` |
| `mode` | 邻域范围：同 `session` 或同 `trace`。 | `session` |
| `tenant_id` | 由服务端从鉴权上下文绑定的租户范围。 | `t_acme` |

**输出**
- `items: Event[]`（按时间排序）

**使用场景**
- chat：找某句话前后语境
- agent：定位失败前后的决策与工具调用

**实现方式（自然语言）**
- 先读取锚点事件得到 `(session_id/trace_id, ts)`，再用 `tenant_id + (session_id/trace_id) + ts` 做范围查询（seek-based）。

---

### 3.7 拉取会话事件（Session Replay）

**接口**：`GET /v1/sessions/{session_id}/events`

**作用**
- 拉取某次对话/某次 agent run 的事件流（可分页），用于回放、复盘、训练数据导出等。

**输入**
- `session_id`
- 分页：`page_size/cursor`
- `scope`：服务端强制绑定 `tenant_id`（可选再校验 user_id 归属）

**字段说明与示例值**

| 字段 | 含义 | 示例值 |
|---|---|---|
| `session_id` | 会话/run ID（路径参数）。 | `sess_20260126_0001` |
| `page_size` | 每页数量。 | `50` |
| `cursor` | 下一页游标。 | `c_ts:2026-01-26T10:40:00Z|evt_01HTZ2...` |
| `tenant_id` | 由服务端从鉴权上下文绑定的租户范围。 | `t_acme` |

**输出**
- `items: Event[]`
- `next_cursor?: string`

**使用场景**
- agent：全链路复盘（decision → tool_call → tool_result → feedback）
- chat：回忆需要完整上下文时按 session 拉取

**实现方式（自然语言）**
- 建 `tenant_id + session_id + ts` 联合索引；按时间流式分页读取。

---

### 3.8 按 trace 拉取（可选）

**接口**：`GET /v1/traces/{trace_id}/events`

**作用**
- 跨会话/跨节点回放一次复杂执行链（trace）。

**输入/输出**
- 同 session replay，但键为 `trace_id`

**字段说明与示例值**

| 字段 | 含义 | 示例值 |
|---|---|---|
| `trace_id` | 一次复杂执行链的 trace 标识（路径参数）。 | `tr_20260126_abcd` |

**使用场景**
- 多 agent 协作任务的全局回放与因果链分析

**实现方式（自然语言）**
- 对 `trace_id` 建二级索引；或维护 refs 关系表做图遍历（实现取决于数据模型）。

---

### 3.9 相似事件（以事件为查询）

**接口**：`GET /v1/events/{event_id}/similar`

**作用**
- 用某条事件作为锚点，找历史相似事件（“类比回忆”）。

**输入**
- `event_id`
- `top_k?: number`
- `filter?`：例如限定 `event_type`、限定时间窗
- `scope`：服务端强制绑定 `tenant_id`

**字段说明与示例值**

| 字段 | 含义 | 示例值 |
|---|---|---|
| `event_id` | 锚点事件 ID。 | `evt_01HTZ2K8J9Q1ZQ4QYQ8M9J7P6A` |
| `top_k` | 返回条数上限。 | `20` |
| `filter.event_types` | 限定相似检索只在某类事件中进行（可选）。 | `["error"]` |
| `tenant_id` | 由服务端从鉴权上下文绑定的租户范围。 | `t_acme` |

**输出**
- `items: (Event & { semantic_score: number })[]`

**使用场景**
- agent：相似错误的历史处理方式、相似环境下的策略
- chat：相似话题下用户曾经的表述（用于一致性）

**实现方式（自然语言）**
- 取锚点事件 embedding，然后走 `semantic_search`；加过滤避免噪声（如只看 `message` 或只看 `error`）。

---

### 3.10 聚合统计（给回忆 Agent 做“先粗后细”）

**接口**：`POST /v1/events/aggregate`

**作用**
- 对事件做分组统计，返回桶（bucket）而非全量原文；可选返回每桶少量示例事件。

**输入**
- `scope`
- `filter`
- `group_by: ("event_type"|"source"|"tag"|"actor_id")[]`
- `time_bucket?: "1m"|"1h"|"1d"`（可选）
- `metrics: ("count"|"distinct_session")[]`
- `top_examples?: { per_bucket: number }`（可选）

**字段说明与示例值**

| 字段 | 含义 | 示例值 |
|---|---|---|
| `group_by` | 分组维度（可多维）。 | `["event_type","actor_id"]` |
| `time_bucket` | 时间分桶粒度（可选）。 | `1h` |
| `metrics` | 要计算的指标列表。 | `["count","distinct_session"]` |
| `top_examples.per_bucket` | 每个桶返回多少条示例事件（可选）。 | `2` |

**输出**
- `buckets: { key: object, metrics: object, examples?: Event[] }[]`

**字段说明与示例值**

| 字段 | 含义 | 示例值 |
|---|---|---|
| `buckets[].key` | 分组键（由 `group_by` 维度组成）。 | `{"event_type":"error","actor_id":"agent_planner"}` |
| `buckets[].metrics` | 指标结果（由 `metrics` 决定）。 | `{"count": 17, "distinct_session": 5}` |
| `buckets[].examples` | 示例事件（可选，用于点进查看证据）。 | `[{"event_id":"evt_01HTZ2...","event_type":"error","ts":"2026-01-26T10:47:00Z"}]` |

**补充说明**
- 如果设置了 `time_bucket`，则 `buckets[].key` 会额外包含时间分桶键（例如 `bucket_start`），用于表示该桶对应的时间片。

**请求示例（按小时统计每个 agent 节点的 error 数，并附带每桶 1 条证据）**

```json
{
  "scope": { "tenant_id": "t_acme" },
  "filter": { "event_types": ["error"], "time_range": { "since": "2026-01-26T00:00:00Z", "until": "2026-01-27T00:00:00Z" } },
  "group_by": ["actor_id"],
  "time_bucket": "1h",
  "metrics": ["count", "distinct_session"],
  "top_examples": { "per_bucket": 1 }
}
```

**响应示例（片段）**

```json
{
  "buckets": [
    {
      "key": { "bucket_start": "2026-01-26T10:00:00Z", "actor_id": "agent_planner" },
      "metrics": { "count": 3, "distinct_session": 2 },
      "examples": [
        { "event_id": "evt_01HTZ2K8J9Q1ZQ4QYQ8M9J7P6A", "ts": "2026-01-26T10:47:00Z", "event_type": "error", "actor_id": "agent_planner", "payload": { "code": "TIMEOUT" } }
      ]
    }
  ]
}
```

**使用场景**
- agent：快速定位“哪里错误最多/哪个节点最不稳定/哪个工具最常失败”
- chat：快速了解近期会话分布（作为检索策略输入，不直接变成记忆结论）

**实现方式（自然语言）**
- 小规模可在线聚合但必须限窗限量；规模更大建议列存/OLAP 或预聚合视图。

---

### 3.11 查询估算（让调用方决定“要不要回忆”）

**接口**：`POST /v1/events/estimate`

**作用**
- 在执行查询前估算命中数/扫描量/耗时，便于调用方做预算控制与缓存策略。

**输入**
- 与 `search/semantic_search/hybrid_search` 相同的 query spec（任选一种）

**输出**
- `estimated_hits: number`
- `estimated_scan: number`
- `estimated_latency_ms: number`
- `notes?: string[]`

**字段说明与示例值**

| 字段 | 含义 | 示例值 |
|---|---|---|
| `estimated_hits` | 预计命中条数（估算）。 | `1200` |
| `estimated_scan` | 预计扫描量/候选量（估算）。 | `50000` |
| `estimated_latency_ms` | 预计耗时毫秒（估算）。 | `180` |
| `notes` | 提示与建议（例如“时间窗过大”）。 | `["time_range missing: consider adding since/until"]` |

**使用场景**
- 上游在“回忆很贵/很慢”时做门控：超过预算就缩小时间窗、降低 top_k 或直接跳过 recall

**实现方式（自然语言）**
- 基于索引统计信息与采样估算；不做全量扫描。

**说明：estimate 输入如何组织**
- `estimate` 的请求体复用 `search/semantic_search/hybrid_search` 的请求结构之一（任选其一）。
- `estimate` 通常会忽略与“结果裁剪/分页”相关的字段（例如 `return_fields/page_size/cursor`），只根据范围与检索条件估算成本。

**请求示例（对 search 进行估算）**

```json
{
  "scope": { "tenant_id": "t_acme", "user_id": "u_12345" },
  "filter": { "time_range": { "since": "2026-01-01T00:00:00Z" }, "event_types": ["message"] },
  "query_text": "不吃辣 火锅"
}
```

**响应示例**

```json
{
  "estimated_hits": 1200,
  "estimated_scan": 50000,
  "estimated_latency_ms": 180,
  "notes": ["consider narrowing time_range", "consider adding session_id or tags"]
}
```

---

### 3.12 查询解释（Explain）

**接口**：`POST /v1/events/explain`

**作用**
- 返回本次查询的执行计划摘要（便于调优与排障）。

**输入**
- 与任意查询相同的 query spec

**输出**
- `plan: { steps: string[], indexes: string[], warnings?: string[] }`

**字段说明与示例值**

| 字段 | 含义 | 示例值 |
|---|---|---|
| `plan.steps` | 执行步骤摘要（人可读）。 | `["route by tenant_id","apply time_range","BM25 search on search_text","rank by score"]` |
| `plan.indexes` | 命中的索引名称（人可读）。 | `["idx_tenant_ts","inv_search_text"]` |
| `plan.warnings` | 可能的性能/质量风险提示。 | `["no time_range; query may be slow"]` |

**使用场景**
- 线上排查召回慢/召回差：是不是没走时间窗、是不是没命中索引、是不是 filter 太宽

**实现方式（自然语言）**
- 暴露 query planner 的摘要，不泄露内部敏感实现细节。

**请求示例（解释一次 search 会怎么执行）**

```json
{
  "scope": { "tenant_id": "t_acme", "user_id": "u_12345" },
  "filter": { "time_range": { "since": "2026-01-01T00:00:00Z" }, "event_types": ["message"] },
  "query_text": "\"不吃辣\" AND 火锅"
}
```

**响应示例**

```json
{
  "plan": {
    "steps": [
      "route by tenant_id",
      "apply time_range",
      "BM25 search on search_text",
      "rank by score",
      "tie-break by ts desc + event_id"
    ],
    "indexes": ["idx_tenant_ts", "inv_search_text"],
    "warnings": []
  }
}
```

## 4. 错误模型（建议）

统一错误结构：

- `error: { code: string, message: string, retryable?: boolean, details?: object }`

**字段说明与示例值**

| 字段 | 含义 | 示例值 |
|---|---|---|
| `error.code` | 错误码（稳定枚举）。 | `FORBIDDEN` |
| `error.message` | 面向调用方的人类可读信息。 | `tenant scope mismatch` |
| `error.retryable` | 是否建议重试。 | `false` |
| `error.details` | 机器可读的补充信息（可选）。 | `{"required":"t_acme","got":"t_other"}` |

常见 `code`：
- `UNAUTHENTICATED`：未认证
- `FORBIDDEN`：越权（scope 不匹配）
- `INVALID_ARGUMENT`：参数错误
- `NOT_FOUND`：资源不存在
- `RESOURCE_EXHAUSTED`：超预算/限流
- `DEADLINE_EXCEEDED`：超时

## 5. 说明：为什么要区分 tenant_id / user_id / session_id

- `tenant_id`：**硬隔离边界**。保证不会把 A 公司的事件返回给 B 公司。
- `user_id`：**用户维度**。chat 记忆通常以用户为主语；同租户下用户之间也不能互相看到。
- `session_id`：**成本与语境边界**。先把范围缩到某次对话/某次 run，召回更准、更便宜；也便于回放与审计。
