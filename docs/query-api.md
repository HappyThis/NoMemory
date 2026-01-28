# v0 技术方案：Chat 用户记忆 Query API

本文定义 v0 方案的查询接口。数据对象为对话消息（message），组织方式为：一个 `user_id` 对应一条按时间排序的消息序列。

使用约定：
- 在真实服务中，建议由服务端从可信上下文绑定 `user_id`，并通过工具装饰器把 `user_id` 注入到查询调用中；回忆 Agent 不应自行选择 `user_id`（见 `docs/recall-service.md`）。

## 1. 数据模型

### 1.1 ChatMessage

| 字段 | 类型 | 含义 | 示例值 |
|---|---|---|---|
| `message_id` | string | 消息唯一标识（用于引用与去重）。 | `m_01HTZ2K8J9Q1ZQ4QYQ8M9J7P6A` |
| `ts` | string | 消息时间戳（RFC3339/ISO8601）。 | `2026-01-26T10:47:00Z` |
| `user_id` | string | 用户标识。 | `u_12345` |
| `role` | string | 消息角色（见下表）。 | `user` |
| `content` | string | 消息正文（检索的核心文本）。 | `我不吃辣` |

**`role` 取值（默认）**

| 取值 | 含义 |
|---|---|
| `user` | 用户输入 |
| `assistant` | 助手输出 |
| `system` | 系统消息（可选） |

### 1.2 分页（Pagination）

接口统一使用游标分页（seek-based），顺序由接口固定定义。

| 字段 | 类型 | 含义 | 示例值 |
|---|---|---|---|
| `page_size` | number | 本页返回最大条数。 | `50` |
| `cursor` | string | 下一页游标（可选，服务端返回的 opaque string）。 | `c_9f2c1b0a...` |

通用输出：
- `items: ChatMessage[]`
- `next_cursor?: string`

约定：
- `cursor/next_cursor` 由服务端生成，客户端应原样回传；不要依赖其内部格式。

### 1.3 过滤器（Filter）

| 字段 | 类型 | 含义 | 示例值 |
|---|---|---|---|
| `time_range.since` | string | 起始时间（含）。 | `2026-01-01T00:00:00Z` |
| `time_range.until` | string | 结束时间（不含）。 | `2026-02-01T00:00:00Z` |
| `role` | string | 限定角色。 | `user` |

## 2. 接口

### 2.1 `GET /v1/users/{user_id}/messages`（范围读取）

**作用**
- 按时间范围读取消息列表（不做关键词/语义匹配）。

**输入**
- 路径参数：`user_id`
- 查询参数：
  - `since?: string`
  - `until?: string`
  - `role?: string`
  - `page_size?: number`
  - `cursor?: string`

**输出**
- `items: ChatMessage[]`
- `next_cursor?: string`

**默认顺序**
- 按 `ts desc`、再按 `message_id`。

**实现方式（自然语言）**
- 对 `(user_id, ts, message_id)` 建索引，基于游标做 seek-based 分页。
- `role` 作为过滤条件在索引上或索引后过滤（实现细节取决于存储）。

---

### 2.2 `POST /v1/messages/lexical_search`（关键词检索）

**作用**
- 对消息正文做关键词/全文检索，返回消息证据。

**输入**
- `user_id: string`
- `query_text: string`
- `filter?: { time_range?: {...}, role?: string }`
- `page_size?: number`
- `cursor?: string`
- `return_fields?: string[]`（可选：裁剪返回字段）

**字段说明**

| 字段 | 含义 | 示例值 |
|---|---|---|
| `query_text` | 关键词/短语查询（全文检索）。 | `"不吃辣" AND 火锅` |
| `return_fields` | 返回字段白名单（从 ChatMessage 字段中选择）。 | `["message_id","ts","role","content"]` |

**输出**
- `items: ChatMessage[]`
- `next_cursor?: string`
- `highlights?: { message_id: string, snippets: string[] }[]`（可选）
- `scores?: { message_id: string, score: number }[]`（可选）

**默认顺序**
- `query_text` 非空：按相关性从高到低；同分按 `ts desc`、再按 `message_id`。

**请求示例**

```json
{
  "user_id": "u_12345",
  "filter": {
    "time_range": { "since": "2026-01-01T00:00:00Z", "until": "2026-02-01T00:00:00Z" },
    "role": "user"
  },
  "query_text": "\"不吃辣\"",
  "page_size": 50
}
```

**实现方式（自然语言）**
- 写入时对 `content` 生成内部 `search_text`（可选：分词/标准化/去噪），并对 `search_text` 建倒排索引。
- 查询时先应用 `filter` 缩小候选集合，再做倒排匹配与排序。

---

### 2.3 `POST /v1/messages/semantic_search`（语义检索）

**作用**
- 用 embedding 相似度召回语义相关消息，返回消息证据。

**输入**
- `user_id: string`
- `filter?: { time_range?: {...}, role?: string }`
- 二选一：
  - `query_text: string`（服务端生成 embedding），或
  - `query_embedding: number[]`
- `top_k?: number`（默认 20）
- `min_score?: number`（可选）
- `return_fields?: string[]`

**输出**
- `items: (ChatMessage & { semantic_score: number })[]`

**默认顺序**
- 按 `semantic_score desc`；同分按 `ts desc`、再按 `message_id`。

**请求示例**

```json
{
  "user_id": "u_12345",
  "filter": { "role": "user", "time_range": { "since": "2026-01-01T00:00:00Z" } },
  "query_text": "不吃辣",
  "top_k": 20
}
```

**实现方式（自然语言）**
- 为每条消息生成并保存 embedding（索引/派生物），在 `user_id` 维度做向量召回。
- 若存在 `time_range/role` 过滤，优先在过滤后的候选集合中召回（减少噪声）。

---

### 2.4 `GET /v1/users/{user_id}/messages/{message_id}/neighbors`（邻域上下文）

**作用**
- 围绕锚点消息取同一用户消息序列内前后 N 条消息，用于补齐上下文避免断章取义。

**输入**
- 路径参数：`user_id`、`message_id`
- `before?: number`（默认 20）
- `after?: number`（默认 0）

**输出**
- `items: ChatMessage[]`（按 `ts asc`、再按 `message_id`）

**实现方式（自然语言）**
- 先读取锚点消息得到 `ts`，再在该用户消息序列内做范围查询（seek-based）。

## 3. 错误模型

- `error: { code: string, message: string, details?: object }`

常见 `code`：
- `INVALID_ARGUMENT`
- `NOT_FOUND`
- `INTERNAL`
