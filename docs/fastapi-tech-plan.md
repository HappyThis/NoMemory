# v0 技术落地方案：FastAPI + Postgres(pgvector) 的 NoMemory

本文把现有 v0 文档（`docs/query-api.md`、`docs/recall-service.md`、`docs/recall-agent-playbook.md`、`docs/retrieval-skill-creator.md`）收敛为一份可直接开工的工程落地方案，并补齐**唯一的写入接口**（消息入库/同步）。

## 0. 边界与原则

- **写入接口只有一个**：仅用于把“用户历史消息”写入存储（服务到服务，可信调用方）。
- **Recall/Query 是读路径**：回忆 Agent 只通过工具读证据，不参与写入。
- **`user_id` 可信绑定**：Recall 运行时由服务端从登录态/token 注入 `user_id`；Agent 永远看不到/传不了 `user_id`（见 `docs/recall-service.md`）。
- **输出必须可追溯**：所有结论必须能指回 `evidence` 原文消息（见 `docs/recall-agent-playbook.md` 的 evidence contract 思路）。
- **使用 ORM**：服务端数据库访问统一走 ORM（v0 选型：SQLAlchemy 2.x），schema 变更用 Alembic 迁移。
- **支持多模型**：向量与 LLM 选择均可配置；默认 LLM 模型为 `glm-4.7-flash`（BigModel）。

## 1. 系统形态（单进程服务）

一个 FastAPI 服务提供三组能力：

1) **Ingest API（写入）**：接收消息批量写入（唯一写接口）。
2) **Query API（读底座）**：实现 `docs/query-api.md` 的 4 个读接口（范围读取/关键词/语义/邻域）。
3) **Recall API（读 + Agent 编排）**：对外提供 “回忆”能力（运行 O/R/D 闭环，输出 `memory_view + evidence + limits`）。

部署上允许拆分，但 v0 推荐同一服务先跑通；后续可将 embedding 生成拆为 worker。

## 1.1 本地开发数据库（Docker）

本方案默认本地使用 Docker 启动 Postgres（含 pgvector 扩展）：

- 配置文件：`docker-compose.yml`
- 初始化 SQL：`docker/db/init/001_extensions.sql`（自动执行 `CREATE EXTENSION vector`）

启动：

```bash
docker compose up -d db
```

连接参数（默认）：

- host: `127.0.0.1`
- port: `5432`
- db: `nomemory`
- user: `nomemory`
- password: `nomemory`

应用侧建议用环境变量拼 DSN，例如：

```text
postgresql+psycopg://nomemory:nomemory@127.0.0.1:5432/nomemory
```

并将其放到环境变量 `DATABASE_URL`（参考 `.env.example`）。

## 1.2 模型与供应商（BigModel 默认）

本方案允许未来接入多家供应商与多模型，但 v0 默认选 BigModel（智谱）：

- 默认 LLM：`glm-4.7-flash`（用于 Recall Agent 的推理与编排）
- 嵌入（embedding）：使用 BigModel 的“文本嵌入” API（用于 `semantic_search` 的向量生成/回填）

配置建议全部走环境变量（参考 `.env.example`）：

- `LLM_PROVIDER=bigmodel`
- `LLM_MODEL=glm-4.7-flash`
- `BIGMODEL_API_KEY=...`
- `BIGMODEL_CHAT_ENDPOINT=...`（按 BigModel 文档配置）
- `BIGMODEL_EMBEDDING_ENDPOINT=...`（按 BigModel 文档配置）
- `BIGMODEL_EMBEDDING_MODEL=...`（按 BigModel 文档配置）

## 1.4 Skills（creator + 默认检索 skill）

本仓库提供两类 Skills（目录位于 `skills/`）：

- `skills/retrieval-skill-creator/`：用于**生成**回忆/检索 skill（规范见 `docs/retrieval-skill-creator.md`）
- `skills/nomemory-recall-default/`：默认的回忆/检索 skill（Recall Agent 运行时加载并遵循）

Recall Agent 会加载 `RECALL_SKILL`（默认 `nomemory-recall-default`），并按 skill 的 Playbook 做 Observe → Reflect → Decide 的闭环检索。

## 1.3 迁移与启动（v0 最小可跑）

v0 推荐用 Alembic 管理 schema：

```bash
alembic upgrade head
```

FastAPI 启动后可用：

- `GET /healthz`：健康检查

## 2. 数据模型（Postgres）

### 2.1 messages（主表）

字段（建议最小集）：

- `user_id TEXT`
- `message_id TEXT`（建议在 user 内唯一；与 `user_id` 组成主键/幂等键）
- `ts TIMESTAMPTZ`
- `role TEXT`（`user|assistant|system`）
- `content TEXT`
- `meta JSONB NULL`（可选：channel、conversation_id、tool_name 等）

约束与索引（核心）：

- 主键（或唯一约束）：`(user_id, message_id)`（幂等写入）
- 范围读取：`INDEX(user_id, ts DESC, message_id DESC)`（seek-based cursor）
- 常用过滤：可选 `INDEX(user_id, role, ts DESC, message_id DESC)`

### 2.2 关键词检索（FTS）

两种做法（二选一即可）：

- A) `messages` 增加生成列 `search_tsv tsvector GENERATED ALWAYS AS (...) STORED`，并建 `GIN(search_tsv)`
- B) 单独建 `message_fts(message_id, user_id, search_tsv)`，由写入/更新时维护

v0 推荐 A（简单）。

### 2.3 向量检索（pgvector）

向量存储做法（二选一）：

- A) `messages` 增加 `embedding vector(d)`（d 取决于 embedding 维度）
- B) 单独表 `message_embeddings(user_id, message_id, provider, model, embedding vector(d), created_at TIMESTAMPTZ)`

v0 推荐 B（便于异步回填、模型升级、重算）。

索引：

- 数据量小可先不加 ANN（直接排序 top_k 也能跑通）
- 数据量大再加近似索引（如 `ivfflat` / `hnsw`，以你所用 pgvector/PG 版本为准）

## 3. API 设计

### 3.1 唯一写接口：Ingest API（服务到服务）

建议：

`POST /v1/users/{user_id}/messages:batch`

用途：把聊天业务系统产生的历史消息/实时消息批量写入本服务。

鉴权：**仅可信调用方**（建议 API Key 或 mTLS；禁止终端用户/Agent 调用）。

v0 开发态（本仓库实现）使用 `X-API-Key` 头做最小鉴权（见 `.env.example` 的 `INGEST_API_KEY`）。

请求体（示意）：

```json
{
  "items": [
    {
      "message_id": "m_...",
      "ts": "2026-01-26T10:47:00Z",
      "role": "user",
      "content": "我不吃辣",
      "meta": { "conversation_id": "c_..." }
    }
  ]
}
```

行为约束：

- **幂等**：基于 `UNIQUE(user_id, message_id)`；重复写入应返回 `ignored_count` 或按 upsert 策略覆盖（v0 推荐：默认忽略重复，避免误改历史）。
- **强校验**：role 枚举、ts 格式、content 长度上限。
- **写入后派生**：
  - FTS：如果用生成列则自动；否则写入时同步维护。
  - embedding：推荐异步生成（见第 4 节）；v0 可先同步生成以验证链路（但会变慢）。

返回（示意）：

```json
{ "inserted": 10, "ignored": 2, "failed": 0 }
```

### 3.2 Query API（读底座，按 `docs/query-api.md`）

这组接口建议仅对内部或受控客户端开放；若必须公网开放，务必验证“路径 user_id == 登录态 user_id”。

- `GET /v1/users/{user_id}/messages`：范围读取（seek-based cursor）
- `POST /v1/messages/lexical_search`：关键词检索（FTS）
- `POST /v1/messages/semantic_search`：语义检索（pgvector）
- `GET /v1/users/{user_id}/messages/{message_id}/neighbors`：邻域上下文

cursor 约定（实现建议）：

- cursor 只包含分页锚点（例如 `last_ts`, `last_message_id`）与方向（desc）
- cursor 需 **opaque + 签名**（防止客户端伪造/回退攻击）

### 3.3 Recall API（对外的“回忆入口”）

`POST /v1/recall`

入参：

- `question: string`
- `context?: { time_range?: { since?: string, until?: string }, role_pref?: "user"|"any" }`

鉴权：

- 终端用户调用：从 token/session 解析 `user_id`（可信）
- 绑定规则：`user_id` 不出现在请求体；只出现在服务端上下文

v0 开发态（本仓库实现）用 `X-User-Id` 头模拟“已绑定 user_id”的可信上下文；生产环境应替换为真实鉴权。

输出（对齐 `docs/recall-agent-playbook.md`）：

```json
{
  "memory_view": { "preferences": [], "profile": [], "constraints": [] },
  "evidence": [{ "message_id": "m_...", "ts": "...", "role": "user", "content": "..." }],
  "limits": { "time_range": { "since": "...", "until": "..." }, "role": "user", "messages_considered": "unknown" }
}
```

## 4. Embedding 生成与回填（推荐异步）

为避免 Ingest 阻塞，推荐异步生成 embedding：

- Ingest 写入 `messages` 后，将 `message_id` 推入队列（Redis/DB outbox/任务表均可）
- worker 拉取任务，调用 embedding 模型生成向量，写入 `message_embeddings`
- Query `semantic_search` 只检索已有向量的消息；若向量尚未生成，Recall 可临时退化为 lexical/messages_list

v0 快速验证链路可先：

- 同步生成 embedding（只用于 demo/小流量），后续再替换为 worker

## 5. Recall Service 的“绑定 user 工具”实现方式（关键安全点）

Recall 运行时对 Agent 暴露的工具签名必须**不含 `user_id`**，并在服务端强制注入绑定 user：

- `messages_list(since?, until?, role?, page_size?, cursor?)`
- `lexical_search(query_text, filter?, page_size?, cursor?)`
- `semantic_search(query_text|query_embedding, filter?, top_k?, min_score?)`
- `neighbors(message_id, before?, after?)`

实现要点：

- 工具层无论收到什么参数，都以服务端绑定的 `user_id` 为准
- 对“夹带的 user_id”直接忽略/覆盖

## 6. Agent 闭环实现（对齐 playbook）

实现为一个有限状态机/循环（可先不做多 agent）：

- **State**：`hypotheses[]`、`limits{time_range, role}`、`evidence[]`、`queries_tried[]`、`budget`
- **Decide**：4.1/4.2/4.3 生成本轮动作（lexical/semantic/messages_list/neighbors）
- **Observe**：读取命中与上下文，形成“可核对证据候选”
- **Reflect**：按 4.4 自检；冲突按 4.6.1/4.6.2/4.6.3 归因与裁决
- **Stop**：满足 4.5 则合成输出（第 7 节）

预算建议：

- `max_tool_calls`（例如 6~12）
- `max_time_range_expand`（例如最多扩大 2 次）
- `max_neighbors_calls`（例如每轮最多 2 次，避免刷上下文）

## 7. 合成与证据契约（Synthesis）

合成目标：**最小化、可追溯**。

- 只输出与当前 question 直接相关的要点
- 每条要点必须能引用 `evidence`（至少 1 条 message 原文片段）
- `limits` 必须明确说明本次覆盖范围（time_range、role 等）
- 冲突：
  - 能裁决：输出裁决结论 + 证据
  - 不能裁决：并列可能性 + 双方证据 + 不确定性标注

## 8. FastAPI 工程结构（建议）

```
app/
  main.py
  api/
    ingest.py
    query.py
    recall.py
  auth/
    user.py        # 终端用户鉴权：解析 user_id
    service.py     # ingest 鉴权：API key/mTLS
  db/
    session.py
    models.py
    migrations/    # alembic
  retrieval/
    messages.py
    lexical.py
    semantic.py
    neighbors.py
  agent/
    state.py
    loop.py
skills/
  recall/
    SKILL.md
    prompts/
      system.txt
```

## 9. 可观测与测试（v0 最小集）

可观测：

- 每次 recall 记录：tool 调用次数、各调用参数（脱敏）、命中数量、停止原因、冲突类型、evidence 数
- 慢查询日志：`lexical_search` / `semantic_search` / `neighbors` 的耗时与 top_k

测试：

- cursor seek 分页一致性（无重复/无漏）
- neighbors 边界（首/尾消息）
- role/time_range 过滤正确性
- recall 端到端金样例（给定 messages，预期 memory_view+evidence+limits）
