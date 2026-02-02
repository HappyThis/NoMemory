# NoMemory

NoMemory 是一个“记忆基础设施”：**只存原始事件（evidence）**，不持久化“现成的记忆结论”；在需要时通过“回忆（recall）”按场景合成可用的记忆视图（memory view）。

## 技术方案版本

- **v0：Chat 用户记忆**（本仓库当前方案）：围绕“用户-消息”构建可检索的对话证据库，并由回忆 Agent 在查询结果之上合成用户记忆视图。
- **v0：检索 skill 生成器（retrieval-skill-creator）**：在不改 Query API 的前提下，为不同回忆场景生成回忆 skill（标准 Agent Skill，见 `docs/retrieval-skill-creator.md`）。

## 核心理念

- **Evidence-first**：事件是唯一权威来源；一切“记忆结论”都可重建、可失效、可追溯。
- **Synthesize on demand**：记忆在读取时由回忆 Agent 合成，而不是写入时抽取固化。
- **Pluggable by Skills**：场景差异通过 Skills（策略/提示词/工具编排/校验器）完成“微调”。

## 三层架构

1. **事件层（Event Log）**
   - 只负责可靠地写入/存储/检索“发生过的原始事件”
   - v0（chat）中事件以对话消息为核心：一个 `user_id` 对应一条按时间排序的消息序列

2. **查询层（Query Layer / Recall API）**
   - 在事件之上提供多维检索能力（可组合）
   - 典型维度：时间窗、关键词检索、语义检索、邻域上下文等

3. **适配层（回忆 Agent + Skills）**
   - 回忆 Agent = Agent + 可加载的回忆 skill（标准 Agent Skill）
   - 回忆 Agent 使用查询层工具进行多轮检索与重写查询
   - 将召回的事件“合成”为一次性记忆视图（memory view），供调用方选择是否采纳
   - 不固化“裁决结果”：回忆时可通过继续检索尽力裁决，但不把结论写回为长期事实（除非调用方另行存储）

## 关键术语

- **Message（消息）**：chat 场景的一条对话消息（用户/助手/system）。
- **Query（查询）**：对消息集合的检索请求（时间/关键词/语义/过滤/分页；顺序通常由接口固定定义）。
- **Recall（回忆）**：基于查询结果，由回忆 Agent 合成“记忆视图”的过程。
- **Memory View（记忆视图）**：一次性生成的“当前可用记忆”（例如：用户偏好/背景），应携带可追溯引用（`message_id`）。
- **Skill（技能）**：一组可配置策略与工具编排，用于约束/提升回忆效果（召回范围、查询改写、输出格式、引用要求、覆盖范围声明等）。
  - 在本文档体系里，Skill 以“标准 Agent Skill”的形式存在（Skill 目录 + `SKILL.md`）。

## 事件模型（最小建议）

> 下面是概念性字段，具体可按你的实现裁剪/扩展。

v0（chat 消息）建议字段：

- `message_id`：消息唯一标识
- `ts`：消息时间戳
- `user_id`：用户标识
- `role`：`user` / `assistant` / `system`
- `content`：消息正文
- `embedding`：可选向量索引（派生物，允许存储）

## 文档（Docs）

- `docs/README.md`：文档目录与阅读顺序
- `docs/query-api.md`：v0（chat）查询接口（范围读取 / lexical_search / semantic_search / neighbors 等）
- `docs/recall-service.md`：v0（组件）检索服务（绑定 user_id 并向 Agent 暴露不含 user_id 的工具）
- `docs/recall-agent-playbook.md`：v0（chat）回忆 Agent 技术方案（如何检索与合成）
- `docs/retrieval-skill-creator.md`：v0（组件）creator skill 设计（生成回忆 skill）

## Skills 能做什么

Skills 的目标是“同一份事件数据，用不同策略生成不同的记忆视图”。常见能力：

- **召回范围**：默认时间窗、是否只看 `role=user`
- **检索编排**：多轮查询、查询重写（更好的 `query_text`）、必要时补 `neighbors`
- **输出约束**：固定结构、必须给出 `message_id` 引用、声明覆盖范围（limits）

## 非目标（Non-goals）

- 不把“总结后的记忆”作为系统内的事实来源长期保存
- 不在存储层固化“真值”（回忆时可通过继续检索尽力裁决，但不把裁决结果写回为长期事实）
- 不强制规定调用方如何缓存/是否回忆（NoMemory 提供原语与契约）

## 状态

该仓库提供 v0 技术方案文档，并包含一个可跑的 FastAPI + Postgres(pgvector) MVP 骨架（Ingest/Query/Recall）。

## Quickstart（本地跑起来）

0) 一键启动（推荐）

```bash
chmod +x ./scripts/dev-up.sh
./scripts/dev-up.sh
```

默认端口为 `8001`（可用环境变量覆盖：`PORT=8000 ./scripts/dev-up.sh`）。

1) 手动启动（可选）

1.1) 启动数据库（Docker）

```bash
docker compose up -d db
```

1.2) 配置环境变量

- 复制 `.env.example` 为 `.env` 并填写 `BIGMODEL_API_KEY`（若你要启用 embedding/LLM）

1.3) 安装依赖并迁移（示例以 `uv` 为例）

```bash
uv sync
uv run alembic upgrade head
```

1.4) 启动服务

```bash
uv run uvicorn app.main:app --reload
```

1.5) 写入消息（唯一写入接口）

- `POST /v1/users/{user_id}/messages:batch`

1.6) 回忆

- `POST /v1/recall`，带 `X-User-Id: <user_id>`
