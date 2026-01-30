# Docs

本目录包含 NoMemory 的版本化技术方案与接口文档。

## 建议阅读顺序

1. `README.md`：总览（理念 + 版本）
2. `docs/fastapi-tech-plan.md`：v0（工程）FastAPI 落地方案（含唯一写入接口）
3. `docs/query-api.md`：v0（chat）Query API
4. `docs/recall-service.md`：v0（组件）检索服务（绑定 user_id 的工具装饰器）
5. `docs/recall-agent-playbook.md`：v0（chat）回忆 Agent 技术方案
6. `docs/retrieval-skill-creator.md`：v0（组件）creator skill 设计

## 文档列表

- `docs/fastapi-tech-plan.md`
  - v0 工程落地方案：FastAPI + Postgres(pgvector)
  - 包含唯一写入接口（Ingest API）与 Recall/Query 的读路径
- `docs/query-api.md`
  - 消息查询接口：范围读取 / `lexical_search` / `semantic_search`
  - 上下文：`neighbors`
- `docs/recall-agent-playbook.md`
  - 回忆 Agent 的自主闭环（假设 → 检索 → 反思 → 补上下文 → 合成与引用）
- `docs/recall-service.md`
  - 检索服务如何绑定 user_id，并向回忆 Agent 暴露不含 user_id 的工具
- `docs/retrieval-skill-creator.md`
  - 如何设计一个 creator skill（标准 Agent Skill），用 Query API 为业务场景生成回忆 skill

## 约定

- **接口返回的是证据**（messages/events），而不是“记忆结论”。
