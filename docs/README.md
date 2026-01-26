# Docs

本目录包含 NoMemory 的概念与接口文档（以“事件证据”为中心，不存放现成记忆结论）。

## 建议阅读顺序

1. `README.md`：项目理念与边界
2. `docs/query-api.md`：查询层接口（如何召回事件证据）
3. `docs/retrieval-skill-creator.md`：`retrieval-skill-creator`（离线生成检索类 skills）

## 文档列表

- `docs/query-api.md`
  - 事件查询接口：`search` / `semantic_search` / `hybrid_search`
  - 上下文与回放：`neighbors` / `session replay` / `trace replay`
  - 统计与可观测：`aggregate` / `estimate` / `explain`
- `docs/retrieval-skill-creator.md`
  - 词表（vocabulary）与别名归一化如何“编译”进 retrieval skills（运行时不额外拉表）

## 约定

- **接口返回的是事件证据**（events），而不是“记忆结论”。
- **硬隔离优先**：任何查询都必须在服务端绑定 `tenant_id`（以及可选 `user_id/session_id`）。
- **字段可控**：哪些字段由写入端/服务端生成、哪些允许客户端提供，应以 `docs/query-api.md` 的“字段所有权”章节为准。

