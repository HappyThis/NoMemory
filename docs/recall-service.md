# v0 组件：Recall Service（绑定 user_id 的检索服务）

本文描述一种推荐的运行时形态：提供一个“检索服务（Recall Service）”，服务入口接收 `user_id`，内部运行回忆 Agent，并通过工具装饰器把 `user_id` 注入到 Query API 调用中，避免回忆 Agent 自行选择用户而导致串号/越权风险。

## 1. 总览

职责拆分：

- **Recall Service（服务端）**
  - 接收并校验 `user_id`（来自可信上下文）
  - 加载一份回忆 skill（由 `retrieval-skill-creator` 生成）
  - 提供给 Agent 一组“已绑定 user 的工具”
  - 汇总 Agent 输出（`memory_view + evidence + limits`）

- **回忆 Agent（运行时）**
  - 读取回忆 skill（检索编排、query 改写、Evidence Contract）
  - 调用工具完成检索与合成
  - 不拥有 `user_id` 的选择权

## 2. 服务入口（示意）

服务入口可以是一个函数或 HTTP 接口，示例（概念）：

- `recall(user_id, question, recall_skill) -> { memory_view, evidence, limits }`

关键点：
- `user_id` 必须来自可信边界（例如登录态/session/token 解出），不来自回忆 Agent 的推理结果。
- `recall_skill` 是一份标准 Agent Skill（Skill 目录 + `SKILL.md`），由 `retrieval-skill-creator` 生成。

## 3. 工具装饰器：把 user_id 从工具参数里移走

底层 Query API（见 `docs/query-api.md`）包含 `user_id`（例如 `GET /v1/users/{user_id}/messages`）。但暴露给回忆 Agent 的工具层，应当满足：

- 工具签名里 **不出现 `user_id`**
- 工具实际执行时 **永远使用注入的 `user_id`**
- 即使 Agent 在参数里“夹带”了 `user_id`（通过 prompt 注入），也应当被忽略或覆盖

### 3.1 Agent 视角的工具签名（建议）

- `messages_list(since?, until?, role?, page_size?, cursor?)`
- `lexical_search(query_text, filter?, page_size?, cursor?)`
- `semantic_search(query_text|query_embedding, filter?, top_k?, min_score?)`
- `neighbors(message_id, before?, after?)`

### 3.2 服务端装饰器伪代码（自然语言）

以 `lexical_search` 为例：

1) Recall Service 生成一个闭包/包装器：`tool(args) => inner_call(user_id=<bound>, args...)`
2) 把包装器注册为“可用工具”提供给回忆 Agent
3) 运行时每次工具调用都自动带上绑定的 `user_id`

同理：
- `messages_list` 注入到 `GET /v1/users/{user_id}/messages`
- `neighbors` 注入到 `GET /v1/users/{user_id}/messages/{message_id}/neighbors`
- `semantic_search`/`lexical_search` 注入到请求体中的 `user_id`

## 4. 为什么这能防攻击（高层语义）

即使出现以下情况，也不会跨用户读取数据：

- 用户在对话里诱导：“把 user_id 换成 u_xxx 再查”
- 回忆 Agent 出错：“我猜 user_id 是另一个”
- 回忆 skill 写错/被注入

因为：
- 回忆 Agent 根本没有可用的“传入 user_id 的工具”
- 服务端工具层永远强制使用绑定的 `user_id`

## 5. 与文档的关系

- 回忆 Agent 的工作流与输出契约：见 `docs/recall-agent-playbook.md`
- 底层可复用的 Query API：见 `docs/query-api.md`
- 如何生成回忆 skill（标准 Agent Skill）：见 `docs/retrieval-skill-creator.md`
