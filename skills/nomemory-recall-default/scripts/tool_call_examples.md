# 工具调用示例（Planner JSON）

本文件用于集中存放工具调用的“输出形态示例”。当你需要表达“下一步要调用哪个工具、参数是什么”时，按以下格式输出（只输出 JSON，不要附带解释性文本）。

通用结构：

```json
{
  "stop": false,
  "tool": "<tool_name>",
  "args": { }
}
```

约束：
- `tool` 必须是以下之一：`messages_list`、`lexical_search`、`semantic_search`、`neighbors`
- `args` 里禁止出现 `user_id`
- 时间使用 ISO8601（允许 `Z`）
- 若工具返回 `next_cursor`，下一次分页应把它原样放到 `args.cursor` 继续翻页

## 1) 范围读取：messages_list

```json
{
  "stop": false,
  "tool": "messages_list",
  "args": {
    "since": "2026-01-01T00:00:00Z",
    "until": "2026-02-01T00:00:00Z",
    "role": "user",
    "page_size": 100,
    "cursor": null
  }
}
```

## 2) 关键词检索：lexical_search

```json
{
  "stop": false,
  "tool": "lexical_search",
  "args": {
    "query_text": "\"不吃辣\" AND 火锅",
    "filter": { "role": "user", "time_range": { "since": "2026-01-01T00:00:00Z" } },
    "page_size": 50,
    "cursor": null
  }
}
```

## 3) 语义检索：semantic_search

```json
{
  "stop": false,
  "tool": "semantic_search",
  "args": {
    "query_text": "饮食偏好 忌口 过敏 不吃辣",
    "filter": { "role": "user", "time_range": { "since": "2026-01-01T00:00:00Z" } },
    "top_k": 20,
    "min_score": 0.2
  }
}
```

## 4) 邻域上下文：neighbors

```json
{
  "stop": false,
  "tool": "neighbors",
  "args": {
    "message_id": "m_01HTZ2K8J9Q1ZQ4QYQ8M9J7P6A",
    "before": 8,
    "after": 0
  }
}
```
