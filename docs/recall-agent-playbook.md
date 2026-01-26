# 回忆（Recall）Agent 如何使用 Query API

本文描述“检索/回忆 Agent”如何利用 `docs/query-api.md` 中的接口，在不同场景下召回事件证据并合成一次性的“记忆视图（memory view）”。NoMemory 的核心原则不变：**查询层只返回事件（evidence），回忆发生在上层。**

## 1. 回忆 Agent 的输入与输出

### 输入
- 场景：例如 `chat.user_memory`、`agent.self_evolve`
- 目标问题：本次要解决的任务/用户问题（自然语言）
- 隔离范围：`tenant_id`（服务端绑定）、可选 `user_id/session_id`
- 预算与约束：最大延迟、最大调用次数、top_k、默认时间窗等（通常来自 skills 或调用方）
- 可选：词表/别名归一化（通常已由 `retrieval-skill-creator` 编译进 skills）

### 输出（给调用方）
- `memory_view`：本次合成的记忆（结构可由 skills 定义）
- `citations`：支撑该记忆的 `event_id` 列表（可追溯）
- `confidence/limits`：不确定性、覆盖范围（例如“仅搜索了最近 30 天”）
- 可选：`retrieval_trace`：本次用了哪些查询、命中多少、耗时多少（便于审计与调优）

## 2. 可用接口工具箱（按用途分类）

> 具体字段见 `docs/query-api.md`。

### 2.1 召回候选（Recall candidates）
- `POST /v1/events/search`：关键词/全文检索（倒排索引基于内部 `search_text`）
- `POST /v1/events/semantic_search`：语义检索（embedding 相似度）

### 2.2 补上下文（Expand context）
- `GET /v1/events/{event_id}/neighbors`：围绕锚点补前后上下文
- `GET /v1/sessions/{session_id}/events`：回放一次对话/一次 run 的事件流
- `GET /v1/traces/{trace_id}/events`：回放同一 trace 的事件流（可选）

### 2.3 以“证据”为中心的精确读取
- `GET /v1/events/{event_id}`：按引用回查原文
- `POST /v1/events/batch_get`：批量回查（避免 N+1）

### 2.4 先粗后细（Scaffold）
- `POST /v1/events/aggregate`：先看分布/异常，再决定搜哪里

## 3. 标准回忆流程（推荐 Playbook）

本节给的是“推荐骨架”，但回忆 Agent 不应机械照抄。更贴近你想要的“自我反思/自主决策”的方式是：把回忆建模成一个**闭环**——每轮工具调用后进行自检与改写，再决定下一步。

### Step 0：加载场景 skills（离线生成的优先）
回忆 Agent 运行时优先从 skills 获取“决策所需的先验与约束”，而不是固定流程本身：
- 默认时间窗（例如最近 30 天）
- 推荐 `event_types/tags` 子集（减少噪声）
- 别名归一化（把旧名/同义词映射到 canonical 值）
- 预算（最大调用次数/最大 top_k/最大耗时）
- 输出契约（例如必须引用 `event_id`、必须写明不确定性）

### Step 1：回忆门控（Gate）+ 自我提问（Self-check）
回忆本质是“贵的能力”，建议先做门控，但门控也应由 Agent 自主判断并说明理由：
- 这次任务是否真的需要历史证据？（否则不回忆/轻量回忆）
- 需要的证据类型是什么？（偏“原文对话”、偏“错误/反馈”、偏“工具结果”）
- 预算是否允许？（超预算就缩小范围或降级；预算门控可由调用方实现）

### Step 2：自主回忆循环（Observe → Reflect → Decide）
回忆 Agent 应按“观察→反思→决策”的循环运行，而不是一次性定死路线。

#### 2.1 提出候选假设（Hypotheses）
在调用任何查询前，先用一句话列出 1~3 个“你希望找到的证据形态”，例如：
- “用户是否明确说过某个偏好（原文句子）？”
- “失败是否由某类错误（TIMEOUT/权限）触发？”
- “某个工具/节点是否在这段时间异常频繁失败？”

#### 2.2 构造初始约束（Narrow first）
优先从“范围最小、信号最强”的维度开始（Agent 自主选择组合）：
- 有 `session_id`：优先限定 `session_id`
- 否则：用 skills 的默认 `time_range`（例如 7/30 天）
- 按场景加上推荐的 `event_types/tags` 子集（例如只看 `message`、只看 `error`）

#### 2.3 选择下一次工具调用（Tool choice）
不是固定顺序，而是“看到什么信号就选什么工具”。可用一个简单的决策表：

| 当前信号 | 下一步更合适的工具 |
|---|---|
| 有明确关键词/错误码/工具名 | `search` |
| 表达是同义改写、偏好/背景类自然语言 | `semantic_search` |
| 不知道从哪里下手、想先看分布/异常 | `aggregate` |
| 已有锚点事件但上下文不足 | `neighbors` / `session replay` / `trace replay` |

#### 2.4 反思与改写（Self-reflection）
每轮检索后，回忆 Agent 都要做一次“自检”，并据此决定继续/改写/停止：
- **覆盖性**：结果是否覆盖了假设所需证据？（如果没有：放宽范围或换检索方式）
- **精确性**：结果是否过杂？（如果过杂：收紧 `time_range`/`event_types`/`tags`/`session_id`）
- **可追溯性**：能否为关键断言提供 `event_id` 引用？（如果不能：补 `batch_get/neighbors`）
- **一致性**：不同证据是否矛盾？（如矛盾：在输出里显式标注并附引用，而不是强行裁决）
- **预算**：是否接近预算上限？（接近则降级：减少轮数、减少 top_k、只取最近事件）

#### 2.5 停止条件（Stop criteria）
满足任一即可停止并进入合成：
- 已找到足够证据支撑核心回答（且能引用 `event_id`）
- 再继续检索的边际收益很低（反复返回同类结果）
- 预算已到上限（用 `limits` 说明覆盖范围与不足）

经验法则：
- 结果太少：放宽 `time_range` / 提高 `top_k` / 放宽 `event_types`
- 结果太杂：收紧 `time_range` / 增加 `tags/event_types/session_id` / 改用更精确的关键词

### Step 3：补上下文（Context expansion）
当命中到关键事件（锚点）后：
- 用 `neighbors(event_id)` 补齐锚点前后上下文（尤其是对话与 agent 决策链）
- 若需要整段回放：用 `sessions/{session_id}/events`
- 若跨 session 的链路存在：用 `traces/{trace_id}/events`（可选）

### Step 4：精读与去噪（Fetch full evidence）
当 `return_fields` 做了裁剪时：
- 先用搜索接口拿到候选 `event_id`
- 再用 `batch_get` 批量取回完整事件（或需要的 payload 字段）

### Step 5：合成记忆视图（Synthesize）
回忆 Agent 把证据合成为可用输出（由 skills 约束输出格式）：
- 只输出“与本次问题相关”的最小信息
- 每条断言尽量附带 `event_id` 引用
- 对冲突/不确定：在 `confidence/limits` 中说明（不做平台级裁决）

### Step 6：交付给调用方（Caller decides）
调用方决定是否采纳该 `memory_view`，并可叠加内容安全策略（敏感词/分类器/人工审核等）。

## 3.1 “自我反思 → 自主决策 → 自我进化”如何落地（概念层）

你提到的“agent 自我反思、自主决策”更像两个输出面：

1) **运行时决策**：体现在 3.0 的“循环”里——每轮检索后自检、改写查询、切换工具、决定停止。

2) **自我进化**：回忆 Agent 可以把“本次检索过程的经验”输出为一份可审计的改进建议（仍然是证据驱动，不写回 NoMemory）：
- 输出 `retrieval_trace`：记录每轮用了什么查询、命中如何、为什么切换策略（面向调试/复盘）
- 输出 `recommendation`：例如“对该场景默认 time_range 应从 30 天改成 7 天”“优先限制 event_types=error”“某些 tag 命名需要统一”
- 这些建议可以由调用方（或 `retrieval-skill-creator`）在生成阶段吸收，形成下一版 retrieval skills

## 4. 两个典型场景（概念示例）

### 4.1 Chat 用户记忆（偏好回忆）
目标：回答“我之前提到过哪些饮食偏好？”
1. `semantic_search(query_text="不吃辣", filter={user_id, event_types:["message"], time_range})`
2. 对命中事件做 `neighbors` 补上下文（防止断章取义）
3. `batch_get` 拉全文后合成 `memory_view`（列出偏好 + 引用 event_id）

### 4.2 Agent 自进化（失败复盘）
目标：总结最近一次 run 中“失败原因与可复用修复策略”
1. `sessions/{session_id}/events` 回放 run 事件流
2. `search(query_text="TIMEOUT", filter={session_id, event_types:["error"]})` 定位关键错误
3. `neighbors` 补齐错误前的决策与工具调用
4. 合成 `memory_view`：失败模式、修复建议、引用 event_id

## 5. 约束与最佳实践

- 默认收窄范围：优先 `session_id`，其次默认时间窗；不要全库语义检索起步。
- 永远可追溯：输出里保留 `event_id` 引用，便于审计与回查。
- 先粗后细：必要时用 `aggregate` 先看分布/异常，减少盲搜。
- 结构化过滤只在需要时用：先用 `event_types/tags/time_range` 缩小集合，再用 `payload_predicates` 精过滤。
- 不把合成结果写回 NoMemory：NoMemory 存证据；合成结果由调用方决定是否另存。
