# v0 技术方案：Chat 用户记忆回忆 Agent

本文给出 v0 方案中“回忆 Agent”的独立技术方案：如何用 `docs/query-api.md` 的接口召回消息证据，并合成用户记忆视图（memory view）。

回忆 Agent 可以理解为两部分的组合：
- 一个可运行工具调用的 Agent
- 一份由 `docs/retrieval-skill-creator.md` 生成并加载的**回忆 skill**（标准 Agent Skill），用于约束检索编排、query 改写、证据契约等

## 1. 设计目标

- 召回结果以“证据消息”为中心：输出里包含可核对的原文证据。
- Agent 具备自主检索能力：每轮检索后自检并改写下一步查询。
- 优先减少噪声：优先限定 `time_range`；做用户偏好/画像时优先只看 `role=user`，避免混入助手复述。

## 2. 输入与输出契约

### 2.1 输入（Input）

- `user_id`
- `question`：本次要回答的问题/任务（自然语言）
- `context`（可选）
  - `time_range`：默认时间窗（例如最近 30 天；若不提供则使用回忆 skill 的默认值）
  - `role_pref`：例如偏好只看 `role=user`（若不提供则使用回忆 skill 的默认值）

### 2.2 输出（Output）

建议输出结构（示意）：

```json
{
  "memory_view": {
    "preferences": [],
    "profile": [],
    "constraints": []
  },
  "evidence": [],
  "limits": {
    "time_range": {"since":"...","until":"..."},
    "role": "user",
    "messages_considered": "unknown"
  }
}
```

约束：
- `evidence` 至少包含支撑主要结论的若干条消息原文（可按需要截断长度，但必须保留可核对的内容）。
- `limits` 必须说明本次回忆覆盖范围（例如是否只看 `role=user`、时间窗是否收窄）。

## 3. 工具集（Query API）

- 范围读取：`GET /v1/users/{user_id}/messages`
- 关键词检索：`POST /v1/messages/lexical_search`
- 语义检索：`POST /v1/messages/semantic_search`
- 邻域上下文：`GET /v1/users/{user_id}/messages/{message_id}/neighbors`

## 4. 自主检索闭环（Observe → Reflect → Decide）

回忆 Agent 的核心不是固定流程，而是闭环决策。

### 4.1 假设（Hypotheses）
先把问题拆成 1~3 个“可被证据支持”的假设：
- 偏好类：饮食/作息/语言/格式偏好
- 背景类：地点/职业/项目/长期目标
- 约束类：禁止做什么、必须怎么做

用户输入（本轮问题/任务）与假设的关系示例：

| 用户输入（question） | 可能的假设（1~3 条） | 直觉上的首选检索 |
|---|---|---|
| “你记得我有什么饮食偏好吗？” | 1) 用户明确提过忌口/过敏/不吃某类食物；2) 偏好可能出现在最近一段时间的点餐/聊天里 | 先 `semantic_search`（同义表达多），必要时再 `lexical_search`（如“不吃辣/过敏/忌口”） |
| “我是不是说过我不吃辣？” | 1) 存在用户原话提到“不吃辣/少辣/怕辣”等；2) 可能伴随具体场景（点菜/聚餐） | 先 `lexical_search`（命中短语最快），命中后用 `neighbors` 校验语境 |
| “按我喜欢的格式总结一下上面的内容” | 1) 用户曾要求过固定输出格式（要点/表格/先结论后细节/中英双语等）；2) 偏好可能在较久以前出现 | 先 `semantic_search`（“格式/写法/输出/风格”等），再用 `GET .../messages` 扩大时间窗回溯 |
| “你还记得我在哪里工作/做什么职业吗？” | 1) 用户曾自述职业/公司/行业；2) 信息可能散落在自我介绍或项目讨论中 | 先 `semantic_search`（表达多样），对高分命中用 `neighbors` 获取完整句子 |
| “我们上次聊到的那个项目进展怎样了？” | 1) 存在“项目 A”的连续讨论；2) 需要找到最近一次提及该项目的上下文串 | 先 `semantic_search`（“上次/那个项目/进展/需求”等），再对关键命中用 `neighbors` 扩展上下文 |
| “我最近有什么重要日程/要办的事？” | 1) 用户在近段时间提到过待办/日程/截止日期；2) 时间线比语义更关键 | 先 `GET .../messages` 做时间范围扫读（例如最近 7/30 天），再在命中的片段上做 `lexical_search` 补关键词（如“截止/会议/预约/DDL”） |

### 4.2 收窄（Narrow）
收窄的目标：在不牺牲召回率的前提下，尽快把候选集压到“可读、可核对”的规模。

**(1) 用时间窗先砍一刀**

从 `question` 里抽取时间信号，映射到 `time_range`：
- 明确“最近/这周/这两天/刚才” → 先用短窗（例如 2~7 天）
- 明确“上次/之前提到过/我们聊过” → 先用中窗（例如 14~30 天）
- 明确“长期偏好/一直/习惯/从小/多年” → 直接用长窗（例如 180 天），或先中窗不够再扩
- 明确具体日期/月份 → 直接定位到对应区间

请求形态（范围读取）示例：

```http
GET /v1/users/u_12345/messages?since=2026-01-01T00:00:00Z&until=2026-02-01T00:00:00Z&role=user&page_size=100
```

**(2) 用 `role` 过滤去掉“复述噪声”**

默认规则（不强制，但通常有效）：
- 做“用户偏好/画像/个人信息” → 优先 `role=user`
- 做“助手承诺/工具输出/系统指令导致的行为” → 需要看 `role=assistant/system`（此时不要锁死 `role=user`）

**(3) 先找锚点，再扩上下文**

当问题指向“某句话/某次讨论”时，最稳的方式是：
1) 先用 `lexical_search` 或 `semantic_search` 找到 1~3 条锚点消息
2) 对锚点用 `neighbors` 拉上下文，避免断章取义

示例（先短语命中，再取上下文）：

```json
POST /v1/messages/lexical_search
{
  "user_id": "u_12345",
  "filter": { "role": "user", "time_range": { "since": "2026-01-01T00:00:00Z" } },
  "query_text": "\"不吃辣\""
}
```

```http
GET /v1/users/u_12345/messages/m_01HTZ2K8J9Q1ZQ4QYQ8M9J7P6A/neighbors?before=10&after=0
```

**(4) “太多/太少”两种情况的收窄动作**

- 命中太多（噪声大）
  - 缩短 `time_range`（30 天 → 7 天）
  - 加强 `query_text` 约束（加引号、加 AND 关键词）
  - 锁定 `role=user`（若场景允许）
- 命中太少（召回不足）
  - 扩大 `time_range`（7 天 → 30/180 天）
  - 从 `lexical_search` 切到 `semantic_search`（同义改写多时）
  - 去掉 `role` 限定（用户可能“让助手转述/总结”导致线索在 assistant 侧）

### 4.3 选择检索方式（Choose）

| 场景信号 | 优先工具 |
|---|---|
| 有明确关键词/短语（人名、地点、菜名、固定表达） | `lexical_search` |
| 同义改写明显（偏好/背景自然语言） | `semantic_search` |

### 4.4 自检与改写（Reflect）
每轮检索后执行自检，并决定下一步：
- **相关性**：命中是否与假设一致？（否→改 query_text 或更换检索方式）
- **上下文**：是否存在断章取义风险？（是→对关键命中用 `neighbors`）
- **覆盖性**：是否遗漏明显线索？（是→扩大时间窗或放宽 `role` 限定）
- **重复性**：是否反复命中同一组消息？（是→停止或切换策略）

### 4.5 停止条件（Stop）
满足任一即可停止检索并进入合成：
- 已有足够证据支撑主要结论（且可引用）
- 继续检索的边际收益很低
- 继续检索会明显引入噪声

## 5. 合成策略（Synthesis）

合成的目标是“可用且可追溯”，而不是“全量总结”：

- **只提取与当前问题相关的最小信息**
- **每条要点带证据**：附 1~N 条原文消息片段（必要时补上邻域上下文）
- **冲突不裁决**：若证据矛盾，输出两种说法并各自引用

## 6. 示例：回忆饮食偏好

目标问题：`“你记得我有什么饮食偏好？”`

1) 假设：用户说过“不吃辣/忌口/过敏”等  
2) 收窄：`role=user` + `time_range=最近30天`  
3) 语义检索：`POST /v1/messages/semantic_search`，`query_text="不吃辣 忌口 过敏 饮食偏好"`  
4) 对 top 命中做 `neighbors`，确认是否为用户原话且语境为真实偏好  
5) 将命中的原文作为证据输出，合成：
- `memory_view.preferences=["不吃辣"]`
- `evidence=[{"role":"user","content":"..."}]`
- `limits.role="user"`
