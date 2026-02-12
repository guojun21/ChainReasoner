# ChainReasoner 痛点优化蓝图（基于 references 项目借鉴 + 全量评测数据驱动）

> 当前成绩：27/100（搜索受限）| 基线：33/100 | 目标：55+ | 原则：正确率高于一切，延迟和资源不敏感
>
> 最近全量评测：2026-02-11，Google+IQS 不可用，仅 Brave+DuckDuckGo

---

## 总览：痛点 × 优先级矩阵（按全量评测数据排序）

| 优先级 | 痛点 | 影响范围 | 状态 | 参考项目 | 核心借鉴 |
|--------|------|----------|------|---------|---------|
| **P0** | 多跳推理链断裂 | 29/52 错误（56%） | 部分已做 | Research_Agent, ReAct, open_deep_research, everything-claude-code, analysis_claude_code | 结构化跳数规划 + 中间校验 + 动态查询 + 迭代检索 + 上下文隔离 + 跳目标聚焦提取 |
| **P0** | 搜索引擎不可用 | Google+IQS 双挂 | 待恢复 | — | 等额度恢复 / 接入替代源（Bright Data, Tavily） |
| **P1** | 格式/语言对齐不足 | 10/52 错误（19%） | 部分已做 | Research_Agent, ReAct | 中英文双输出 + 前缀去除 + 昵称映射 |
| **P1** | 搜索证据不足 | 13/52 错误（25%） | 部分已做 | crawl4ai, firecrawl | 搜索策略多样化 + 学术/媒体特定源 |
| **P2** | 验证环节太弱 | 错误答案未拦截 | 待实施 | Enhancing-MH-QA, Research_Agent | self-verification + 证据回查 |

---

## 已完成项（归档概要）

| 痛点 | 实施内容 |
|------|---------|
| 答案格式不精确 | 7 步后处理管线 + 格式感知 prompt + 格式提示解析器 |
| 网页抓取质量差 | 三层清洗管线（规则 + 密度剪枝 + LLM 精炼）+ 域名感知提取器 |
| LLM 非确定性 | 多候选一致性投票 + Jaccard 相似度聚类 + LLM 仲裁 |

---

## P1-b：多跳推理链断裂

### 现状问题
3-4 跳问题几乎全错，原因：
- query decomposition 质量不稳定
- 第一跳找到的中间实体不够精确 → 第二跳搜偏
- 没有中间结果校验，链条一旦偏了就一路错到底

### 借鉴来源

#### 1. Research_Agent `planning.yaml` → `multi_hop_decomposition`
```yaml
# 最核心的借鉴！结构化输出：
{
  "hop_count": "两跳/三跳及以上",
  "hops": [
    {"hop_num": 1, "target": "xxx", "tool": "xxx", "stop_condition": "获取到xxx信息即停止"},
    {"hop_num": 2, "target": "xxx", "tool": "xxx", "stop_condition": "获取到xxx信息即停止"}
  ],
  "total_stop_condition": "所有跳完成且获取到足够信息即停止"
}
```
关键设计：
- **每跳有明确目标和终止条件**，而不是盲目搜索
- **反推类多跳适配**：结果→原因→前置条件，按逆序拆解
- **跨语言多跳**：中文问题查英文数据源时，先翻译再搜索

#### 2. Research_Agent 执行节点 → 中间结果校验与纠错
```python
# 每跳执行完都验证一次
if hasattr(execution_agent, "validate_hop_result"):
    is_valid, correct_result = execution_agent.validate_hop_result(...)
    if not is_valid:
        correct_result = await execution_agent.correct_hop_result(...)
```

#### 3. ReAct 的 Thought-Action-Observation 循环
```
Thought 1: 我需要找到...
Action 1: Search[RepRap project]
Observation 1: (搜索结果)
Thought 2: 根据结果，我现在需要找...
Action 2: Search[RepRapPro company]
Observation 2: (搜索结果)
...
Action N: Finish[RepRapPro Ltd]
```
关键：每一步都有 Thought（反思当前信息是否足够），而不是机械执行预定计划。

#### 4. open_deep_research 的 think_tool
```
一个"空操作工具"，强制 Agent 在搜索之间停下来思考
→ 把已有信息整合一下，再决定下一步搜什么
→ 防止盲目搜索，提高搜索效率
```

### 落地方案

**A. 结构化多跳规划器（替换现有 decompose_question）**
```
输入：原始问题
输出：
{
  "hop_count": 3,
  "hops": [
    {"hop": 1, "target": "找到RepRap项目的创始人", "query_zh": "...", "query_en": "...", "stop_when": "找到人名"},
    {"hop": 2, "target": "找到该学者创立的商业实体", "query_template": "{人名} commercial company", "stop_when": "找到公司名"},
    {"hop": 3, "target": "验证公司的欧洲和亚洲业务状态", "query_template": "{公司名} ceased trading Europe Asia", "stop_when": "确认时间和地区"}
  ],
  "total_stop": "获取完整公司英文名称"
}
```

**B. 中间结果校验（每跳之后）**
```
hop_1_result = search(...)
# 校验：结果是否回答了 hop_1 的 target？
validation = LLM("hop_1 的目标是'找到人名'。搜索结果是：{result}。是否找到了人名？")
if validation == "否":
    retry_with_different_query(...)
```

**C. 动态查询生成（后跳依赖前跳）**
```
# hop_2 的查询模板中使用 hop_1 的结果
hop_2_query = hop_2.query_template.format(人名=hop_1_result)
```

---

### P0-d/P0-e 实施情况（2026-02-11）

**已完成的改动（3 个子阶段）：**

#### P0-d: 迭代检索 + 合并 LLM 调用 + 上下文隔离

借鉴 `everything-claude-code` 的迭代检索模式和 `analysis_claude_code v3` 的子代理上下文隔离：

- **迭代检索循环**（`_execute_iterative_retrieval_for_single_hop`）：每跳内执行 DISPATCH → EVALUATE → REFINE → LOOP 循环（最多 `MAX_RETRIEVAL_CYCLES_PER_HOP=2` 轮），根据评估反馈精炼查询
- **合并 3 个 LLM 调用为 1 个**（`evaluate_hop_result_and_decide_next_action`）：将 validate_hop + reflection + completion_check 合并为单次 LLM 调用，每跳减少 3-5 次 LLM 开销
- **上下文隔离**（`_build_isolated_hop_context`）：每跳仅获得其目标 + 前跳摘要（截断至 200 字）+ 本跳证据，防止 LLM 注意力被无关信息稀释
- **随机选题测试**：新增 `--random-question` 参数，支持从 100 题中随机选 1 题测试

#### P0-e: 修复中间跳提取 Bug（关键架构修复）

通过分析 Q45 随机测试日志发现的 Bug：

- **问题**：`_execute_iterative_retrieval_for_single_hop` 中 `llm_answer_fn` 始终传入 `question`（原始用户问题），而非 `hop_target`。导致中间跳的 LLM 试图直接回答最终问题，而非提取中间实体。
- **修复**：`extraction_question = question if is_last_hop else hop_target`
  - 中间跳（非最后一跳）：传入 `hop_target`，如 "找到灵感源于元胞自动机的开源硬件项目"
  - 最后一跳：传入原始 `question`，保留格式提示解析能力
- **参考**：`analysis_claude_code v3` 的 `sub_messages = [{"role": "user", "content": prompt}]` — 每个子代理只看到自己的聚焦任务
- **增强日志**：每个 cycle 记录 `extraction_question` 类型（original/hop_target）和提取结果

#### P0-e 目录清理

- 更新 `src/agents/__init__.py`，添加模块分类文档（核心模块 9 个 / legacy 管线 5 个 / 兼容 shim 5 个）

#### 验证结果

**Q45 对比（P0-e 修复前 vs 修复后，不同问题但验证逻辑生效）：**

| 指标 | 修复前（Q45, run_232032） | 修复后（Q0, run_233803） |
|------|--------------------------|-------------------------|
| 中间跳 extract_answer 的 question | 原始问题（Bug!） | hop_target（已修复） |
| Hop 1 提取结果 | "Stamp Stories"（直接猜最终答案） | "Unknown"（正确拒绝，但搜索未命中） |
| Hop 2 提取结果 | "2025 Stamp Yearbook"（又猜） | "Hon Hai"→"Arduino LLC"（搜索偏差） |
| 最终得分 | 0/1 | 0/1 |

**P0-e 修复效果分析：**
- P0-e 修复已**验证生效** — 中间跳现在收到正确的聚焦问题
- Q0 仍得 0 分的根因不是 P0-e，而是：
  1. **搜索证据不足**：中文查询词未能找到 RepRap 英文内容（Google 禁用），Hop 1 返回 "Unknown"
  2. **链式级联失败**：Hop 1 无结果 → Hop 2 盲搜 → 错误实体 "Arduino LLC"
  3. **验证发现了真相但无法利用**：`verify_answer` 返回 REFUTES(0.92)，reasoning 指出应为 "RepRapPro Ltd"，但当前逻辑仅回退到备选答案，不提取验证中的正确实体

**残余问题 & 下一步：**
1. 搜索质量（P0 依赖 Google/IQS 恢复或 BrightData 增强英文搜索覆盖）
2. 验证环节应能从 REFUTES reasoning 中提取正确实体作为修正答案
3. 中文问题应自动生成英文搜索查询（跨语言多跳），目前中文查询 "开源硬件项目 元胞自动机" 搜不到 RepRap

---

## P2-a：IQS 延迟过高

### 现状问题
- IQS 12 次调用中 7 次超过 10 秒，3 次超过 30 秒（最长 47 秒）
- 在 3 worker 并发 + 超时限制下，很多搜索结果还没回来就被截断

### 借鉴来源

#### 1. crawl4ai — 异步浏览器池 + 缓存
```
- 异步并发抓取，不阻塞等待
- 内置缓存层，相同 URL 不重复抓取
```

#### 2. firecrawl — 批量处理 + 并行
```
- 支持 batch 模式批量提交 URL
- 内部自动并行化和重试
```

#### 3. bm25s — 本地检索
```
- 对已抓取的内容建本地 BM25 索引
- 后续查询先查本地缓存，命中就不走网络
```

### 落地方案

**A. 搜索结果缓存**
```
已搜索过的 query → 缓存 response（文件级 JSON 缓存）
同一 run 内如果多跳产生类似 query → 先查缓存
```

**B. 并行化搜索**
```
当前：中文 IQS → 等结果 → 英文 Brave → 等结果
改为：中文 IQS + 英文 Brave 同时发出 → 等最先返回的
```

**C. IQS 超时重试 + 降级**
```
IQS 超时 15s → 重试一次
重试仍超时 → 降级到 DuckDuckGo + 全文抓取
```

**D. 预抓取（Prefetch）**
```
在 hop_1 搜索时，同时预抓取 hop_1 结果中 top URL 的全文
→ hop_2 需要深入内容时，已经有缓存了
```

---

## P2-b：验证环节太弱

### 现状问题
- verify_answer 只做了 5 次（每题 1 次）
- Q4 验证出 `INSUFFICIENT|0.2` 后仍然提交了错误答案
- 没有"拒绝回答"机制：即使验证不通过也强行输出

### 借鉴来源

#### 1. Enhancing-Multi-Hop-QA `self_verification.py`
```python
# 最直接可用的验证框架
verdict = verify(question, answer, evidence_snippets)
# SUPPORTS → 保留答案
# REFUTES → 重新搜索/推理
# INSUFFICIENT → 追加搜索，补充证据

def should_abstain(result):
    if result.label in [INSUFFICIENT, ABSTAIN]:
        return True
    if result.confidence < threshold:
        return True
    return False
```

#### 2. Enhancing-Multi-Hop-QA 的验证 prompt
```
VERIFICATION_SYSTEM_PROMPT:
- SUPPORTS: 证据明确支持答案
- REFUTES: 证据与答案矛盾
- INSUFFICIENT: 证据不足以判断
+ confidence score 0-1
+ brief reasoning
```

#### 3. Research_Agent `toolhub.yaml` → 多工具结果冲突检测
```yaml
# "如果工具结果有冲突，请指出并说明"
# 多源搜索结果不一致时，不是随机选一个，而是标注冲突
```

### 落地方案

**A. 强制验证（每个答案必须过验证）**
```
answer = extract_answer(evidence)
verification = verify(question, answer, evidence)
if verification.verdict == "REFUTES":
    answer = re_search_and_extract(different_queries)
elif verification.verdict == "INSUFFICIENT":
    answer = supplement_search_and_extract(more_queries)
# 只有 SUPPORTS 才最终输出
```

**B. 多源交叉验证**
```
answer_from_search = extract_answer(search_evidence)
answer_from_knowledge = knowledge_answer(question)
if answer_from_search == answer_from_knowledge:
    confidence = HIGH → 直接输出
elif answer_from_search != answer_from_knowledge:
    trigger_verification_search(answer_from_search)
```

**C. 验证失败后的重试策略**
```
attempt = 0
while attempt < 3:
    answer = generate_answer()
    if verify(answer).verdict == "SUPPORTS":
        return answer
    attempt += 1
    # 每次重试用不同的搜索查询
return best_answer_so_far  # 退而求其次
```

---

## P1-c：推理过程暂存缺失（跨跳证据丢失、LLM 无法回溯前序结果）

### 现状问题
当前 `ConstrainedMultiHopSearchAgent.answer()` 的中间结果全靠**局部变量**传递：
- `knowledge_answer`、`search_answer`、`hop2_answer` 都是裸字符串，互相看不到对方的证据来源
- `reasoning_steps: List[str]` 是只写不读的 append-only 列表——纯粹为 post-mortem 日志，推理过程中不回读
- `search_traces` 同理，只写不读
- hop-2 只拿到 hop-1 的答案字符串，**看不到支撑该答案的原始证据和搜索结果**
- LLM 做 verify 时没有完整证据链可回溯
- 没有 per-question 的文件目录，事后无法分析某道题的推理过程

**直接后果**：多跳推理无法"翻看前面抓到的文章"，每一跳都是从零开始，前序信息全部丢失。

### 借鉴来源

#### 1. Researcher (zlb22) — 文件系统级暂存（完全匹配）

```
每次推理 → 创建独立工作目录
  ├── raw/              ← sub-agent 抓取的原始网页/文档
  ├── summaries/        ← 每个文档的摘要
  ├── INDEX.md          ← 文件目录索引（agent 靠这个知道有什么资料）
  └── FINAL_REPORT.md   ← 最终报告
```

核心模式：sub-agent 把长内容写入文件 → 返回"摘要 + 文件路径"给主 agent → 主 agent 需要详情时按需 `read_file`。

**关键洞察**：主 agent 的上下文窗口里只有摘要和索引，不会被原始内容撑爆；需要详情时精准读取。

#### 2. deepagents (LangChain, ★9.2k) — Agent 虚拟文件系统

```python
# agent 可用的工具集
tools = [
    ls(workspace_dir),          # 列出工作区文件
    read_file(path),            # 读取文件内容
    write_file(path, content),  # 写入文件
    edit_file(path, edits),     # 编辑文件
    glob(pattern),              # 模式匹配搜索
    grep(pattern, path),        # 内容搜索
]
```

核心模式：agent 拥有一个 per-session 的虚拟文件系统。大结果（搜索结果、网页全文）自动 offload 到文件，LLM 上下文只保留引用。Agent 通过 `ls` + `grep` 自主发现和检索已有资料。

**关键洞察**：大工具返回值自动 eviction — 超过阈值的结果不塞进对话历史，而是写入文件并在对话中留一个 `[Result saved to workspace/search_result_001.md]` 引用。

#### 3. A-mem (★786, NeurIPS 2025) — Zettelkasten 卡片索引

```python
class MemoryNote:
    content: str              # 证据原文
    keywords: List[str]       # 关键词标签
    context: str              # 来源上下文（哪个查询、哪一跳产生的）
    links: List[str]          # 与其他卡片的语义链接
    created_at: datetime
```

核心模式：每条证据变成一张"卡片"，卡片之间通过语义相似度自动链接。检索时不是全文搜索，而是沿着卡片网络导航。

**关键洞察**：新证据加入时会触发已有卡片的更新 — 如果新信息补充或修正了旧卡片，旧卡片内容会被更新。这实现了"知识持续积累"而非简单堆叠。

### 落地方案

**A. per-question 推理暂存目录**

```
logs/run_{timestamp}/scratchpad/
  ├── q001/                           ← 题目维度的文件夹
  │   ├── _INDEX.md                   ← 目录索引（agent 入口）
  │   ├── hop1_evidence_001.md        ← hop-1 搜到的第 1 条证据
  │   ├── hop1_evidence_002.md        ← hop-1 搜到的第 2 条证据
  │   ├── hop1_summary.md             ← hop-1 的结论摘要
  │   ├── hop2_evidence_001.md        ← hop-2 搜到的证据
  │   ├── hop2_summary.md             ← hop-2 的结论摘要
  │   └── verification_result.md      ← 验证结果
  ├── q002/
  │   └── ...
```

**B. ReasoningScratchpad dataclass**

```python
@dataclass
class ReasoningScratchpad:
    question_id: str
    question_text: str
    workspace_dir: Path                    # 指向 scratchpad/q{id}/
    
    # 结构化证据链
    hops: List[HopResult]                  # 每跳的证据 + 结论
    current_index: str                     # _INDEX.md 的实时内容
    
    def write_evidence(self, hop: int, evidence: Dict) -> str:
        """写入证据文件，返回文件路径"""
        
    def get_index(self) -> str:
        """返回当前索引（LLM 用这个知道有什么资料）"""
        
    def read_evidence(self, filename: str) -> str:
        """按需读取某条证据的详情"""
        
    def update_index(self):
        """每次写入后自动更新 _INDEX.md"""
```

**C. LLM Prompt 集成**

```
你正在回答一个多跳推理问题。以下是你的工作台索引：

{scratchpad.get_index()}

你已经完成了 {len(scratchpad.hops)} 跳推理。
如果需要查看某条证据的详情，请说明文件名。
现在请根据已有证据，决定下一步行动。
```

---

## P1-d：规划与进度追踪缺失（无法动态调整推理策略、无 TODO 机制）

### 现状问题
当前推理流程是**硬编码的 5 阶段顺序执行**：
```
knowledge → hop1-search → hop2-search → select-best → verify
```

问题：
- **没有可查看的"推理计划"**：agent 不知道自己接下来要做什么，每个阶段都是盲目执行
- **没有逐步 check 机制**：无法追踪哪些步骤完成了、哪些失败了、失败原因是什么
- **无法动态调整**：即使 hop-1 就找到了高置信度答案，也必须走完 hop-2 和 verify
- **无法跳过或重试**：某个搜索超时后没有重试逻辑，也无法跳过该步骤换一个策略

### 借鉴来源

#### 1. LLMCompiler (★1.8k, ICML 2024) — DAG 任务图 + 自动并行

```
Planner:   生成任务 DAG（有向无环图），标注依赖关系
             Task 1: search("Einstein birthday")
             Task 2: search("Einstein Nobel Prize")  [可与 Task 1 并行]
             Task 3: extract(Task 1 + Task 2 结果)   [依赖 Task 1, 2]
             Task 4: verify(Task 3 结果)              [依赖 Task 3]

Dispatcher: 检测哪些任务的依赖已满足 → 立即并行执行
            Task 1 ✓, Task 2 ✓ → Task 3 可执行
            
Executor:   执行单个任务，结果写回 DAG
```

**核心优势**：自动检测可并行的任务，比线性执行快 3.7 倍。任务间通过占位符 `$1`, `$2` 传递结果。

#### 2. ReWOO (★935) — 先规划后执行（Plan-then-Execute）

```
Phase 1 - Plan（1 次 LLM 调用）:
  #P1 = search("Einstein birth year")
  #P2 = search("Nobel Prize Physics " + #E1)     ← #E1 是 #P1 的执行结果的占位符
  #P3 = extract_answer(#E1 + #E2, question)

Phase 2 - Execute（无 LLM，纯工具调用）:
  #E1 = search("Einstein birth year") → "1879"
  #E2 = search("Nobel Prize Physics 1879") → "1921 for photoelectric effect"
  #E3 = extract_answer("1879" + "1921 for photoelectric effect", question) → "1921"
```

**核心优势**：只需 2 次 LLM 调用（1 次规划 + 1 次总结），token 消耗减少 5 倍。代价是无法根据中间结果调整计划。

#### 3. babyagi (★22k) — 动态任务队列 + 优先级排序

```python
# 核心循环
while task_list:
    task = task_list.popleft()                    # 取最高优先级任务
    result = execute_task(task)                    # 执行
    store_result(task, result)                     # 存入向量 DB
    new_tasks = create_new_tasks(task, result)     # LLM 生成后续任务
    task_list = prioritize_tasks(task_list + new_tasks)  # 重排优先级
```

**核心优势**：任务列表是动态的——执行一个任务后，LLM 可以根据结果新增或取消后续任务。

#### 4. llm-tools-todo (★3) — Agent 自管 Checklist（最直接）

```python
# 给 agent 的工具集
tools = {
    "todo_begin":    lambda title: create_session(title),
    "todo_add":      lambda item, priority: add_item(item, priority),
    "todo_complete": lambda item_id: mark_complete(item_id),
    "todo_list":     lambda: show_all_items_with_status(),
    "todo_end":      lambda: finalize_session(),
}

# Agent 自主使用：
# 1. todo_begin("回答：爱因斯坦获诺贝尔奖是哪年？")
# 2. todo_add("搜索爱因斯坦基本信息", priority=1)
# 3. todo_add("搜索诺贝尔物理学奖历史", priority=2)
# 4. todo_add("交叉验证答案", priority=3)
# 5. [执行搜索] → todo_complete(1)
# 6. [执行搜索] → todo_complete(2)
# 7. todo_list() → 查看进度："1 ✓, 2 ✓, 3 待完成"
# 8. [验证] → todo_complete(3)
# 9. todo_end()
```

#### 5. OpenManus (已有 reference) — `[✓][→][ ][!]` 视觉状态标记

```python
# app/tool/planning.py 中的 PlanningTool
STATUS_MARKERS = {
    "completed":   "[✓]",
    "in_progress": "[→]",
    "pending":     "[ ]",
    "blocked":     "[!]",
}
```

### 落地方案

**A. ReasoningPlanTracker — 推理计划追踪器**

```python
@dataclass
class PlanStep:
    id: int
    description: str           # "搜索爱因斯坦诺贝尔奖年份"
    status: str                # pending | in_progress | completed | failed | skipped
    depends_on: List[int]      # 依赖哪些步骤
    result_summary: str        # 执行结果摘要
    confidence: float          # 0-1 置信度
    
@dataclass
class ReasoningPlanTracker:
    question_id: str
    steps: List[PlanStep]
    
    def generate_initial_plan(self, question: str, llm_fn) -> None:
        """LLM 生成初始推理计划"""
        
    def mark_step(self, step_id: int, status: str, result: str = "") -> None:
        """标记某步骤完成/失败"""
        
    def should_replan(self) -> bool:
        """检测是否需要重新规划（如：高置信度可提前终止）"""
        
    def get_plan_summary(self) -> str:
        """返回当前计划状态（给 LLM 看的格式化文本）"""
        # [✓] Step 1: 搜索爱因斯坦基本信息 → "1879年生于德国"
        # [→] Step 2: 搜索诺贝尔物理学奖记录
        # [ ] Step 3: 交叉验证答案
        
    def get_next_actionable_steps(self) -> List[PlanStep]:
        """返回所有依赖已满足的待执行步骤（支持并行）"""
```

**B. 与 Scratchpad 联动**

```
ReasoningPlanTracker（规划层）
  ↕ 双向通信
ReasoningScratchpad（数据层）

Plan 决定"下一步搜什么" → Scratchpad 存储"搜到了什么"
Scratchpad 的证据 → 反馈给 Plan 判断"是否需要重规划"
```

**C. 动态重规划触发条件**

```python
REPLAN_TRIGGERS = [
    "某步骤连续失败 2 次",
    "当前答案置信度已达 0.9+（可提前终止）",
    "搜索结果与预期方向完全不符",
    "发现问题比预想更简单/更复杂",
    "新证据推翻了之前的中间结论",
]
```

---

## 额外：并行探索策略（Cursor 分线程借鉴）

### 核心洞察
Cursor 新版的分线程策略本质是：
**同一个问题 → 多条独立探索路径 → 各自深入 → 汇聚最优结果**

这和 MetaGPT 的多 Agent 协作、Ralph Wiggum 的循环执行、以及学术界的 Self-Consistency 是同一个思路。

### 借鉴来源

| 来源 | 模式 | 特点 |
|------|------|------|
| **Cursor 分线程** | 同一问题开 N 个 thread 并行探索 | 各 thread 独立上下文、不互相干扰 |
| **MetaGPT** | 多角色 Agent（产品经理/架构师/工程师） | SOP 驱动、角色间有明确交接协议 |
| **nanobot** | 20 轮迭代循环，每轮读取上一轮状态 | 简单粗暴但有效 |
| **Ralph Wiggum** | 同一 prompt 反复执行直到完成 | 文件和 git 作为"记忆" |
| **OpenManus** | 30 步上限 + stuck 检测 | 卡住时自动换策略 |

### 落地方案

**A. 每题三线程并行推理**
```
Thread 1 (快速路径): knowledge_answer → 直接用 LLM 内部知识回答
Thread 2 (搜索路径): decompose → search → extract
Thread 3 (深度路径): decompose → multi-hop search → verify → extract

三个 thread 并行执行 → 汇聚投票
```

**B. 搜索策略多样化**
```
Thread A: 用中文关键词搜 IQS
Thread B: 用英文关键词搜 Brave
Thread C: 用问题原文搜 DuckDuckGo
→ 合并三个搜索结果去重后，统一排序
```

**C. 汇聚投票器**
```python
def aggregate_answers(answers: List[str]) -> str:
    # 1. 完全一致 → 直接返回
    if len(set(answers)) == 1:
        return answers[0]
    
    # 2. 多数一致 → 返回多数
    counter = Counter(normalize(a) for a in answers)
    most_common = counter.most_common(1)[0]
    if most_common[1] >= 2:
        return most_common[0]
    
    # 3. 全不一致 → 用 LLM 做最终裁决
    return llm_judge(question, answers, evidences)
```

---

## 全量评测结果与痛点分析（2026-02-11 全量 100 题）

### 成绩：27/100（较基线 33 分下降 6 分）

> **注意**：本次运行 Google API 和 IQS 均不可用（quota 耗尽），仅靠 Brave + DuckDuckGo。基线 33 分时 Google/IQS 可能可用，因此分数下降主要归因于搜索引擎可用性而非代码改动。

### 一致性投票模块统计

| 指标 | 值 |
|------|------|
| 共识检测成功 | 85/100（85%） |
| LLM 仲裁触发 | 15/100（15%） |
| 候选答案 4 个 | 90/100 |
| 候选答案 3 个 | 8/100 |
| 候选答案 2 个 | 2/100 |
| 最终来源 hop2 | 87 |
| 最终来源 search | 8 |
| 最终来源 knowledge | 5 |

**分析**：85% 的题目多个推理路径达成共识，投票机制正常。但 hop2 占 87%（太高），说明系统过于依赖二跳搜索结果，当搜索质量差时 hop2 答案也差。

### 错误分类（49 错 + 3 部分匹配 = 52 题）

| 错误类型 | 数量 | 占比 | 说明 |
|---------|------|------|------|
| **多跳推理错误** | **29** | **56%** | 中间实体找错 → 后续全偏（最大痛点） |
| **搜索证据不足** | **13** | **25%** | Brave/DDG 没搜到关键信息 |
| **格式/语言差异** | **10** | **19%** | 中英文名、昵称/全名、单位格式 |

### 典型错误模式

**A. 多跳推理链断裂（29 题，56%）— 最大痛点**
- Q13: 汤若望 → 找错朝代（万历 vs 顺治）
- Q97: 顺治的父亲 → 给了顺治本人（福临 vs 皇太极）
- Q98: 达芬奇的赞助人 → 给了达芬奇本人（Leonardo vs Ludovico Sforza）
- Q10: 马伯庸作品改编 → 找错了电视剧（陈情令 vs 长安十二时辰）
- Q66: 岩崎家族 → 找错了家族成员（久弥 vs 小弥太）

**共性**：找到了正确领域但选错了具体实体，说明推理链的最后一跳缺乏精准定位。

**B. 搜索证据不足（13 题，25%）**
- Q3: FBI 文件页数（241 vs 591）— 需要特定 PDF 文档
- Q4/Q14: 播客/纪录片名称 — 需要特定媒体平台搜索
- Q37: 细菌属鉴定（Burkholderia vs Frankia）— 需要学术论文

**共性**：信息存在于特定小众平台，Brave/DDG 搜不到。

**C. 格式/语言差异（10 题，19%）**
- Q7: "Washington Union Station" vs "Union Station"（多了前缀）
- Q63: "能效比" vs "GigaFlops/Watt"（中文概念 vs 英文单位）
- Q67/Q73/Q87: 中文答案 vs 英文标准答案

**共性**：题目语言或格式要求理解不精确，部分可通过后处理修复。

### 痛点优先级重排

基于全量评测数据，**痛点排序发生根本变化**：

| 优先级 | 痛点 | 错误占比 | 预期提升 | 理由 |
|--------|------|----------|----------|------|
| **P0** | 多跳推理链断裂 | 56%（29题） | +15~20 分 | 最大错误来源，找对领域但选错实体 |
| **P0** | 搜索引擎不可用 | 间接影响全部 | +5~10 分 | Google+IQS 不可用，信息获取能力腰斩 |
| **P1** | 格式/语言对齐 | 19%（10题） | +5~8 分 | 中英文名、前缀冗余、昵称 vs 全名 |
| **P1** | 搜索证据不足 | 25%（13题） | +3~5 分 | 小众信息 Brave/DDG 搜不到 |
| **P2** | 验证环节太弱 | 间接 | +3~5 分 | 错误答案未被拦截 |

### 下一步行动

1. **P0: 多跳推理链断裂**（最高优先级）
   - 结构化多跳规划器（每跳有明确目标和终止条件）
   - 中间结果校验（每跳后验证是否回答了该跳目标）
   - 动态查询生成（后跳依赖前跳的精确结果）

2. **P0: 搜索引擎恢复**
   - Google API：等待每日额度重置或更换密钥
   - IQS（阿里云）：需充值恢复服务
   - 备选：接入 Bright Data SERP API / Tavily 作为替代搜索源

3. **P1: 格式后处理增强**
   - 中英文双输出（检测题目语言要求）
   - 前缀去除增强（"Washington Union Station" → "Union Station"）
   - 昵称→全名映射（"Kit" → "Christopher"）

---

## 实施优先级与路线图

### 已完成（Phase 1）
1. ✅ **答案后处理管线** — P0 已实施
2. ✅ **改进 extract_answer prompt** — P0 已实施
3. ✅ **三层网页清洗管线** — P0-b 已实施
4. ✅ **多候选一致性投票** — P1-a 已实施
5. ✅ **域名感知提取器** — P0-b 已实施
6. ✅ **搜索后端预检+动态降级** — P0-a 已实施

### Phase 2: 推理层（下一步，预期 +15~25 分）
7. **结构化多跳规划器** — 借鉴 Research_Agent multi_hop_decomposition（最高优先级！）
8. **中间结果校验** — 每跳校验 + 失败重试
9. **格式/语言智能对齐** — 中英文双输出 + 前缀去除 + 昵称映射
10. **验证环节强化** — 借鉴 Enhancing-MH-QA self_verification

### Phase 3: 搜索层（预期 +5~10 分）
11. **搜索引擎恢复/替代** — Google/IQS 额度恢复，或接入 Bright Data/Tavily
12. **搜索策略多样化** — 多引擎并行 + 结果合并
13. **查询感知抓取** — BM25 只保留与 query 相关段落

### 总计预期：27 分（当前）→ 搜索恢复后 33+ → 推理层优化后 50-65 分
