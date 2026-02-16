# ChainReasoner 痛点优化蓝图（基于 references 项目借鉴 + 全量评测数据驱动）

> 当前成绩：27/100（搜索受限）→ Wing 架构改造后待验证 | 基线：33/100 | Wing 对手：42/100 | 目标：70+ | 原则：正确率高于一切，延迟和资源不敏感
>
> 最近全量评测：2026-02-11，Google+IQS 不可用，仅 Brave+DuckDuckGo
> Wing 架构改造：2026-02-13，启用 BrightData + 领域感知路由 + 高跳数(3-5) + 置信度放弃 + 结构化实体传递

---

## 总览：痛点 × 优先级矩阵（按全量评测数据排序）

| 优先级 | 痛点 | 影响范围 | 状态 | 参考项目 | 核心借鉴 |
|--------|------|----------|------|---------|---------|
| **P0** | 多跳推理链断裂 | 29/52 错误（56%） | ✅ P0-d/e/f + Wing 架构 | Research_Agent, ReAct, Wing | 结构化跳数规划(3-5跳) + 领域感知路由 + 结构化实体传递 + 迭代检索 + 置信度放弃 |
| **P0** | 搜索引擎不可用 | Google+IQS 双挂 | ✅ BrightData 已启用 | Wing | BrightData SERP API → Google 质量结果，绕过配额限制 |
| **P0-next** | 知识-搜索仲裁缺失 | Q0 级联偏差 | 🔴 待实施 | — | 知识答案验证搜索 + 双链推理 + 第一跳恢复 |
| **P0-next** | 答案命名标准化 | Wing 4题格式错 | 🔴 待实施 | — | official name 二次搜索 + 中英文对齐 |
| **P1** | 格式/语言对齐不足 | 10/52 错误（19%） | 部分已做 | Research_Agent, ReAct | 中英文双输出 + 前缀去除 + 昵称映射 |
| **P1** | 搜索证据不足 | 13/52 错误（25%） | 部分已做 | crawl4ai, firecrawl | 搜索策略多样化 + 学术/媒体特定源 |
| **P1-next** | 本地知识库缓存 | 跨题重复搜索 | ✅ per-question scratchpad + BM25 | bm25s, A-mem | 搜索结果持久化 + BM25 本地检索 |
| **P1-next** | Agentic 证据决策 | 管线答案可能错误 | ✅ Phase 2.7 模块化重构完成 | nanobot, deepagents | 工具注册表 + 大结果 offload + 三重协议容错 + 地理语言路由 |
| **P2** | 验证环节太弱 | 错误答案未拦截 | 部分(置信度放弃+agentic) | Enhancing-MH-QA, Research_Agent | self-verification + 证据回查 |
| **P2** | 密度剪枝过激 | 网页内容被过度删除 | ✅ 阈值+文本长度保护 | crawl4ai | DENSITY_THRESHOLD 0.48→0.32 |
| **P2** | PDF 内容丢失 | PDF 文件被直接丢弃 | ✅ PyPDF2 文本提取 | — | PDF 下载+提取+存入 scratchpad |

---

## 已完成项（归档概要）

| 痛点 | 实施内容 |
|------|---------|
| 答案格式不精确 | 7 步后处理管线 + 格式感知 prompt + 格式提示解析器 |
| 网页抓取质量差 | 三层清洗管线（规则 + 密度剪枝 + LLM 精炼）+ 域名感知提取器 |
| LLM 非确定性 | 多候选一致性投票 + Jaccard 相似度聚类 + LLM 仲裁 |
| **Wing 架构改造** (2026-02-13) | 5 项核心改动，详见下方 Wing 章节 |
| **网页抓取管线加固** (2026-02-15) | NoneType Bug 修复 + 大文件下载保护（stream=True + 2MB 上限） |
| **Agentic 证据决策循环** (2026-02-15) | nanobot 风格 tool-use loop（7 工具 × 8 轮上限），Phase 4.5 注入 |
| **密度剪枝修复** (2026-02-15) | 阈值 0.48→0.32 + 默认标签权重 0.5→0.7 + 200 字符最小保护 |
| **PDF 文本提取** (2026-02-15) | PyPDF2 提取 PDF 文本（5MB 上限）+ 自动路由 + 存入 scratchpad |
| **Agentic Loop 工程化重构** (2026-02-15) | 单文件→10 文件模块化，工具注册表 + 大结果 offload + 三重协议容错 + 地理语言路由 + 迭代上限 12 |

### Wing 架构改造详情（2026-02-13）

**背景**：Wing 对手得分 42（我们 27），分析其架构后实施全面改造。

**改动 1: BrightData 搜索引擎启用** ✅
- `enhanced_multi_hop_api_server.py`: `_build_hybrid_search()` 中初始化 `BrightDataSerpApiClient`
- preflight 检查中加入 BrightData 探针
- 效果：Google 质量搜索结果，绕过配额限制，5s 延迟 10 个结果

**改动 2: 领域感知路由** ✅
- 新建 `src/agents/question_domain_classifier.py`: LLM + 关键词混合分类器
- 5 个领域: academic / business / government / culture / news
- `language_aware_hybrid_search_dispatcher.py`: 按 (language, domain) 选不同引擎优先链
- `constrained_multi_hop_search_agent.py`: 每跳分类领域 → 传入 search meta

**改动 3: 高跳数规划 (3-5 跳)** ✅
- `structured_multi_hop_reasoning_planner.py`: prompt 改为 3-5 跳示例 + domain 字段
- hop_count 上限 4 → 5
- 解析器增加 domain 字段默认值

**改动 4: 置信度驱动放弃** ✅
- `constrained_multi_hop_search_agent.py`: `_apply_confidence_driven_abstention()` 方法
- 三条规则: Unknown/refusal → 空; 全候选 Unknown → 空; believe<0.4+无共识 → 空
- 返回值新增 `believe` 字段

**改动 5: 跨跳结构化实体传递** ✅
- hop 循环中构建 `structured_hop_results` 列表 (entity + evidence_snippet + confidence)
- previous_hop_results 改为 entity-highlighted 格式 ("**Entity** — evidence")
- `_build_isolated_hop_context` 和 `generate_queries_for_hop` 都利用精确实体名

**Q0 Mini 测试观察**:
- BrightData preflight 通过 ✅
- 5 跳规划成功生成 (academic→academic→business→business→business) ✅
- 领域分类工作 ✅
- 问题: Hop 1 搜索证据将 "元胞自动机" 误导向 "Arduino Esplora" 而非 "RepRap"，知识答案正确但被搜索覆盖

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

#### P0-f: REFUTES 实体提取 + 双语查询 + 检索轮次增加（2026-02-13）

**已完成改动（5 项）：**

1. **REFUTES 实体提取**（最高 ROI）：
   - 新增 `extract_corrected_entity_from_refutation_via_llm()` 函数（`large_language_model_call_handlers.py`）
   - 修改 `verify_answer_against_evidence_via_llm()` 返回 `(label, confidence, reasoning)` 三元组
   - 修改 `_run_answer_verification_phase()` 在 REFUTES 时调用提取函数，从 reasoning 中获取正确实体
   - 注入 `llm_refute_extract_fn` 到 search agent

2. **跨语言双语查询**：
   - 修改 `generate_queries_for_hop()` prompt 强制要求 `[ZH]` 和 `[EN]` 双语查询
   - 解析时去除语言标记前缀，上限从 3 → 6（3 ZH + 3 EN）
   - 主 agent 中每跳查询上限从 `max_queries` 提升到 `max(max_queries, 6)`

3. **检索轮次增加**：`MAX_RETRIEVAL_CYCLES_PER_HOP` 2 → 3
4. **早停阈值提高**：`confidence >= 0.75` → `0.85`（减少过早终止）

5. **关键修复：注入 llm_generic_fn 和 llm_fuse_fn**：
   - **发现**：`llm_generic_fn` 和 `llm_fuse_fn` 从未被注入到 search agent！导致 hop planning 退化为默认 1-hop、query generation 无 LLM 参与、evidence fusion 被跳过
   - **修复**：在 `enhanced_multi_hop_api_server.py` 中注入 `llm_generic_fn → send_chat_completion_request_with_retry` 和 `llm_fuse_fn → fuse_multi_hop_evidence_via_llm`
   - **影响**：此修复比 P0-f 的三项改动加起来还重要——之前所有的 hop planning/evaluation/refinement 代码都是形同虚设

6. **安装 lxml**：修复多个 `readpage_scrape` 的 "Couldn't find a tree builder" 报错

**P0-f Q0 测试结果（run_20260213_030343）：**

| 指标 | P0-f 之前（run_025502） | P0-f 之后（run_030343） |
|------|------------------------|------------------------|
| Hop plan | 1 跳（DEFAULT_HOP_PLAN，无 LLM） | 3 跳（LLM 生成目标） |
| 查询语言 | 纯中文 | 中文 + 英文双语 |
| 检索轮次 | 2 轮/跳 | 3 轮/跳 |
| Evidence fusion | 跳过（llm_fuse_fn=None） | 生效（合并 3 跳证据） |
| Evaluation/Refine | 跳过 | 生效（多次 refine 查询） |
| 英文搜索 | 无 | Brave 搜索 RepRap/cellular automaton |
| 最终答案 | RepRap Ltd（错误） | Ha Engineering... （错误） |
| 得分 | 0/1 | 0/1 |

**分析**：所有代码架构改动均已验证生效，但 Q0 仍错误。根因是 hop 1 的 LLM 幻觉了错误实体 "Johan Ha"（正确应为 "Adrian Bowyer"），这是因为：
- IQS 中文搜索 "欧洲学者 开源硬件 元胞自动机" 没命中 RepRap 相关页面
- LLM 在证据不足时编造了一个不存在的人名

**残余问题 & 下一步：**
1. LLM 幻觉实体的防护（当搜索证据不足时，不应接受 LLM 编造的实体名作为中间结果）
2. 搜索质量仍为核心瓶颈：Google 不可用，IQS 中文结果与英文话题不匹配
3. 需要更好的搜索结果评估——当证据完全没提到目标实体时应触发"证据不足"而非填入幻觉

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

### 已完成（Phase 1-2）
1. ✅ **答案后处理管线** — P0 已实施
2. ✅ **改进 extract_answer prompt** — P0 已实施
3. ✅ **三层网页清洗管线** — P0-b 已实施
4. ✅ **多候选一致性投票** — P1-a 已实施
5. ✅ **域名感知提取器** — P0-b 已实施
6. ✅ **搜索后端预检+动态降级** — P0-a 已实施
7. ✅ **结构化多跳规划器 (3-5 跳)** — P0-d + Wing 架构
8. ✅ **中间结果校验 + 迭代检索** — P0-d/e/f
9. ✅ **BrightData 搜索引擎** — Wing 架构改动 1
10. ✅ **领域感知搜索路由** — Wing 架构改动 2
11. ✅ **置信度驱动放弃机制** — Wing 架构改动 4
12. ✅ **跨跳结构化实体传递** — Wing 架构改动 5

### Phase 2.5: 管线加固 + Agentic 决策层（2026-02-15）

**已完成改动（6 项）：**

#### 2.5-a: 网页抓取管线加固 ✅
- **NoneType Bug 修复**：修复 `three_layer_html_content_cleaning_pipeline.py` 中 3 处 NoneType 错误（zhihu/baike JSON 解析、github repo_title）
- **大文件下载保护**：`direct_http_web_page_content_fetcher.py` 改用 `stream=True`，header 检查 Content-Type/Content-Length 后再下载 body，2MB 上限
- **效果**：页面抓取耗时从 89s 降至 5.4s，NoneType 错误消除

#### 2.5-b: Agentic 证据决策循环 ✅
- **新建 `src/mini_agent/mini_agent_loop.py`**（原 `agentic_evidence_decision_loop.py`）：nanobot 风格 while-tool-calls 循环
- **7 个工具**：`list_files`、`get_index`、`read_file`、`grep_evidence`、`search_local`、`web_search`（限 2 次）、`submit_answer`
- **XML 标签协议**：LLM 输出 `<tool_call>{"name": "...", "args": {...}}</tool_call>`，系统解析执行
- **注入位置**：`constrained_multi_hop_search_agent.py` Phase 4.5（投票之后、验证之前）
- **Q0 测试结果**：6 次工具调用（get_index → grep×3 → web_search → submit_answer），confidence=0.95，成功将管线错误答案 "RepRap Ltd" 纠正为 "RepRapPro Limited"，耗时 24s

#### 2.5-c: 密度剪枝修复 ✅
- **降低阈值**：`DENSITY_THRESHOLD` 0.48 → 0.32
- **提高默认标签权重**：未知标签 0.5 → 0.7
- **最小文本长度保护**：节点文本 > 200 字符时不剪枝
- **效果**：L1/L2 字符数基本一致（如 L1=2969 L2=2969），不再过度剪枝

#### 2.5-d: PDF 文本提取 ✅
- **从 `_BINARY_FILE_SUFFIXES` 移除 `.pdf`**
- **新增 `_fetch_and_extract_pdf_text()`**：PyPDF2 提取文本，5MB 上限，截断到 5000 字符
- **Content-Type 路由**：URL 以 `.pdf` 结尾或 Content-Type 为 `application/pdf` 时自动走 PDF 提取
- **依赖**：`PyPDF2>=3.0.0` 已加入 requirements.txt

#### Q0 Mini 测试观察（run_20260215_043429）
- Agentic loop 触发 ✅，6 次工具调用，confidence=0.95
- Agentic loop 纠正管线答案 ✅（RepRap Ltd → RepRapPro Limited → RepRapPro Ltd）
- 密度剪枝不再过激 ✅（L1≈L2，内容保留完整）
- 55 个 evidence 文件 + 22 个 page 文件写入 scratchpad ✅
- BM25 索引 572 个文档，25 次本地搜索 19 次命中 ✅
- 总耗时 1444s（24 分钟），其中 agentic loop 仅 24s
- GitHub extractor NoneType 错误仍存在但被 try/except 捕获，自动 fallback 到 generic pipeline

---

### Phase 3: 冲击 70 分（下一步，预期 +25~35 分）

**Tier 1: 必须做（预期 +10~15 分）**

13. 🔴 **知识-搜索双链仲裁** — knowledge_answer 与搜索答案不一致时，用知识答案做验证搜索，分裂两条推理链并行推进
14. 🔴 **第一跳错误恢复** — 对比 knowledge_answer 和 hop_1_answer，差异巨大时并行分裂两条链
15. 🔴 **答案命名标准化** — 找到实体后额外搜 "official name"/"registered name" 确认精确格式

**Tier 2: 高 ROI（预期 +5~8 分）**

16. 🔴 **数值/年份精确验证** — 数字类答案从多源提取后取众数
17. 🔴 **问题类型格式预判** — 提取前预判答案格式（数字/人名/机构名），格式不匹配强制重提取
18. ✅ **本地知识库缓存** — 搜索结果 + 抓取页面持久化到本地 BM25 索引（per-question scratchpad 已实现）
19. ✅ **Agentic 证据决策循环** — LLM 自主查阅 scratchpad 做最终决策（Phase 2.5-b 已实现）

**Tier 3: 锦上添花（预期 +3~5 分）**

20. 🔴 **Wikipedia 直达检索** — 人名/地名/机构名直接抓 Wikipedia 页面做精确提取
21. 🔴 **并行模型投票** — 多温度/seed 生成候选答案
22. 🔴 **答案后处理 pipeline 强化** — 中英文名对齐、格式约束执行
23. ✅ **PDF 文本提取** — 不丢弃 PDF，PyPDF2 提取文本存入 scratchpad（Phase 2.5-d 已实现）

### Phase 2.6: 8 轮迭代优化 (2026-02-15)

**目标**: 通过 8 轮 mini-test 迭代，系统性地识别和修复管线痛点。

#### 修复清单

| Round | 痛点 | 修复 | 文件 |
|-------|------|------|------|
| R1 | 5跳×3轮=68次LLM调用(24min) + 9次NoneType错误 | MAX_RETRIEVAL_CYCLES 3→2, 最大跳数 5→4, 修复 attrs=None 导致的 NoneType | per_hop_result_validator_and_corrector.py, structured_multi_hop_reasoning_planner.py, three_layer_html_content_cleaning_pipeline.py |
| R2 | Layer3 LLM精炼暴增到24次/96s | 禁用 Layer3 LLM page_content_refinement（agentic loop 已补偿） | enhanced_multi_hop_api_server.py |
| R3 | (Q0连续3轮正确，无新痛点) | 无代码改动，进入Q1测试 | - |
| R4 | MiniAgent 不调用 submit_answer, conf=0.5 | 改进 prompt 强调必须 submit_answer, max_iterations 8→6 | mini_agent_loop.py |
| R5 | Page enrichment 30次/115s 太慢 | enrichment top_n 3→2 | language_aware_hybrid_search_dispatcher.py |
| R6 | generic_hop_planning 仍是最大消耗 | max_tokens 1500→1200, FETCH_TIMEOUT 10→8s | enhanced_multi_hop_api_server.py, direct_http_web_page_content_fetcher.py |
| R7 | max_tokens=800截断推理 + confidence boost有害 | max_tokens恢复1200, 移除confidence boost, 改进entity精确度prompt | mini_agent_loop.py, enhanced_multi_hop_api_server.py |
| R8 | INSUFFICIENT验证结果错误触发fallback到knowledge_answer | 只在REFUTES时fallback，INSUFFICIENT保持当前答案 | constrained_multi_hop_search_agent.py |

#### 指标变化

| 指标 | Round 1 (基线) | Round 3 (Q0最佳) | Round 6 (Q1最佳) | 改善幅度 |
|------|---------------|-----------------|-----------------|---------|
| 总耗时 | 1444s (24min) | 1002s (16.7min) | 555s (9.3min) | **-62%** |
| LLM调用 | 68 | 44 | 29 | **-57%** |
| LLM耗时 | 275s | 182s | 114s | **-59%** |
| NoneType错误 | 9 | 0 | 0 | **-100%** |
| Page enrichment | 35次/151s | 59次/141s | 19次/61s | **-60%** |
| Q0答案 | RepRapPro Ltd ✅ | RepRapPro Ltd ✅ | - | 稳定正确 |
| Q1答案 | - | - | Magnus Mörner | 一致稳定 |

#### 关键发现

1. **INSUFFICIENT fallback bug（R8发现）**: 这是一个影响所有题目的严重 bug。当 reverse verification 返回 INSUFFICIENT（证据不足），系统错误地 fallback 到 knowledge_answer，导致正确答案被覆盖。修复后预计对整体得分有显著提升。
2. **Layer3 LLM精炼的性价比极低**: 24次调用消耗96s但对答案质量无贡献。禁用后节省约100s且答案不变。
3. **Agentic loop 是关键纠错机制**: 在 Q0 中，管线投票选了错误的 "RepRap Ltd"，agentic loop 纠正为 "RepRapPro Limited"。
4. **BrightData 超时是不可控的外部因素**: 多次 timeout retry 增加了 20-60s 的不可控延迟。

### 总计预期路线：27 分（旧）→ Wing 架构 ~42 分 → Phase 2.5 管线加固 ~45 分 → Phase 2.6 迭代优化 ~50 分 → Tier 1 完成 ~60 分 → Tier 2 完成 ~70 分 → Tier 3 完成 75+ 分

---

## Phase 3: ChainReasoner 全面优化（基于 Q83 失败复盘）

### 优化内容（7 大模块，30+ 处代码修改）

#### Phase 1: 多跳规划优化
- **约束条件分解 prompt**: 在 hop planning system prompt 中增加"先列出所有独立约束条件，再为每个约束设计搜索跳"
- **depends_on_hop 字段**: hop plan JSON 增加跳间依赖关系字段
- **max_hops 5→5**: 允许复杂多约束题使用 5 跳

#### Phase 2: 搜索查询优化
- **多角度查询**: 要求 LLM 从 DIRECT 和 INDIRECT 两个角度生成查询
- **约束覆盖**: 搜索"哪家公司"类问题时要求包含所有已知约束条件

#### Phase 3: 证据收集优化
- **enrichment top_n 2→3**: 恢复页面内容抓取数量
- **内容哈希去重**: scratchpad 写入前检查 MD5 哈希，跳过重复证据（Q83 跳过了 249-312 条重复）
- **BM25 min_score 1.5→1.0**: 降低本地检索阈值，减少漏召回

#### Phase 4: 答案提取优化
- **反确认偏差 prompt**: hop 评估增加"如果证据不明确，列出所有可能候选"指令
- **alternative_entities 字段**: 提取实体时同时保留 top-2 替代候选
- **ANTI-CONFIRMATION BIAS 规则**: 明确要求 LLM 不要只选第一个实体

#### Phase 5: Agentic 决策层深度优化
- **决策树**: grep<200字符→必须web_search, read_file<100字符→必须web_search
- **自检清单**: submit_answer 前要求验证所有约束条件
- **反思机制**: 第 3 次工具调用后插入反思提示
- **证据不足检测**: 前 2 次本地工具<300字符时自动建议 web_search
- **countdown 优化**: 第 4 轮建议 web_search，第 5 轮必须 submit
- **无 tool_call 重试**: LLM 不输出 tool_call 时给出示例重试而非直接退出
- **候选全同警告**: 所有候选答案相同时警告确认偏差
- **max_iterations 6→8**: 增加迭代空间

#### Phase 6: 投票与仲裁优化
- **来源权重**: agentic_loop=1.5, search=1.2, hop2=1.0, knowledge=0.8, heuristic=0.6
- **仲裁 prompt**: 增加"先识别所有约束条件，再逐一检查候选"指令

#### Phase 7: 反向验证优化
- **对抗性查询**: 验证阶段同时搜索排除当前答案的查询
- **替代答案搜索**: confidence<0.8 时额外验证第二候选
- **Better_entity 字段**: 验证 prompt 要求指出更匹配的不同实体

### 测试结果

| 题目 | 答案 | 正确? | Agentic Loop 行为 | 关键发现 |
|------|------|-------|-------------------|---------|
| Q1 | James Lockhart | ✅ agentic纠正 | 7次迭代, grep→read→grep→**web_search**→**fetch_page**→**web_search**→**submit_answer**, conf=0.95 | **Pipeline给出Werner Baer(错), agentic loop通过web_search+fetch_page成功纠正** |
| Q10 | 靳东和雷佳音 | 待确认 | 7次迭代, grep→grep→search_local→web_search×3, 未submit | LLM搜索了"阿耐 本名 瓦特"但未找到足够证据 |
| Q0 | RepRap Ltd | ❌ | 3次迭代, grep(247字符)→search_local, 未submit | Pipeline候选全部不含"RepRapPro", LLM随机性 |
| Q83 | 通用汽车 | ❌ | 3次迭代, grep→grep("Stellantis"), 未submit | 搜索引擎结果中无"福特"相关证据 |

### 关键发现

1. **Q1 证明 agentic loop 优化有效**: Pipeline 投票选了错误的 Werner Baer，agentic loop 通过 web_search + fetch_page 找到正确答案 James Lockhart (conf=0.95)。这是优化前不可能实现的。
2. **submit_answer 问题仍存在**: qwen3-max 在部分场景下不输出 `<tool_call>` 标签，导致 agentic loop 默认回退。已增加重试机制但效果有限。
3. **证据去重生效**: 每次运行跳过 249-312 条重复证据，减少了 scratchpad 噪声。
4. **Q83 是搜索引擎级别的问题**: 所有搜索引擎对 UAW 罢工查询都主要返回 GM/Stellantis 结果，"福特"需要极精准的查询才能找到。

---

### Phase 2.7: Agentic Loop 工程化重构（2026-02-15）

**背景**：Phase 2.5-b 的 agentic 决策循环是单文件 637 行，存在三个核心差距 + 一个搜索语言缺陷：
1. 迭代次数不足（硬编码 8 次，nanobot 默认 20 次）
2. 工具结果硬截断 3000 字符（deepagents 用 offload + 头尾预览）
3. XML 工具调用协议不稳定（qwen 可能输出格式不规范）
4. 搜索语言判断缺失（中文问题+英语世界事件仍用中文搜索）

**已完成改动（10 个新文件 + 1 个 shim）：**

#### 2.7-a: 模块化拆分 ✅
- 将 `src/agents/agentic_evidence_decision_loop.py`（637 行）拆分为 `src/mini_agent/` 模块（10 个文件）
- 旧文件改为 backward-compatibility shim，import 路径不变
- 遵循 `代码命名与重构规范.md` 的命名规则

#### 2.7-b: 工具注册表模式 ✅
- **`tool_base_and_registry.py`**：`MiniAgentToolBase` 抽象基类 + `MiniAgentToolRegistry` 注册表
- 每个工具是独立类，自带 name/description/schema/when_to_use/when_not_to_use
- 参数验证、执行、prompt 定义生成全部封装在工具类内
- 借鉴 nanobot `tools/base.py` + `tools/registry.py`

#### 2.7-c: 工具类实现 ✅
- **`local_evidence_browsing_tools.py`**：5 个本地工具类（ListFiles, GetIndex, ReadFile, GrepEvidence, SearchLocal）
- **`web_search_and_fetch_tools.py`**：3 个网络工具类（WebSearch, FetchPage, DeepWikiSearch）
- **`answer_submission_tool.py`**：终止工具 `SubmitFinalAnswerTool`（is_terminal=True）

#### 2.7-d: 多协议工具调用解析器 ✅
- **`tool_call_protocol_parser.py`**：三重容错解析（XML tag > JSON block > bare JSON）
- JSON 修复：处理 qwen 常见错误（尾逗号、单引号、缺失闭括号）
- 每次解析记录使用的协议（xml_tag/json_block/freeform），便于分析

#### 2.7-e: 大结果 offload 中间件 ✅
- **`large_result_offload_middleware.py`**：超过 4000 字符的工具结果写入 scratchpad 文件
- 生成头 5 行 + 尾 5 行预览，替换原始结果
- 排除列表：list_files, get_index, submit_answer
- 借鉴 deepagents `_process_large_message()` + `_create_content_preview()`

#### 2.7-f: 动态系统提示构建 ✅
- **`prompt_builder.py`**：从注册表动态生成工具定义和系统提示
- 添加新工具时自动更新 prompt，无需手动编辑
- 借鉴 nanobot `context.py` 的 `build_system_prompt()`

#### 2.7-g: 问题地理-语言智能路由 ✅
- **`question_geographic_language_router.py`**：三层判断（信号检测 → 规则引擎 → LLM 精细判断）
- 输出语言优先级：english_first / chinese_first / bilingual_equal
- 中文问题+英语世界事件 → 自动切换为英文优先搜索

#### 2.7-h: 增强主循环 ✅
- **`mini_agent_loop.py`**：迭代上限 8→12，自适应提前终止
- 连续 2 次无工具调用 → 强制 submit（nanobot 风格）
- 增强日志：每轮记录 tool/args/result_length/was_offloaded/protocol
- 循环结束记录：iterations/parse_failures/offloaded_files/tool_breakdown/language_priority

#### Q0 Mini 测试结果（run_20260215_173405）

| 指标 | Phase 2.5-b（旧单文件） | Phase 2.7（新模块化） |
|------|------------------------|----------------------|
| 迭代次数 | 6 | 7（12 次上限内自适应终止） |
| 工具调用 | 6 | 7（grep×3 + read×3 + submit×1） |
| 解析失败 | 未记录 | 0（XML 协议全部成功） |
| Offload 事件 | 无（硬截断） | 1 次（read_file 结果 offload 到文件） |
| 语言路由 | 无 | english_first（正确识别英国公司话题） |
| 置信度 | 0.95 | 0.95 |
| 答案 | RepRapPro Ltd ✅ | RepRapPro Ltd ✅ |
| Agentic 耗时 | 24s | 24.7s |

#### 新增文件清单

```
src/mini_agent/
├── __init__.py                              (~20 行)
├── tool_base_and_registry.py                (~130 行)
├── local_evidence_tools.py                  (~170 行)
├── web_search_tools.py                      (~190 行)
├── answer_submission_tool.py                (~40 行)
├── tool_call_protocol_parser.py             (~190 行)
├── large_result_offload.py                  (~130 行)
├── prompt_builder.py                        (~120 行)
├── language_router.py                       (~240 行)
└── mini_agent_loop.py                       (~300 行)
```

`src/agents/agentic_evidence_decision_loop.py` → backward-compatibility shim（~18 行）
`src/agentic/__init__.py` → backward-compatibility shim（~13 行）

---

### Phase 3.1: 语言路由 + IQS 优化 + Fetch 优先级 + Hop 日志增强 + 日志实时化（2026-02-16）

**背景**：Q20 随机测试暴露多个问题 — 英文话题被中文搜索淹没、IQS 抓取低价值页面、FetchPage 优先级错误、回溯日志缺失、日志文件不实时更新。

**已完成改动（8 项）：**

#### 3.1-a: 智能语言路由注入搜索阶段 ✅
- `constrained_multi_hop_search_agent.py`: 在 `answer()` 入口调用 `classify_question_geographic_language_priority`，将结果存入 `_language_priority`
- `_perform_search_with_result_caching`: 在 `meta` 中传递 `language_priority`
- `language_aware_hybrid_search_dispatcher.py`: `execute_search_query` 优先使用 `meta["language_priority"]` 而非字符级语言检测
- **效果**：中文问题 + 英语话题 → 自动切换英文优先搜索

#### 3.1-b: IQS 搜索效率优化 ✅
- `alibaba_iqs_search_client.py`: `readpage_top_n` 5→2，减少无关页面抓取
- 新增 `_LOW_VALUE_DOMAINS` 黑名单（音乐、视频、小众百科等 15+ 域名）
- 新增 `_is_low_value_domain()` 方法，跳过黑名单域名的 readpage 抓取
- `language_priority == "english_first"` 时跳过 IQS readpage enrichment
- **效果**：IQS enrichment 调用量减少 60%+

#### 3.1-c: FetchPage 优先级反转 ✅
- `web_search_and_fetch_tools.py`: 直接 HTTP fetcher 改为主路径，IQS readpage 改为 fallback
- `MAX_FETCH_PAGE_CALLS_PER_QUESTION` 2→4
- **效果**：页面抓取成功率提升，减少对 IQS 的依赖

#### 3.1-d: BrightData 韧性增强 ✅
- `language_aware_hybrid_search_dispatcher.py`: `_MAX_FAIL_STREAK` 3→5
- **效果**：BrightData 偶发超时不会过早禁用搜索引擎

#### 3.1-e: Hop 回溯日志增强 ✅
- `constrained_multi_hop_search_agent.py`: 每跳后输出 `[HOP {n} RESULT]` 日志（valid/confidence/entity）
- 新增 `[BACKTRACK CHECK]` 日志，记录回溯条件检查结果
- 收集 `hop_evaluation_summaries` 列表，传入 MiniAgent

#### 3.1-f: Agent 置信度注入 ✅
- `prompt_builder.py`: 新增 `hop_evaluations` 参数
- 将 hop 评估结果（hop_num/entity/valid/confidence）注入 MiniAgent 系统提示
- 低置信度 hop 会触发警告，防止 agent 过度自信
- `mini_agent_loop.py`: 传递 `hop_evaluations` 参数

#### 3.1-g: 日志完全实时化 ✅
- `per_run_faithful_api_call_trace_logger.py`: 所有 trace 文件改用 `buffering=1`（行缓冲）
- `_write_record()`: 在 `flush()` 后增加 `os.fsync()` 强制落盘
- `logger_config.py`: `RotatingFileHandler` 改用 `_FlushingRotatingFileHandler`（每条 emit 后 flush）
- `logger_config.py`: `StreamHandler` 输出流包装为 `_FlushingStreamWrapper`（每次 write 后 flush）
- `run_progressive_evaluation_with_regression_guard.py`: `DualOutputStreamWriter` 加 `buffering=1` + `os.fsync()`
- `run_progressive_evaluation_with_regression_guard.py`: 入口设置 `PYTHONUNBUFFERED=1`
- **效果**：所有日志文件（trace JSONL + eval_progress.txt + logging 输出）写入后立即可见

#### 3.1-h: PyPDF2 安装修复 ✅
- 发现 `.venv` 中缺少 PyPDF2，手动安装

#### Q0 Mini 测试结果（run_20260216_202643）

| 指标 | Phase 3（之前） | Phase 3.1（之后） |
|------|----------------|------------------|
| 最终答案 | RepRap Ltd ❌ | **RepRapPro Ltd ✅** |
| 得分 | 0/1 | **1/1** |
| 总耗时 | ~24min | 1236s (20.6min) |
| 回溯触发 | 无日志记录 | Hop 2 触发回溯（Conway → RepRap Ltd） |
| 语言路由 | 未注入搜索 | english_first 正确传递 |
| 投票候选 | 3 个 | 5 个（hop2/search/knowledge/heuristic/knowledge_chain） |
| 投票仲裁 | — | LLM 仲裁选 RepRapPro Limited（knowledge） |
| Agentic loop | 3 次迭代 | 11 次迭代（grep×4 + read×2 + web_search×3 + fetch×1 + submit） |

**关键发现**：
1. **投票机制是答对的核心** — 5 个 hop 都没直接找到 RepRapPro，但 knowledge 候选给出了正确答案，LLM 仲裁正确选择
2. **回溯日志可追溯** — Hop 2 的 `[BACKTRACK CHECK]` 记录了 Conway → RepRap Ltd 的回溯过程
3. **IQS 优化生效** — 英文话题下跳过了 IQS readpage enrichment
4. **耗时瓶颈在 Hop 3-4** — 系统知道 Conway 不对但搜不到 Adrian Bowyer（RepRap 创始人），白费 300+ 秒
