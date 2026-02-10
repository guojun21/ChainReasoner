# ChainReasoner 八大痛点优化蓝图（基于 references 项目借鉴）

> 当前成绩：33/100 | 目标：50+ | 原则：正确率高于一切，延迟和资源不敏感

---

## 总览：痛点 × 参考项目矩阵

| 痛点 | 影响 | 参考项目 | 核心借鉴 |
|------|------|---------|---------|
| P0 答案格式不精确 | "搜到了但输出错" | Research_Agent, ReAct | 答案后处理管线 + 精准提取 prompt |
| P0 网页抓取质量差 | 搜到了但抓不到/抓到噪音 | crawl4ai, firecrawl, MediaCrawler | LLM 清洗 + 密度剪枝 + 结构化提取 |
| P1 LLM 非确定性 | 同题多跑不同结果 | MetaGPT (ToT), OpenManus, Cursor 分线程 | 多次采样 + 投票 + 并行探索 |
| P1 多跳推理链断裂 | 3-4 跳全错 | Research_Agent, ReAct, open_deep_research | 结构化跳数规划 + 中间校验 |
| P2 IQS 延迟过高 | 中文证据不足 | crawl4ai, firecrawl, bm25s | 并行抓取 + 本地缓存 + 重试 |
| P2 验证环节太弱 | 错误答案未拦截 | Enhancing-MH-QA, Research_Agent | self-verification + 证据回查 |
| P1 推理暂存缺失 | 跨跳证据丢失、无法回溯 | Researcher, deepagents, A-mem | per-question 文件目录 + 索引 + LLM 可查 |
| P1 规划与进度追踪缺失 | 无法动态调整推理策略 | LLMCompiler, ReWOO, babyagi, llm-tools-todo | todo-list 机制 + 动态重规划 |
| 额外 并行探索策略 | 提升整体效率和质量 | Cursor 分线程, Ralph Wiggum, nanobot | 多路径并行搜索 + 结果汇聚 |

---

## P0：答案格式不精确

### 现状问题
精确文本匹配下，差一个字就 0 分：
- `RepRapPro Limited` vs 标准答案 `RepRapPro Ltd` → 0 分
- `radio receiver` vs 标准答案 `radio` → 0 分
- `Washington Union Station` vs 标准答案 `Union Station` → 0 分
- `今日俄罗斯国际通讯社` vs `今日俄罗斯国际新闻通讯社` → 0 分

### 借鉴来源

#### 1. Research_Agent `src/utils/normalize.py`
```python
# 核心流程：
# 1. 去 "Answer:" / "答案：" 前缀
# 2. 数值格式处理（去逗号、去小数点、四舍五入）
# 3. 多实体分隔符统一（；→ ,）
# 4. 去引号、去多余空格
```

#### 2. ReAct `wrappers.py` → `normalize_answer()`
```python
# 更激进：去冠词(a/an/the)、去标点、转小写、压空格
def normalize_answer(s):
    remove_articles → white_space_fix → remove_punc → lower
```

#### 3. Research_Agent `synthesis.yaml` → 精准提取 prompt
```yaml
# "请直接给出最标准的答案，不要包含推理过程或多余解释"
# 这句话在他们的每个 prompt 里都重复出现
```

### 落地方案

**A. 改进 answer extraction prompt**
```
当前问题要求的答案格式是 {format_hint}。
请从以下证据中提取最精确的答案。
规则：
1. 如果题目要求"格式形如：Alibaba Group Limited"，则严格模仿该格式
2. 不要多余的修饰词、解释、推理过程
3. 如果是英文公司名，优先使用缩写形式（Ltd 而非 Limited）
4. 如果是人名，只给名字，不加头衔
5. 只输出答案本身，一个字都不要多
```

**B. 答案后处理管线（新增模块）**
```
raw_answer
  → 去前缀（"Answer:", "答案：", "The answer is"）
  → 去引号、去括号内容
  → 数值归一化（"2.40" → "2.40", 保留原始精度）
  → 去多余空格
  → 格式二次校验（如果题目给了格式示例，用 LLM 做一次格式对齐）
```

**C. 格式感知提取**
- 解析题目中的格式提示（"要求格式形如：XXX"）
- 把格式提示作为强约束传入 extract_answer prompt
- 如果答案不符合格式，触发重新提取

---

### 🔧 P0 实施情况（2026-02-11）

**已完成的改动：**

#### A. 格式提示解析器 + 后处理管线（`question_answer_format_hint_parsing_and_alignment.py`）
- ✅ 已有 `ParsedQuestionAnswerFormatHint` 数据类（`has_format_hint`, `format_example`, `expected_language`, `expected_answer_style`）
- ✅ 新增 `apply_format_aware_answer_postprocessing_pipeline()` — 7 步后处理管线：
  1. 去 LLM 前缀（中英文：`Answer:`, `答案：`, `The answer is`, `该商业实体的名称是` 等 10+ 模式）
  2. 去引号、去尾部标点
  3. 空白归一化
  4. 句子包装提取（`The company is called X` → `X`）
  5. 数值归一化（纯数字答案去逗号、处理小数）
  6. 格式提示对齐（年份/数字/中文双实体和连接符等）
  7. 最终清理
- ✅ 借鉴 Research_Agent `normalize.py` + ReAct `wrappers.py` 设计

#### B. 增强 extract_answer prompt（格式感知）
- ✅ `large_language_model_call_handlers.py` + `pipeline_llm_prompt_factory_functions.py` 双路径统一
- ✅ 系统 prompt 新增 `FORMAT AWARENESS (CRITICAL)` 段：
  - 格式示例仅为答案类型参考，不要复制示例中的具体词语
  - 证据中的原形优先（`Ltd` → `Ltd`，`Limited` → `Limited`）
  - 初步答案与证据冲突时，始终以证据为准
- ✅ 用户 prompt 新增 `FORMAT CONSTRAINTS` 段（由 `build_format_sensitive_answer_constraints_for_prompt()` 动态生成）
- ✅ knowledge_answer prompt 新增第 7 条规则：格式示例仅为类型参考

#### C. 知识答案上下文去偏
- ✅ 修改 `constrained_multi_hop_search_agent.py`：传给 extract_answer 的上下文中，知识答案加 NOTE 警示
  - 旧：`Preliminary answer from reasoning: RepRapPro Limited`
  - 新：`NOTE: A preliminary answer 'RepRapPro Limited' was generated from LLM knowledge alone. This may have incorrect formatting. Always verify using evidence.`
- ✅ knowledge_answer `max_tokens` 从 500 提升至 1200（避免推理截断）

#### D. 评估基础设施
- ✅ `--question-limit N` 参数（自动跳过 stage 门控）
- ✅ `answer_postprocess_trace.jsonl` — 新增第 5 个日志视角，记录完整后处理链路
- ✅ 日志可回答三个关键问题：LLM 原始返回了什么 → 后处理如何变换 → 最终答案为何

#### 验证结果（Q0 mini 测试，3 轮）

| 轮次 | 改动阶段 | 输出答案 | 得分 | 说明 |
|------|----------|----------|------|------|
| Run 1 | 改动前基线 | RepRapPro **Limited** | 0/1 | 格式错误：`Limited` 应为 `Ltd` |
| Run 2 | 格式 prompt 修复后 | RepRap **Ltd** | 0/1 | 格式修复生效（`Ltd`），但 Google API 挂掉，搜索证据不含 "RepRapPro"，LLM 知识错误丢了 "Pro" |
| Run 3 | 增加 max_tokens | RepRap **Ltd** | 0/1 | 同上：搜索结果非确定性 + Google API 失效导致证据质量低 |

**分析：**
- ✅ **格式修复有效**：`Limited` → `Ltd` 转换成功（Run 1 vs Run 2 对比）
- ❌ **实体精度问题**（非 P0）：Run 2-3 中搜索引擎未返回含 "RepRapPro" 的关键文章（3DPrint.com），LLM 知识把 "RepRapPro Ltd" 错认为 "RepRap Ltd"
- ❌ **Google API 限额耗尽**（`密钥总次数达到最大值`），严重降低英文搜索质量
- ⚠️ 如果搜索能稳定返回 3DPrint.com 的关键文章（Run 1 的证据），格式修复后应能正确输出 `RepRapPro Ltd`（得 1 分）

**下一步（依赖其他痛点优化）：**
- P0-b（网页抓取质量差）：修复后搜索证据质量提升 → 实体精度提升
- P1-a（LLM 非确定性）：多路径投票 → 减少实体名混淆
- P2-a（IQS 延迟过高）：搜索缓存 + 并行化 → 更稳定的证据获取

---

## P0-b：网页抓取质量差（搜到了但抓不到关键信息 / 抓到一堆噪音）

### 现状问题
当前 `direct_http_web_page_content_fetcher.py` 的做法：
- 纯正则 HTML 清洗（去 script/style/tags）
- 只做 6 个硬编码 HTML 实体解码
- 空白压缩后直接截断前 3000 字符
- **没有去噪音**：导航栏、广告、页脚、Cookie 提示全混在里面
- **没有用 LLM 清洗**：抓到的 3000 字符中可能只有 200 字是有用信息
- **不区分网站**：Wikipedia、百度百科、新闻网站用同一套逻辑

结果：Brave/DuckDuckGo 搜到的 URL 抓回来的内容质量极低，LLM 在一堆噪音里找不到关键证据。

### 借鉴来源

#### 1. crawl4ai — 五层内容提取管线（最完整）

```
Raw HTML
  → [1] 噪音移除（去 script/style/nav/footer/aside/iframe/noscript）
  → [2] 空元素剪枝（< 5 词的元素自底向上删除）
  → [3] 属性清理（只保留 src/href/alt/title）
  → [4] 文本密度剪枝 / BM25 相关性过滤 / LLM 清洗（三选一）
  → [5] Markdown 转换 + 引用格式化
```

**核心：文本密度剪枝（PruningContentFilter）— 不需要 LLM 就能去掉 80% 噪音**
```python
# 五维评分，低于 0.48 的节点直接删
score = (
    0.4 * text_density +          # 文本长度 / 标签长度
    0.2 * (1 - link_density) +    # 链接文本占比越低越好
    0.2 * tag_weight +            # article=1.5, p=1.0, div=0.5, span=0.3
    0.1 * class_id_penalty +      # 含 nav/ads/footer/sidebar 的扣分
    0.1 * log(text_len + 1)       # 越长越好
)
```

**噪音检测正则**
```python
NEGATIVE = re.compile(r"nav|footer|header|sidebar|ads|comment|promo|advert|social|share", re.I)
EXCLUDED_TAGS = {"nav", "footer", "header", "aside", "script", "style", "form", "iframe", "noscript"}
```

**LLM 清洗 Prompt（LLMContentFilter）— 用 LLM 把 HTML 变成干净 Markdown**
```
Your task is to filter and convert HTML content into clean, focused markdown
that's optimized for use with LLMs and information retrieval systems.

DO: Keep essential information, main content, key details
DO: Preserve hierarchical structure using markdown headers
DON'T: Include navigation menus, ads, footers, cookie notices
DON'T: Keep social media widgets, sidebars, related content
```

**BM25 相关性过滤（BM25ContentFilter）— 根据搜索查询保留相关段落**
```python
# 用搜索 query 对页面段落做 BM25 打分
# 加上标签权重加成：h1=5x, h2=4x, h3=3x, strong=2x, code=2x
# 低分段落直接丢弃
```

#### 2. firecrawl — 规则化噪音过滤 + LLM 结构化提取

```
Phase 1（规则层）: ~50 条 CSS 选择器黑名单
  → 删除 header, footer, nav, cookie-banner, social-share, ad-*, promo-* 等
  → 如果删完之后没内容了 → fallback 到不删的版本

Phase 2（转换层）: HTML → Markdown（html2text / TurndownService）
  → 然后正则后处理：修复断行、去多余空行、清理破损链接

Phase 3（LLM 层）: Markdown → 结构化 JSON
  → 不是给 HTML！是给清洗后的 Markdown
  → 配合 JSON Schema 定义要提取什么字段
  → 关键：先规则清洗再 LLM 提取，节省 token、提高精度
```

**核心洞察：firecrawl 不是直接把 HTML 扔给 LLM，而是先规则清洗 → 转 Markdown → 再 LLM 提取。这样 LLM 看到的已经是干净的 Markdown 了。**

#### 3. MediaCrawler — 特定网站的结构化提取

对于中文知识网站（知乎、百度百科、小红书），MediaCrawler 用了一个巧妙的方法：

```python
# 直接从 HTML 里提取嵌入的 JSON 状态对象
state = re.findall(r"window.__INITIAL_STATE__=({.*})</script>", html)[0]
note_dict = json.loads(state.replace("undefined", '""'))
# → 拿到的直接就是结构化数据，比 DOM 解析干净 10 倍
```

**这对中文题特别有用**：知乎、百度百科等网站都有 `window.__INITIAL_STATE__` 或 `<script id="js-initialData">`，直接提取比解析 HTML 精准得多。

其他工具：
- `html.unescape()` 替代硬编码实体（处理所有 HTML 实体，不只 6 个）
- `parsel`（Scrapy 选择器引擎）做精确 XPath/CSS 提取
- `jieba` 中文分词 + 停用词过滤

### 落地方案

**A. 三层内容清洗管线（替换当前的纯正则方案）**

```
原始 HTML
  → Layer 1: 规则清洗（不需要 LLM，毫秒级）
     • 删除 EXCLUDED_TAGS（nav/footer/header/aside/script/style/iframe/noscript）
     • 删除 class/id 匹配 NEGATIVE_PATTERNS 的元素
     • 空元素剪枝（< 5 词的叶节点删除）
     • html.unescape() 处理所有实体
  → Layer 2: 文本密度剪枝（不需要 LLM，毫秒级）
     • 对剩余元素算 5 维评分
     • 低于阈值的节点删除
     • 结果转 Markdown
  → Layer 3: LLM 精炼（可选，仅对重要 URL 启用）
     • 把 Markdown 发给 LLM
     • Prompt: "从以下网页内容中，提取与查询 {query} 相关的关键信息，去除所有无关内容"
     • 这一步的输入已经很干净了（经过 Layer 1+2），LLM 效率很高
```

**B. 特定网站优化路径（域名感知）**

```python
DOMAIN_EXTRACTORS = {
    "zh.wikipedia.org": extract_wikipedia_content,     # 提取正文 div#mw-content-text
    "en.wikipedia.org": extract_wikipedia_content,
    "baike.baidu.com":  extract_baidu_baike_content,   # 提取 window.__INITIAL_STATE__
    "zhihu.com":        extract_zhihu_content,          # 提取 js-initialData JSON
    "default":          three_layer_pipeline,           # 默认三层管线
}

def fetch_and_clean(url, query):
    domain = urlparse(url).netloc
    extractor = DOMAIN_EXTRACTORS.get(domain, DOMAIN_EXTRACTORS["default"])
    return extractor(html, query)
```

**C. 查询感知抓取（Query-Aware Fetching）**

不只是抓全文，而是带着搜索查询去抓：

```python
def query_aware_extract(html, query):
    """Layer 2 的增强版：用 BM25 只保留与 query 相关的段落"""
    paragraphs = extract_paragraphs(html)
    scores = bm25_score(query, paragraphs)
    # 只保留 top-K 相关段落，而不是整个页面的前 3000 字
    relevant = [p for p, s in zip(paragraphs, scores) if s > threshold]
    return "\n".join(relevant)
```

**D. 抓取结果质量评分（决定是否需要 LLM 精炼）**

```python
def should_use_llm_refinement(raw_text, query):
    """判断规则清洗后的文本是否需要 LLM 再精炼"""
    # 1. 文本太短（< 100 字）→ 可能抓取失败，不浪费 LLM
    # 2. query 关键词在文本中出现率 < 10% → 可能抓偏了，需要 LLM 过滤
    # 3. 文本超长（> 5000 字）→ 需要 LLM 精炼为关键段落
    # 4. 来自未知域名 → 保守起见用 LLM 清洗
```

### P0-b 实施情况（2026-02-11）

**已完成的改动：**

#### A. 三层内容清洗管线（`three_layer_html_content_cleaning_pipeline.py` — 新文件）

借鉴 crawl4ai `PruningContentFilter` + firecrawl CSS 黑名单 + MediaCrawler JSON 提取：

- Layer 1: 规则清洗（BeautifulSoup + lxml）
  - 删除 EXCLUDED_TAGS（nav/footer/header/aside/script/style/form/iframe/noscript/svg/canvas）
  - 42 个 CSS 选择器黑名单（.header/.footer/.sidebar/.ad/#cookie 等，来自 firecrawl）
  - NEGATIVE_PATTERNS 正则匹配 class/id（nav|footer|sidebar|ads|comment|promo|advert|social|share|cookie|widget|modal|popup|overlay|breadcrumb|menu|navigation|related|recommend|toolbar|banner|sponsor|newsletter|signup|subscription|pagination|pager|copyright|disclaimer）
  - 空元素剪枝（< 5 词的叶节点自底向上删除，最多 3 轮）
  - `html.unescape()` 处理所有 HTML 实体（替代硬编码 6 个）

- Layer 2: 文本密度剪枝（crawl4ai 5 维评分）
  - text_density(0.4) + link_density(0.2) + tag_weight(0.2) + class_id_weight(0.1) + text_length(0.1)
  - tag_weights: article=1.5, main=1.5, p=1.0, h1=1.2, div=0.5, span=0.3 等
  - 低于阈值（0.48）的节点递归删除

- Layer 3: LLM 精炼（可选，由启发式判断触发）
  - 触发条件：文本 > 5000 字或 query 关键词出现率 < 10%
  - Prompt：提取与 query 相关的关键事实
  - 若返回 NO_RELEVANT_CONTENT 则保持原文

#### B. 域名感知提取器（同一文件中）
- **Wikipedia**: 提取 `div#mw-content-text` 正文段落，去除 infobox/navbox/sidebar/reference/toc
- **百度百科**: `window.__INITIAL_STATE__` JSON 提取 → DOM 提取 fallback
- **知乎**: `js-initialData` JSON 提取 → RichContent DOM fallback
- **CSDN**: `div#content_views` 或 `article.baidu_pl` 提取

路由覆盖：en/zh/ja/de/fr/es.wikipedia.org, baike.baidu.com, wapbaike.baidu.com, zhihu.com, blog.csdn.net, m.blog.csdn.net

#### C. 现有文件改动
- `direct_http_web_page_content_fetcher.py`: 旧的纯正则 `_extract_clean_text_from_html()` 替换为调用三层管线；`enrich_search_results_with_full_page_content()` 新增 query/llm_refine_fn/enable_llm_refinement 参数；MAX_PAGE_TEXT_LENGTH 从 3000 提升到 5000
- `language_aware_hybrid_search_dispatcher.py`: `_enrich_non_iqs_results_with_page_content()` 传入 query 和 llm_refine_fn
- `alibaba_iqs_search_client.py`: readpage_scrape 结果经过 `apply_layer1_only()` 清洗；截断从 2000 提升到 3000
- `enhanced_multi_hop_api_server.py`: 新增 `_llm_refine_page_content()` 方法注入 dispatcher

#### D. Trace 日志增强
- `per_run_faithful_api_call_trace_logger.py`: 新增 `record_page_content_cleaning_trace()` 方法
- 输出到 `page_cleaning_trace.jsonl`，记录：url, domain, extractor_used, raw_html_chars, layer1_chars, layer2_chars, layer3_chars, final_chars, cleaning_elapsed_ms

#### 验证结果（Q0 mini 测试）

| URL | 提取器 | 原始HTML | 清洗后 | 减少率 | 耗时 |
|-----|--------|----------|--------|--------|------|
| paper.sciencenet.cn | generic_3layer | 39,865 | 3,534 | -91.1% | 1,917ms |
| yunthe.com | generic_3layer | 10,356 | 150 | -98.6% | 997ms |
| blog.csdn.net | extract_csdn_content | 152,089 | 1,553 | -99.0% | 12ms |
| arxiv.org | generic_3layer | 103,234 | 42,230 | -59.1% | 1,469ms |
| en.wikipedia.org/wiki/RepRap | extract_wikipedia_content | 166,373 | 16,896 | -89.8% | 25ms |

**分析：**
- 三层清洗管线和域名感知提取器均正常工作
- 噪音去除率 59%-99%，域名提取器极快（12-25ms）
- Layer 3 LLM 精炼被触发 3 次，均正确识别为不相关内容（NO_RELEVANT_CONTENT）
- Q0 仍为 0/1（答案 "RepRap Ltd" vs 标准 "RepRapPro Ltd"），根因是 Google+IQS 均不可用（quota 耗尽），Brave/DuckDuckGo 未搜到含 "RepRapPro" 的关键页面
- 当搜索引擎恢复后，清洗管线将显著提升证据质量（去噪后 LLM 更容易提取精确实体名）

**下一步：**
- 待 Google API / IQS 额度恢复后重新验证
- 考虑增加 BM25 相关性过滤（Layer 2 增强版）
- 特定网站提取器可按需扩展（thepaper.cn, 360doc.com 等）

---

## P1-a：LLM 非确定性

### 现状问题
同一题 4 次运行出 4 个不同答案：
- Q0: RepRapPro Limited / RepRap Limited / RepRapPro Limited / RepRap Professional Ltd
- Q3: 322 / 546 / 乱文 / 238（标准答案 591）
- Q4: Love, Actually / Love in the Time... / Dateable Podcast / Love Me

### 借鉴来源

#### 1. MetaGPT Tree-of-Thought BFS
```
生成 N 条候选推理路径 → 用 LLM 对每条打分 → 选最高分
核心：不是跑一次，而是跑 N 次，投票选最稳定的答案
```

#### 2. OpenManus `is_stuck()` 机制
```python
# 当连续给出相同答案时，注入"换个策略"prompt
# 防止陷入局部最优
```

#### 3. Cursor 分线程策略（核心洞察！）
```
Cursor 新版对同一个问题可以并行开多个 thread：
- Thread A：用搜索引擎 1 + prompt 风格 1
- Thread B：用搜索引擎 2 + prompt 风格 2
- Thread C：纯知识回答（不搜索）
→ 汇聚三个 thread 的结果，投票 / 交叉验证
```

#### 4. Ralph Wiggum 循环执行
```
同一个 prompt 反复执行，每次看到上一轮的结果
→ 自我纠正，逐轮收敛到正确答案
```

### 落地方案

**A. 多路径并行推理（Self-Consistency Decoding）**
```
对每个问题，并行跑 K=3 条独立推理路径：
  路径 1：knowledge_answer（纯 LLM 知识）
  路径 2：search → extract_answer（搜索 + 提取）
  路径 3：search（不同查询词）→ extract_answer

汇聚：
  - 如果 3 条路径答案一致 → 高置信度输出
  - 如果 2:1 → 采用多数答案
  - 如果 3 条都不同 → 触发第 4 轮验证搜索
```

**B. 温度控制**
```
temperature=0（已有，确认）
但对于"探索性"搜索查询生成，可以用 temperature=0.3 增加多样性
```

**C. 答案收敛检测**
```
如果多次采样的答案高度相似但不完全相同（如 "RepRapPro Ltd" vs "RepRapPro Limited"）
→ 触发 LLM 做最终格式选择，而不是随机选一个
```

### P1-a 实施情况（2026-02-11）

**已完成的改动：**

#### A. 多候选答案一致性投票模块（`multi_candidate_answer_consistency_voter.py` — 新文件）

借鉴 MetaGPT ScEnsemble + Research_Agent `_check_consistency` 设计：

- `normalize_answer_text_for_comparison()`: 归一化答案文本（小写、去标点、去停用词、合并空白）
- `calculate_answer_pair_jaccard_similarity()`: 两两 Jaccard 相似度计算
- `detect_consensus_among_answer_candidates()`: 共识检测 — 找到 Jaccard >= 0.7 的一致对，按优先级选择
- `select_best_answer_via_llm_arbitration()`: LLM 仲裁 — 借鉴 ScEnsemble prompt 格式，从候选中选最可靠答案
- `select_final_answer_with_consistency_voting()`: 主入口 — 两阶段：一致性检测（零 LLM 调用）→ LLM 仲裁（仅不一致时+1调用）

#### B. 替换 constrained_multi_hop_search_agent.py 的答案选择

- 旧逻辑（priority chain: hop2 > search > knowledge > heuristic）保留为 static fallback
- Phase 3 改为调用 `select_final_answer_with_consistency_voting()`
- 新增 `llm_arbitrate_fn` 和 `trace_logger` 属性（由 API server 注入）
- `reasoning_steps` 中记录投票决策原因

#### C. LLM 仲裁函数

- `large_language_model_call_handlers.py`: 新增 `arbitrate_among_candidate_answers_via_llm()`（temp=0.0, max_tokens=64）
- `enhanced_multi_hop_api_server.py`: 注入 `llm_arbitrate_fn` lambda 和 `trace_logger` 到 search_agent

#### D. Trace 日志增强

- `per_run_faithful_api_call_trace_logger.py`: 新增 `record_answer_consistency_voting_trace()` 方法
- 输出到 `answer_voting_trace.jsonl`，记录：question, candidates, similarity_matrix, consensus_found, llm_arbitration_used, final_answer, answer_source, decision_reason

#### 验证结果（Q0 mini 测试）

| 指标 | 结果 |
|------|------|
| 候选答案 | hop2="RepRap Ltd", search="RepRap Ltd", knowledge="RepRap Ltd", heuristic="2005" |
| 相似度矩阵 | hop2 vs search = 1.0, hop2 vs knowledge = 1.0, search vs knowledge = 1.0 |
| 共识检测 | 成功（3个来源完美一致，Jaccard=1.0） |
| LLM 仲裁 | 未触发（节省 1 次 LLM 调用） |
| 最终决策 | `consensus:hop2+search(sim=1.00)->prefer:hop2` |
| 最终答案 | "RepRap Ltd"（0/1，标准答案 "RepRapPro Ltd"） |

**分析：**
- 一致性投票机制正常工作：3 个推理路径（hop2/search/knowledge）完美一致 → 直接走共识路径，零额外 LLM 调用
- heuristic="2005" 被正确排除（与其他 3 个候选的 Jaccard = 0.0）
- Q0 仍为 0/1 的根因：Google+IQS 均不可用（quota 耗尽），仅靠 Brave，搜索证据不含 "RepRapPro" → 所有推理路径都收敛到错误的 "RepRap Ltd"
- 当搜索引擎恢复后，一致性投票将显著提升鲁棒性：若不同路径返回 "RepRapPro Ltd" 和 "RepRap Ltd"，共识检测 + LLM 仲裁可以选出更精确的答案

**LLM 调用开销：**
- 一致时：+0 LLM 调用（本次测试验证）
- 不一致时：+1 LLM 调用（max_tokens=64，成本极低）

**下一步：**
- 待 Google API / IQS 额度恢复后重新验证多源不一致场景
- 考虑增加 CharacterN-gram 相似度作为 Jaccard 的补充（对短答案更鲁棒）
- P1-b（多跳推理链断裂）可进一步提升中间结果精度

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

## 实施优先级与路线图

### Phase 1（立即可做，预期 +8~15 分）
1. **答案后处理管线** — 借鉴 Research_Agent normalize.py + ReAct normalize_answer
2. **改进 extract_answer prompt** — 加入格式约束、精简要求
3. **三层网页清洗管线** — 替换当前纯正则方案（规则去噪 → 密度剪枝 → LLM 精炼）
4. **强制验证** — 借鉴 Enhancing-MH-QA self_verification

### Phase 2（1-2 天，预期 +10~15 分）
5. **结构化多跳规划器** — 借鉴 Research_Agent multi_hop_decomposition
6. **中间结果校验** — 每跳校验 + 失败重试
7. **多路径并行推理** — 3 条路径 + 投票
8. **域名感知提取器** — Wikipedia/百度百科/知乎等特定网站用专用提取逻辑
9. **per-question 推理暂存目录** — 借鉴 Researcher + deepagents，每题一个 scratchpad 文件夹 + INDEX.md
10. **TODO-list 式推理计划器** — 借鉴 LLMCompiler + ReWOO，plan → execute → check → replan

### Phase 3（3-5 天，预期 +5~10 分）
11. **搜索缓存 + 并行化** — 降低 IQS 延迟影响
12. **预抓取** — hop 间 URL 预抓取
13. **搜索策略多样化** — 多引擎并行 + 结果合并
14. **查询感知抓取** — BM25 只保留与 query 相关段落，不截断前 3000 字

### 总计预期：从 33 分提升到 60-75 分
