"""Three-layer HTML content cleaning pipeline + domain-aware extractors.

Why: The original regex-only HTML cleaner in direct_http_web_page_content_fetcher.py
keeps navigation, ads, footers, and cookie banners — flooding the LLM context with
noise and reducing evidence quality.  This module replaces that approach with:

  Layer 1 — Rule-based cleaning (BeautifulSoup, milliseconds)
    * Delete excluded tags (nav/footer/header/aside/script/style/form/iframe/noscript)
    * Delete elements whose class/id match NEGATIVE_PATTERNS
    * Prune empty leaf nodes (< 5 words)
    * html.unescape() for all entities

  Layer 2 — Text-density pruning (inspired by crawl4ai PruningContentFilter)
    * 5-dimension scoring: text_density, link_density, tag_weight, class_id_weight, text_length
    * Nodes below threshold are removed
    * Result converted to clean text with paragraph structure

  Layer 3 — LLM refinement (optional, for noisy or very long content)
    * Send cleaned text + query to LLM to extract only relevant facts
    * Triggered by heuristic: text too long (>5000), or query keywords sparse

Domain-aware extractors provide optimised paths for high-frequency sources:
  Wikipedia, 百度百科, 知乎, CSDN — using site-specific selectors/JSON extraction.

References:
  - crawl4ai PruningContentFilter: references/04-网页爬虫/crawl4ai_★59k/crawl4ai/content_filter_strategy.py
  - firecrawl CSS blacklist: references/04-网页爬虫/firecrawl_★81k/apps/api/src/scraper/scrapeURL/lib/removeUnwantedElements.ts
  - MediaCrawler JSON extraction: references/04-网页爬虫 (window.__INITIAL_STATE__ approach)
"""

import html as _html_module
import json as _json
import logging
import math
import re as _re
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from bs4 import BeautifulSoup, Tag, NavigableString, XMLParsedAsHTMLWarning

# Suppress XMLParsedAsHTMLWarning when lxml parses non-HTML content
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — borrowed from crawl4ai + firecrawl
# ---------------------------------------------------------------------------

EXCLUDED_TAGS = frozenset({
    "nav", "footer", "header", "aside", "script", "style",
    "form", "iframe", "noscript", "svg", "canvas",
})

# Regex pattern for class/id values that indicate noise (crawl4ai negative_patterns)
NEGATIVE_PATTERNS = _re.compile(
    r"nav|footer|header|sidebar|ads|advert|comment|promo|social|share|"
    r"cookie|widget|modal|popup|overlay|breadcrumb|menu|navigation|"
    r"related|recommend|toolbar|banner|sponsor|newsletter|signup|"
    r"subscription|pagination|pager|copyright|disclaimer",
    _re.IGNORECASE,
)

# CSS selector blacklist (from firecrawl excludeNonMainTags — 42 selectors)
FIRECRAWL_CSS_BLACKLIST = [
    "header", "footer", "nav", "aside",
    ".header", ".top", ".navbar", "#header",
    ".footer", ".bottom", "#footer",
    ".sidebar", ".side", ".aside", "#sidebar",
    ".modal", ".popup", "#modal", ".overlay",
    ".ad", ".ads", ".advert", "#ad",
    ".lang-selector", ".language", "#language-selector",
    ".social", ".social-media", ".social-links", "#social",
    ".menu", ".navigation", "#nav",
    ".breadcrumbs", "#breadcrumbs",
    ".share", "#share",
    ".widget", "#widget",
    ".cookie", "#cookie",
]

# Tag weights for density scoring (crawl4ai)
TAG_WEIGHTS: Dict[str, float] = {
    "article": 1.5, "main": 1.5,
    "section": 1.0, "p": 1.0,
    "h1": 1.2, "h2": 1.1, "h3": 1.0, "h4": 0.9, "h5": 0.8, "h6": 0.7,
    "div": 0.5, "span": 0.3,
    "li": 0.5, "ul": 0.5, "ol": 0.5,
    "td": 0.5, "th": 0.6,
    "blockquote": 0.8, "pre": 0.7, "code": 0.7,
    "table": 0.6, "tr": 0.4,
    "figure": 0.4, "figcaption": 0.5,
    "dl": 0.5, "dt": 0.5, "dd": 0.5,
}

DENSITY_THRESHOLD = 0.48

# Metric weights (crawl4ai defaults)
METRIC_WEIGHTS = {
    "text_density": 0.4,
    "link_density": 0.2,
    "tag_weight": 0.2,
    "class_id_weight": 0.1,
    "text_length": 0.1,
}


# ---------------------------------------------------------------------------
# Layer 1 — Rule-based cleaning
# ---------------------------------------------------------------------------

def _remove_css_blacklist_elements(soup: BeautifulSoup) -> None:
    """Remove elements matching firecrawl CSS blacklist selectors."""
    for selector in FIRECRAWL_CSS_BLACKLIST:
        for element in soup.select(selector):
            element.decompose()


def _remove_excluded_tags(soup: BeautifulSoup) -> None:
    """Remove all instances of EXCLUDED_TAGS."""
    for tag_name in EXCLUDED_TAGS:
        for element in soup.find_all(tag_name):
            element.decompose()


def _remove_negative_pattern_elements(soup: BeautifulSoup) -> None:
    """Remove elements whose class or id matches NEGATIVE_PATTERNS."""
    for element in soup.find_all(True):
        if not isinstance(element, Tag):
            continue
        classes = " ".join(element.get("class", []))
        element_id = element.get("id", "")
        if classes and NEGATIVE_PATTERNS.search(classes):
            element.decompose()
        elif element_id and NEGATIVE_PATTERNS.search(element_id):
            element.decompose()


def _prune_empty_leaf_nodes(soup: BeautifulSoup, min_words: int = 5) -> None:
    """Remove leaf elements with fewer than min_words words (bottom-up)."""
    changed = True
    passes = 0
    while changed and passes < 3:
        changed = False
        passes += 1
        for element in soup.find_all(True):
            if not isinstance(element, Tag):
                continue
            # Skip if it has child tags (not a leaf)
            if element.find(True):
                continue
            text = element.get_text(strip=True)
            word_count = len(text.split())
            if word_count < min_words:
                element.decompose()
                changed = True


def layer1_rule_based_cleaning(html: str) -> BeautifulSoup:
    """Layer 1: Rule-based HTML cleaning. Returns cleaned BeautifulSoup tree.

    Steps:
      1. Parse with lxml (fast, tolerant)
      2. Remove excluded tags
      3. Remove CSS blacklist elements
      4. Remove negative-pattern class/id elements
      5. Prune empty leaf nodes
      6. html.unescape() all text nodes
    """
    soup = BeautifulSoup(html, "lxml")

    _remove_excluded_tags(soup)
    _remove_css_blacklist_elements(soup)
    _remove_negative_pattern_elements(soup)
    _prune_empty_leaf_nodes(soup)

    # html.unescape() all remaining text nodes
    for text_node in soup.find_all(string=True):
        if isinstance(text_node, NavigableString):
            unescaped = _html_module.unescape(str(text_node))
            if unescaped != str(text_node):
                text_node.replace_with(unescaped)

    return soup


def layer1_extract_text(soup: BeautifulSoup) -> str:
    """Extract clean text from a Layer-1-cleaned soup, preserving paragraph breaks."""
    parts: List[str] = []
    body = soup.find("body") or soup
    for element in body.descendants:
        if isinstance(element, NavigableString):
            text = str(element).strip()
            if text:
                parts.append(text)
        elif isinstance(element, Tag) and element.name in (
            "p", "div", "br", "h1", "h2", "h3", "h4", "h5", "h6",
            "li", "tr", "blockquote", "section", "article",
        ):
            parts.append("\n")
    raw_text = " ".join(parts)
    # Collapse multiple whitespace / newlines
    raw_text = _re.sub(r"[ \t]+", " ", raw_text)
    raw_text = _re.sub(r"\n\s*\n+", "\n\n", raw_text)
    return raw_text.strip()


# ---------------------------------------------------------------------------
# Layer 2 — Text-density pruning (inspired by crawl4ai)
# ---------------------------------------------------------------------------

def _compute_class_id_weight(node: Tag) -> float:
    """Negative weight for noise-indicating class/id values."""
    score = 0.0
    classes = " ".join(node.get("class", []))
    if classes and NEGATIVE_PATTERNS.search(classes):
        score -= 0.5
    element_id = node.get("id", "")
    if element_id and NEGATIVE_PATTERNS.search(element_id):
        score -= 0.5
    return score


def _compute_composite_score(node: Tag) -> float:
    """Compute the 5-dimension composite score for a DOM node."""
    text = node.get_text(strip=True)
    text_len = len(text)
    if text_len == 0:
        return -1.0

    try:
        tag_len = len(node.encode_contents().decode("utf-8", errors="replace"))
    except Exception:
        tag_len = text_len + 10

    link_text_len = sum(
        len(s.strip()) for s in
        (a.get_text(strip=True) for a in node.find_all("a", recursive=False))
        if s
    )

    score = 0.0
    total_weight = 0.0

    # text_density
    density = text_len / tag_len if tag_len > 0 else 0
    score += METRIC_WEIGHTS["text_density"] * density
    total_weight += METRIC_WEIGHTS["text_density"]

    # link_density (lower link ratio is better)
    link_ratio = 1 - (link_text_len / text_len if text_len > 0 else 0)
    score += METRIC_WEIGHTS["link_density"] * link_ratio
    total_weight += METRIC_WEIGHTS["link_density"]

    # tag_weight
    tag_score = TAG_WEIGHTS.get(node.name, 0.5)
    score += METRIC_WEIGHTS["tag_weight"] * tag_score
    total_weight += METRIC_WEIGHTS["tag_weight"]

    # class_id_weight
    class_score = _compute_class_id_weight(node)
    score += METRIC_WEIGHTS["class_id_weight"] * max(0, class_score)
    total_weight += METRIC_WEIGHTS["class_id_weight"]

    # text_length
    score += METRIC_WEIGHTS["text_length"] * math.log(text_len + 1)
    total_weight += METRIC_WEIGHTS["text_length"]

    return score / total_weight if total_weight > 0 else 0


def _prune_tree_by_density(node: Tag, threshold: float = DENSITY_THRESHOLD) -> None:
    """Recursively prune subtrees with composite score below threshold."""
    if not isinstance(node, Tag) or node.name is None:
        return

    score = _compute_composite_score(node)

    if score < threshold and node.name not in ("body", "html", "[document]"):
        node.decompose()
        return

    # Process children (copy list to allow mutation)
    children = [child for child in node.children if isinstance(child, Tag)]
    for child in children:
        _prune_tree_by_density(child, threshold)


def layer2_density_pruning(soup: BeautifulSoup, threshold: float = DENSITY_THRESHOLD) -> str:
    """Layer 2: Remove low-density subtrees and return clean text.

    Operates on an already Layer-1-cleaned soup.
    """
    body = soup.find("body")
    if not body:
        return layer1_extract_text(soup)

    _prune_tree_by_density(body, threshold)
    return layer1_extract_text(soup)


# ---------------------------------------------------------------------------
# Layer 3 — LLM refinement (optional)
# ---------------------------------------------------------------------------

def _should_use_llm_refinement(text: str, query: str) -> bool:
    """Heuristic: decide whether LLM refinement is beneficial.

    Returns True when:
      - text > 5000 chars (too long, needs summarisation)
      - query keyword overlap < 10% (content may be off-topic)
    Returns False when:
      - text < 100 chars (too short, no point wasting an LLM call)
      - query keyword overlap >= 10% (content is already relevant)
    """
    if len(text) < 100:
        return False
    if len(text) > 5000:
        return True

    # Check query keyword overlap
    query_words = set(query.lower().split())
    if not query_words:
        return False
    text_lower = text.lower()
    hits = sum(1 for w in query_words if w in text_lower)
    overlap = hits / len(query_words) if query_words else 0
    return overlap < 0.10


def layer3_llm_refinement(
    text: str,
    query: str,
    llm_refine_fn: Optional[Callable[[str, str], str]] = None,
) -> str:
    """Layer 3: Use LLM to extract only query-relevant facts from cleaned text.

    If llm_refine_fn is None or refinement is not deemed necessary, returns text as-is.
    """
    if not llm_refine_fn:
        return text
    if not _should_use_llm_refinement(text, query):
        return text

    try:
        refined = llm_refine_fn(query, text)
        if refined and len(refined.strip()) > 20:
            return refined.strip()
    except Exception as exc:
        logger.warning("Layer 3 LLM refinement failed: %s", exc)

    return text


# ---------------------------------------------------------------------------
# Domain-aware extractors
# ---------------------------------------------------------------------------

def extract_wikipedia_content(html: str, url: str = "", query: str = "") -> str:
    """Extract main article text from Wikipedia pages.

    Targets: en.wikipedia.org, zh.wikipedia.org
    Strategy: Extract div#mw-content-text paragraphs, skip infobox/navbox/sidebar/references.
    """
    soup = BeautifulSoup(html, "lxml")
    content_div = soup.find("div", id="mw-content-text")
    if not content_div:
        # Fallback to generic pipeline
        return _generic_three_layer_pipeline(html, url, query)

    # Remove noise elements within Wikipedia content
    for cls in ["infobox", "navbox", "sidebar", "mw-editsection",
                "reference", "reflist", "toc", "mw-jump-link",
                "mbox-small", "ambox", "tmbox", "thumb",
                "mw-empty-elt", "shortdescription"]:
        for el in content_div.find_all(class_=cls):
            el.decompose()
    for el in content_div.find_all("table", class_=_re.compile(r"infobox|navbox|sidebar|metadata")):
        el.decompose()
    for el in content_div.find_all("sup", class_="reference"):
        el.decompose()
    # Remove "See also", "References", "External links" sections
    for heading in content_div.find_all(["h2", "h3"]):
        span = heading.find("span", class_="mw-headline")
        if span and span.get_text(strip=True).lower() in (
            "see also", "references", "external links", "further reading",
            "notes", "bibliography", "参见", "参考文献", "外部链接", "注释",
        ):
            # Remove this heading and everything after it until next h2
            for sibling in list(heading.find_next_siblings()):
                if sibling.name == "h2":
                    break
                sibling.decompose()
            heading.decompose()

    parts: List[str] = []
    for p in content_div.find_all(["p", "li", "dd"]):
        text = p.get_text(strip=True)
        if text and len(text) > 10:
            parts.append(_html_module.unescape(text))

    result = "\n\n".join(parts)
    return result.strip() if result.strip() else _generic_three_layer_pipeline(html, url, query)


def extract_baidu_baike_content(html: str, url: str = "", query: str = "") -> str:
    """Extract main content from Baidu Baike pages.

    Strategy 1: Try window.__INITIAL_STATE__ JSON extraction.
    Strategy 2: Fallback to div.main-content CSS selectors.
    """
    # Strategy 1: JSON state extraction
    state_match = _re.search(r"window\.__INITIAL_STATE__\s*=\s*(\{.*?\})\s*[;<]", html, _re.DOTALL)
    if state_match:
        try:
            state_text = state_match.group(1).replace("undefined", '""')
            state = _json.loads(state_text)
            # Navigate to article content — handle None values safely
            article = state.get("lemmaData") or state.get("article") or {}
            if not isinstance(article, dict):
                article = {}
            abstract = article.get("abstract", "")
            content_parts = []
            if abstract:
                content_parts.append(abstract)
            sections = article.get("sections", []) or article.get("content", []) or []
            for section in sections:
                if isinstance(section, dict):
                    title = section.get("title", "")
                    text = section.get("content", "") or section.get("text", "")
                    if title:
                        content_parts.append(f"\n{title}\n")
                    if text:
                        # Strip HTML tags from content
                        clean = _re.sub(r"<[^>]+>", "", text)
                        content_parts.append(_html_module.unescape(clean))
            if content_parts:
                return "\n".join(content_parts).strip()
        except (_json.JSONDecodeError, KeyError, TypeError):
            pass

    # Strategy 2: DOM extraction
    soup = BeautifulSoup(html, "lxml")
    # Try multiple possible content containers
    content = (soup.find("div", class_="main-content") or
               soup.find("div", class_="body-wrapper") or
               soup.find("div", class_="content") or
               soup.find("div", id="content"))
    if content:
        _remove_excluded_tags(BeautifulSoup(str(content), "lxml"))
        parts = []
        for p in content.find_all(["p", "div", "li"]):
            text = p.get_text(strip=True)
            if text and len(text) > 10:
                parts.append(_html_module.unescape(text))
        if parts:
            return "\n\n".join(parts).strip()

    return _generic_three_layer_pipeline(html, url, query)


def extract_zhihu_content(html: str, url: str = "", query: str = "") -> str:
    """Extract answer/article content from Zhihu pages.

    Strategy 1: Try js-initialData JSON extraction.
    Strategy 2: Fallback to RichContent selectors.
    """
    # Strategy 1: JSON state extraction
    state_match = _re.search(
        r'<script\s+id="js-initialData"\s*[^>]*>(.*?)</script>', html, _re.DOTALL)
    if state_match:
        try:
            state = _json.loads(state_match.group(1))
            entities = state.get("initialState", {}).get("entities", {})
            answers = entities.get("answers", {})
            content_parts = []
            for answer_id, answer in answers.items():
                content = answer.get("content", "")
                if content:
                    clean = _re.sub(r"<[^>]+>", "", content)
                    content_parts.append(_html_module.unescape(clean))
            if not content_parts:
                # Try article entities
                articles = entities.get("articles", {})
                for art_id, article in articles.items():
                    content = article.get("content", "")
                    if content:
                        clean = _re.sub(r"<[^>]+>", "", content)
                        content_parts.append(_html_module.unescape(clean))
            if content_parts:
                return "\n\n".join(content_parts).strip()
        except (_json.JSONDecodeError, KeyError, TypeError):
            pass

    # Strategy 2: DOM extraction
    soup = BeautifulSoup(html, "lxml")
    content_parts = []
    for rich in soup.find_all("div", class_=_re.compile(r"RichContent|RichText|Post-RichText")):
        text = rich.get_text(strip=True)
        if text and len(text) > 20:
            content_parts.append(_html_module.unescape(text))
    if content_parts:
        return "\n\n".join(content_parts).strip()

    return _generic_three_layer_pipeline(html, url, query)


def extract_csdn_content(html: str, url: str = "", query: str = "") -> str:
    """Extract article content from CSDN blog pages.

    Targets: blog.csdn.net, m.blog.csdn.net
    Strategy: Extract div#content_views or article.baidu_pl.
    """
    soup = BeautifulSoup(html, "lxml")

    content = (soup.find("div", id="content_views") or
               soup.find("article", class_="baidu_pl") or
               soup.find("div", id="article_content") or
               soup.find("div", class_="article_content"))

    if content:
        # Remove code blocks, they're usually not useful for fact extraction
        for code_block in content.find_all(["pre", "code"]):
            code_block.decompose()
        # Remove copyright notices
        for el in content.find_all(class_=_re.compile(r"copyright|article-copyright")):
            el.decompose()

        parts = []
        for element in content.find_all(["p", "h1", "h2", "h3", "h4", "li", "blockquote"]):
            text = element.get_text(strip=True)
            if text and len(text) > 5:
                parts.append(_html_module.unescape(text))
        if parts:
            return "\n\n".join(parts).strip()

    return _generic_three_layer_pipeline(html, url, query)


# ---------------------------------------------------------------------------
# New domain-aware extractors (2026-02-13)
# ---------------------------------------------------------------------------

def extract_arxiv_content(html: str, url: str = "", query: str = "") -> str:
    """Extract paper metadata + abstract from arXiv abstract/HTML pages.

    Targets: arxiv.org/abs/*, arxiv.org/html/*
    Strategy:
      1. Try ``#abs`` block (abstract page): title + authors + abstract.
      2. Try ``.ltx_document`` (HTML-rendered paper): title + abstract + first
         N paragraphs of the body.
      3. Try ``blockquote.abstract`` (older arXiv layout).
    """
    soup = BeautifulSoup(html, "lxml")
    parts: List[str] = []

    # ── Title ──
    title_el = (
        soup.find("h1", class_="title") or
        soup.find("h1", class_="ltx_title") or
        soup.find("meta", attrs={"name": "citation_title"})
    )
    if title_el:
        if title_el.name == "meta":
            parts.append(title_el.get("content", ""))
        else:
            parts.append(title_el.get_text(strip=True).replace("Title:", "").strip())

    # ── Authors ──
    authors_el = (
        soup.find("div", class_="authors") or
        soup.find("div", class_="ltx_authors")
    )
    if authors_el:
        parts.append("Authors: " + authors_el.get_text(strip=True).replace("Authors:", "").strip())

    # ── Abstract ──
    abstract_el = (
        soup.find("blockquote", class_="abstract") or
        soup.find("div", class_="ltx_abstract") or
        soup.find("div", id="abs")
    )
    if abstract_el:
        abstract_text = abstract_el.get_text(strip=True)
        # Remove leading "Abstract:" label
        abstract_text = _re.sub(r"^Abstract\s*[:：]?\s*", "", abstract_text, flags=_re.IGNORECASE)
        parts.append("Abstract: " + abstract_text)

    # ── Body (first ~3000 chars from ltx_document or main content) ──
    body_el = soup.find("div", class_="ltx_document") or soup.find("article")
    if body_el:
        body_parts: List[str] = []
        char_budget = 3000
        for p in body_el.find_all(["p", "div"], recursive=True):
            # Skip if inside abstract (already extracted)
            if p.find_parent(class_=_re.compile(r"abstract|ltx_abstract")):
                continue
            text = p.get_text(strip=True)
            if text and len(text) > 15:
                body_parts.append(_html_module.unescape(text))
                char_budget -= len(text)
                if char_budget <= 0:
                    break
        if body_parts:
            parts.append("\n".join(body_parts))

    result = "\n\n".join(parts).strip()
    return result if len(result) > 50 else _generic_three_layer_pipeline(html, url, query)


def extract_github_content(html: str, url: str = "", query: str = "") -> str:
    """Extract README / issue / discussion content from GitHub pages.

    Targets: github.com
    Strategy:
      - README pages: ``.markdown-body`` inside ``#readme``.
      - Issue/PR pages: ``.comment-body`` elements (question + top answers).
      - Repo landing: ``.markdown-body`` (first one is usually the README).
    """
    soup = BeautifulSoup(html, "lxml")
    parts: List[str] = []

    # ── Repo name / title ──
    repo_title = soup.find("strong", class_="mr-2") or soup.find("h1", class_="gh-header-title")
    if repo_title:
        parts.append(repo_title.get_text(strip=True))

    # ── README ──
    readme = soup.find("div", id="readme")
    if readme:
        md_body = readme.find("article", class_="markdown-body") or readme
        text = md_body.get_text(separator="\n", strip=True)
        if text and len(text) > 30:
            parts.append(text[:4000])

    # ── Issue / PR comments ──
    if not readme:
        comments = soup.find_all("div", class_="comment-body")
        for i, comment in enumerate(comments[:5]):  # top 5 comments
            text = comment.get_text(separator="\n", strip=True)
            if text and len(text) > 20:
                parts.append(f"[Comment {i+1}] {text[:2000]}")

    # ── Fallback: any .markdown-body ──
    if not parts:
        for md in soup.find_all("article", class_="markdown-body"):
            text = md.get_text(separator="\n", strip=True)
            if text and len(text) > 30:
                parts.append(text[:4000])
                break

    result = "\n\n".join(parts).strip()
    return result if len(result) > 50 else _generic_three_layer_pipeline(html, url, query)


def extract_stackoverflow_content(html: str, url: str = "", query: str = "") -> str:
    """Extract question + top answers from StackOverflow pages.

    Targets: stackoverflow.com, *.stackexchange.com
    Strategy:
      - Question: ``.question .js-post-body`` or ``#question .s-prose``.
      - Answers: ``.answer .js-post-body`` sorted by vote count.
      - Keep question title from ``#question-header h1``.
    """
    soup = BeautifulSoup(html, "lxml")
    parts: List[str] = []

    # ── Question title ──
    header = soup.find("div", id="question-header")
    if header:
        h1 = header.find("h1")
        if h1:
            parts.append("Q: " + h1.get_text(strip=True))

    # ── Question body ──
    question_div = soup.find("div", class_="question")
    if question_div:
        body = question_div.find("div", class_="js-post-body") or question_div.find("div", class_="s-prose")
        if body:
            text = body.get_text(separator="\n", strip=True)
            if text:
                parts.append(text[:2000])

    # ── Answers (sorted by vote, take top 3) ──
    answers = soup.find_all("div", class_="answer")
    # Sort by data-score attribute if available
    scored_answers: List[Tuple[int, Any]] = []
    for ans in answers:
        score_str = ans.get("data-answerid", "0")
        vote_cell = ans.find("div", class_="js-vote-count")
        vote = 0
        if vote_cell:
            try:
                vote = int(vote_cell.get_text(strip=True))
            except ValueError:
                pass
        scored_answers.append((vote, ans))
    scored_answers.sort(key=lambda x: x[0], reverse=True)

    for rank, (vote, ans) in enumerate(scored_answers[:3], 1):
        body = ans.find("div", class_="js-post-body") or ans.find("div", class_="s-prose")
        if body:
            text = body.get_text(separator="\n", strip=True)
            if text and len(text) > 20:
                # Remove code blocks for brevity — facts are usually in prose
                clean_text = _re.sub(r"```[\s\S]*?```", "[code]", text)
                parts.append(f"[Answer {rank}, votes={vote}] {clean_text[:2000]}")

    result = "\n\n".join(parts).strip()
    return result if len(result) > 50 else _generic_three_layer_pipeline(html, url, query)


def extract_news_generic_content(html: str, url: str = "", query: str = "") -> str:
    """Extract article body from generic Chinese/international news sites.

    Targets: sohu.com, sina.cn, ifeng.com, cnr.cn, 163.com, thepaper.cn,
             and any site with standard ``<article>`` or ``[itemprop=articleBody]``.

    Strategy (priority order):
      1. ``[itemprop="articleBody"]`` — Schema.org standard.
      2. ``<article>`` tag — HTML5 semantic.
      3. Common class patterns: ``.article-content``, ``.post-content``,
         ``.article_content``, ``#artibody``, ``.art_content``.
      4. Fallback to generic 3-layer pipeline.
    """
    soup = BeautifulSoup(html, "lxml")

    # Remove noise first
    for tag_name in ("nav", "footer", "header", "aside", "script", "style",
                     "form", "iframe", "noscript"):
        for el in soup.find_all(tag_name):
            el.decompose()
    for el in soup.find_all(class_=_re.compile(
            r"comment|recommend|related|sidebar|share|ad[sv]|copyright|disclaimer")):
        el.decompose()

    # ── Title ──
    parts: List[str] = []
    title_el = soup.find("h1") or soup.find("title")
    if title_el:
        title_text = title_el.get_text(strip=True)
        if title_text:
            parts.append(title_text)

    # ── Article body — try multiple selectors ──
    content_el = (
        soup.find(attrs={"itemprop": "articleBody"}) or
        soup.find("article") or
        soup.find("div", class_=_re.compile(
            r"article[_-]?content|post[_-]?content|art[_-]?content|"
            r"article[_-]?body|main[_-]?content|entry[_-]?content")) or
        soup.find("div", id=_re.compile(
            r"artibody|article[_-]?content|main[_-]?content")) or
        soup.find("div", class_="content")
    )

    if content_el:
        for p in content_el.find_all(["p", "h2", "h3", "h4", "li", "blockquote"]):
            text = p.get_text(strip=True)
            if text and len(text) > 8:
                parts.append(_html_module.unescape(text))

    result = "\n\n".join(parts).strip()
    return result if len(result) > 80 else _generic_three_layer_pipeline(html, url, query)


# ---------------------------------------------------------------------------
# Domain routing
# ---------------------------------------------------------------------------

def _match_domain(netloc: str, pattern: str) -> bool:
    """Check if netloc matches a domain pattern (allows subdomain matching)."""
    return netloc == pattern or netloc.endswith("." + pattern)


DOMAIN_EXTRACTORS: List[Tuple[str, Callable[[str, str, str], str]]] = [
    # ── Existing extractors ──
    ("en.wikipedia.org", extract_wikipedia_content),
    ("zh.wikipedia.org", extract_wikipedia_content),
    ("ja.wikipedia.org", extract_wikipedia_content),
    ("de.wikipedia.org", extract_wikipedia_content),
    ("fr.wikipedia.org", extract_wikipedia_content),
    ("es.wikipedia.org", extract_wikipedia_content),
    ("baike.baidu.com", extract_baidu_baike_content),
    ("wapbaike.baidu.com", extract_baidu_baike_content),
    ("zhihu.com", extract_zhihu_content),
    ("blog.csdn.net", extract_csdn_content),
    # ── New extractors (2026-02-13) ──
    ("arxiv.org", extract_arxiv_content),
    ("github.com", extract_github_content),
    ("stackoverflow.com", extract_stackoverflow_content),
    ("stackexchange.com", extract_stackoverflow_content),
    # Chinese news sites
    ("news.sohu.com", extract_news_generic_content),
    ("m.sohu.com", extract_news_generic_content),
    ("sohu.com", extract_news_generic_content),
    ("sina.cn", extract_news_generic_content),
    ("sina.com.cn", extract_news_generic_content),
    ("finance.sina.cn", extract_news_generic_content),
    ("ifeng.com", extract_news_generic_content),
    ("cnr.cn", extract_news_generic_content),
    ("163.com", extract_news_generic_content),
    ("thepaper.cn", extract_news_generic_content),
    # International news
    ("bbc.com", extract_news_generic_content),
    ("bbc.co.uk", extract_news_generic_content),
    ("reuters.com", extract_news_generic_content),
    ("cnn.com", extract_news_generic_content),
    ("nytimes.com", extract_news_generic_content),
    ("theguardian.com", extract_news_generic_content),
]


def _get_domain_extractor(url: str) -> Optional[Callable[[str, str, str], str]]:
    """Return the domain-specific extractor for a URL, or None for generic pipeline."""
    if not url:
        return None
    try:
        netloc = urlparse(url).netloc.lower()
    except Exception:
        return None
    for pattern, extractor in DOMAIN_EXTRACTORS:
        if _match_domain(netloc, pattern):
            return extractor
    return None


# ---------------------------------------------------------------------------
# Generic three-layer pipeline
# ---------------------------------------------------------------------------

def _generic_three_layer_pipeline(
    html: str,
    url: str = "",
    query: str = "",
    llm_refine_fn: Optional[Callable[[str, str], str]] = None,
    enable_llm_refinement: bool = False,
) -> str:
    """Apply the full 3-layer cleaning pipeline to arbitrary HTML.

    Layer 1: Rule-based cleaning
    Layer 2: Text-density pruning
    Layer 3: LLM refinement (optional)
    """
    soup = layer1_rule_based_cleaning(html)
    text = layer2_density_pruning(soup)

    if enable_llm_refinement and llm_refine_fn and query:
        text = layer3_llm_refinement(text, query, llm_refine_fn)

    return text


# ---------------------------------------------------------------------------
# Public API — main entry point
# ---------------------------------------------------------------------------

def clean_html_with_three_layer_pipeline(
    html: str,
    url: str = "",
    query: str = "",
    llm_refine_fn: Optional[Callable[[str, str], str]] = None,
    enable_llm_refinement: bool = False,
    trace_logger: Optional[Any] = None,
) -> str:
    """Main entry: clean raw HTML using domain-aware routing + 3-layer pipeline.

    Args:
        html: Raw HTML string.
        url: Source URL (used for domain routing).
        query: Search query (used for Layer 3 relevance and BM25).
        llm_refine_fn: Optional callable(query, text) -> refined_text for Layer 3.
        enable_llm_refinement: Whether to enable Layer 3 LLM refinement.
        trace_logger: Optional trace logger for recording cleaning metrics.

    Returns:
        Cleaned text string.
    """
    start_time = time.time()
    extractor_used = "generic_3layer"

    # Try domain-specific extractor first
    domain_extractor = _get_domain_extractor(url)
    if domain_extractor:
        try:
            netloc = urlparse(url).netloc.lower()
        except Exception:
            netloc = "unknown"
        extractor_used = domain_extractor.__name__
        try:
            text = domain_extractor(html, url, query)
            if text and len(text.strip()) > 50:
                elapsed_ms = int((time.time() - start_time) * 1000)
                logger.info("Domain extractor %s: url=%s chars=%d elapsed_ms=%d",
                            extractor_used, url[:80], len(text), elapsed_ms)
                if trace_logger and hasattr(trace_logger, "record_page_content_cleaning_trace"):
                    trace_logger.record_page_content_cleaning_trace(
                        url=url, domain=netloc, extractor_used=extractor_used,
                        raw_html_chars=len(html),
                        layer1_chars=len(text), layer2_chars=len(text),
                        layer3_chars=0, final_chars=len(text),
                        cleaning_elapsed_ms=elapsed_ms,
                    )
                return text
        except Exception as exc:
            logger.warning("Domain extractor %s failed for %s: %s, falling back to generic",
                           extractor_used, url[:60], exc)
            extractor_used = "generic_3layer"

    # Generic 3-layer pipeline
    try:
        netloc = urlparse(url).netloc.lower() if url else "unknown"
    except Exception:
        netloc = "unknown"

    soup = layer1_rule_based_cleaning(html)
    layer1_text = layer1_extract_text(soup)
    layer1_chars = len(layer1_text)

    layer2_text = layer2_density_pruning(soup)
    layer2_chars = len(layer2_text)

    layer3_chars = 0
    final_text = layer2_text

    if enable_llm_refinement and llm_refine_fn and query:
        refined = layer3_llm_refinement(layer2_text, query, llm_refine_fn)
        if refined != layer2_text:
            layer3_chars = len(refined)
            final_text = refined

    elapsed_ms = int((time.time() - start_time) * 1000)
    logger.info("3-layer pipeline: url=%s L1=%d L2=%d L3=%d final=%d elapsed_ms=%d",
                url[:80], layer1_chars, layer2_chars, layer3_chars, len(final_text), elapsed_ms)

    if trace_logger and hasattr(trace_logger, "record_page_content_cleaning_trace"):
        trace_logger.record_page_content_cleaning_trace(
            url=url, domain=netloc, extractor_used=extractor_used,
            raw_html_chars=len(html),
            layer1_chars=layer1_chars, layer2_chars=layer2_chars,
            layer3_chars=layer3_chars, final_chars=len(final_text),
            cleaning_elapsed_ms=elapsed_ms,
        )

    return final_text


def apply_layer1_only(html: str) -> str:
    """Lightweight Layer-1-only cleaning — for IQS readpage_scrape results.

    IQS already returns scraped markdown, but sometimes with HTML residue.
    This applies just the rule-based cleaning without density pruning.
    """
    if not html or len(html) < 20:
        return html
    # Quick check: if there are no HTML tags, return as-is
    if "<" not in html:
        return html
    soup = layer1_rule_based_cleaning(html)
    return layer1_extract_text(soup)
