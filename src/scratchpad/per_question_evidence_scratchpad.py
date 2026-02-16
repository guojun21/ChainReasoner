"""Per-question evidence scratchpad with BM25 local retrieval.

Why: The multi-hop pipeline currently discards search results after each hop.
When later hops need related information, the system re-searches the web,
wasting time and API calls.  This module persists every search result and
cleaned page to a per-question directory, then provides BM25-based local
retrieval so later hops can re-use evidence already fetched.

Architecture:
  - Each question gets a scratchpad directory: ``scratchpad/q{id}/``
  - Search results → ``evidence/hop{N}_{hash}.md`` (structured markdown)
  - Cleaned pages → ``pages/{domain}_{hash}.md`` (full text)
  - Entity index → ``INDEX.md`` (hop-by-hop entity + summary)
  - In-memory BM25 inverted index for fast local retrieval

References:
  - Researcher (zlb22): workspace/raw + summaries + INDEX.md pattern
  - deepagents: read_file/write_file/grep tool interface
  - bm25s: pure-Python BM25 scoring
"""

import hashlib
import logging
import math
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EvidenceDocument:
    """A single evidence item stored in the scratchpad."""

    file_path: str          # Relative path within scratchpad dir
    hop_num: int
    query: str
    title: str
    url: str
    content: str            # Cleaned text content
    tokens: List[str] = field(default_factory=list)  # Tokenised for BM25
    timestamp: float = 0.0
    relevance_score: float = 0.0  # Original search relevance score


# ---------------------------------------------------------------------------
# Tokeniser — bilingual (Chinese bigram + English word split)
# ---------------------------------------------------------------------------

_EN_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+(?:[-'][a-zA-Z0-9]+)*")
_ZH_CHAR_PATTERN = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")


def tokenize_text_for_bm25_index(text: str) -> List[str]:
    """Tokenise text into terms for BM25 indexing.

    Why two strategies: English words are space-delimited; Chinese has no
    spaces so we use character bigrams (overlapping pairs) which capture
    most compound words without requiring a segmentation library.
    """
    if not text:
        return []
    text_lower = text.lower()
    tokens: List[str] = []

    # English tokens — full words
    for match in _EN_TOKEN_PATTERN.finditer(text_lower):
        word = match.group()
        if len(word) >= 2:  # Skip single chars
            tokens.append(word)

    # Chinese bigrams — overlapping character pairs
    zh_chars = [c for c in text if _ZH_CHAR_PATTERN.match(c)]
    for i in range(len(zh_chars) - 1):
        tokens.append(zh_chars[i] + zh_chars[i + 1])
    # Also add individual CJK chars for single-char matching
    for c in zh_chars:
        tokens.append(c)

    return tokens


# ---------------------------------------------------------------------------
# BM25 Engine — pure Python, no external dependencies
# ---------------------------------------------------------------------------

class InMemoryBM25Index:
    """Lightweight BM25 index over EvidenceDocument objects.

    Why not use an external library: The competition environment may not
    have bm25s/rank_bm25 installed, and the document count per question
    is small (typically 20-80 documents) so a pure-Python implementation
    is both fast enough and dependency-free.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: List[EvidenceDocument] = []
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0.0
        # Inverted index: token -> set of document indices
        self._inverted: Dict[str, set] = {}
        # Per-document token frequency: doc_idx -> {token: count}
        self._tf: List[Dict[str, int]] = []

    @property
    def document_count(self) -> int:
        return len(self.documents)

    def add_document(self, doc: EvidenceDocument) -> None:
        """Add a document to the index (incremental update)."""
        doc_idx = len(self.documents)
        self.documents.append(doc)

        # Tokenise if not already done
        if not doc.tokens:
            doc.tokens = tokenize_text_for_bm25_index(
                f"{doc.title} {doc.content}")

        self.doc_lengths.append(len(doc.tokens))
        total = sum(self.doc_lengths)
        self.avg_doc_length = total / len(self.doc_lengths) if self.doc_lengths else 0

        # Build term frequency for this doc
        tf: Dict[str, int] = {}
        for token in doc.tokens:
            tf[token] = tf.get(token, 0) + 1
        self._tf.append(tf)

        # Update inverted index
        for token in set(doc.tokens):
            if token not in self._inverted:
                self._inverted[token] = set()
            self._inverted[token].add(doc_idx)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[EvidenceDocument, float]]:
        """Search the index and return top-k (document, score) pairs.

        Uses standard BM25 scoring:
          score(D, Q) = sum over q in Q of:
            IDF(q) * tf(q,D) * (k1 + 1) / (tf(q,D) + k1 * (1 - b + b * |D| / avgdl))
        """
        if not self.documents:
            return []

        query_tokens = tokenize_text_for_bm25_index(query)
        if not query_tokens:
            return []

        n = len(self.documents)
        scores: Dict[int, float] = {}

        for token in set(query_tokens):
            if token not in self._inverted:
                continue
            doc_indices = self._inverted[token]
            df = len(doc_indices)
            # IDF with smoothing (BM25 Lucene variant)
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))

            for doc_idx in doc_indices:
                tf_val = self._tf[doc_idx].get(token, 0)
                doc_len = self.doc_lengths[doc_idx]
                denom = tf_val + self.k1 * (
                    1 - self.b + self.b * doc_len / max(self.avg_doc_length, 1))
                term_score = idf * tf_val * (self.k1 + 1) / max(denom, 0.001)
                scores[doc_idx] = scores.get(doc_idx, 0.0) + term_score

        # Sort by score descending, take top_k
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(self.documents[idx], score) for idx, score in ranked]


# ---------------------------------------------------------------------------
# Main scratchpad class
# ---------------------------------------------------------------------------

class PerQuestionEvidenceScratchpad:
    """Manages a per-question local knowledge base on disk + in-memory BM25 index.

    Why: Multi-hop QA often searches for overlapping topics across hops.
    By persisting all search results and cleaned pages to a structured
    directory, later hops can retrieve already-fetched evidence locally
    instead of re-querying the web — saving time and improving evidence
    utilisation.

    Usage:
        sp = PerQuestionEvidenceScratchpad(base_dir="logs/run_.../scratchpad", question_id="0")
        sp.write_search_evidence(hop_num=1, query="RepRap founder", results=[...])
        hits = sp.search_local("who founded RepRap", top_k=3)
    """

    def __init__(
        self,
        base_dir: str,
        question_id: str,
        question_text: str = "",
    ):
        self.question_id = str(question_id)
        self.question_text = question_text
        self.root = Path(base_dir) / f"q{self.question_id}"
        self.evidence_dir = self.root / "evidence"
        self.pages_dir = self.root / "pages"
        self.index_path = self.root / "INDEX.md"

        # Create directories
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.pages_dir.mkdir(parents=True, exist_ok=True)

        # In-memory BM25 index
        self._bm25 = InMemoryBM25Index()

        # Content hash set for deduplication (avoids storing identical evidence)
        self._content_hashes: set = set()

        # Statistics
        self._stats = {
            "evidence_files_written": 0,
            "page_files_written": 0,
            "local_search_queries": 0,
            "local_search_hits": 0,  # queries that returned >= 1 result
            "total_evidence_chars": 0,
            "dedup_skipped": 0,
        }

        # Initialise INDEX.md
        if not self.index_path.exists():
            header = f"# Question {self.question_id}"
            if question_text:
                header += f": {question_text[:120]}"
            header += "\n\n"
            self.index_path.write_text(header, encoding="utf-8")

        logger.info("Scratchpad created: %s", self.root)

    # ── Write operations ──────────────────────────────────────────────

    def write_search_evidence(
        self,
        hop_num: int,
        query: str,
        results: List[Dict[str, Any]],
    ) -> str:
        """Persist search results as a structured markdown file.

        Returns the file path written (relative to scratchpad root).
        """
        if not results:
            return ""

        query_hash = hashlib.md5(query.encode()).hexdigest()[:6]
        filename = f"hop{hop_num}_{query_hash}.md"
        filepath = self.evidence_dir / filename

        # Build markdown content with YAML frontmatter
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        lines = [
            "---",
            f"hop: {hop_num}",
            f'query: "{query[:200]}"',
            f"timestamp: {timestamp}",
            f"result_count: {len(results)}",
            "---",
            "",
        ]

        for item in results:
            title = item.get("title", "Untitled")
            url = item.get("url", "")
            content = item.get("content", "")
            score = item.get("_score", 0.0)

            # Content-hash dedup: skip if we already stored identical content
            content_hash = hashlib.md5(
                (url + content[:2000]).encode(errors="replace")
            ).hexdigest()
            if content_hash in self._content_hashes:
                self._stats["dedup_skipped"] += 1
                logger.debug("Dedup skip: %s (hash=%s)", title[:60], content_hash[:8])
                continue
            self._content_hashes.add(content_hash)

            lines.append(f"## [{title}]({url})")
            if content:
                # Truncate very long content for file storage
                lines.append(content[:3000])
            if score:
                lines.append(f"_score: {score:.3f}")
            lines.append("")

            # Add to BM25 index
            doc = EvidenceDocument(
                file_path=str(filepath.relative_to(self.root)),
                hop_num=hop_num,
                query=query,
                title=title,
                url=url,
                content=content[:3000],
                relevance_score=score,
                timestamp=time.time(),
            )
            self._bm25.add_document(doc)

        text = "\n".join(lines)
        filepath.write_text(text, encoding="utf-8")
        self._stats["evidence_files_written"] += 1
        self._stats["total_evidence_chars"] += len(text)

        logger.debug("Scratchpad evidence written: %s (%d results, %d chars)",
                      filename, len(results), len(text))
        return filename

    def write_page_content(
        self,
        url: str,
        cleaned_text: str,
        hop_num: int = 0,
    ) -> str:
        """Persist a cleaned web page to the pages/ directory.

        Returns the filename written, or empty string if nothing written.
        """
        if not cleaned_text or len(cleaned_text) < 50:
            return ""

        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.hostname or "unknown"
        except Exception:
            domain = "unknown"

        url_hash = hashlib.md5(url.encode()).hexdigest()[:6]
        filename = f"{domain}_{url_hash}.md"
        filepath = self.pages_dir / filename

        # Don't overwrite if already exists (same URL)
        if filepath.exists():
            return filename

        header = f"---\nurl: {url}\nhop: {hop_num}\n---\n\n"
        text = header + cleaned_text[:10000]  # Cap at 10k chars
        filepath.write_text(text, encoding="utf-8")
        self._stats["page_files_written"] += 1
        self._stats["total_evidence_chars"] += len(text)

        # Add to BM25 index
        doc = EvidenceDocument(
            file_path=f"pages/{filename}",
            hop_num=hop_num,
            query="",
            title=domain,
            url=url,
            content=cleaned_text[:5000],
            timestamp=time.time(),
        )
        self._bm25.add_document(doc)

        logger.debug("Scratchpad page written: %s (%d chars)", filename, len(text))
        return filename

    def update_index(
        self,
        hop_num: int,
        entity: str,
        summary: str,
        target: str = "",
    ) -> None:
        """Append a hop entity summary to INDEX.md."""
        entry = f"\n## Hop {hop_num}"
        if target:
            entry += f": {target}"
        entry += f"\n- Entity: **{entity}**\n"
        if summary:
            entry += f"- Evidence: {summary[:300]}\n"

        with open(self.index_path, "a", encoding="utf-8") as fh:
            fh.write(entry)

        logger.debug("Scratchpad index updated: Hop %d entity=%s", hop_num, entity[:60])

    # ── Read / search operations ─────────────────────────────────────

    def search_local(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """Search the local BM25 index for evidence matching the query.

        Returns results in the same format as search engine results
        (list of dicts with title/url/content/_score) so they can be
        directly merged into the hop pipeline's ranked results.

        Args:
            query: Search query.
            top_k: Maximum results to return.
            min_score: Minimum BM25 score threshold.  Documents below
                       this score are filtered out to avoid noise.
        """
        self._stats["local_search_queries"] += 1

        if self._bm25.document_count == 0:
            return []

        results = self._bm25.search(query, top_k=top_k)

        # Filter by minimum score
        filtered = [(doc, score) for doc, score in results if score >= min_score]

        if filtered:
            self._stats["local_search_hits"] += 1

        # Convert to standard search result format
        output: List[Dict[str, Any]] = []
        for doc, score in filtered:
            output.append({
                "title": doc.title,
                "url": doc.url,
                "content": doc.content[:3000],
                "_score": score,
                "_source": "local_scratchpad",
                "_hop_origin": doc.hop_num,
            })

        logger.debug("Scratchpad search: query='%s' -> %d hits (top_score=%.2f)",
                      query[:60], len(output),
                      output[0]["_score"] if output else 0.0)
        return output

    def grep_evidence(self, pattern: str, max_results: int = 20) -> List[Dict[str, str]]:
        """Regex search across all evidence and page files.

        Why: Sometimes the agent needs to find a specific entity name
        or date that BM25 tokenisation might miss (e.g. "RepRapPro"
        as a single token).
        """
        matches: List[Dict[str, str]] = []
        try:
            compiled = re.compile(pattern, re.IGNORECASE)
        except re.error:
            return matches

        for subdir in [self.evidence_dir, self.pages_dir]:
            for filepath in sorted(subdir.glob("*.md")):
                try:
                    text = filepath.read_text(encoding="utf-8")
                except Exception:
                    continue
                for match in compiled.finditer(text):
                    start = max(0, match.start() - 100)
                    end = min(len(text), match.end() + 100)
                    context = text[start:end].replace("\n", " ")
                    matches.append({
                        "file": str(filepath.relative_to(self.root)),
                        "match": match.group(),
                        "context": context,
                    })
                    if len(matches) >= max_results:
                        return matches
        return matches

    def read_evidence(self, file_path: str) -> str:
        """Read a specific evidence file by relative path."""
        full_path = self.root / file_path
        if full_path.exists():
            return full_path.read_text(encoding="utf-8")
        return ""

    def get_index(self) -> str:
        """Return the full INDEX.md content."""
        if self.index_path.exists():
            return self.index_path.read_text(encoding="utf-8")
        return ""

    def get_all_evidence_as_context(self, max_chars: int = 8000) -> str:
        """Return all evidence concatenated as a single context string.

        Why: For the evidence fusion phase, the LLM may benefit from
        seeing ALL collected evidence across all hops, not just what
        was passed per-hop.
        """
        parts: List[str] = []
        total = 0
        for doc in self._bm25.documents:
            entry = f"[{doc.title}]({doc.url})\n{doc.content[:1500]}\n"
            if total + len(entry) > max_chars:
                break
            parts.append(entry)
            total += len(entry)
        return "\n".join(parts)

    def get_stats(self) -> Dict[str, Any]:
        """Return scratchpad usage statistics for trace logging."""
        return {
            **self._stats,
            "bm25_document_count": self._bm25.document_count,
            "scratchpad_root": str(self.root),
        }
