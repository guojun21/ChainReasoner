"""Direct HTTP web page content fetcher — enrich search snippets with full page text.

Why: Brave and DuckDuckGo only return short snippets (200-500 chars),
while many competition questions require specific facts (names, numbers,
dates) buried deep in the page. This fetcher grabs the full HTML, runs it
through a 3-layer cleaning pipeline (rule-based + density pruning + optional
LLM refinement) with domain-aware extractors, giving the LLM high-quality
evidence per result.

P0-b upgrade: Replaced the old regex-only cleaner with the three-layer
pipeline in three_layer_html_content_cleaning_pipeline.py.
"""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse

import requests

from src.search.three_layer_html_content_cleaning_pipeline import (
    clean_html_with_three_layer_pipeline,
)

logger = logging.getLogger(__name__)

# Limits — raised from 3000 to 5000 because cleaned content has better signal-to-noise
MAX_PAGE_TEXT_LENGTH = 5000  # chars to keep per page
FETCH_TIMEOUT_SECONDS = 8  # Reduced from 10: faster fail on slow pages
MAX_CONCURRENT_FETCHES = 3

# ── Large-file protection ──
# Why: PDF/DOC/media files can be 20-30 MB.  requests.get() downloads the
# entire body before Content-Type is checked, wasting 30-90 seconds per file.
# With stream=True we inspect headers first and abort before reading the body.
MAX_RESPONSE_BYTES = 2_000_000  # 2 MB — anything larger is almost certainly not a useful HTML page

# URL suffixes that are never useful HTML pages — skip immediately
_BINARY_FILE_SUFFIXES = frozenset({
    # NOTE: .pdf intentionally excluded — PDFs are now downloaded and text-extracted
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".zip", ".tar", ".gz", ".bz2", ".rar", ".7z",
    ".mp3", ".mp4", ".avi", ".mov", ".wmv", ".flv", ".mkv", ".webm",
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg", ".webp", ".ico",
    ".woff", ".woff2", ".ttf", ".eot",
})

# PDF-specific constants
_MAX_PDF_BYTES = 5_000_000  # 5 MB — PDFs are larger than HTML but may contain valuable text

# ── URL blacklist — domains that are known to be useless for content extraction ──
# Why: These sites either require login (Facebook, LinkedIn, Instagram),
# return mostly JS-rendered noise (TikTok), contain huge irrelevant datasets
# (HuggingFace), or are video-only (YouTube).  Attempting to fetch them
# wastes time and pollutes the LLM context with garbage.
_URL_BLACKLIST_DOMAINS = frozenset({
    "facebook.com", "www.facebook.com", "m.facebook.com",
    "instagram.com", "www.instagram.com",
    "twitter.com", "www.twitter.com", "x.com", "www.x.com",
    "linkedin.com", "www.linkedin.com",
    "tiktok.com", "www.tiktok.com",
    "huggingface.co", "www.huggingface.co",
    "youtube.com", "www.youtube.com", "m.youtube.com",
})


def _is_url_blacklisted(url: str) -> bool:
    """True if the URL's domain is in the blacklist."""
    try:
        host = urlparse(url).hostname or ""
        # Strip leading 'www.' for matching (already in set, but just in case)
        return host in _URL_BLACKLIST_DOMAINS or host.lstrip("www.") in _URL_BLACKLIST_DOMAINS
    except Exception:
        return False


def _has_binary_file_suffix(url: str) -> bool:
    """True if the URL path ends with a known binary file extension (PDF, DOC, etc.).

    Why: Downloading a 30 MB PDF only to discard it after Content-Type check
    wastes 30-90 seconds.  Checking the URL suffix first is instant.
    """
    try:
        path = urlparse(url).path.lower()
        _, ext = os.path.splitext(path)
        return ext in _BINARY_FILE_SUFFIXES
    except Exception:
        return False


def _is_pdf_url(url: str) -> bool:
    """True if the URL path ends with .pdf (case-insensitive)."""
    try:
        path = urlparse(url).path.lower()
        return path.endswith(".pdf")
    except Exception:
        return False


def _fetch_and_extract_pdf_text(
    url: str,
    timeout: int = FETCH_TIMEOUT_SECONDS,
    trace_logger: Optional[Any] = None,
) -> str:
    """Download a PDF and extract its text content using PyPDF2.

    Why: PDFs often contain valuable information (academic papers, reports,
    official documents) that the multi-hop pipeline should not discard.
    We cap downloads at 5 MB and truncate extracted text to MAX_PAGE_TEXT_LENGTH.

    Returns extracted text or empty string on failure.
    """
    import io

    try:
        from PyPDF2 import PdfReader
    except ImportError:
        logger.warning("PyPDF2 not installed — cannot extract PDF text from %s", url[:80])
        return ""

    try:
        response = requests.get(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0",
            },
            timeout=timeout,
            stream=True,
        )
        if response.status_code != 200:
            response.close()
            return ""

        # Check Content-Length before downloading
        content_length_str = response.headers.get("Content-Length", "")
        if content_length_str:
            try:
                if int(content_length_str) > _MAX_PDF_BYTES:
                    logger.debug("Skipping oversized PDF (%s bytes): %s",
                                 content_length_str, url[:80])
                    response.close()
                    return ""
            except ValueError:
                pass

        # Download with size cap
        chunks = []
        bytes_read = 0
        for chunk in response.iter_content(chunk_size=65536, decode_unicode=False):
            chunks.append(chunk)
            bytes_read += len(chunk)
            if bytes_read > _MAX_PDF_BYTES:
                logger.debug("Aborting PDF download after %d bytes: %s",
                             bytes_read, url[:80])
                response.close()
                return ""
        response.close()

        # Extract text from PDF
        pdf_bytes = b"".join(chunks)
        reader = PdfReader(io.BytesIO(pdf_bytes))
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)

        full_text = "\n".join(text_parts).strip()
        if not full_text:
            logger.debug("PDF has no extractable text: %s", url[:80])
            return ""

        # Truncate to max length
        result = full_text[:MAX_PAGE_TEXT_LENGTH]
        logger.info("PDF text extracted: %d chars from %d pages (%d bytes): %s",
                     len(result), len(reader.pages), bytes_read, url[:80])

        if trace_logger:
            trace_logger.record_search_api_call(
                service_name="pdf-text-extractor",
                query=f"extract text from {url[:60]}",
                request_params={"url": url, "pdf_bytes": bytes_read, "pages": len(reader.pages)},
                response_data={"text_length": len(result)},
                result_count=1,
                elapsed_ms=0,
                status="success",
            )

        return result

    except Exception as exc:
        logger.debug("PDF extraction failed for %s: %s", url[:80], exc)
        return ""


def _fetch_and_clean_single_page(
    url: str,
    query: str = "",
    timeout: int = FETCH_TIMEOUT_SECONDS,
    llm_refine_fn: Optional[Callable[[str, str], str]] = None,
    enable_llm_refinement: bool = False,
    trace_logger: Optional[Any] = None,
) -> str:
    """Fetch a single URL and return cleaned text via the 3-layer pipeline.

    Why (large-file protection): Uses ``stream=True`` so the HTTP response
    headers are read first.  If Content-Type is not HTML/text, or
    Content-Length exceeds MAX_RESPONSE_BYTES (2 MB), the connection is
    closed immediately — no body bytes are downloaded.  This prevents the
    old behaviour where a 30 MB PDF was fully downloaded before being
    discarded, wasting 30-90 seconds per file.

    Returns empty string on any error (timeout, 4xx/5xx, encoding issues).
    """
    if not url or not url.startswith("http"):
        return ""
    if _is_url_blacklisted(url):
        logger.debug("Skipping blacklisted URL: %s", url[:80])
        return ""
    # Route PDFs to dedicated text extractor
    if _is_pdf_url(url):
        return _fetch_and_extract_pdf_text(url, timeout=timeout, trace_logger=trace_logger)
    if _has_binary_file_suffix(url):
        logger.debug("Skipping binary-suffix URL: %s", url[:80])
        return ""
    try:
        # stream=True: only download headers first, body is deferred
        response = requests.get(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0",
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
            },
            timeout=timeout,
            allow_redirects=True,
            stream=True,
        )
        if response.status_code != 200:
            response.close()
            return ""

        # ── Header-based guards (before downloading body) ──
        content_type = response.headers.get("Content-Type", "").lower()
        # If server returns PDF content type, route to PDF extractor
        if "application/pdf" in content_type:
            response.close()
            return _fetch_and_extract_pdf_text(url, timeout=timeout, trace_logger=trace_logger)
        if "html" not in content_type and "text" not in content_type:
            logger.debug("Skipping non-HTML Content-Type (%s): %s", content_type[:40], url[:80])
            response.close()
            return ""

        content_length_str = response.headers.get("Content-Length", "")
        if content_length_str:
            try:
                content_length = int(content_length_str)
                if content_length > MAX_RESPONSE_BYTES:
                    logger.debug(
                        "Skipping oversized response (%d bytes > %d max): %s",
                        content_length, MAX_RESPONSE_BYTES, url[:80])
                    response.close()
                    return ""
            except ValueError:
                pass  # Malformed Content-Length — proceed cautiously

        # ── Download body with size cap ──
        # Even if Content-Length header is missing/wrong, cap actual bytes read
        chunks = []
        bytes_read = 0
        for chunk in response.iter_content(chunk_size=65536, decode_unicode=False):
            chunks.append(chunk)
            bytes_read += len(chunk)
            if bytes_read > MAX_RESPONSE_BYTES:
                logger.debug(
                    "Aborting download after %d bytes (max %d): %s",
                    bytes_read, MAX_RESPONSE_BYTES, url[:80])
                response.close()
                return ""
        response.close()

        # Decode to text
        encoding = response.encoding or "utf-8"
        html_text = b"".join(chunks).decode(encoding, errors="replace")

        return clean_html_with_three_layer_pipeline(
            html=html_text,
            url=url,
            query=query,
            llm_refine_fn=llm_refine_fn,
            enable_llm_refinement=enable_llm_refinement,
            trace_logger=trace_logger,
        )
    except Exception as exc:
        logger.debug("Fetch failed for %s: %s", url[:80], exc)
        return ""


# Keep the old name as a thin wrapper for backward compatibility
def fetch_single_web_page_as_clean_text(url: str, timeout: int = FETCH_TIMEOUT_SECONDS) -> str:
    """Fetch a single URL and return cleaned text content (legacy API).

    Note: This wrapper does not pass query/llm_refine_fn — use
    _fetch_and_clean_single_page() directly for full pipeline features.
    """
    return _fetch_and_clean_single_page(url=url, query="", timeout=timeout)


def enrich_search_results_with_full_page_content(
    results: List[Dict[str, str]],
    top_n: int = 3,
    max_content_length: int = MAX_PAGE_TEXT_LENGTH,
    query: str = "",
    llm_refine_fn: Optional[Callable[[str, str], str]] = None,
    enable_llm_refinement: bool = False,
    trace_logger: Optional[Any] = None,
    scratchpad: Optional[Any] = None,
) -> List[Dict[str, str]]:
    """Fetch full page text for the top N search results and replace/extend content.

    Why: Search engine snippets are typically 200-500 chars. For multi-hop
    reasoning questions, we need the actual page content to find specific
    facts. This function fetches the top N URLs concurrently, cleans them
    via the 3-layer pipeline, and replaces the short snippet with the
    cleaned text (truncated to max_content_length).

    Only replaces content if the fetched text is longer than the existing snippet.

    Args:
        results: List of search result dicts with 'url' and 'content' keys.
        top_n: Number of top results to fetch.
        max_content_length: Max chars to keep per page.
        query: Search query for query-aware cleaning.
        llm_refine_fn: Optional LLM refinement callable for Layer 3.
        enable_llm_refinement: Whether Layer 3 is active.
        trace_logger: Optional trace logger for cleaning metrics.
        scratchpad: Optional PerQuestionEvidenceScratchpad for persisting
                    fetched page content to the local knowledge base.
    """
    if not results:
        return results

    urls_to_fetch = []
    skipped_blacklist = 0
    for idx in range(min(top_n, len(results))):
        if results[idx] is None:
            continue
        url = results[idx].get("url", "")
        if not url or not url.startswith("http"):
            continue
        if _is_url_blacklisted(url):
            skipped_blacklist += 1
            logger.info("Skipping blacklisted URL in enrichment: %s", url[:80])
            continue
        urls_to_fetch.append((idx, url))

    if not urls_to_fetch:
        return results

    start_time = time.time()

    # Fetch and clean pages concurrently
    fetched = {}
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_FETCHES) as executor:
        future_to_idx = {
            executor.submit(
                _fetch_and_clean_single_page,
                url=url,
                query=query,
                llm_refine_fn=llm_refine_fn,
                enable_llm_refinement=enable_llm_refinement,
                trace_logger=trace_logger,
            ): idx
            for idx, url in urls_to_fetch
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                text = future.result()
                if text:
                    fetched[idx] = text
            except Exception:
                pass

    # Replace content if fetched text is more informative
    enriched_count = 0
    for idx, full_text in fetched.items():
        if results[idx] is None:
            continue
        existing_content = results[idx].get("content", "")
        if len(full_text) > len(existing_content):
            results[idx]["content"] = full_text[:max_content_length]
            enriched_count += 1
            # Persist cleaned page to scratchpad for local retrieval
            if scratchpad and hasattr(scratchpad, "write_page_content"):
                page_url = results[idx].get("url", "")
                if page_url:
                    scratchpad.write_page_content(
                        url=page_url,
                        cleaned_text=full_text[:max_content_length],
                    )

    elapsed_ms = int((time.time() - start_time) * 1000)
    logger.info("Page enrichment: fetched=%d/%d enriched=%d elapsed_ms=%d",
                len(fetched), len(urls_to_fetch), enriched_count, elapsed_ms)

    if trace_logger and fetched:
        trace_logger.record_search_api_call(
            service_name="page-content-fetcher",
            query=f"enrich top {top_n} URLs (query={query[:40]})",
            request_params={"urls": [url for _, url in urls_to_fetch],
                            "query": query, "llm_refinement": enable_llm_refinement},
            response_data={"fetched_count": len(fetched), "enriched_count": enriched_count,
                           "avg_length": sum(len(t) for t in fetched.values()) // max(len(fetched), 1)},
            result_count=enriched_count,
            elapsed_ms=elapsed_ms, status="success",
        )

    return results
