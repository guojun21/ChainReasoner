"""Bright Data SERP API client — stable Google search via commercial data platform.

Why: Google Custom Search API has strict daily quota limits (100 free/day)
and the self-hosted proxy at google.ydcloud.org is unreliable.  Bright Data
provides pay-per-success Google SERP results with automatic IP rotation,
CAPTCHA solving, and no rate limiting — giving us a stable high-quality
search backend when Google direct access is exhausted.

Response format: Bright Data returns the Google search results page as
**markdown text** (format="raw").  We parse the markdown to extract
structured search results [{title, url, content}].

API docs: https://docs.brightdata.com/scraping-automation/serp-api/introduction
"""

import logging
import re
import time
from typing import Any, Dict, List, Optional
from urllib.parse import unquote

import requests

from src.search.abstract_search_client_interface import AbstractSearchClientInterface

logger = logging.getLogger(__name__)

BRIGHT_DATA_SERP_API_URL = "https://api.brightdata.com/request"
BRIGHT_DATA_UNAVAILABLE_ERROR_CODE = "BRIGHTDATA_UNAVAILABLE"

# ---------- Markdown SERP parser ----------

# Pattern: markdown link with title and URL, followed by description text
# Google SERP markdown format from Bright Data looks like:
#   [### Title\n\n![...](...)\nSiteName\nhttps://... ](https://actual-url)
#   SiteName\nhttps://...\nDescription text...
_RESULT_BLOCK_PATTERN = re.compile(
    r'\[\s*\n*###\s+(.+?)(?:\n|\r)',  # Capture title after ###
    re.DOTALL,
)


def _parse_google_serp_markdown_into_results(markdown_text: str) -> List[Dict[str, str]]:
    """Parse Google SERP results from Bright Data markdown response.

    Why: Bright Data SERP API (format=raw) returns the Google search page as
    markdown.  Each organic result appears as a markdown link block starting
    with '### Title' followed by site name, URL, and description snippet.
    We extract title, URL, and description from these blocks.
    """
    results: List[Dict[str, str]] = []
    if not markdown_text:
        return results

    # Split by the result link pattern: [### Title ... ](url)
    # Each result block in the markdown looks like:
    #
    # [\n### RepRap - RepRap\n\n  \n\n![](base64...)\n\nRepRap\n\nhttps://reprap.org\n\n...](https://reprap.org/)
    # \nRepRap\n\nhttps://reprap.org\n\nOct 15, 2025 ... description text...
    #
    # Strategy: find all markdown links that contain ### headers (these are search results)
    # Then extract the URL from the link target and description from the following text.

    # Find sections starting with "# Search Results" to skip navigation noise
    search_start = markdown_text.find("# Search Results")
    if search_start == -1:
        search_start = 0
    else:
        search_start += len("# Search Results")

    # Cut off footer sections
    footer_markers = ["# Related searches", "# Footer", "# Page navigation", "## Pagination"]
    search_end = len(markdown_text)
    for marker in footer_markers:
        pos = markdown_text.find(marker, search_start)
        if pos != -1 and pos < search_end:
            search_end = pos

    content = markdown_text[search_start:search_end]

    # Pattern to match result blocks: [ ### Title ... ](URL)
    # followed by display URL and description
    link_pattern = re.compile(
        r'\[\s*\n*\s*###\s+(.+?)\n'   # [### Title
        r'.*?'                          # ... icon, site name, etc.
        r'\]\('                         # ](
        r'(https?://[^\)]+)'            # capture URL
        r'\)',                           # )
        re.DOTALL,
    )

    matches = list(link_pattern.finditer(content))

    for i, match in enumerate(matches):
        title = match.group(1).strip()
        url = match.group(2).strip()

        # Clean title: remove markdown formatting
        title = re.sub(r'\*+', '', title)  # remove bold/italic markers
        title = re.sub(r'\s+', ' ', title).strip()

        # Extract description: text between end of this match and start of next match
        desc_start = match.end()
        desc_end = matches[i + 1].start() if i + 1 < len(matches) else search_end - search_start
        desc_block = content[desc_start:desc_end].strip()

        # The description block typically looks like:
        #   SiteName\nhttps://url\n\nDate ... description text ...
        # We want to extract the actual description text, skipping the site name and URL echo.
        desc_lines = desc_block.split('\n')
        description_parts = []
        skip_header = True
        for line in desc_lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            # Skip lines that are just the site name or URL
            if skip_header:
                if line_stripped.startswith('http') or line_stripped.startswith('www.'):
                    continue
                if '://' in line_stripped and len(line_stripped) < 200:
                    continue
                # Skip short site name lines (before the URL)
                if len(line_stripped) < 40 and not any(c in line_stripped for c in '.!?;:,'):
                    continue
                skip_header = False
            # Skip markdown image references and navigation links
            if line_stripped.startswith('![') or line_stripped.startswith('['):
                continue
            if line_stripped.startswith('Missing:') or line_stripped.startswith('Show results with:'):
                continue
            if '| Show results with:' in line_stripped:
                line_stripped = line_stripped.split('| Show results with:')[0].strip()
                if line_stripped.startswith('Missing:'):
                    continue
            # Skip "Translate this page" links
            if 'Translate this page' in line_stripped:
                continue
            description_parts.append(line_stripped)

        description = ' '.join(description_parts).strip()
        # Clean up excessive whitespace
        description = re.sub(r'\s+', ' ', description)

        # Remove Google tracking parameters from URL
        if '/url?q=' in url:
            try:
                url = unquote(url.split('/url?q=')[1].split('&')[0])
            except (IndexError, ValueError):
                pass

        if title and url:
            results.append({
                "title": title,
                "url": url,
                "content": description[:1000],  # Cap description length
            })

    return results


class BrightDataSerpApiClient(AbstractSearchClientInterface):
    """Bright Data SERP API — stable Google search results via REST API.

    Why: When Google Custom Search quota is exhausted, this provides an
    identical-quality alternative using Bright Data's proxy infrastructure.
    Pay-per-success billing means failed requests cost nothing.
    """

    def __init__(self, api_key: str, zone: str = "serp_api1",
                 country: str = "", count: int = 10, timeout: int = 45):
        self.api_key = api_key
        self.zone = zone
        self.country = country
        self.count = count
        self.timeout = timeout

    @classmethod
    def from_mcp_config(cls, mcp_config: dict) -> Optional["BrightDataSerpApiClient"]:
        """Factory: build from ``mcpServers.bright-data-serp`` in mcp_config.json."""
        servers = mcp_config.get("mcpServers", {})
        bd_cfg = servers.get("bright-data-serp")
        if not bd_cfg:
            return None
        api_key = bd_cfg.get("api_key", "")
        if not api_key:
            return None
        zone = bd_cfg.get("zone", "serp_api1")
        country = bd_cfg.get("country", "")
        logger.info("BrightDataSerpApiClient: zone=%s country=%s", zone, country or "auto")
        return cls(api_key=api_key, zone=zone, country=country)

    def execute_search_query(self, query: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Send query to Bright Data SERP API and return standardised results.

        Uses format='raw' to get Google SERP as markdown, then parses
        the markdown to extract structured search results.
        """
        start_time = time.time()
        logger.info("BrightData SERP search: query=%s", query[:80])

        google_url = f"https://www.google.com/search?q={requests.utils.quote(query)}&num={self.count}"

        payload: Dict[str, Any] = {
            "zone": self.zone,
            "url": google_url,
            "format": "raw",   # Returns markdown text of the SERP page
        }
        if self.country:
            payload["country"] = self.country

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                BRIGHT_DATA_SERP_API_URL,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )

            if response.status_code == 401:
                logger.warning("BrightData SERP: authentication failed (401)")
                return self._error_result(query, "auth_failed", start_time)

            if response.status_code == 403:
                logger.warning("BrightData SERP: forbidden (403)")
                return self._error_result(query, "forbidden", start_time)

            if response.status_code == 429:
                logger.warning("BrightData SERP: rate limited (429)")
                return self._error_result(query, "rate_limited", start_time)

            if response.status_code >= 400:
                logger.warning("BrightData SERP: HTTP %d — %s",
                               response.status_code, response.text[:200])
                return self._error_result(query, f"http_{response.status_code}", start_time)

            # Response is markdown text (format=raw).
            # Bright Data returns Content-Type: text/markdown without charset,
            # causing requests to default to ISO-8859-1.  Force UTF-8 decoding
            # since Google SERP pages are always UTF-8.
            response.encoding = "utf-8"
            markdown_text = response.text

        except requests.exceptions.Timeout:
            logger.error("BrightData SERP: timeout after %ds", self.timeout)
            return self._error_result(query, "timeout", start_time)
        except requests.exceptions.RequestException as exc:
            logger.error("BrightData SERP: request error: %s", exc)
            return self._error_result(query, str(exc), start_time)

        results = _parse_google_serp_markdown_into_results(markdown_text)
        elapsed_ms = int((time.time() - start_time) * 1000)

        top_titles = [r["title"][:60] for r in results[:3]]
        logger.info("BrightData SERP ok: elapsed_ms=%d query=%s results=%d top=%s",
                     elapsed_ms, query[:80], len(results), top_titles)

        result_dict: Dict[str, Any] = {
            "service": "brightdata-serp",
            "query": query,
            "results": results,
            "count": len(results),
        }

        if self.trace_logger:
            self.trace_logger.record_search_api_call(
                service_name="brightdata-serp", query=query,
                request_params={"zone": self.zone, "country": self.country, "url": google_url},
                response_data=result_dict, result_count=len(results),
                elapsed_ms=elapsed_ms, status="success",
            )

        return result_dict

    def _error_result(self, query: str, error: str, start_time: float) -> Dict[str, Any]:
        """Build a standardised error result dict."""
        elapsed_ms = int((time.time() - start_time) * 1000)
        error_result: Dict[str, Any] = {
            "service": "brightdata-serp",
            "query": query,
            "results": [],
            "count": 0,
            "error": error,
            "error_code": BRIGHT_DATA_UNAVAILABLE_ERROR_CODE,
        }
        if self.trace_logger:
            self.trace_logger.record_search_api_call(
                service_name="brightdata-serp", query=query,
                request_params={"zone": self.zone, "country": self.country},
                response_data=error_result, result_count=0,
                elapsed_ms=elapsed_ms, status="error", error=error,
            )
        return error_result
