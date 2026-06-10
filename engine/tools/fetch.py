from __future__ import annotations

import re

import httpx
from bs4 import BeautifulSoup
from markdownify import markdownify as md

_MAX_CHARS = 8_000
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; DeepResearch/1.0)"}


def fetch(url: str, max_chars: int = _MAX_CHARS) -> str:
    """Fetch a URL and return cleaned Markdown text, truncated to `max_chars`.

    Returns an empty string on network errors so callers can skip gracefully.
    """
    try:
        resp = httpx.get(url, timeout=15, follow_redirects=True, headers=_HEADERS)
        resp.raise_for_status()
    except Exception:
        return ""

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()

    body = soup.body or soup
    try:
        text: str = md(str(body), strip=["a", "img"])
    except RecursionError:
        # Deeply nested HTML overflows markdownify's recursion; fall back to plain text
        text = body.get_text(separator="\n", strip=True)

    # Collapse 3+ consecutive blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text[:max_chars]
