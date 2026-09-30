# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""Bounded public-web search."""
from __future__ import annotations

from html.parser import HTMLParser
from html import unescape
from typing import Any, Dict
from urllib.parse import parse_qs, unquote, urlparse

import requests

from .base_tool import BaseTool, ToolMetadata


class _DuckDuckGoParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.results = []
        self._field = None
        self._parts = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        classes = set(attributes.get("class", "").split())
        if tag == "a" and "result__a" in classes:
            href = attributes.get("href", "")
            parsed = urlparse(href)
            href = unquote(parse_qs(parsed.query).get("uddg", [href])[0])
            self.results.append({"title": "", "url": href, "snippet": ""})
            self._field, self._parts = "title", []
        elif self.results and ("result__snippet" in classes):
            self._field, self._parts = "snippet", []

    def handle_data(self, data):
        if self._field:
            self._parts.append(data)

    def handle_endtag(self, tag):
        if self._field and tag in {"a", "div"}:
            self.results[-1][self._field] = unescape(" ".join("".join(self._parts).split()))
            self._field, self._parts = None, []


@BaseTool.register("web_search")
class WebSearchTool(BaseTool):
    metadata = ToolMetadata(name="web_search")

    def execute(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Search the public web for papers, documentation, and scientific context.

        Args:
            query: Specific search query.
            max_results: Maximum number of results, between 1 and 10.
        """
        if not query.strip():
            raise ValueError("query must not be empty")
        max_results = max(1, min(int(max_results), 10))
        callback = self.context.get("web_search_callback")
        if callback is not None:
            results = callback(query, max_results)
            return {"query": query, "results": list(results)[:max_results], "provider": "callback"}
        response = requests.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query},
            timeout=15,
            headers={"User-Agent": "SRAgent/1.0 (research assistant)"},
        )
        response.raise_for_status()
        parser = _DuckDuckGoParser()
        parser.feed(response.text)
        return {
            "query": query,
            "results": parser.results[:max_results],
            "provider": "duckduckgo-html",
        }


"""
Real invocation and output captured from DuckDuckGo HTML search:

>>> WebSearchTool().execute("EIC symbolic regression paper", max_results=2)
{'query': 'EIC symbolic regression paper', 'results': [
 {'title': '[2509.21780v1] Beyond Formula Complexity: Effective Information ...',
  'url': 'https://arxiv.org/abs/2509.21780v1',
  'snippet': 'Combining EIC with various search-based symbolic regression ...'},
 {'title': 'Beyond Accuracy and Complexity: The Effective Information Criterion for ...',
  'url': 'https://arxiv.org/html/2509.21780',
  'snippet': 'This paper presents the Effective Information Criterion (EIC) ...'}],
 'provider': 'duckduckgo-html'}
"""
