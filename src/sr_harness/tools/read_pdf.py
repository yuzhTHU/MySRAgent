# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""Bounded PDF text extraction from workspace files or public URLs."""
from __future__ import annotations

import ipaddress
import socket
from io import BytesIO
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlparse

import requests

from .base_tool import BaseTool, ToolMetadata

MAX_DOWNLOAD_BYTES = 25 * 1024 * 1024


def _validate_public_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Only absolute http(s) URLs are supported.")
    for info in socket.getaddrinfo(
        parsed.hostname,
        parsed.port or 443,
        type=socket.SOCK_STREAM,
    ):
        if not ipaddress.ip_address(info[4][0]).is_global:
            raise ValueError("Refusing to access a non-public network address.")


@BaseTool.register("read_pdf")
class PDFReadTool(BaseTool):
    metadata = ToolMetadata(name="read_pdf")

    def execute(self, source: str, start_page: int = 1, max_pages: int = 10) -> Dict[str, Any]:
        """Extract text from selected pages of a local or public PDF.

        Args:
            source: Local PDF path or an absolute public http(s) URL.
            start_page: First page to read, one-indexed.
            max_pages: Maximum pages to extract, between 1 and 50.
        """
        start_page = max(1, int(start_page))
        max_pages = max(1, min(int(max_pages), 50))
        raw = self._load(source)
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("PDF support requires: pip install -e '.[tools]'") from exc
        reader = PdfReader(BytesIO(raw))
        begin = start_page - 1
        end = min(len(reader.pages), begin + max_pages)
        if begin >= len(reader.pages):
            raise ValueError(f"start_page exceeds PDF page count ({len(reader.pages)}).")
        pages = [
            {"page": i + 1, "text": reader.pages[i].extract_text() or ""}
            for i in range(begin, end)
        ]
        return {"source": source, "page_count": len(reader.pages), "pages": pages}

    def _load(self, source: str) -> bytes:
        if source.startswith(("http://", "https://")):
            _validate_public_url(source)
            response = requests.get(
                source,
                timeout=30,
                stream=True,
                headers={"User-Agent": "SRAgent/1.0"},
            )
            response.raise_for_status()
            data = bytearray()
            for chunk in response.iter_content(64 * 1024):
                data.extend(chunk)
                if len(data) > MAX_DOWNLOAD_BYTES:
                    raise ValueError("PDF exceeds the 25 MiB download limit.")
            return bytes(data)
        path = Path(source).expanduser().resolve()
        allowed_root = Path(self.context.get("workspace_dir") or Path.cwd()).resolve()
        try:
            path.relative_to(allowed_root)
        except ValueError as exc:
            raise ValueError(f"Local PDFs must be inside {allowed_root}.") from exc
        if path.stat().st_size > MAX_DOWNLOAD_BYTES:
            raise ValueError("PDF exceeds the 25 MiB read limit.")
        return path.read_bytes()

    @classmethod
    def format_result_dict(cls, result: Dict[str, Any]) -> str:
        return "\n\n".join(
            f"## Page {page['page']}\n{page['text']}" for page in result["pages"]
        )


"""
Real invocation and selected output captured from the public EIC paper:

>>> PDFReadTool().execute(
...     "https://arxiv.org/pdf/2509.21780", start_page=1, max_pages=1
... )
{'source': 'https://arxiv.org/pdf/2509.21780', 'page_count': 29,
 'pages': [{'page': 1,
            'text': 'BEYOND ACCURACY AND COMPLEXITY: THE EFFECTIVE '
                    'INFORMATION CRITERION FOR STRUCTURALLY STABLE ...'}]}
"""
