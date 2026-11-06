"""Web search + page fetch for the background model.

Modelled on how Unsloth Studio does it (studio/backend/core/inference/tools.py and _html_to_md.py,
re-implemented here from scratch): DuckDuckGo through the `ddgs` library, results handed to the model
as plain "Title / URL / Snippet" blocks with a reminder to fetch the page for full text; a fetcher
with SSRF protection, a download cap, GitHub repo roots rewritten to the README API, HTML converted
to Markdown with a main-content heuristic (<article> then <main>), PDFs extracted, and a page budget
of 16,000 characters (about 4k tokens) with a visible truncation note.
"""
from __future__ import annotations

import asyncio
import ipaddress
import re
import socket
from html.parser import HTMLParser
from urllib.parse import urlparse

import httpx

MAX_PAGE_CHARS = 16_000
MIN_PAGE_CHARS = 2_000
MAX_FETCH_BYTES = 512 * 1024
MAX_PDF_BYTES = 10 * 1024 * 1024
MAX_PDF_PAGES = 50
MAX_REDIRECTS = 5
FETCH_TIMEOUT = 30
SEARCH_TIMEOUT = 20
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
EMPTY_SEARCH = "No results found. Try different or fewer keywords."
FETCH_REMINDER = ("IMPORTANT: These are only short snippets. To get the full page content, call fetch_page "
                  "with the URL (or web_search with the url parameter).")


# --------------------------------------------------------------------------- #
# search
# --------------------------------------------------------------------------- #
def _search_sync(query: str, max_results: int) -> str:
    from ddgs import DDGS  # lazy: import cost + optional
    try:
        rows = DDGS(timeout=SEARCH_TIMEOUT).text(query, max_results=max_results)
    except Exception as e:  # ddgs raises on an empty sweep as well as on refusals
        msg = str(e)
        if "No results" in msg:
            return EMPTY_SEARCH
        if "Ratelimit" in type(e).__name__ or "ratelimit" in msg.lower():
            return "Search rate-limited by the engine. Wait a moment and try a narrower query."
        return f"Search failed: {type(e).__name__}: {msg[:160]}"
    parts = []
    for r in rows or []:
        href = str(r.get("href") or "").strip()
        if not href:
            continue
        title = " ".join(str(r.get("title") or "").split())
        snippet = " ".join(str(r.get("body") or "").split())
        parts.append(f"Title: {title}\nURL: {href}\nSnippet: {snippet}")
        if len(parts) >= max_results:
            break
    if not parts:
        return EMPTY_SEARCH
    return "\n\n---\n\n".join(parts) + "\n\n---\n\n" + FETCH_REMINDER


async def web_search(query: str, max_results: int = 5, url: str | None = None) -> str:
    """Search DuckDuckGo, or fetch `url` directly when given (same shape as Studio's tool)."""
    if url and url.strip():
        return await fetch_page(url.strip())
    if not query or not query.strip():
        return "No query provided."
    max_results = max(1, min(int(max_results or 5), 10))
    return await asyncio.to_thread(_search_sync, query.strip(), max_results)


# --------------------------------------------------------------------------- #
# fetch
# --------------------------------------------------------------------------- #
def normalize_url(url: str) -> str:
    url = url.strip()
    if not re.match(r"^[a-z][a-z0-9+.-]*://", url, re.I):
        url = "https://" + url.lstrip("/")
    return url


def _host_is_public(hostname: str) -> tuple[bool, str]:
    """SSRF guard: refuse private, loopback, link-local, reserved and multicast targets."""
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return False, f"could not resolve {hostname}"
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified:
            return False, f"{hostname} resolves to a non-public address ({ip}); refusing"
    return True, ""


def github_readme_api_url(url: str) -> str | None:
    """https://github.com/<owner>/<repo>[/] -> README via the API, so the model reads the README, not the UI."""
    p = urlparse(url)
    if p.netloc.lower() not in ("github.com", "www.github.com"):
        return None
    parts = [x for x in p.path.split("/") if x]
    if len(parts) != 2:
        return None
    return f"https://api.github.com/repos/{parts[0]}/{parts[1]}/readme"


async def _get(url: str, extra_headers: dict | None = None, max_bytes: int = MAX_FETCH_BYTES) -> tuple[str | None, bytes, str, str]:
    """Manual redirect loop so every hop is SSRF-checked. Returns (error, body, content_type, final_url)."""
    headers = {"User-Agent": UA, "Accept": "text/html,application/xhtml+xml,application/pdf,text/plain;q=0.9,*/*;q=0.5"}
    headers.update(extra_headers or {})
    async with httpx.AsyncClient(timeout=FETCH_TIMEOUT, follow_redirects=False, headers=headers) as c:
        for _ in range(MAX_REDIRECTS + 1):
            p = urlparse(url)
            if p.scheme not in ("http", "https"):
                return f"Blocked: unsupported URL scheme '{p.scheme}'", b"", "", url
            ok, why = _host_is_public(p.hostname or "")
            if not ok:
                return f"Blocked: {why}", b"", "", url
            try:
                async with c.stream("GET", url) as r:
                    if r.is_redirect and r.headers.get("location"):
                        url = str(r.next_request.url) if r.next_request else r.headers["location"]
                        continue
                    if r.status_code >= 400:
                        return f"Failed to fetch URL: HTTP {r.status_code}", b"", "", url
                    ctype = r.headers.get("content-type", "").lower()
                    limit = MAX_PDF_BYTES if "pdf" in ctype else max_bytes
                    body = bytearray()
                    async for chunk in r.aiter_bytes():
                        body.extend(chunk)
                        if len(body) > limit:
                            break
                    return None, bytes(body), ctype, str(r.url)
            except httpx.HTTPError as e:
                return f"Failed to fetch URL: {type(e).__name__}: {str(e)[:120]}", b"", "", url
    return "Failed to fetch URL: too many redirects", b"", "", url


def _decode(body: bytes, ctype: str) -> str:
    m = re.search(r"charset=([\w-]+)", ctype)
    for enc in ([m.group(1)] if m else []) + ["utf-8", "latin-1"]:
        try:
            return body.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return body.decode("utf-8", "replace")


def looks_like_html(text: str) -> bool:
    head = text[:2000].lower()
    return "<html" in head or "<!doctype html" in head or "<body" in head or "<div" in head


def pdf_text(data: bytes) -> str:
    from io import BytesIO

    from pypdf import PdfReader
    reader = PdfReader(BytesIO(data))
    pages = []
    for i, page in enumerate(reader.pages[:MAX_PDF_PAGES]):
        pages.append(f"[page {i + 1}]\n" + (page.extract_text() or "").strip())
    note = f"\n\n(showing {MAX_PDF_PAGES} of {len(reader.pages)} pages)" if len(reader.pages) > MAX_PDF_PAGES else ""
    return "\n\n".join(pages) + note


def truncate_page(text: str, max_chars: int) -> str:
    if not text.strip():
        return "(page returned no readable text)"
    if len(text) > max_chars:
        return text[:max_chars] + f"\n\n... (truncated, {len(text)} chars total)"
    return text


async def fetch_page(url: str, max_chars: int | None = None) -> str:
    """Fetch a URL and return readable text: HTML -> Markdown (main content), PDF -> text, text as-is."""
    max_chars = MAX_PAGE_CHARS if not max_chars else max(MIN_PAGE_CHARS, min(int(max_chars), MAX_PAGE_CHARS))
    url = normalize_url(url)
    api = github_readme_api_url(url)
    if api:
        err, body, ctype, _ = await _get(api, {"Accept": "application/vnd.github.raw+json", "X-GitHub-Api-Version": "2022-11-28"})
        if err is None and body.strip():
            text = _decode(body, ctype)
            if looks_like_html(text):
                text = html_to_markdown(text, main_content=True)
            return truncate_page(f"README of {url} (fetched via the GitHub README API):\n\n{text}", max_chars)
    err, body, ctype, final = await _get(url)
    if err:
        return err
    if "pdf" in ctype or body[:5] == b"%PDF-":
        try:
            return truncate_page(f"PDF at {final}\n\n{pdf_text(body)}", max_chars)
        except Exception as e:  # noqa: BLE001
            return f"Failed to read PDF: {type(e).__name__}: {e}"
    if ctype and not (ctype.startswith("text/") or "html" in ctype or "xml" in ctype or "json" in ctype):
        return f"Blocked: binary content-type {ctype.split(';')[0]} is not readable text"
    text = _decode(body, ctype)
    if "html" in ctype or looks_like_html(text):
        text = html_to_markdown(text, main_content=True)
    return truncate_page(text.strip(), max_chars)


# --------------------------------------------------------------------------- #
# HTML -> Markdown (standard library only)
# --------------------------------------------------------------------------- #
_SKIP = {"script", "style", "head", "noscript", "svg", "math", "template", "iframe", "button", "select",
         "datalist", "form", "nav", "footer", "aside", "dialog"}
_BLOCK = {"p", "div", "section", "article", "main", "header", "ul", "ol", "li", "table", "tr", "blockquote",
          "pre", "hr", "br", "dd", "dt", "dl", "figure", "figcaption", "h1", "h2", "h3", "h4", "h5", "h6"}
_MIN_MAIN_CHARS = 400


def _hidden(attrs: dict) -> bool:
    if "hidden" in attrs or (attrs.get("aria-hidden") or "").strip().lower() == "true":
        return True
    style = (attrs.get("style") or "").lower().replace(" ", "")
    return "display:none" in style or "visibility:hidden" in style


class _MD(HTMLParser):
    def __init__(self, scope: str | None):
        super().__init__(convert_charrefs=True)
        self.scope = scope                # "article" / "main" / None
        self.in_scope = scope is None
        self.scope_depth = 0
        self.out: list[str] = []
        self.skip = 0                     # depth inside skipped/hidden subtrees
        self.stack: list[tuple[str, bool]] = []  # (tag, counted_as_skip)
        self.lists: list[tuple[str, int]] = []   # (ul/ol, counter)
        self.href: str | None = None
        self.link_text: list[str] = []
        self.pre = 0
        self.title = ""
        self.in_title = False

    def _emit(self, s: str):
        if self.in_scope:
            self.out.append(s)

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "title":
            self.in_title = True
        skip = tag in _SKIP or _hidden(a)
        self.stack.append((tag, skip))
        if skip:
            self.skip += 1
            return
        if self.skip:
            return
        if self.scope and tag == self.scope and not self.in_scope:
            self.in_scope = True
            self.scope_depth = len(self.stack)
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self._emit("\n\n" + "#" * int(tag[1]) + " ")
        elif tag == "p" or tag == "div" or tag in ("section", "article", "main", "header", "figure", "figcaption", "dl", "table", "blockquote"):
            self._emit("\n\n")
        elif tag == "br":
            self._emit("\n")
        elif tag == "hr":
            self._emit("\n\n---\n\n")
        elif tag in ("ul", "ol"):
            self.lists.append((tag, 0)); self._emit("\n")
        elif tag == "li":
            kind, n = self.lists[-1] if self.lists else ("ul", 0)
            if self.lists:
                self.lists[-1] = (kind, n + 1)
            self._emit("\n" + "  " * (len(self.lists) - 1) + (f"{n + 1}. " if kind == "ol" else "- "))
        elif tag == "tr":
            self._emit("\n| ")
        elif tag in ("td", "th"):
            pass
        elif tag == "pre":
            self.pre += 1; self._emit("\n\n```\n")
        elif tag == "code" and not self.pre:
            self._emit("`")
        elif tag in ("b", "strong"):
            self._emit("**")
        elif tag in ("i", "em"):
            self._emit("*")
        elif tag == "a":
            self.href = a.get("href"); self.link_text = []
        elif tag == "img":
            alt = (a.get("alt") or "").strip()
            if alt:
                self._emit(f"![{alt}]")

    def handle_endtag(self, tag):
        # pop to the matching open tag (tolerates unclosed elements)
        while self.stack:
            t, skipped = self.stack.pop()
            if skipped:
                self.skip -= 1
            if t == tag:
                break
        if tag == "title":
            self.in_title = False
        if self.skip:
            return
        if self.scope and self.in_scope and tag == self.scope and len(self.stack) < self.scope_depth:
            self.in_scope = False
            return
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6", "p", "div", "li", "blockquote", "tr", "figure", "figcaption"):
            self._emit("\n")
        elif tag in ("td", "th"):
            self._emit(" | ")
        elif tag in ("ul", "ol"):
            if self.lists:
                self.lists.pop()
            self._emit("\n")
        elif tag == "pre":
            self.pre = max(0, self.pre - 1); self._emit("\n```\n\n")
        elif tag == "code" and not self.pre:
            self._emit("`")
        elif tag in ("b", "strong"):
            self._emit("**")
        elif tag in ("i", "em"):
            self._emit("*")
        elif tag == "a":
            text = "".join(self.link_text).strip()
            href = (self.href or "").strip()
            if text and href and not href.startswith(("#", "javascript:")):
                self.out[len(self.out) - len(self.link_text):] = []  # replace raw link text
                self._emit(f"[{text}]({href})")
            self.href = None; self.link_text = []

    def handle_data(self, data):
        if self.in_title:
            self.title += data
        if self.skip or not self.in_scope:
            return
        text = data if self.pre else re.sub(r"\s+", " ", data)
        if not text.strip() and not self.pre:
            if self.out and not self.out[-1].endswith((" ", "\n")):
                self.out.append(" ")
            return
        if self.href is not None:
            self.link_text.append(text)
        self.out.append(text)

    def markdown(self) -> str:
        s = "".join(self.out)
        s = re.sub(r"[ \t]+\n", "\n", s)
        s = re.sub(r"\n{3,}", "\n\n", s)
        s = re.sub(r"(\| \n)+", "|\n", s)
        return s.strip()


def html_to_markdown(html: str, *, main_content: bool = False) -> str:
    html = html.replace("\r\n", "\n")
    if main_content:
        for scope in ("article", "main"):
            p = _MD(scope); p.feed(html)
            md = p.markdown()
            if len(md) >= _MIN_MAIN_CHARS:
                return (f"# {p.title.strip()}\n\n" if p.title.strip() and not md.startswith("#") else "") + md
    p = _MD(None); p.feed(html)
    md = p.markdown()
    return (f"# {p.title.strip()}\n\n" if p.title.strip() and not md.startswith("#") else "") + md
