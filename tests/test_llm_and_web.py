"""The chat helpers and the background model's web tools (no network)."""
from __future__ import annotations

import pytest

from interaction_harness.llm import assistant_tool_message, thinking_body, tool_result_message
from interaction_harness.tools import web


@pytest.mark.parametrize("mode, on, body", [
    ("chat_template_kwargs", True, {"chat_template_kwargs": {"enable_thinking": True}}),
    ("chat_template_kwargs", False, {"chat_template_kwargs": {"enable_thinking": False}}),
    ("reasoning_effort", True, {"reasoning_effort": "high"}),
    ("reasoning_effort", False, {"reasoning_effort": "low"}),
    ("off", True, {}),
])
def test_thinking_body(mode, on, body):
    assert thinking_body(mode, on) == body


def test_tool_messages():
    calls = [{"id": "c1", "name": "get_time", "arguments": ""}]
    msg = assistant_tool_message("", calls)
    assert msg["content"] is None and msg["tool_calls"][0]["function"] == {"name": "get_time", "arguments": "{}"}
    assert tool_result_message("c1", "get_time", {"iso": "x"})["content"] == '{"iso": "x"}'


def test_html_to_markdown_prefers_the_article():
    body = "<p>" + "Real content sentence. " * 30 + "</p>"
    html = (f"<html><head><title>T</title><script>x()</script></head><body><nav>menu</nav>"
            f"<article><h2>Heading</h2>{body}<a href='https://a.b'>link</a></article><footer>f</footer></body></html>")
    md = web.html_to_markdown(html, main_content=True)
    assert md.startswith("## Heading") and "[link](https://a.b)" in md  # the page title only when no heading leads
    assert "menu" not in md and "x()" not in md
    short = web.html_to_markdown("<html><head><title>T</title></head><body><p>hi</p></body></html>", main_content=True)
    assert short == "# T\n\nhi"


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "10.0.0.1", "169.254.169.254"])
def test_fetch_refuses_private_addresses(host):
    ok, why = web._host_is_public(host)
    assert not ok and "non-public" in why


def test_urls():
    assert web.normalize_url("example.com/x") == "https://example.com/x"
    assert web.github_readme_api_url("https://github.com/a/b") == "https://api.github.com/repos/a/b/readme"
    assert web.github_readme_api_url("https://github.com/a/b/issues") is None
    assert web.truncate_page("x" * 50, 10).startswith("x" * 10 + "\n\n... (truncated")
