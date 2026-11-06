"""Text prep for speech: strip markdown, and cut a token stream into speakable sentences."""
from __future__ import annotations

import re

_MD = [
    (re.compile(r"```.*?```", re.S), " "),
    (re.compile(r"`([^`]*)`"), r"\1"),
    (re.compile(r"^\s*#{1,6}\s*", re.M), ""),
    (re.compile(r"\*\*([^*]*)\*\*"), r"\1"),
    (re.compile(r"\*([^*]*)\*"), r"\1"),
    (re.compile(r"\[([^\]]*)\]\([^)]*\)"), r"\1"),
    (re.compile(r"https?://\S+"), " "),
    (re.compile(r"^\s*[-*]\s+", re.M), ""),
]


def strip_markdown(text: str) -> str:
    for pat, repl in _MD:
        text = pat.sub(repl, text)
    return re.sub(r"[ \t]+", " ", text).strip()


# A period that ends the text so far only ends a sentence when no digit precedes it: "at 5." may be "at 5.5%".
_SENT_END = re.compile(r"(?<=[.!?…])\s+|(?<=[!?…])$|(?<=\D\.)$|\n+")
_CLAUSE = re.compile(r"[,;:—]\s")


class SentenceChunker:
    """Incrementally cuts a token stream into speakable sentences (for low-latency TTS).

    push(delta) -> completed sentences; flush() -> the remainder.
    The first chunk may be cut early at a clause boundary so time-to-first-audio does not
    wait for a long opening sentence. Very short "sentences" are merged with the next one.
    """

    def __init__(self, min_first: int = 28, min_len: int = 12):
        self.buf = ""
        self.first_out = False
        self.min_first = min_first
        self.min_len = min_len

    def push(self, delta: str) -> list[str]:
        self.buf += delta
        out: list[str] = []
        while True:
            m = _SENT_END.search(self.buf)
            if m:
                sent, rest = self.buf[: m.end()].strip(), self.buf[m.end():]
                if len(sent) < self.min_len:
                    if not rest.strip():
                        break  # wait for more text; flush() will emit it
                    m2 = _SENT_END.search(self.buf, m.end())
                    if not m2:
                        break
                    sent, rest = self.buf[: m2.end()].strip(), self.buf[m2.end():]
                self.buf = rest
                if sent:
                    out.append(sent)
                    self.first_out = True
                continue
            if not self.first_out and len(self.buf) >= self.min_first:
                cm = _CLAUSE.search(self.buf, self.min_first // 2)
                if cm:
                    out.append(self.buf[: cm.end()].strip())
                    self.buf = self.buf[cm.end():]
                    self.first_out = True
                    continue
            break
        return out

    def flush(self) -> list[str]:
        rest, self.buf = self.buf.strip(), ""
        self.first_out = True
        return [rest] if rest else []
