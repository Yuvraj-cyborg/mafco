"""Parsers for the LLM output protocol.

The LLM speaks two control verbs:
  - ```repl ... ```        → Python code to execute in the REPL.
  - FINAL(answer) / FINAL_VAR(varname)  → terminate the loop.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ```repl ... ```  (also accept "python" since models slip into that)
_REPL_BLOCK_RE = re.compile(
    r"```(?:repl|python|py)\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


def extract_repl_blocks(text: str) -> list[str]:
    """Return all repl/python code blocks from a model response, in order."""
    return [m.group(1).rstrip() for m in _REPL_BLOCK_RE.finditer(text or "")]


@dataclass
class FinalAnswer:
    kind: str  # "final" or "final_var"
    value: str


def extract_final_answer(text: str) -> FinalAnswer | None:
    """Find a top-level FINAL(...) or FINAL_VAR(...) in the response."""
    if not text:
        return None
    stripped = _REPL_BLOCK_RE.sub("", text)
    for keyword, kind in (("FINAL_VAR", "final_var"), ("FINAL", "final")):
        match = _scan_call(stripped, keyword)
        if match is not None:
            return FinalAnswer(kind=kind, value=match.strip())
    return None


def _scan_call(text: str, keyword: str) -> str | None:
    """Find `keyword(...)` and return the inner content using paren matching."""
    pattern = re.compile(rf"\b{re.escape(keyword)}\s*\(", re.IGNORECASE)
    for m in pattern.finditer(text):
        start = m.end()
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            ch = text[i]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    return text[start:i]
            i += 1
    return None


def truncate(s: str | None, limit: int = 4000, marker: str = "\n…[truncated]…\n") -> str:
    """Truncate s to ~limit chars, keeping head+tail so context isn't lost.

    `None` is accepted defensively (callers in error/edge paths sometimes
    forward optional fields) and is treated as the empty string.
    """
    if s is None:
        return ""
    if len(s) <= limit:
        return s
    head = limit // 2
    tail = limit - head - len(marker)
    if tail <= 0:
        return s[:limit] + "…"
    return s[:head] + marker + s[-tail:]
