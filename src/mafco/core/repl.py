"""Persistent REPL executor for the RLM loop.

The REPL is the LLM's hands. It holds:
  - `context`: the (potentially huge) input string / list / dict.
  - `memory`: the MemoryStore — first-class, programmable.
  - `llm_query(prompt, **kw)`: sub-LLM call.
  - Standard library access (the LLM is trusted at this layer; the OUTER
     boundary — what prompts run at all — is the policy boundary).

State persists across calls within a session. Each `exec_block` returns the
captured stdout plus any exception traceback so the LLM can self-correct.
"""

from __future__ import annotations

import builtins
import contextlib
import io
import traceback
from dataclasses import dataclass
from typing import Any, Callable

from mafco.core.memory import MemoryStore
from mafco.types.schema import BeliefType


@dataclass
class ReplResult:
    stdout: str
    error: str | None = None  # full traceback if exec raised, else None

    @property
    def ok(self) -> bool:
        return self.error is None

    def render(self, *, stdout_limit: int = 6000) -> str:
        """Format for the next-turn message back to the LLM."""
        from mafco.utils.parsing import truncate

        parts: list[str] = []
        out = truncate(self.stdout, limit=stdout_limit) if self.stdout else ""
        if out:
            parts.append(f"[stdout]\n{out}")
        if self.error:
            parts.append(f"[error]\n{self.error}")
        if not parts:
            parts.append("[stdout]\n(no output)")
        return "\n\n".join(parts)


class ReplExecutor:
    """Executes Python code blocks against a persistent namespace."""

    def __init__(
        self,
        *,
        context: Any,
        memory: MemoryStore,
        llm_query: Callable[..., str],
        extra_globals: dict[str, Any] | None = None,
    ) -> None:
        self.memory = memory
        self.context = context
        self._globals: dict[str, Any] = {
            "__builtins__": builtins,
            "__name__": "mafco_repl",
            "context": context,
            "memory": memory,
            "llm_query": llm_query,
            "BeliefType": BeliefType,
            "FACT": BeliefType.FACT,
            "HYPOTHESIS": BeliefType.HYPOTHESIS,
            "CONCLUSION": BeliefType.CONCLUSION,
        }
        if extra_globals:
            self._globals.update(extra_globals)

    def exec_block(self, code: str) -> ReplResult:
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                exec(compile(code, "<repl>", "exec"), self._globals)
        except SystemExit:
            raise
        except BaseException:
            return ReplResult(stdout=buf.getvalue(), error=traceback.format_exc())
        return ReplResult(stdout=buf.getvalue(), error=None)

    def get_var(self, name: str) -> Any:
        if name not in self._globals:
            raise KeyError(f"variable {name!r} not in REPL namespace")
        return self._globals[name]

    @property
    def namespace(self) -> dict[str, Any]:
        return self._globals
