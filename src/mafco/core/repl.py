"""Persistent REPL executor for the RLM loop.

The REPL is the LLM's hands. It holds:
  - `context`: the (potentially huge) input string / list / dict.
  - `memory`: the MemoryStore — first-class, programmable.
  - `llm_query(prompt, **kw)`: sub-LLM call.
  - Standard library access (the LLM is trusted at this layer).

State persists across calls within a session. Each `exec_block` returns the
captured stdout plus any exception traceback so the LLM can self-correct.

Security / threat model
-----------------------
This is a deliberately UNSANDBOXED Python REPL. The whole point of MAFCO is
that the LLM authors and runs Python code as its primary cognitive primitive,
so it has access to full `builtins` — including `open`, `os`, `subprocess`,
`importlib`, `socket`, network and filesystem. There is no syscall filter and
no resource limit at this layer.

This means:

  - Prompt injection IS an arbitrary-code-execution vector. Any actor who
    can influence `context`, the system prompt, `llm_query` results, or any
    other text the model reads can in principle cause the model to emit a
    ```repl``` block that executes OS-level commands as the host process.
  - The trust boundary is the OUTER perimeter — the operator who decides
    which prompts and which contexts get passed in. Treat MAFCO like
    `python -c "$user_input"`: only run it on inputs you'd run as code.
  - Operators wanting safety SHOULD run MAFCO inside an OS-level sandbox
    (container, VM, ephemeral user, seccomp/landlock, or similar) and
    should not pass untrusted third-party text into `context` without
    out-of-band scrubbing.

Restricting `__builtins__` here was considered and rejected: it would not
meaningfully stop a determined injected instruction (the model can rebuild
most attack primitives from `bytes`, `compile`, `eval`, `getattr`, etc.) and
would break legitimate code the model needs to write. The honest control is
the outer sandbox, not an in-process allowlist.
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
