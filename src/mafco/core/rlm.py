"""The RLM loop with first-class Memory.

Algorithm (from arXiv:2512.24601, extended for MAFCO):

    state ← InitREPL(context=P, memory=MemoryStore())
    hist  ← [Metadata(state)]
    while True:
        msg ← LLM(hist)
        if msg contains FINAL(...) | FINAL_VAR(...):
            return resolved_answer(state)
        for each repl block in msg:
            (state, stdout) ← REPL.exec(state, code)
            hist ← hist ∥ msg ∥ Metadata(stdout)

We keep `hist` short by only feeding back metadata (truncated stdout / errors)
rather than dumping intermediate state into the model's context. The full
audit trail lives in `RLMTrace` for inspection / debugging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from mafco.core.memory import MemoryStore
from mafco.core.prompts import (
    SUB_LLM_SYSTEM_PROMPT,
    render_root_prompt,
)
from mafco.core.repl import ReplExecutor, ReplResult
from mafco.model.client import LLMClient, ModelConfig
from mafco.utils.parsing import (
    extract_final_answer,
    extract_repl_blocks,
    truncate,
)


@dataclass
class RLMConfig:
    max_iterations: int = 12
    stdout_feedback_chars: int = 6000
    sub_llm_max_chars: int = 200_000  # rough char cap per sub-LLM call
    verbose: bool = False
    model: ModelConfig = field(default_factory=ModelConfig)


@dataclass
class RLMTurn:
    iteration: int
    assistant_message: str
    repl_blocks: list[str]
    repl_results: list[ReplResult]
    final: str | None = None


@dataclass
class RLMTrace:
    turns: list[RLMTurn] = field(default_factory=list)
    sub_calls: int = 0
    sub_call_chars: int = 0
    iterations_used: int = 0


@dataclass
class RLMResult:
    answer: str
    memory: MemoryStore
    trace: RLMTrace
    terminated: str  # "final" | "final_var" | "max_iterations" | "error"


def _summarise_context(context: Any) -> str:
    t = type(context).__name__
    if isinstance(context, str):
        return f"str of {len(context)} chars (first 240 chars: {context[:240]!r})"
    if isinstance(context, list):
        sizes = [
            len(x) if hasattr(x, "__len__") else None for x in context[:8]
        ]
        return f"list of {len(context)} items (first sizes: {sizes})"
    if isinstance(context, dict):
        return f"dict with keys: {list(context.keys())[:20]}"
    if hasattr(context, "__len__"):
        return f"{t} of length {len(context)}"
    return f"{t} (no length info)"


class RLM:
    """A minimal, MAFCO-aware Recursive Language Model."""

    def __init__(
        self,
        config: RLMConfig | None = None,
        *,
        client: LLMClient | None = None,
    ) -> None:
        self.config = config or RLMConfig()
        self.client = client or LLMClient(self.config.model)

    def run(
        self,
        query: str,
        *,
        context: Any = "",
        memory: MemoryStore | None = None,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> RLMResult:
        memory = memory if memory is not None else MemoryStore()
        trace = RLMTrace()

        repl = ReplExecutor(
            context=context,
            memory=memory,
            llm_query=self._make_llm_query(trace),
        )

        system_prompt = render_root_prompt(
            context_summary=_summarise_context(context)
        )
        history: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": _initial_user_message(query, memory)},
        ]

        terminated = "max_iterations"
        answer = ""

        for i in range(1, self.config.max_iterations + 1):
            self._emit(on_event, "iteration_start", {"iteration": i})
            resp = self.client.complete(history)
            assistant_msg = resp.content or ""
            self._emit(
                on_event,
                "assistant_message",
                {"iteration": i, "content": assistant_msg},
            )

            history.append({"role": "assistant", "content": assistant_msg})

            final = extract_final_answer(assistant_msg)
            blocks = extract_repl_blocks(assistant_msg)

            results: list[ReplResult] = []
            for code in blocks:
                self._emit(on_event, "repl_exec", {"iteration": i, "code": code})
                result = repl.exec_block(code)
                results.append(result)
                self._emit(
                    on_event,
                    "repl_result",
                    {
                        "iteration": i,
                        "ok": result.ok,
                        "stdout": result.stdout,
                        "error": result.error,
                    },
                )

            turn = RLMTurn(
                iteration=i,
                assistant_message=assistant_msg,
                repl_blocks=blocks,
                repl_results=results,
                final=final.value if final else None,
            )
            trace.turns.append(turn)
            trace.iterations_used = i

            if final is not None:
                if final.kind == "final":
                    # Heuristic: models sometimes write FINAL(varname) when
                    # they mean FINAL_VAR(varname). If the payload is a single
                    # bare identifier that resolves in the REPL, unwrap it.
                    bare = final.value.strip()
                    if bare.isidentifier():
                        try:
                            answer = _stringify(repl.get_var(bare))
                            terminated = "final_var"
                            break
                        except KeyError:
                            pass
                    answer = final.value
                    terminated = "final"
                    break
                # final_var
                try:
                    value = repl.get_var(final.value)
                except KeyError:
                    history.append(
                        {
                            "role": "user",
                            "content": (
                                f"FINAL_VAR({final.value}) refers to a name not "
                                "in the REPL namespace. Either set it via a "
                                "repl block, or emit FINAL(...) directly."
                            ),
                        }
                    )
                    continue
                answer = _stringify(value)
                terminated = "final_var"
                break

            if not blocks:
                history.append(
                    {
                        "role": "user",
                        "content": (
                            "No repl block detected in your last message. Either "
                            "emit a ```repl ... ``` block or terminate with "
                            "FINAL(...)/FINAL_VAR(name)."
                        ),
                    }
                )
                continue

            feedback = "\n\n---\n\n".join(
                r.render(stdout_limit=self.config.stdout_feedback_chars)
                for r in results
            )
            mem_digest = truncate(memory.summary(), limit=2000)
            history.append(
                {
                    "role": "user",
                    "content": (
                        f"REPL output:\n{feedback}\n\n"
                        f"--- memory snapshot ({len(memory)} live units) ---\n"
                        f"{mem_digest}\n--- end snapshot ---\n\n"
                        "Continue. Either run another repl block or emit FINAL."
                    ),
                }
            )

        return RLMResult(
            answer=answer,
            memory=memory,
            trace=trace,
            terminated=terminated,
        )

    # -------------------------------------------------------------- helpers

    def _make_llm_query(self, trace: RLMTrace) -> Callable[..., str]:
        cfg = self.config
        client = self.client

        def llm_query(
            prompt: str,
            *,
            system: str | None = None,
            model: str | None = None,
            temperature: float | None = None,
        ) -> str:
            if not isinstance(prompt, str):
                prompt = str(prompt)
            if len(prompt) > cfg.sub_llm_max_chars:
                prompt = (
                    prompt[: cfg.sub_llm_max_chars // 2]
                    + f"\n…[truncated {len(prompt) - cfg.sub_llm_max_chars} chars]…\n"
                    + prompt[-cfg.sub_llm_max_chars // 2 :]
                )
            trace.sub_calls += 1
            trace.sub_call_chars += len(prompt)
            return client.query(
                prompt,
                system=system or SUB_LLM_SYSTEM_PROMPT,
                model=model or cfg.model.sub_model,
                temperature=temperature,
                enable_reasoning=False,
            )

        return llm_query

    def _emit(
        self,
        on_event: Callable[[str, dict[str, Any]], None] | None,
        kind: str,
        payload: dict[str, Any],
    ) -> None:
        if on_event is not None:
            on_event(kind, payload)
        elif self.config.verbose:
            _default_print_event(kind, payload)


def _initial_user_message(query: str, memory: MemoryStore) -> str:
    seed = memory.summary() if len(memory) > 0 else "(memory is empty)"
    return (
        f"QUERY:\n{query}\n\n"
        f"Initial memory snapshot:\n{seed}\n\n"
        "Begin. Inspect the context, then build memory through code."
    )


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    return repr(value)


def _default_print_event(kind: str, payload: dict[str, Any]) -> None:
    if kind == "assistant_message":
        print(f"\n=== iter {payload['iteration']} :: assistant ===")
        print(payload["content"])
    elif kind == "repl_exec":
        print(f"\n--- iter {payload['iteration']} :: repl ---\n{payload['code']}")
    elif kind == "repl_result":
        marker = "ok" if payload["ok"] else "ERR"
        out = payload.get("stdout") or ""
        err = payload.get("error") or ""
        print(f"\n--- iter {payload['iteration']} :: {marker} ---")
        if out:
            print(out)
        if err:
            print(err)
