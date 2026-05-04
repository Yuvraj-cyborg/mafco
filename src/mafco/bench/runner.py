"""Benchmark runner: MAFCO vs. flat-baseline on MAFCO-targeted tasks.

Each task carries a query, a context (deliberately small so we exercise memory
semantics, not long-context scaling), and a grading rubric:

  must_contain               substrings the answer MUST include
  must_not_contain_in_answer substrings the answer MUST NOT include
  expects.min_revisions      MAFCO-only: required `revise` events
  expects.min_invalidations  MAFCO-only: required `invalidate` events
  expects.min_facts          MAFCO-only: required `fact` units

We compare two systems on the same model:
  MAFCO     full RLM loop with the typed memory primitive.
  baseline  one direct LLM call with the context inlined into the prompt.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from mafco.core.memory import MemoryStore
from mafco.core.rlm import RLM, RLMConfig
from mafco.model.client import LLMClient, ModelConfig
from mafco.types.schema import BeliefType, MemoryEventKind


@dataclass
class TaskResult:
    task_id: str
    category: str
    system: str  # "mafco" | "baseline"
    answer: str
    answer_correct: bool
    correct: bool
    rubric_misses: list[str]
    latency_s: float
    iterations: int = 0
    sub_calls: int = 0
    sub_call_chars: int = 0
    facts: int = 0
    hypotheses: int = 0
    conclusions: int = 0
    revises: int = 0
    invalidates: int = 0
    merges: int = 0
    memory_expectations_met: bool = True
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    error: str | None = None


@dataclass
class BenchmarkResult:
    model: str
    sub_model: str
    started_at: float = field(default_factory=time.time)
    finished_at: float = 0.0
    tasks: list[TaskResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "sub_model": self.sub_model,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "tasks": [asdict(t) for t in self.tasks],
        }


def _grade_answer(answer: str, must: list[str], must_not: list[str]) -> tuple[bool, list[str]]:
    a = (answer or "").lower()
    misses: list[str] = []
    for m in must:
        if m.lower() not in a:
            misses.append(f"missing:{m}")
    for n in must_not:
        if n and n.lower() in a:
            misses.append(f"present_but_forbidden:{n}")
    return (len(misses) == 0), misses


def _check_memory_expectations(expects: dict[str, int], memory: MemoryStore) -> bool:
    counts = _memory_counts(memory)
    if "min_facts" in expects and counts["facts"] < expects["min_facts"]:
        return False
    if "min_revisions" in expects and counts["revises"] < expects["min_revisions"]:
        return False
    if "min_invalidations" in expects and counts["invalidates"] < expects["min_invalidations"]:
        return False
    if "min_merges" in expects and counts["merges"] < expects["min_merges"]:
        return False
    return True


def _memory_counts(memory: MemoryStore) -> dict[str, int]:
    facts = sum(1 for u in memory if u.type is BeliefType.FACT)
    hyps = sum(1 for u in memory if u.type is BeliefType.HYPOTHESIS)
    concs = sum(1 for u in memory if u.type is BeliefType.CONCLUSION)
    revises = sum(1 for e in memory.events if e.kind is MemoryEventKind.REVISE)
    invs = sum(1 for e in memory.events if e.kind is MemoryEventKind.INVALIDATE)
    merges = sum(1 for e in memory.events if e.kind is MemoryEventKind.MERGE)
    return {
        "facts": facts,
        "hypotheses": hyps,
        "conclusions": concs,
        "revises": revises,
        "invalidates": invs,
        "merges": merges,
    }


def run_mafco(task: dict[str, Any], rlm: RLM) -> TaskResult:
    t0 = time.perf_counter()
    try:
        result = rlm.run(task["query"], context=task["context"])
        ans = result.answer
        memory = result.memory
        iters = result.trace.iterations_used
        subs = result.trace.sub_calls
        sub_chars = result.trace.sub_call_chars
        err = None
    except Exception as e:
        ans = ""
        memory = MemoryStore()
        iters = subs = sub_chars = 0
        err = f"{type(e).__name__}: {e}"

    latency = time.perf_counter() - t0
    correct, misses = _grade_answer(
        ans,
        task.get("must_contain", []),
        task.get("must_not_contain_in_answer", []),
    )
    counts = _memory_counts(memory)
    expects = task.get("expects", {})
    mem_ok = _check_memory_expectations(expects, memory)

    return TaskResult(
        task_id=task["id"],
        category=task["category"],
        system="mafco",
        answer=ans,
        answer_correct=correct,
        correct=correct and mem_ok,
        rubric_misses=misses + ([] if mem_ok else ["memory_expectations_unmet"]),
        latency_s=round(latency, 2),
        iterations=iters,
        sub_calls=subs,
        sub_call_chars=sub_chars,
        facts=counts["facts"],
        hypotheses=counts["hypotheses"],
        conclusions=counts["conclusions"],
        revises=counts["revises"],
        invalidates=counts["invalidates"],
        merges=counts["merges"],
        memory_expectations_met=mem_ok,
        error=err,
    )


_BASELINE_SYSTEM = (
    "You answer the user's question using only the supplied CONTEXT. "
    "If the context contains corrections or retractions, prefer the corrected "
    "value. Be concise."
)


def run_baseline(task: dict[str, Any], client: LLMClient, model: str | None = None) -> TaskResult:
    t0 = time.perf_counter()
    prompt = (
        f"CONTEXT:\n{task['context']}\n\n"
        f"QUESTION:\n{task['query']}\n\n"
        "Answer:"
    )
    err: str | None = None
    ans = ""
    pt = ct = tt = 0
    try:
        resp = client.complete(
            [
                {"role": "system", "content": _BASELINE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            model=model,
            enable_reasoning=False,
        )
        ans = resp.content
        if resp.usage:
            pt = int(resp.usage.get("prompt_tokens", 0) or 0)
            ct = int(resp.usage.get("completion_tokens", 0) or 0)
            tt = int(resp.usage.get("total_tokens", 0) or 0)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"

    latency = time.perf_counter() - t0
    correct, misses = _grade_answer(
        ans,
        task.get("must_contain", []),
        task.get("must_not_contain_in_answer", []),
    )
    return TaskResult(
        task_id=task["id"],
        category=task["category"],
        system="baseline",
        answer=ans,
        answer_correct=correct,
        correct=correct,
        rubric_misses=misses,
        latency_s=round(latency, 2),
        prompt_tokens=pt,
        completion_tokens=ct,
        total_tokens=tt,
        error=err,
    )


def load_tasks(path: str | Path) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text())


def run_suite(
    tasks: list[dict[str, Any]],
    *,
    model: str | None = None,
    sub_model: str | None = None,
    max_iterations: int = 8,
    max_tokens: int = 4096,
    temperature: float = 0.2,
    on_task_done: Any = None,
) -> BenchmarkResult:
    mc = ModelConfig(temperature=temperature, max_tokens=max_tokens)
    if model:
        mc.model = model
    if sub_model:
        mc.sub_model = sub_model
    rlm = RLM(RLMConfig(model=mc, max_iterations=max_iterations, verbose=False))
    client = rlm.client
    out = BenchmarkResult(model=mc.model, sub_model=mc.sub_model)

    for task in tasks:
        for runner in ("mafco", "baseline"):
            if runner == "mafco":
                tr = run_mafco(task, rlm)
            else:
                tr = run_baseline(task, client, model=mc.model)
            out.tasks.append(tr)
            if on_task_done is not None:
                on_task_done(tr)

    out.finished_at = time.time()
    return out


def summarise(result: BenchmarkResult) -> dict[str, Any]:
    by_sys: dict[str, list[TaskResult]] = {}
    for t in result.tasks:
        by_sys.setdefault(t.system, []).append(t)

    summary: dict[str, Any] = {"model": result.model, "per_system": {}}
    for sys_name, rows in by_sys.items():
        n = len(rows)
        ans_ok = sum(1 for r in rows if r.answer_correct)
        full_ok = sum(1 for r in rows if r.correct)
        agg = {
            "n": n,
            "answer_correct": ans_ok,
            "answer_accuracy": round(ans_ok / n, 3) if n else 0.0,
            "strict_correct": full_ok,
            "strict_accuracy": round(full_ok / n, 3) if n else 0.0,
            "avg_latency_s": round(sum(r.latency_s for r in rows) / n, 2) if n else 0.0,
        }
        if sys_name == "mafco":
            mem_ok = sum(1 for r in rows if r.memory_expectations_met)
            agg.update(
                {
                    "memory_ops_met": mem_ok,
                    "memory_ops_rate": round(mem_ok / n, 3) if n else 0.0,
                    "avg_iterations": round(sum(r.iterations for r in rows) / n, 2) if n else 0,
                    "total_facts": sum(r.facts for r in rows),
                    "total_hypotheses": sum(r.hypotheses for r in rows),
                    "total_conclusions": sum(r.conclusions for r in rows),
                    "total_revises": sum(r.revises for r in rows),
                    "total_invalidates": sum(r.invalidates for r in rows),
                    "total_merges": sum(r.merges for r in rows),
                }
            )
        summary["per_system"][sys_name] = agg
    return summary
