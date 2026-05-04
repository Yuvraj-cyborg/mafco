# MAFCO

**Memory As First-Class Computational Object** - a typed, revision-aware,
auditable memory primitive layered on top of a minimal Recursive Language
Model (RLM).

The RLM gives the model a Python REPL with the prompt loaded as a variable.
MAFCO adds a `memory` object inside that REPL that the model is required to
program against every belief is stored as a `fact`, `hypothesis`, or
`conclusion`, with provenance, confidence, and full revision history.

## Setup

This project uses [uv] as package manager and CLI runner. Install it globally if you don’t have it (yes do it, it’s great):

```bash
uv sync 
```

Create a `.env` with your OpenAI API key:

```
OPENAI_API_KEY=sk-...
```

To point at an OpenAI-compatible provider instead (OpenRouter, Together,
etc.), set `OPENAI_BASE_URL` in `.env` — the OpenAI SDK reads it
automatically.

## Usage

```bash
# Inline query
uv run mafco run "What is the current best yield?" \
    --context "Note A: 62%. Correction: actual is 41%. Xylene tried: 37%."

# From a file
uv run mafco run "Summarise the current state" -c notes.md 

# create one to test for above 

# Stream the trace (see every repl block + result)
uv run mafco run "..." -c notes.md --verbose

# Save / reload memory across sessions
uv run mafco run "..." -c notes.md --save-memory session.memory.json
uv run mafco run "follow-up question" --load-memory session.memory.json
```

Inspect a saved memory snapshot:

```bash
uv run mafco memory show   session.memory.json   # human-readable digest
uv run mafco memory events session.memory.json   # audit log table
uv run mafco memory dump   session.memory.json   # raw JSON
```

## Configuration

All settings can be passed as flags or environment variables.

| Setting           | Flag             | Env var                  | Default       |
| ----------------- | ---------------- | ------------------------ | ------------- |
| Root model        | `--model`        | `MAFCO_MODEL`            | `gpt-5.5`     |
| Sub-LLM model     | `--sub-model`    | `MAFCO_SUB_MODEL`        | (same as root)|
| Max tokens        | `--max-tokens`   | `MAFCO_MAX_TOKENS`       | `4096`        |
| Reasoning effort  | —                | `MAFCO_REASONING_EFFORT` | `low`         |
| Iterations        | `--max-iter, -n` | —                        | `12`          |
| Temperature       | `--temperature`  | —                        | `0.6` (legacy)|
| Reasoning toggle  | `--no-reasoning` | —                        | enabled       |

GPT-5 family (`gpt-5`, `gpt-5-mini`, `gpt-5.5`, `o1`, `o3`, `o4`) is
auto-detected and routed via `reasoning_effort` + `max_completion_tokens`.
Legacy models (`gpt-4o`, `gpt-4-turbo`) keep `temperature` + `max_tokens`.

To switch models:

```bash
export MAFCO_MODEL=gpt-5            # full quality
export MAFCO_MODEL=gpt-5-nano       # cheapest
export MAFCO_REASONING_EFFORT=high  # raise the reasoning budget
```

## Benchmarking

The bench compares MAFCO against a flat one-shot baseline on the same model.
Tasks live in `benchmarks/tasks.json` and target the MAFCO primitives
(revision, invalidation, merge, aggregation).

```bash
uv run mafco bench                                  # uses benchmarks/tasks.json
uv run mafco bench --tasks my_tasks.json -o out.json
```

Output: per-task pass/fail, latency, and a count of memory operations
(facts/hypotheses/conclusions/revises/invalidates/merges). See
`Report.tex` for the writeup with results and TikZ diagrams.

## Project layout

```
src/mafco/
  types/schema.py     MemoryUnit, BeliefType, MemoryEvent
  core/memory.py      MemoryStore - the first-class memory object
  core/repl.py        Persistent REPL executor
  core/rlm.py         RLM loop (root LLM ↔ REPL ↔ memory)
  core/prompts.py     Root + sub-LLM system prompts
  model/client.py     OpenRouter client (OpenAI SDK pointed at OR)
  utils/parsing.py    Parse repl-blocks + FINAL/FINAL_VAR
  bench/runner.py     Benchmark runner (MAFCO vs baseline)
  cli.py              Typer CLI
benchmarks/tasks.json Default task suite
main.py               Entry point
Report.tex            Latex report (compile with pdflatex / Overleaf)
```

## What the model sees inside the REPL

```python
context        # the (potentially huge) input
memory         # MemoryStore — programmable, typed, revision-aware
llm_query()    # call a sub-LLM on a slice of context
```

The model writes ` ```repl ... ``` ` blocks; output (truncated) plus a
memory snapshot is fed back next turn. The loop ends when the model emits
`FINAL(answer)` or `FINAL_VAR(varname)`.

## MemoryStore API (what the model programs against)

```python
memory.add_fact(content, confidence=0.9, provenance=["chunk:42"])
memory.add_hypothesis(content, confidence=0.5)
memory.add_conclusion(content, provenance=[fact_id])

memory.revise(unit_id, new_content, new_confidence=0.7, reason="...")
memory.invalidate(unit_id, reason="falsified by ...")
memory.merge([id1, id2], merged_content, reason="duplicates")

memory.facts()         # list[MemoryUnit]
memory.hypotheses()
memory.conclusions()
memory.query(type=..., min_confidence=..., contains=..., tag=...)
memory.summary()       # human-readable digest
memory.save(path)      # JSON snapshot
```

Every mutation appends a `MemoryEvent` — the audit trail is complete.
`revise()` creates a new unit linked via `revision_of`; the old one stays for
provenance.
