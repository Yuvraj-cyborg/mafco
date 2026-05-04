"""MAFCO command-line interface.

Commands:
  mafco run      Run the RLM on a query, optionally with a context file.
  mafco bench    Run the benchmark suite (MAFCO vs. baseline).
  mafco memory   Inspect a saved memory snapshot.
  mafco version
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from mafco.core.memory import MemoryStore
from mafco.core.rlm import RLM, RLMConfig
from mafco.model.client import ModelConfig

load_dotenv()

app = typer.Typer(
    add_completion=False,
    help="MAFCO — Memory As First-Class Computational Object on top of an RLM.",
    no_args_is_help=True,
)
memory_app = typer.Typer(help="Inspect persisted memory snapshots.", no_args_is_help=True)
app.add_typer(memory_app, name="memory")

console = Console()


def _make_rlm(
    model: str | None,
    sub_model: str | None,
    temperature: float,
    max_tokens: int | None,
    max_iterations: int,
    verbose: bool,
    no_reasoning: bool,
) -> RLM:
    mc = ModelConfig(temperature=temperature, enable_reasoning=not no_reasoning)
    if model:
        mc.model = model
    if sub_model:
        mc.sub_model = sub_model
    if max_tokens is not None:
        mc.max_tokens = max_tokens
    return RLM(RLMConfig(model=mc, max_iterations=max_iterations, verbose=verbose))


def _load_context(path: Optional[Path], inline: Optional[str]) -> Any:
    if path is not None:
        if not path.exists():
            raise typer.BadParameter(f"context file not found: {path}")
        text = path.read_text()
        if path.suffix.lower() == ".json":
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text
        return text
    if inline is not None:
        return inline
    return ""


@app.command()
def run(
    query: str = typer.Argument(..., help="The question to answer."),
    context_file: Optional[Path] = typer.Option(
        None, "--context-file", "-c", help="Path to a context file (txt/md/json)."
    ),
    context: Optional[str] = typer.Option(
        None, "--context", help="Inline context string."
    ),
    save_memory: Optional[Path] = typer.Option(
        None, "--save-memory", help="Persist final MemoryStore as JSON to this path."
    ),
    load_memory: Optional[Path] = typer.Option(
        None, "--load-memory", help="Seed the run with a saved MemoryStore."
    ),
    model: Optional[str] = typer.Option(None, "--model", help="Root model id (override default/env)."),
    sub_model: Optional[str] = typer.Option(None, "--sub-model", help="Sub-LLM model id."),
    temperature: float = typer.Option(0.6, "--temperature", "-t"),
    max_tokens: int = typer.Option(4096, "--max-tokens", help="Cap output tokens per turn."),
    max_iterations: int = typer.Option(12, "--max-iter", "-n"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Stream the trace."),
    no_reasoning: bool = typer.Option(
        False, "--no-reasoning", help="Disable thinking on root calls."
    ),
    show_memory: bool = typer.Option(
        True, "--show-memory/--no-show-memory", help="Print memory at end."
    ),
) -> None:
    """Run the RLM on a query with optional context."""
    ctx = _load_context(context_file, context)
    mem = MemoryStore.load(load_memory) if load_memory else None

    rlm = _make_rlm(
        model, sub_model, temperature, max_tokens, max_iterations, verbose, no_reasoning
    )
    console.rule("[bold cyan]MAFCO RLM[/bold cyan]")
    console.print(f"[dim]model:[/dim] {rlm.config.model.model}")
    console.print(f"[dim]sub:  [/dim] {rlm.config.model.sub_model}")
    console.print(
        f"[dim]ctx:  [/dim] {type(ctx).__name__}, "
        f"{len(ctx) if hasattr(ctx, '__len__') else '?'} units/chars"
    )
    console.rule()

    result = rlm.run(query, context=ctx, memory=mem)

    console.rule("[bold green]Final Answer[/bold green]")
    console.print(Panel.fit(result.answer or "(empty)", border_style="green"))

    if show_memory:
        console.rule("[bold magenta]Memory[/bold magenta]")
        console.print(result.memory.summary(), markup=False, highlight=False)

    console.rule("[dim]trace[/dim]")
    console.print(
        f"iterations={result.trace.iterations_used} "
        f"sub_calls={result.trace.sub_calls} "
        f"sub_call_chars={result.trace.sub_call_chars} "
        f"terminated={result.terminated}"
    )

    if save_memory:
        result.memory.save(save_memory)
        console.print(f"[dim]memory saved → {save_memory}[/dim]")


@app.command()
def bench(
    tasks_path: Path = typer.Option(
        Path("benchmarks/tasks.json"), "--tasks", help="Path to tasks JSON."
    ),
    output: Path = typer.Option(
        Path("benchmarks/results.json"), "--output", "-o", help="Where to dump results."
    ),
    model: Optional[str] = typer.Option(None, "--model"),
    sub_model: Optional[str] = typer.Option(None, "--sub-model"),
    max_iterations: int = typer.Option(8, "--max-iter", "-n"),
    max_tokens: int = typer.Option(4096, "--max-tokens"),
    temperature: float = typer.Option(0.2, "--temperature", "-t"),
) -> None:
    """Run the MAFCO-vs-baseline benchmark suite over the task file."""
    from mafco.bench.runner import load_tasks, run_suite, summarise

    tasks = load_tasks(tasks_path)
    console.print(
        f"[bold]bench[/bold]: {len(tasks)} tasks × 2 systems "
        f"(model={model or 'default'})"
    )

    def _on_done(tr: Any) -> None:
        ops = (
            f"f={tr.facts} h={tr.hypotheses} c={tr.conclusions} "
            f"rev={tr.revises} inv={tr.invalidates} mrg={tr.merges}"
            if tr.system == "mafco"
            else "—"
        )
        status = "OK" if tr.correct else "MISS"
        err_tag = ""
        if tr.error:
            short = tr.error.split(":", 1)[0]
            if "RateLimit" in short:
                err_tag = " [red](rate-limited)[/red]"
            else:
                err_tag = f" [red]({short})[/red]"
        console.print(
            f"  [{tr.system:8s}] {tr.task_id:25s} {status} "
            f"({tr.latency_s}s)  {ops}{err_tag}"
        )

    result = run_suite(
        tasks,
        model=model,
        sub_model=sub_model,
        max_iterations=max_iterations,
        max_tokens=max_tokens,
        temperature=temperature,
        on_task_done=_on_done,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result.to_dict(), indent=2))

    console.rule("[bold]Summary[/bold]")
    s = summarise(result)
    for sys_name, agg in s["per_system"].items():
        console.print(f"[bold]{sys_name}[/bold]: {json.dumps(agg)}")
    console.print(f"[dim]results → {output}[/dim]")


@memory_app.command("show")
def memory_show(
    path: Path = typer.Argument(..., help="Path to a memory JSON snapshot."),
) -> None:
    """Pretty-print a persisted memory snapshot."""
    store = MemoryStore.load(path)
    panel = Panel.fit(store.summary(), title=f"MemoryStore @ {path}")
    console.print(panel, markup=False, highlight=False)


@memory_app.command("dump")
def memory_dump(
    path: Path = typer.Argument(...),
) -> None:
    """Dump the raw JSON of a memory snapshot (with syntax highlighting)."""
    text = path.read_text()
    console.print(Syntax(text, "json", line_numbers=False))


@memory_app.command("events")
def memory_events(
    path: Path = typer.Argument(...),
) -> None:
    """Print the audit-event log of a memory snapshot."""
    store = MemoryStore.load(path)
    table = Table(title="Audit events")
    table.add_column("ts"); table.add_column("kind"); table.add_column("unit_id"); table.add_column("payload")
    for ev in store.events:
        table.add_row(
            ev.timestamp,
            ev.kind.value,
            ev.unit_id,
            json.dumps(ev.payload, default=str),
        )
    console.print(table)


@app.command()
def version() -> None:
    """Print package version."""
    try:
        from importlib.metadata import version as _v
        console.print(f"mafco {_v('mafco')}")
    except Exception:
        console.print("mafco (dev)")


def main() -> None:
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[red]interrupted[/red]")
        sys.exit(130)


if __name__ == "__main__":
    main()
