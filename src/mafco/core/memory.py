"""MemoryStore — the first-class computational object the LLM programs against.

Design goals (from MAFCO.md):
  - Explicit belief typing (fact / hypothesis / conclusion).
  - Revision-aware: every change is an auditable event, never a silent overwrite.
  - Provenance: every unit knows where it came from.
  - Programmable: exposed as a Python object inside the REPL the LLM drives.
  - Persistable: snapshot/load to JSON for cross-session continuity.

This module is the SINGLE source of truth for memory semantics. The REPL hands
the LLM a `memory` instance of `MemoryStore`, and that is the only legitimate
way to mutate beliefs.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Iterable, Iterator

from mafco.types.schema import (
    BeliefType,
    MemoryEvent,
    MemoryEventKind,
    MemoryUnit,
)


class MemoryError(Exception):
    pass


class MemoryStore:
    """Programmable, typed, revision-tracked memory.

    All mutating methods append to `self.events` so the audit log is complete.
    Reading methods never mutate state.
    """

    def __init__(self, session_id: str | None = None) -> None:
        self.session_id: str = session_id or _short_id()
        self._units: dict[str, MemoryUnit] = {}
        self.events: list[MemoryEvent] = []

    def add(
        self,
        content: str,
        type: BeliefType | str = BeliefType.FACT,
        *,
        confidence: float = 1.0,
        provenance: list[str] | str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        valid_until: str | None = None,
    ) -> str:
        """Insert a new MemoryUnit and return its id."""
        belief = _coerce_type(type)
        prov = _coerce_provenance(provenance)
        unit = MemoryUnit(
            id=_short_id(),
            content=content.strip(),
            type=belief,
            confidence=_clamp_conf(confidence),
            provenance=prov,
            tags=list(tags or []),
            metadata=dict(metadata or {}),
            valid_until=valid_until,
        )
        self._units[unit.id] = unit
        self.events.append(
            MemoryEvent(
                kind=MemoryEventKind.ADD,
                unit_id=unit.id,
                payload={"type": belief.value, "confidence": unit.confidence},
            )
        )
        return unit.id

    def get(self, unit_id: str) -> MemoryUnit:
        if unit_id not in self._units:
            raise MemoryError(f"unknown memory unit id: {unit_id!r}")
        return self._units[unit_id]

    def __contains__(self, unit_id: object) -> bool:
        return isinstance(unit_id, str) and unit_id in self._units

    def __len__(self) -> int:
        return len(self._units)

    def __iter__(self) -> Iterator[MemoryUnit]:
        return iter(self._units.values())

    def all(self, *, include_invalidated: bool = False) -> list[MemoryUnit]:
        return [
            u for u in self._units.values() if include_invalidated or not u.invalidated
        ]

    def query(
        self,
        *,
        type: BeliefType | str | None = None,
        min_confidence: float | None = None,
        contains: str | None = None,
        tag: str | None = None,
        include_invalidated: bool = False,
    ) -> list[MemoryUnit]:
        """Filter units by belief type, confidence floor, substring or tag."""
        belief = _coerce_type(type) if type is not None else None
        needle = contains.lower() if contains else None
        out: list[MemoryUnit] = []
        for u in self._units.values():
            if u.invalidated and not include_invalidated:
                continue
            if belief is not None and u.type is not belief:
                continue
            if min_confidence is not None and u.confidence < min_confidence:
                continue
            if needle is not None and needle not in u.content.lower():
                continue
            if tag is not None and tag not in u.tags:
                continue
            out.append(u)
        return out

    # -------------------------------------------------------------- mutate

    def revise(
        self,
        unit_id: str,
        new_content: str,
        *,
        new_confidence: float | None = None,
        reason: str | None = None,
    ) -> str:
        """Create a NEW unit that supersedes `unit_id`. Old unit is kept (audit)."""
        old = self.get(unit_id)
        new_id = self.add(
            content=new_content,
            type=old.type,
            confidence=new_confidence if new_confidence is not None else old.confidence,
            provenance=list(old.provenance) + [f"revision_of:{unit_id}"],
            tags=list(old.tags),
            metadata={**old.metadata, "revised_from": unit_id, "reason": reason},
        )
        revised = self._units[new_id]
        revised.revision_of = unit_id
        revised.revision_history = [*old.revision_history, unit_id]
        old.metadata["superseded_by"] = new_id
        self.events.append(
            MemoryEvent(
                kind=MemoryEventKind.REVISE,
                unit_id=new_id,
                payload={"revised_from": unit_id, "reason": reason},
            )
        )
        return new_id

    def invalidate(self, unit_id: str, reason: str) -> None:
        unit = self.get(unit_id)
        if unit.invalidated:
            return
        unit.invalidated = True
        unit.invalidation_reason = reason
        self.events.append(
            MemoryEvent(
                kind=MemoryEventKind.INVALIDATE,
                unit_id=unit_id,
                payload={"reason": reason},
            )
        )

    def merge(
        self,
        unit_ids: Iterable[str],
        merged_content: str,
        *,
        type: BeliefType | str | None = None,
        confidence: float | None = None,
        reason: str | None = None,
    ) -> str:
        """Combine several units into a single new one and invalidate the sources."""
        ids = list(unit_ids)
        if len(ids) < 2:
            raise MemoryError("merge requires at least 2 unit ids")
        sources = [self.get(i) for i in ids]
        merged_type = _coerce_type(type) if type is not None else sources[0].type
        merged_conf = (
            confidence
            if confidence is not None
            else min(s.confidence for s in sources)
        )
        prov = [f"merge_of:{i}" for i in ids]
        new_id = self.add(
            content=merged_content,
            type=merged_type,
            confidence=merged_conf,
            provenance=prov,
            metadata={"merged_from": ids, "reason": reason},
        )
        for s in sources:
            s.invalidated = True
            s.invalidation_reason = f"merged into {new_id}"
            s.metadata["merged_into"] = new_id
        self.events.append(
            MemoryEvent(
                kind=MemoryEventKind.MERGE,
                unit_id=new_id,
                payload={"merged_from": ids, "reason": reason},
            )
        )
        return new_id

    def tag(self, unit_id: str, *tags: str) -> None:
        unit = self.get(unit_id)
        added: list[str] = []
        for t in tags:
            if t not in unit.tags:
                unit.tags.append(t)
                added.append(t)
        if added:
            self.events.append(
                MemoryEvent(
                    kind=MemoryEventKind.TAG, unit_id=unit_id, payload={"tags": added}
                )
            )

    # -------------------------------------------------------- convenience

    def add_fact(self, content: str, **kw: Any) -> str:
        return self.add(content, BeliefType.FACT, **kw)

    def add_hypothesis(self, content: str, **kw: Any) -> str:
        return self.add(content, BeliefType.HYPOTHESIS, **kw)

    def add_conclusion(self, content: str, **kw: Any) -> str:
        return self.add(content, BeliefType.CONCLUSION, **kw)

    def facts(self, **kw: Any) -> list[MemoryUnit]:
        return self.query(type=BeliefType.FACT, **kw)

    def hypotheses(self, **kw: Any) -> list[MemoryUnit]:
        return self.query(type=BeliefType.HYPOTHESIS, **kw)

    def conclusions(self, **kw: Any) -> list[MemoryUnit]:
        return self.query(type=BeliefType.CONCLUSION, **kw)

    # -------------------------------------------------------- presentation

    def summary(self, *, max_units: int = 30, max_chars: int = 220) -> str:
        """Compact, deterministic textual summary for showing to an LLM."""
        live = self.all()
        lines = [f"# MemoryStore (session={self.session_id}) — {len(live)} live units"]
        by_type: dict[BeliefType, list[MemoryUnit]] = {b: [] for b in BeliefType}
        for u in live:
            by_type[u.type].append(u)
        plurals = {
            BeliefType.FACT: "facts",
            BeliefType.HYPOTHESIS: "hypotheses",
            BeliefType.CONCLUSION: "conclusions",
        }
        for b in (BeliefType.FACT, BeliefType.HYPOTHESIS, BeliefType.CONCLUSION):
            bucket = by_type[b]
            if not bucket:
                continue
            lines.append(f"\n## {plurals[b]} ({len(bucket)})")
            for u in bucket[:max_units]:
                snippet = u.content if len(u.content) <= max_chars else u.content[: max_chars - 1] + "…"
                rev = f" rev_of={u.revision_of}" if u.revision_of else ""
                lines.append(
                    f"- [{u.id} c={u.confidence:.2f}{rev}] {snippet}"
                )
            if len(bucket) > max_units:
                lines.append(f"- … (+{len(bucket) - max_units} more)")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"MemoryStore(session={self.session_id!r}, units={len(self._units)}, "
            f"events={len(self.events)})"
        )

    # ------------------------------------------------------------ persist

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "units": [u.to_dict() for u in self._units.values()],
            "events": [e.to_dict() for e in self.events],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MemoryStore":
        store = cls(session_id=d.get("session_id"))
        for ud in d.get("units", []):
            unit = MemoryUnit.from_dict(ud)
            store._units[unit.id] = unit
        for ed in d.get("events", []):
            kind = MemoryEventKind(ed["kind"])
            store.events.append(
                MemoryEvent(
                    kind=kind,
                    unit_id=ed["unit_id"],
                    timestamp=ed.get("timestamp", ""),
                    payload=ed.get("payload", {}),
                )
            )
        return store

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "MemoryStore":
        return cls.from_dict(json.loads(Path(path).read_text()))


# --------------------------------------------------------------- helpers


def _short_id() -> str:
    return uuid.uuid4().hex[:10]


def _clamp_conf(c: float) -> float:
    return max(0.0, min(1.0, float(c)))


def _coerce_type(t: BeliefType | str | None) -> BeliefType:
    if isinstance(t, BeliefType):
        return t
    if isinstance(t, str):
        return BeliefType(t.lower())
    raise MemoryError(f"invalid belief type: {t!r}")


def _coerce_provenance(p: list[str] | str | None) -> list[str]:
    if p is None:
        return []
    if isinstance(p, str):
        return [p]
    return list(p)
