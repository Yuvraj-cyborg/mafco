"""Schemas for memory units and audit events.

A `MemoryUnit` is the atomic, programmable, typed object the LLM reasons against.
A `MemoryEvent` records every lifecycle change to make the store revision-aware.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class BeliefType(str, Enum):
    FACT = "fact"
    HYPOTHESIS = "hypothesis"
    CONCLUSION = "conclusion"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class MemoryUnit:
    id: str
    content: str
    type: BeliefType
    confidence: float = 1.0
    provenance: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=_utcnow)
    valid_until: str | None = None
    revision_of: str | None = None
    revision_history: list[str] = field(default_factory=list)
    invalidated: bool = False
    invalidation_reason: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["type"] = self.type.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MemoryUnit":
        d = dict(d)
        d["type"] = BeliefType(d["type"])
        return cls(**d)


class MemoryEventKind(str, Enum):
    ADD = "add"
    REVISE = "revise"
    INVALIDATE = "invalidate"
    MERGE = "merge"
    TAG = "tag"


@dataclass
class MemoryEvent:
    kind: MemoryEventKind
    unit_id: str
    timestamp: str = field(default_factory=_utcnow)
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["kind"] = self.kind.value
        return d
