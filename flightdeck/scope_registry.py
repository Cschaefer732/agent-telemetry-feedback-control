"""A merit-ranked overlay on the gate's hand-written keyword registries.

The gate's registries are edited by hand and nothing measures them. This file is how a
phrase earns its way in from evidence instead -- and how it earns its way back out.

Retention is by MERIT, never by recency. The measured failure mode for an accumulating
memory of this shape is that FIFO eviction is WORSE than keeping everything (P@5 15.8% ->
3.8%), and that an unbounded store degrades on its own (2,400 records -> 13% accuracy vs
248 -> 39%). So every entry carries the counters that justify it, and eviction reads those
counters rather than a timestamp.

The overlay is an OVERLAY. It only ever adds to the literal tuples in scope_gate, and a
missing, unreadable or inconsistent file leaves the hand-written defaults in force. That
makes `rm scope_registry.json` a complete, instant rollback, and it matches the fail-open
posture the hook already has: a broken registry must not out-decide the tuned baseline.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

OVERLAY_PATH = Path(__file__).resolve().parent / "scope_registry.json"

#: Set this to refuse every write. The kill switch is an env var so it can be flipped
#: without a deploy, and it is checked in the writer rather than the reader so a frozen
#: registry still SERVES its current contents.
FREEZE_ENV = "FLIGHTDECK_SCOPE_REGISTRY_FROZEN"

#: Promotion needs repetition, not a single lucky hit; demotion needs sustained harm.
#: These are no-regret bars sized for tiny N, not significance thresholds, and they are
#: deliberately stated as such -- at this arrival rate nothing here will be publishable.
PROMOTE_MIN_HELPFUL = 3
PROMOTE_MIN_RATIO = 0.7
DEMOTE_MIN_TOTAL = 5
DEMOTE_MAX_RATIO = 0.5

ACTIVE, RETIRED = "active", "retired"


@dataclass
class Entry:
    phrase: str
    registry: str
    helpful: int = 0
    harmful: int = 0
    added_at: int = field(default_factory=lambda: int(time.time()))
    status: str = ACTIVE
    evidence: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.helpful + self.harmful

    @property
    def ratio(self) -> float | None:
        return self.helpful / self.total if self.total else None

    @property
    def promotable(self) -> bool:
        return (
            self.helpful >= PROMOTE_MIN_HELPFUL
            and (self.ratio or 0) >= PROMOTE_MIN_RATIO
        )

    @property
    def demotable(self) -> bool:
        return self.total >= DEMOTE_MIN_TOTAL and (self.ratio or 1) < DEMOTE_MAX_RATIO


def load(path: Path | str = OVERLAY_PATH) -> list[Entry]:
    """Never raises. An unreadable overlay means "no overlay", not "no gate"."""
    try:
        raw = json.loads(Path(path).read_text())
        return [Entry(**e) for e in raw.get("entries", [])]
    except (OSError, ValueError, TypeError):
        return []


def save(entries: list[Entry], path: Path | str = OVERLAY_PATH) -> Path:
    if os.environ.get(FREEZE_ENV):
        raise RuntimeError(f"{FREEZE_ENV} is set; refusing to write the registry overlay")
    target = Path(path)
    if target.exists():
        backup = target.with_name(f"{target.stem}-{int(time.time())}{target.suffix}")
        backup.write_text(target.read_text())
    target.write_text(
        json.dumps(
            {"version": 1, "written_at": int(time.time()),
             "entries": [asdict(e) for e in entries]},
            indent=1, sort_keys=True,
        )
    )
    return target


def apply_overlay(
    registry: str, base: tuple[str, ...], path: Path | str = OVERLAY_PATH
) -> tuple[str, ...]:
    """Extend one hand-written tuple with the active phrases promoted for it.

    Additive only: an overlay can never REMOVE a hand-written phrase. Learning that deletes
    the baseline it was measured against cannot be rolled back by deleting a file.
    """
    extra = [
        e.phrase for e in load(path)
        if e.registry == registry and e.status == ACTIVE and e.phrase not in base
    ]
    return base + tuple(sorted(extra))


def summary(path: Path | str = OVERLAY_PATH) -> dict[str, Any]:
    entries = load(path)
    return {
        "entries": len(entries),
        "active": sum(1 for e in entries if e.status == ACTIVE),
        "retired": sum(1 for e in entries if e.status == RETIRED),
        "frozen": bool(os.environ.get(FREEZE_ENV)),
        "path": str(path),
        "verdict": "silent" if not entries else "pass",
    }
