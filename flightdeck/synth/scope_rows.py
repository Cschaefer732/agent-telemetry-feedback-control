"""ScopeRecord generation -- the row half of what `sample_data.py` used to do, split from
label generation per report 06. Pure: no store, no filesystem, deterministic in params+seed
only.

# generator
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from flightdeck.models import DISPOSITIONS, SCOPE_TIERS, ScopeRecord
from flightdeck.synth.profiles import (
    REAL_CORRECTION_RATE,
    REAL_REWORK_RATE,
    TIER_PROFILE,
    TIER_WEIGHTS,
)
from flightdeck.synth.provenance import stamp_row


@dataclass(frozen=True)
class ScopeRowParams:
    count: int = 60
    seed: int = 20260905
    host: str = "synth"
    tier_weights: tuple[float, float, float] = TIER_WEIGHTS  # none/mini/full
    rework_rate: float = REAL_REWORK_RATE
    correction_rate: float = REAL_CORRECTION_RATE


def _record(rng: random.Random, params: ScopeRowParams, index: int) -> ScopeRecord:
    tier = rng.choices(list(SCOPE_TIERS), weights=list(params.tier_weights))[0]
    profile = TIER_PROFILE[tier]
    found = rng.randint(3, 18)
    late = min(found, round(found * rng.uniform(*profile["late"])))
    committed = rng.randint(1, max(1, found - late))
    remaining = found - committed
    non_goals = rng.randint(0, remaining)
    assumptions = remaining - non_goals
    files = rng.randint(1, 14)
    asked = rng.randint(*profile["questions"])
    flagged = rng.randint(0, 8)
    return ScopeRecord(
        record_id=f"synth-{index:04d}",
        session_id=f"synth-session-{index // 4:03d}",
        created_at=1_756_900_000 + index * 3600,
        host=params.host,
        tier=tier,
        verdict="pass",
        found_total=found,
        committed=committed,
        non_goals=non_goals,
        assumptions=assumptions,
        late_discovered=late,
        ceremony_tokens=rng.randint(*profile["ceremony"]),
        ceremony_ms=rng.randint(500, 90_000),
        questions_asked=asked,
        questions_valuable=rng.randint(0, asked),
        files_edited=files,
        rework_files=sum(1 for _ in range(files) if rng.random() < params.rework_rate),
        turns_to_first_edit=rng.randint(1, 6),
        turns_to_done=rng.randint(4, 40),
        corrections=sum(
            1 for _ in range(rng.randint(2, 12)) if rng.random() < params.correction_rate
        ),
        correction_families={"missed": rng.randint(0, 2), "negate": rng.randint(0, 2)},
        assumptions_overridden=rng.randint(0, assumptions) if assumptions else 0,
        divergence_flagged=flagged,
        divergence_kept=rng.randint(0, flagged),
        provenance=stamp_row(
            {"generator": "flightdeck.synth.scope_rows", "label_source": "generator_asserted"}
        ),
    )


def generate_rows(params: ScopeRowParams) -> list[ScopeRecord]:
    """Pure -- no store, no filesystem. Deterministic in params+seed only."""
    rng = random.Random(params.seed)
    return [_record(rng, params, index) for index in range(params.count)]


def to_corpus(rows: list[ScopeRecord]) -> list[dict[str, Any]]:
    """rows -> to_row() dicts, ready for content_hash() and Store.add_scope_record()."""
    return [row.to_row() for row in rows]


def invariants_hold(record: ScopeRecord) -> bool:
    """Every found item lands in exactly one disposition."""
    assert len(DISPOSITIONS) == 3
    return record.committed + record.non_goals + record.assumptions == record.found_total
