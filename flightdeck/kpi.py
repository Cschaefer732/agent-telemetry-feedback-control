"""Deterministic per-turn KPI scoring, zero model calls.

Every component is a pure function of a Turn plus its Events so the nightly review can trust the
number without re-deriving it. `score_and_persist` is the only impure entry point: it pulls the
trailing-7-day baseline/p95 out of the store and writes the result back onto the turn row.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from flightdeck.models import Event, Turn

if TYPE_CHECKING:
    from flightdeck.store import Store

DEFAULT_WEIGHTS: dict[str, float] = {
    "completion": 1 / 6,
    "tool_reliability": 1 / 6,
    "efficiency": 1 / 6,
    "focus": 1 / 6,
    "context_health": 1 / 6,
    "autonomy": 1 / 6,
}

DEFAULT_THRESHOLDS: dict[str, float] = {
    "tool_error_rate": 0.25,
    "low_kpi": 0.5,
}

_TRAILING_WINDOW_MS = 7 * 24 * 60 * 60 * 1000


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass
class Components:
    completion: float
    tool_reliability: float
    efficiency: float
    focus: float
    context_health: float
    autonomy: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def _completion(turn: Turn, events: list[Event]) -> float:
    # outcome None (a turn closed by orphan-backfill, or a source blind to outcome) is NOT a
    # failure — only "error"/"cancelled" are. Unmeasurable completion defaults optimistically,
    # matching should_flag's null-tolerance and _context_health's "full health when blind" rule;
    # scoring None as 0.0 silently dragged every outcome-blind/in-flight turn down ~1/6.
    failed = turn.outcome in ("error", "cancelled")
    todo_events = [e for e in events if e.kind == "todo"]
    if todo_events:
        opened = sum(e.payload.get("opened", 0) or 0 for e in todo_events)
        closed = sum(e.payload.get("closed", 0) or 0 for e in todo_events)
        base = _clamp(closed / opened) if opened > 0 else (0.0 if failed else 1.0)
    else:
        base = 0.0 if failed else 1.0

    if any(e.kind == "critic" and e.payload.get("verdict") == "fail" for e in events):
        base = min(base, 0.5)
    if failed:
        base = 0.0
    return _clamp(base)


def _tool_reliability(events: list[Event]) -> float:
    tool_calls = [e for e in events if e.kind == "tool_call"]
    if not tool_calls:
        return 1.0
    failed = sum(1 for e in tool_calls if e.ok == 0)
    return _clamp(1 - failed / len(tool_calls))


def _efficiency(turn: Turn, baseline_tokens: float | None) -> float:
    total_tokens = turn.total_tokens
    if baseline_tokens is None or total_tokens == 0:
        return 1.0
    return _clamp(baseline_tokens / total_tokens)


def _focus(events: list[Event]) -> float:
    edits = [e for e in events if e.kind == "edit"]
    reverts = [e for e in events if e.kind == "revert"]
    seen_paths: set[object] = set()
    repeat_edits = 0
    for e in edits:
        path = e.payload.get("path")
        if path in seen_paths:
            repeat_edits += 1
        else:
            seen_paths.add(path)
    churn = (repeat_edits + 2 * len(reverts)) / max(1, len(edits))
    return _clamp(1 - churn)


def _context_health(turn: Turn, events: list[Event]) -> float:
    occupancy = turn.context_occupancy
    # A source that cannot observe occupancy (Claude Code hooks) still observes compaction, so an
    # unmeasurable occupancy starts from full health rather than skipping the penalty entirely.
    base = 1.0 if occupancy is None else 1 - occupancy
    # Only real compactions. Context snapshots are observations, not events worth penalizing, and
    # they used to share this kind — which quietly zeroed this component on any well-instrumented
    # turn, exactly inverting what the score was supposed to reward.
    compactions = sum(1 for e in events if e.kind == "compaction")
    return _clamp(base - 0.2 * compactions)


def _autonomy(events: list[Event]) -> float:
    interrupts = sum(1 for e in events if e.kind == "interrupt")
    denies = sum(
        1 for e in events if e.kind == "permission" and e.payload.get("decision") == "deny"
    )
    return _clamp(1 - min(1, (interrupts + denies) / 3))


def score_components(
    turn: Turn, events: list[Event], *, baseline_tokens: float | None = None
) -> Components:
    return Components(
        completion=_completion(turn, events),
        tool_reliability=_tool_reliability(events),
        efficiency=_efficiency(turn, baseline_tokens),
        focus=_focus(events),
        context_health=_context_health(turn, events),
        autonomy=_autonomy(events),
    )


def composite(components: Components, weights: dict[str, float] | None = None) -> float:
    weights = weights if weights is not None else DEFAULT_WEIGHTS
    return sum(weights.get(name, 0.0) * value for name, value in components.as_dict().items())


def score_turn(
    turn: Turn,
    events: list[Event],
    *,
    baseline_tokens: float | None = None,
    weights: dict[str, float] | None = None,
) -> tuple[float, Components]:
    components = score_components(turn, events, baseline_tokens=baseline_tokens)
    return composite(components, weights), components


def should_flag(
    turn: Turn,
    events: list[Event],
    components: Components,
    *,
    p95_wall_ms: float | None = None,
    thresholds: dict[str, float] | None = None,
    weights: dict[str, float] | None = None,
) -> tuple[bool, list[str]]:
    thresholds = thresholds if thresholds is not None else DEFAULT_THRESHOLDS
    reasons: list[str] = []

    if turn.outcome is not None and turn.outcome != "ok":
        reasons.append("outcome_not_ok")
    if any(e.kind == "critic" and e.payload.get("verdict") == "fail" for e in events):
        reasons.append("critic_fail")

    tool_calls = [e for e in events if e.kind == "tool_call"]
    if tool_calls:
        failed = sum(1 for e in tool_calls if e.ok == 0)
        if failed / len(tool_calls) > thresholds["tool_error_rate"]:
            reasons.append("tool_error_rate")

    if p95_wall_ms is not None and turn.wall_ms is not None and turn.wall_ms > p95_wall_ms:
        reasons.append("slow")
    if any(e.kind == "compaction" for e in events):
        reasons.append("midturn_compaction")
    if any(e.kind == "revert" for e in events):
        reasons.append("edit_revert")
    if any(e.kind == "interrupt" for e in events):
        reasons.append("user_interrupt")
    if composite(components, weights) < thresholds["low_kpi"]:
        reasons.append("low_kpi")

    return bool(reasons), reasons


def trailing_baseline(
    store: Store,
    *,
    source: str | None,
    tier: str | None,
    before_ms: int,
    exclude_turn_id: str | None = None,
) -> tuple[float | None, float | None]:
    """(median total_tokens, p95 wall_ms) over the trailing 7-day window ending at `before_ms`,
    for turns matching `source`/`tier`. Backs score_and_persist's default per-turn lookup; a
    caller scoring many turns for the same (source, tier) in one run (cmd_rollup) can compute
    this once and pass it to every turn instead — the two SQL percentile calls per turn were the
    same cost regardless of which turn asked, so recomputing per turn bought nothing but O(turns)
    redundant queries."""
    since_ms = before_ms - _TRAILING_WINDOW_MS
    baseline_tokens = store.percentile(
        "total_tokens",
        0.5,
        tier=tier,
        source=source,
        since_ms=since_ms,
        until_ms=before_ms,
        exclude_turn_id=exclude_turn_id,
    )
    p95_wall_ms = store.percentile(
        "wall_ms",
        0.95,
        tier=tier,
        source=source,
        since_ms=since_ms,
        until_ms=before_ms,
        exclude_turn_id=exclude_turn_id,
    )
    return baseline_tokens, p95_wall_ms


def score_and_persist(
    store: Store,
    turn: Turn,
    *,
    baseline: tuple[float | None, float | None] | None = None,
    weights: dict[str, float] | None = None,
    thresholds: dict[str, float] | None = None,
) -> tuple[float, Components, list[str]]:
    """Score `turn` and write it back. `baseline` is (baseline_tokens, p95_wall_ms); when omitted
    (the default) it's computed exactly for this turn's own trailing window. Pass a precomputed
    baseline to skip that lookup — see `trailing_baseline`. `weights`/`thresholds` default to
    DEFAULT_WEIGHTS/DEFAULT_THRESHOLDS (via score_turn/should_flag) when omitted; the rollup path
    passes the governor.toml [kpi.weights]/[thresholds] tables so the nightly reviewer's edits to
    those tables actually take effect — see governor.load_kpi_tuning."""
    events = store.events_for(turn.turn_id)

    if baseline is None:
        baseline_tokens, p95_wall_ms = trailing_baseline(
            store,
            source=turn.source,
            tier=turn.tier,
            before_ms=turn.started_at,
            exclude_turn_id=turn.turn_id,
        )
    else:
        baseline_tokens, p95_wall_ms = baseline

    score, components = score_turn(turn, events, baseline_tokens=baseline_tokens, weights=weights)
    flagged, reasons = should_flag(
        turn, events, components, p95_wall_ms=p95_wall_ms, thresholds=thresholds, weights=weights
    )

    turn.kpi_score = score
    turn.kpi_components = json.dumps(components.as_dict(), separators=(",", ":"))
    turn.flagged = 1 if flagged else 0
    store.upsert_turn(turn)

    return score, components, reasons
