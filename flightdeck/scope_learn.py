"""Mine evidence for gate registry changes, and refuse the ones that cannot be justified.

The loop this closes: the gate decides, the decision is recorded, and until now nothing
ever found out whether it was right. Here the signal is the NEXT-TURN CORRECTION -- a turn
the gate let through with no scoping (tier NONE, no scope already open) whose very next
human turn corrects the work. That is weak, biased evidence, not ground truth, and it is
treated as such: it only ever proposes a candidate, and a candidate only lands after a
deterministic counterfactual replay says it would have helped more than it hurt.

Nothing here asks a model to rewrite anything. Candidate mining is the only step with any
judgement in it, and the judgement is a regex or a judge verdict over text that already
exists. Promotion, demotion and rejection are arithmetic. A loop that lets an LLM edit its
own decision rules based on its own scoring of its own output has no fixed point worth
reaching.

WHY SUPPRESSION IS IMPOSSIBLE HERE, structurally rather than by threshold: the overlay is
ADDITIVE ONLY (`scope_registry.apply_overlay` can extend a tuple, never shrink one), so no
update this module can produce is capable of moving a turn from MINI/FULL to NONE. The
degenerate strategy -- learn to say NONE about everything, drive ceremony and rows to zero,
and report a clean sheet - is unreachable, not merely discouraged. The tier-share check
below is defence in depth for the day someone allows removals.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from flightdeck.models import TuningChange
from flightdeck.scope_gate import NONE, classify
from flightdeck.scope_registry import ACTIVE, RETIRED, Entry, load, save
from flightdeck.store import Store

#: An update may not raise the share of turns that get no scoping by more than this.
MAX_NONE_SHARE_INCREASE = 0.10

#: Candidate phrases are drawn from the offending prompt. Single words are too blunt to
#: target and long spans never recur, so mining stays inside this window.
NGRAM_MIN, NGRAM_MAX = 2, 3

#: Which registry a candidate is proposed for. Only these are learnable: they are the
#: ambiguity signals. BUILD_VERBS and the question/approval registries decide what is NOT
#: a task, and widening those SUPPRESSES scoping, which this module must never do.
LEARNABLE = ("VAGUE_MARKERS", "HEAVY_VERBS", "MULTI_SURFACE", "DISCOVERY_VERBS")


@dataclass
class Turn:
    session: str
    index: int
    text: str
    corrected_next: bool = False


@dataclass
class Event:
    """One replayed decision, with what happened on the turn after it."""

    turn: Turn
    tier: str
    had_active_scope: bool
    false_none: bool = False


@dataclass
class Candidate:
    phrase: str
    registry: str
    helpful: int = 0
    harmful: int = 0
    evidence: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.helpful + self.harmful

    @property
    def ratio(self) -> float | None:
        return self.helpful / self.total if self.total else None


def session_turns(root: Path | str) -> Iterator[list[Turn]]:
    """Human turns per session, in order, with the next-turn correction flag attached."""
    from flightdeck.scope_baseline import (
        classify_correction,
        human_prompt_text,
        is_human_prompt,
    )

    base = Path(root).expanduser()
    for path in sorted(base.glob("*/*.jsonl")):
        texts: list[str] = []
        try:
            lines = path.read_text().splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if record.get("type") != "user" or not is_human_prompt(record):
                continue
            text = human_prompt_text(record) or ""
            if text.strip():
                texts.append(text)
        if not texts:
            continue
        turns = [Turn(session=path.stem, index=i, text=t) for i, t in enumerate(texts)]
        for i, turn in enumerate(turns[:-1]):
            turn.corrected_next = bool(classify_correction(turns[i + 1].text))
        yield turns


def replay(turns: Sequence[Turn], classifier: Callable[..., Any] = classify) -> list[Event]:
    """Re-run the gate over a session exactly as the hook would have.

    A scope opens on the first non-NONE decision and stays open for the rest of the
    session, which is how `ScopePass` behaves inside its freshness window. Sessions longer
    than that window are replayed as one, which slightly UNDER-counts task starts -- stated
    because it biases the mined evidence toward fewer candidates, not more.
    """
    events: list[Event] = []
    active = False
    for turn in turns:
        decision = classifier(turn.text, has_active_scope=active)
        event = Event(turn=turn, tier=decision.tier, had_active_scope=active)
        event.false_none = (
            decision.tier == NONE and not active and turn.corrected_next
        )
        events.append(event)
        if decision.tier != NONE:
            active = True
    return events


def _ngrams(text: str) -> set[str]:
    words = [w for w in text.lower().split() if w.isalpha() or "-" in w]
    out = set()
    for size in range(NGRAM_MIN, NGRAM_MAX + 1):
        for i in range(len(words) - size + 1):
            out.add(" ".join(words[i : i + size]))
    return out


def mine(events: Sequence[Event], *, registry: str = "VAGUE_MARKERS") -> list[Candidate]:
    """Propose phrases from the turns the gate let through and should not have."""
    from flightdeck.scope_gate import DISCOVERY_VERBS, HEAVY_VERBS, MULTI_SURFACE, VAGUE_MARKERS

    known = set(HEAVY_VERBS) | set(VAGUE_MARKERS) | set(MULTI_SURFACE) | set(DISCOVERY_VERBS)
    counts: dict[str, Candidate] = {}
    for event in events:
        if not event.false_none:
            continue
        for phrase in _ngrams(event.turn.text):
            if phrase in known or any(k in phrase for k in known):
                continue
            candidate = counts.setdefault(phrase, Candidate(phrase=phrase, registry=registry))
            candidate.evidence.append(f"{event.turn.session}#{event.turn.index}")
    return sorted(counts.values(), key=lambda c: -len(c.evidence))


def score(candidate: Candidate, events: Sequence[Event]) -> Candidate:
    """Counterfactual replay: what would this phrase have changed?

    helpful -- a turn the gate wrongly let through would now be scoped.
    harmful -- a turn the gate rightly let through would now cost ceremony.
    Both are counted over the SAME window; a candidate is never scored on its own evidence
    alone, which is how a phrase that fires everywhere looks good until it ships.
    """
    candidate.helpful = candidate.harmful = 0
    needle = candidate.phrase
    for event in events:
        if event.tier != NONE or event.had_active_scope:
            continue
        if needle not in event.turn.text.lower():
            continue
        if event.turn.corrected_next:
            candidate.helpful += 1
        else:
            candidate.harmful += 1
    return candidate


def none_share(events: Sequence[Event]) -> float:
    return sum(1 for e in events if e.tier == NONE) / len(events) if events else 0.0


def suppression_check(before: Sequence[Event], after: Sequence[Event]) -> dict[str, Any]:
    """Runs BEFORE any quality arithmetic.

    A quality figure computed over a window the update emptied is a small-sample artifact,
    not a result -- so the regime is checked first and the rate is never consulted if the
    regime moved. Rejecting here costs a good update occasionally; not rejecting costs the
    ability to notice the system has stopped measuring itself.
    """
    was, now = none_share(before), none_share(after)
    delta = now - was
    return {
        "none_share_before": round(was, 4),
        "none_share_after": round(now, 4),
        "delta": round(delta, 4),
        "limit": MAX_NONE_SHARE_INCREASE,
        "passed": delta <= MAX_NONE_SHARE_INCREASE,
    }


def propose(
    root: Path | str, *, registry: str = "VAGUE_MARKERS", limit: int = 10
) -> dict[str, Any]:
    """Mine, score and rank candidates. Reads only; writes nothing."""
    events: list[Event] = []
    sessions = 0
    for turns in session_turns(root):
        sessions += 1
        events.extend(replay(turns))
    candidates = [score(c, events) for c in mine(events, registry=registry)]
    promotable = [c for c in candidates if c.helpful >= 1]
    promotable.sort(key=lambda c: (-(c.ratio or 0), -c.helpful))
    from flightdeck.scope_registry import PROMOTE_MIN_HELPFUL, PROMOTE_MIN_RATIO

    clearing = [
        c for c in promotable
        if c.helpful >= PROMOTE_MIN_HELPFUL and (c.ratio or 0) >= PROMOTE_MIN_RATIO
    ]
    return {
        "sessions": sessions,
        "turns": len(events),
        "false_none_events": sum(1 for e in events if e.false_none),
        "false_none_rate": round(
            sum(1 for e in events if e.false_none)
            / max(1, sum(1 for e in events if e.tier == NONE and not e.had_active_scope)),
            4,
        ),
        "candidates": len(candidates),
        "clearing_threshold": [
            {"phrase": c.phrase, "helpful": c.helpful, "harmful": c.harmful,
             "ratio": round(c.ratio or 0, 3), "evidence": c.evidence[:4]}
            for c in clearing[:limit]
        ],
        "best_below_threshold": [
            {"phrase": c.phrase, "helpful": c.helpful, "harmful": c.harmful,
             "ratio": round(c.ratio or 0, 3)}
            for c in promotable[:limit] if c not in clearing
        ],
        "verdict": "pass" if clearing else "silent",
        "reason": (
            f"{len(clearing)} candidates cleared the bar"
            if clearing
            else "no candidate cleared the promotion bar; shipping nothing is the "
                 "correct outcome most weeks at this arrival rate"
        ),
    }


def apply(
    root: Path | str,
    *,
    registry: str = "VAGUE_MARKERS",
    store: Store | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Promote what cleared the bar, demote what has stopped earning its place.

    Every accepted change writes a TuningChange row -- the schema for exactly this was
    already in the database and had no writer, so a registry that drifted left no trail.
    """
    proposal = propose(root, registry=registry)
    if not proposal["clearing_threshold"]:
        proposal["applied"] = False
        return proposal

    events: list[Event] = []
    for turns in session_turns(root):
        events.extend(replay(turns))

    entries = load()
    by_phrase = {(e.registry, e.phrase): e for e in entries}
    promoted: list[str] = []
    for row in proposal["clearing_threshold"]:
        key = (registry, row["phrase"])
        entry = by_phrase.get(key) or Entry(phrase=row["phrase"], registry=registry)
        entry.helpful, entry.harmful = row["helpful"], row["harmful"]
        entry.evidence = row["evidence"]
        entry.status = ACTIVE
        by_phrase[key] = entry
        promoted.append(row["phrase"])

    demoted = [e.phrase for e in by_phrase.values() if e.demotable and e.status == ACTIVE]
    for entry in by_phrase.values():
        if entry.demotable:
            entry.status = RETIRED

    # Regime check before any quality arithmetic, per the module docstring.
    extra = tuple(sorted(p for (r, p), e in by_phrase.items()
                         if r == registry and e.status == ACTIVE))
    after = _replay_with(root, registry, extra)
    check = suppression_check(events, after)
    proposal["suppression_check"] = check
    if not check["passed"]:
        proposal["applied"] = False
        proposal["verdict"] = "fail"
        proposal["reason"] = (
            f"rejected: would raise the un-scoped share by {check['delta']:.2%}, "
            f"over the {MAX_NONE_SHARE_INCREASE:.0%} limit"
        )
        return proposal

    proposal["promoted"] = promoted
    proposal["demoted"] = demoted
    if dry_run:
        proposal["applied"] = False
        proposal["reason"] = "dry run; nothing written"
        return proposal

    save(list(by_phrase.values()))
    if store is not None:
        store.add_tuning_change(
            TuningChange(
                change_id=f"scope-registry-{int(time.time())}",
                applied_at=int(time.time()),
                domain="scope_gate",
                path="flightdeck/scope_registry.json",
                summary=f"promoted {len(promoted)}, retired {len(demoted)} in {registry}",
                evidence={"promoted": promoted, "demoted": demoted,
                          "suppression_check": check,
                          "false_none_rate": proposal["false_none_rate"]},
                kpi_before=check["none_share_before"],
                kpi_after=check["none_share_after"],
            )
        )
    proposal["applied"] = True
    return proposal


def _replay_with(root: Path | str, registry: str, extra: tuple[str, ...]) -> list[Event]:
    """Replay the corpus as if `extra` were already promoted into `registry`."""
    import flightdeck.scope_gate as gate

    original = getattr(gate, registry)
    setattr(gate, registry, tuple(sorted(set(original) | set(extra))))
    try:
        events: list[Event] = []
        for turns in session_turns(root):
            events.extend(replay(turns))
        return events
    finally:
        setattr(gate, registry, original)
