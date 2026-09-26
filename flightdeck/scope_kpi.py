"""Aggregation over scope_records, and the checks that stop a dead pipeline reading green."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from flightdeck.models import HEALTHY_SCOPE_VERDICTS, SCOPE_TIERS

#: A counter that has been zero for this many consecutive windows is a FAILING check, not
#: a quiet one. The dreamer logged "playbooks": 0 nightly for weeks and exited 0 each time.
ZERO_STREAK_LIMIT = 3

#: Below this many OBSERVED rows a rate is noise dressed as a number. Rates are still
#: reported -- hiding them would lose the only signal a small window has -- but they are
#: named in `sample_warning` so nothing downstream reads them as settled.
MIN_ROWS_FOR_RATE = 20


def _rate(numerator: float, denominator: float) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def aggregate(rows: Sequence[Any]) -> dict[str, Any]:
    """Roll scope_records into the KPI pairs. Every quality figure is returned beside the
    cost figure that bounds it -- reporting late_discovered without ceremony_tokens invites
    fixing coverage by scoping forever."""
    rows = list(rows)
    if not rows:
        return {"rows": 0, "verdict": "silent", "reason": "no scope_records in window"}

    # A thin row ingested from the gate log carries verdict "silent": the gate fired and
    # nothing observed whether the scoping that followed was any good. Those rows are not
    # evidence of health OR of failure, so they are excluded from the verdict and counted
    # separately -- a window of nothing but unobserved decisions is "silent", not "fail".
    observed = [r for r in rows if r["verdict"] != "silent"]
    healthy = [r for r in observed if r["verdict"] in HEALTHY_SCOPE_VERDICTS]
    failed = [r for r in observed if r["verdict"] not in HEALTHY_SCOPE_VERDICTS]
    found = sum(r["found_total"] for r in rows)
    late = sum(r["late_discovered"] for r in rows)
    files = sum(r["files_edited"] for r in rows)
    rework = sum(r["rework_files"] for r in rows)
    asked = sum(r["questions_asked"] for r in rows)
    valuable = sum(r["questions_valuable"] for r in rows)
    ceremony = [r["ceremony_tokens"] for r in rows if r["ceremony_tokens"] is not None]
    ceremony_ms = [r["ceremony_ms"] for r in rows if r["ceremony_ms"] is not None]

    by_tier: dict[str, int] = {t: 0 for t in SCOPE_TIERS}
    for row in rows:
        if row["tier"] in by_tier:
            by_tier[row["tier"]] += 1

    # The old rule read "pass" whenever ANY row was healthy: one pass among ninety-nine
    # failures reported pass. A recorded failure is a fact rather than an estimate, so it
    # is never averaged away and never suppressed by a small-sample rule.
    if not observed:
        verdict = "silent"
    elif failed:
        verdict = "fail"
    else:
        verdict = "pass"

    return {
        "rows": len(rows),
        "observed_rows": len(observed),
        "unobserved_rows": len(rows) - len(observed),
        "healthy_rows": len(healthy),
        "failed_rows": len(failed),
        "healthy_rate": _rate(len(healthy), len(observed)),
        "verdict": verdict,
        "verdict_reason": (
            f"{len(rows)} gate decisions recorded, none with an observed outcome"
            if not observed
            else f"{len(failed)} of {len(observed)} observed rows failed"
        ),
        "sample_warning": (
            None
            if len(observed) >= MIN_ROWS_FOR_RATE
            else f"rates computed on {len(observed)} observed rows, "
            f"below the {MIN_ROWS_FOR_RATE}-row floor"
        ),
        "tiers": by_tier,
        # Tiers cost and cover wildly different amounts, so a pooled rate over a window
        # whose tier mix moved is comparing two different populations and calling the
        # difference a trend.
        "by_tier": _per_tier(rows),
        # KPI 1 vs 2 -- coverage against what it cost
        "late_discovery_rate": _rate(late, found),
        "ceremony_tokens_mean": round(sum(ceremony) / len(ceremony)) if ceremony else None,
        "ceremony_ms_mean": round(sum(ceremony_ms) / len(ceremony_ms)) if ceremony_ms else None,
        # A coverage figure with no cost figure beside it can be improved by scoping
        # forever. Say so in the output rather than letting the pair look complete.
        "unpaired_metrics": _unpaired(late, found, ceremony, ceremony_ms),
        # KPI 3 vs 4
        "rework_rate": _rate(rework, files),
        "turns_to_done_mean": _mean(rows, "turns_to_done"),
        # KPI 5 vs 6
        "clarification_value_rate": _rate(valuable, asked),
        "questions_per_task": _rate(asked, len(rows)),
        # KPI 7, 8
        "corrections_per_task": _rate(sum(r["corrections"] for r in rows), len(rows)),
        "assumption_override_rate": _rate(
            sum(r["assumptions_overridden"] for r in rows),
            sum(r["assumptions"] for r in rows),
        ),
        # KPI 9
        "divergence_precision": _rate(
            sum(r["divergence_kept"] for r in rows),
            sum(r["divergence_flagged"] for r in rows),
        ),
    }


def _per_tier(rows: Sequence[Any]) -> dict[str, dict[str, Any]]:
    """The headline pair, stratified. Pooling hides a tier-mix shift as a quality change."""
    out: dict[str, dict[str, Any]] = {}
    for tier in SCOPE_TIERS:
        group = [r for r in rows if r["tier"] == tier]
        if not group:
            continue
        ceremony = [r["ceremony_tokens"] for r in group if r["ceremony_tokens"] is not None]
        out[tier] = {
            "rows": len(group),
            "late_discovery_rate": _rate(
                sum(r["late_discovered"] for r in group), sum(r["found_total"] for r in group)
            ),
            "ceremony_tokens_mean": (round(sum(ceremony) / len(ceremony)) if ceremony else None),
        }
    return out


def _unpaired(late: int, found: int, ceremony: list[Any], ceremony_ms: list[Any]) -> list[str]:
    """Names every quality metric currently reported without its bounding cost metric."""
    unpaired = []
    if found and not (ceremony or ceremony_ms):
        unpaired.append("late_discovery_rate has no ceremony cost recorded")
    return unpaired


def _mean(rows: Sequence[Any], column: str) -> float | None:
    values = [r[column] for r in rows if r[column] is not None]
    return round(sum(values) / len(values), 2) if values else None


def zero_streak_verdict(history: Sequence[int], limit: int = ZERO_STREAK_LIMIT) -> str:
    """A promotion/aggregation counter stuck at zero is a failing check.

    `history` is the count produced by consecutive runs, oldest first. Returns 'fail' once
    the trailing run of zeroes reaches `limit`, 'silent' when there is no history at all,
    'pass' otherwise. Never returns 'pass' for an all-zero history, which is the shape the
    dreamer's green timer had.
    """
    if not history:
        return "silent"
    streak = 0
    for value in reversed(history):
        if value:
            break
        streak += 1
    return "fail" if streak >= limit else "pass"


def health_verdict(summary: dict[str, Any], *, rows_history: Sequence[int] | None = None) -> str:
    """The actionable verdict for a caller that needs a nonzero exit code -- 'pass' from
    `aggregate()` stays 'pass', 'fail' stays 'fail'. 'silent' is where the real work is:

    aggregate()'s verdict_reason correctly separates "no rows at all" from "rows present,
    none observed", but nothing consumed that distinction as a health signal before this --
    a stream that has been silent for weeks reported exactly the same as a stream that has
    been silent for one quiet Sunday, and both exited 0.

    `rows_history` is `rows` (or `observed_rows`, pick one and be consistent across calls)
    from the caller's own previous windows, oldest first, NOT including the current
    `summary`. Pass it and a streak of `ZERO_STREAK_LIMIT` all-zero windows (this one
    included) turns 'silent' into 'fail' via `zero_streak_verdict` -- the same rule that
    exists for the dreamer's playbook counter, applied here to scope's own row count.

    Without `rows_history` this returns 'silent' for a silent summary, same as today. That
    is deliberate, not a shortcut: a sibling change is landing a verdict-closure path that
    will let 'silent' rows resolve to real pass/fail over time, and at that point a
    permanently-silent stream becomes a real symptom while a single legitimately-quiet
    window must not. This function has to be correct in both worlds, so it never hardcodes
    "silent is fine" -- it only says so when there isn't enough history to say otherwise.
    """
    verdict = summary.get("verdict")
    if verdict != "silent":
        return verdict
    if rows_history is None:
        return "silent"
    streak = zero_streak_verdict([*rows_history, summary.get("rows", 0)])
    return "fail" if streak == "fail" else "silent"


def health_exit_code(verdict: str) -> int:
    """0 for 'pass' and (still-inconclusive) 'silent', 1 for 'fail'. A silent-but-not-yet-
    a-streak result must not fail a nightly job -- that is exactly the false alarm this
    module exists to avoid on the other side of the ledger."""
    return 1 if verdict == "fail" else 0


def dumps(summary: dict[str, Any]) -> str:
    return json.dumps(summary, indent=2, sort_keys=True)
