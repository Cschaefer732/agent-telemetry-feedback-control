"""Nightly review: sampling, rollup, guardrails, and regression revert.

This module contains no model calls and makes no decisions about what to change. It decides what
the reviewer is allowed to look at and what it is allowed to touch; the judgment happens in the
`nightly-review` skill run by real Claude Code on the review host.

The split matters. The reviewer writes to `main` unattended, so the boundary on its authority has
to be enforced by code that is tested, not by an instruction in a prompt that a model may
reinterpret at 3am.
"""

from __future__ import annotations

import fnmatch
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from flightdeck.models import Event, Probe, Turn
from flightdeck.store import Store, hostname, now_ms

DAY_MS = 24 * 60 * 60 * 1000

# Paths the reviewer may write. Anything not matched here is denied, so adding a capability is an
# explicit act rather than an oversight. These are AN EXAMPLE layout (a config repo vendoring a
# "crush-fork" checkout beside Claude Code skills/rules) -- edit for your own repo.
WRITABLE_PATTERNS = (
    "crush-fork/skills/**/*.md",
    "crush-fork/agents/*.md",
    "crush-fork/modes/*.md",
    "crush-fork/CRUSH.md",
    "crush-fork/MEMORY.md",
    "crush-fork/governor.toml",
    "flightdeck/governor.toml",
    "claude/skills/**/*.md",
    "claude/rules/*.md",
    "docs/reviews/*.md",
)

# Denied even if a writable pattern would otherwise match. The reviewer cannot rebuild and
# cross-compile the crush fork at 3am, so it must not be able to break the build. Everything here
# is either compiled, executed at boot, or controls what the reviewer itself may do.
DENIED_PATTERNS = (
    "**/*.go",
    "**/*.patch",
    "**/build.sh",
    "**/install.sh",
    "**/*.service",
    "**/*.timer",
    "**/crush.json",
    "**/settings.json",
    ".git/**",
    "**/.git/**",
    "flightdeck/review.py",
    "flightdeck/*.py",
)


@dataclass
class Sample:
    reason: str
    turns: list[Turn]
    dropped: int = 0
    note: str = ""


@dataclass
class SweepSignal:
    signal: str
    session_id: str
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str]:
        return (self.signal, self.session_id)


def complexity(turn: Turn, events: list[Event]) -> float:
    """How much was going on in this turn.

    Multiplicative rather than additive: a turn with many tools AND many files AND many tokens is
    qualitatively harder than one that merely scores high on a single axis, and those are the turns
    where a mis-tuned system costs the most.
    """
    tools = len({e.name for e in events if e.kind == "tool_call" and e.name})
    files = len({e.payload.get("path") for e in events if e.kind in ("edit", "revert")} - {None})
    tokens = max(1, turn.total_tokens)
    return (1 + tools) * (1 + files) * (tokens / 1000.0)


class Sampler:
    def __init__(self, store: Store, *, window_ms: int = DAY_MS, now: int | None = None) -> None:
        self.store = store
        self.now = now or now_ms()
        self.since = self.now - window_ms

    def _turns(self) -> list[Turn]:
        return list(self.store.iter_turns(since_ms=self.since, until_ms=self.now))

    def complex_turns(self, limit: int = 10) -> Sample:
        scored = []
        for turn in self._turns():
            events = self.store.events_for(turn.turn_id)
            scored.append((complexity(turn, events), turn))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        chosen = [turn for _, turn in scored[:limit]]
        return Sample(
            reason="complexity",
            turns=chosen,
            dropped=max(0, len(scored) - limit),
            note=f"top {len(chosen)} of {len(scored)} by complexity",
        )

    def low_kpi_turns(self, limit: int = 15, *, skip_reviewed: bool = True) -> Sample:
        candidates = [t for t in self._turns() if t.kpi_score is not None]
        if skip_reviewed:
            candidates = [t for t in candidates if not self.store.judgment_for(t.turn_id)]
        candidates.sort(key=lambda t: t.kpi_score or 0.0)
        chosen = candidates[:limit]
        return Sample(
            reason="low_kpi",
            turns=chosen,
            dropped=max(0, len(candidates) - limit),
            note=f"bottom {len(chosen)} of {len(candidates)} by KPI",
        )

    def sweep(self) -> list[SweepSignal]:
        """Cheap scan of every session for problem signals.

        Samples A and B look at the extremes. This looks at everything, because the failures worth
        catching are usually not the slowest or the lowest-scoring — they are the quiet ones.
        """
        signals: list[SweepSignal] = []
        by_session: dict[str, list[Turn]] = defaultdict(list)
        for turn in self._turns():
            by_session[turn.session_id].append(turn)

        for session_id, turns in by_session.items():
            tool_failures: Counter[str] = Counter()
            denials = 0
            hook_errors: Counter[str] = Counter()
            reverted_paths: Counter[str] = Counter()
            interrupts = 0

            for turn in turns:
                for event in self.store.events_for(turn.turn_id):
                    if event.kind == "tool_call" and event.ok == 0 and event.name:
                        tool_failures[event.name] += 1
                    elif event.kind == "permission" and event.payload.get("decision") == "deny":
                        denials += 1
                    elif event.kind == "hook" and event.ok == 0 and event.name:
                        hook_errors[event.name] += 1
                    elif event.kind == "revert":
                        path = event.payload.get("path")
                        if path:
                            reverted_paths[path] += 1
                    elif event.kind == "interrupt":
                        interrupts += 1

            for tool, count in tool_failures.items():
                if count >= 3:
                    signals.append(
                        SweepSignal(
                            "repeated_tool_failure", session_id, {"tool": tool, "count": count}
                        )
                    )
            if denials >= 2:
                signals.append(SweepSignal("permission_denials", session_id, {"count": denials}))
            for hook, count in hook_errors.items():
                signals.append(
                    SweepSignal("hook_error", session_id, {"hook": hook, "count": count})
                )
            for path, count in reverted_paths.items():
                if count >= 2:
                    signals.append(
                        SweepSignal("edit_revert_loop", session_id, {"path": path, "count": count})
                    )
            if interrupts >= 2:
                signals.append(
                    SweepSignal("repeated_interrupts", session_id, {"count": interrupts})
                )

        signals.extend(self._probe_signals())
        return signals

    def _probe_signals(self) -> list[SweepSignal]:
        """Failed probes are session-less findings, but they belong in the same escalation path.

        A collector that produced nothing is the single most important thing this review can find:
        it means every other number in the report is understated, and silence is what let the last
        dead hook sit unnoticed for weeks.
        """
        signals: list[SweepSignal] = []
        rows = self.store.conn.execute(
            "SELECT ts, host, kind, ok, total, detail FROM probes WHERE ts >= ? ORDER BY ts",
            (self.since,),
        ).fetchall()
        latest: dict[tuple[str, str], Probe] = {}
        for row in rows:
            probe = Probe(
                ts=row["ts"],
                host=row["host"],
                kind=row["kind"],
                ok=row["ok"],
                total=row["total"],
                detail=json.loads(row["detail"]) if row["detail"] else {},
            )
            latest[(probe.host, probe.kind)] = probe
        for (host, kind), probe in sorted(latest.items()):
            if not probe.healthy:
                signals.append(
                    SweepSignal(
                        f"probe_{kind}",
                        f"host:{host}",
                        {"ok": probe.ok, "total": probe.total, **probe.detail},
                    )
                )
        if not latest:
            signals.append(
                SweepSignal("probe_never_ran", f"host:{hostname()}", {"window_start": self.since})
            )
        return signals


class Guardrails:
    """Path authority for the nightly reviewer.

    Deny wins over allow, always. A pattern list that can be satisfied by adding an allow entry
    would let the reviewer widen its own authority, which is precisely the failure mode that makes
    unattended writes to main dangerous.
    """

    def __init__(
        self,
        repo_root: Path,
        *,
        writable: tuple[str, ...] = WRITABLE_PATTERNS,
        denied: tuple[str, ...] = DENIED_PATTERNS,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.writable = writable
        self.denied = denied

    def _relative(self, path: Path | str) -> str | None:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.repo_root / candidate
        # Resolve without requiring existence: the reviewer may create a file that is not there
        # yet, and a not-yet-existing path must still be checked, not waved through.
        resolved = Path(candidate).resolve()
        try:
            return str(resolved.relative_to(self.repo_root))
        except ValueError:
            # Outside the repo entirely — including anything reached via `..` or a symlink that
            # escapes the tree.
            return None

    def is_writable(self, path: Path | str) -> tuple[bool, str]:
        rel = self._relative(path)
        if rel is None:
            return False, "outside_repo"
        for pattern in self.denied:
            if _matches(rel, pattern):
                return False, f"denied:{pattern}"
        for pattern in self.writable:
            if _matches(rel, pattern):
                return True, f"allowed:{pattern}"
        return False, "not_in_allowlist"

    def check(self, paths: list[Path | str]) -> dict[str, str]:
        """Return only the rejections, keyed by path. An empty dict means the change may proceed."""
        rejected: dict[str, str] = {}
        for path in paths:
            allowed, reason = self.is_writable(path)
            if not allowed:
                rejected[str(path)] = reason
        return rejected


def _matches(rel_path: str, pattern: str) -> bool:
    # fnmatch does not treat `**` as crossing directory separators, so `a/**/b.md` misses `a/b.md`.
    # Expanding to both forms is simpler and more predictable than a bespoke matcher.
    if fnmatch.fnmatch(rel_path, pattern):
        return True
    collapsed = pattern.replace("/**/", "/")
    return collapsed != pattern and fnmatch.fnmatch(rel_path, collapsed)


@dataclass
class RegressionVerdict:
    change_id: str
    domain: str
    commit_sha: str | None
    before: float
    after: float
    delta_pct: float
    n_before: int
    n_after: int
    should_revert: bool
    reason: str


def regression_check(
    store: Store,
    *,
    now: int | None = None,
    window_days: int = 7,
    min_age_days: int = 2,
    max_age_days: int = 14,
    min_samples: int = 30,
    drop_threshold: float = 0.05,
) -> list[RegressionVerdict]:
    """Find tuning changes that made things worse.

    Only changes between min_age and max_age days old are judged: younger than that and there is
    not enough post-change evidence, older and the world has moved on enough that attributing a
    drift to one commit is guesswork.
    """
    now = now or now_ms()
    verdicts: list[RegressionVerdict] = []
    rows = store.conn.execute(
        "SELECT * FROM tuning_changes WHERE reverted_at IS NULL AND applied_at BETWEEN ? AND ?",
        (now - max_age_days * DAY_MS, now - min_age_days * DAY_MS),
    ).fetchall()

    for row in rows:
        applied = row["applied_at"]
        before = _mean_kpi(store, applied - window_days * DAY_MS, applied)
        after = _mean_kpi(store, applied, min(now, applied + window_days * DAY_MS))
        if before[1] < min_samples or after[1] < min_samples:
            verdicts.append(
                RegressionVerdict(
                    change_id=row["change_id"],
                    domain=row["domain"],
                    commit_sha=row["commit_sha"],
                    before=before[0],
                    after=after[0],
                    delta_pct=0.0,
                    n_before=before[1],
                    n_after=after[1],
                    should_revert=False,
                    reason="insufficient_samples",
                )
            )
            continue
        delta = (after[0] - before[0]) / before[0] if before[0] else 0.0
        regressed = delta < -drop_threshold
        verdicts.append(
            RegressionVerdict(
                change_id=row["change_id"],
                domain=row["domain"],
                commit_sha=row["commit_sha"],
                before=before[0],
                after=after[0],
                delta_pct=delta * 100.0,
                n_before=before[1],
                n_after=after[1],
                should_revert=regressed,
                reason="kpi_regression" if regressed else "within_tolerance",
            )
        )
    return verdicts


def _mean_kpi(store: Store, since: int, until: int) -> tuple[float, int]:
    row = store.conn.execute(
        "SELECT AVG(kpi_score) AS mean, COUNT(kpi_score) AS n FROM turns "
        "WHERE kpi_score IS NOT NULL AND started_at >= ? AND started_at < ?",
        (since, until),
    ).fetchone()
    return (row["mean"] or 0.0, row["n"] or 0)


def aggregates(store: Store, *, since_ms: int, until_ms: int | None = None) -> dict[str, Any]:
    """The numbers the reviewer reads before it reads any individual turn."""
    until = until_ms or now_ms()
    turns = list(store.iter_turns(since_ms=since_ms, until_ms=until))
    if not turns:
        # An empty window is itself the finding. Returning zeros without saying so would let a
        # dead collector read as a quiet night.
        return {"turns": 0, "empty_window": True, "since": since_ms, "until": until}

    by_tier: dict[str, list[float]] = defaultdict(list)
    by_source: dict[str, list[float]] = defaultdict(list)
    by_host: dict[str, list[float]] = defaultdict(list)
    tool_failures: Counter[str] = Counter()
    tool_totals: Counter[str] = Counter()
    compactions = 0
    flagged = 0
    judged = 0
    tokens = 0

    for turn in turns:
        if turn.kpi_score is not None:
            by_tier[turn.tier or "unknown"].append(turn.kpi_score)
            by_source[turn.source].append(turn.kpi_score)
            by_host[turn.host].append(turn.kpi_score)
        flagged += turn.flagged
        judged += turn.judged
        tokens += turn.total_tokens
        for event in store.events_for(turn.turn_id):
            if event.kind == "tool_call" and event.name:
                tool_totals[event.name] += 1
                if event.ok == 0:
                    tool_failures[event.name] += 1
            elif event.kind == "compaction" and event.name != "context_snapshot":
                compactions += 1

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    return {
        "turns": len(turns),
        "empty_window": False,
        "since": since_ms,
        "until": until,
        "total_tokens": tokens,
        "flagged": flagged,
        "judged": judged,
        "unjudged_flagged": flagged - judged,
        "compactions": compactions,
        "kpi_by_tier": {k: round(mean(v), 4) for k, v in sorted(by_tier.items())},
        "kpi_by_source": {k: round(mean(v), 4) for k, v in sorted(by_source.items())},
        "kpi_by_host": {k: round(mean(v), 4) for k, v in sorted(by_host.items())},
        "tool_failure_leaderboard": [
            {"tool": tool, "failures": count, "calls": tool_totals[tool]}
            for tool, count in tool_failures.most_common(10)
        ],
    }
