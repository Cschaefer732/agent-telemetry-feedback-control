"""Record types shared by every component.

These mirror the schema exactly. Serialization is explicit rather than reflective so a schema
change is a compile-time-ish failure in tests instead of a silently dropped column.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any

SOURCES = ("crush", "claude-code", "arch-loop")

TIERS = ("fast", "balanced", "deep", "frontier")

OUTCOMES = ("ok", "error", "cancelled", "interrupted")

EVENT_KINDS = (
    "tool_call",
    "model_req",
    "compaction",
    # Distinct from "compaction": a snapshot is a periodic reading of how full the window is, and
    # counting one as a compaction event would penalize a turn for being observed.
    "context_snapshot",
    "mode_change",
    "skill_load",
    "skill_use",
    "mcp_call",
    "lsp_event",
    "hook",
    # bridge-emitted training telemetry (2026-08-23): per-hook execution, measured
    # injected-context size, and per-turn process/plugin provenance
    "hook_run",
    "context_inject",
    "provenance",
    # a web-editor save (POST /file/write, sparky release 2026.08.29-1051) logged by the bridge
    # beside the agent's own edits: payload {path, bytes, sha256, source}
    "file_write",
    # the write-policy's refusals (sparky release 2026.08.30-0353): payload {path, reason, source}.
    # Recorded because a denial leaves no file and no tool call -- without it the audit trail
    # shows only the saves that succeeded.
    "file_write_denied",
    "permission",
    "recall",
    "critic",
    "queue",
    "todo",
    "edit",
    "revert",
    "interrupt",
    "worktree",
    "delegate",
)

TEXT_KINDS = (
    "prompt",
    "response",
    "context_snapshot",
    "summary",
    "tool_arg",
    "tool_result",
)

PROBE_KINDS = ("symlink", "migration", "hook_liveness", "collector_heartbeat", "judge_queue")

GOVERNOR_DOMAINS = ("model_tier", "skills", "compaction", "mode_delegation")

# --- scoping subsystem ---------------------------------------------------------------
# One definition, read by writer AND aggregator. The dreamer ran for weeks promoting on
# outcome == "succeeded" while the writer emitted ok|error|cancelled|unknown|truncated:
# 183 rows in, zero out, timer green the whole time. Any module that filters scope rows
# imports these tuples; test_scope asserts every such filter is a subset of them.
SCOPE_TIERS = ("none", "mini", "full")

# Three-valued on purpose. A scoping pass that never ran is SILENT, not PASS -- a binary
# verdict scores a dead writer as success, which is how the capture layer stayed unwired.
SCOPE_VERDICTS = ("pass", "fail", "silent")

# Where a discovered item landed. Every FOUND item must reach exactly one of these:
# discovery is unbounded and free, commitment is budgeted and loud.
DISPOSITIONS = ("committed", "non_goal", "assumption")

# Correction families, tagged separately because precision differs per family and
# `missed` is the scope-specific one -- `negate`/`redirect` are often taste iteration.
CORRECTION_FAMILIES = ("negate", "missed", "repeat", "redirect")

#: The verdicts an aggregator may treat as healthy. Declared HERE, beside the full set,
#: so a filter can never key on a string the writer does not emit -- the dreamer promoted
#: on "succeeded" against a writer emitting ok|error|... and shipped 183 rows into a void.
HEALTHY_SCOPE_VERDICTS = ("pass",)


@dataclass
class Turn:
    turn_id: str
    session_id: str
    source: str
    host: str
    started_at: int
    parent_session_id: str | None = None
    cwd: str | None = None
    git_sha: str | None = None
    turn_idx: int | None = None
    ended_at: int | None = None
    wall_ms: int | None = None
    agent_name: str | None = None
    is_subagent: int = 0
    mode: str | None = None
    provider: str | None = None
    model: str | None = None
    tier: str | None = None
    model_ms: int | None = None
    requests: int | None = None
    retries: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    estimated: int = 0
    context_peak: int | None = None
    context_window: int | None = None
    ttft_ms: int | None = None
    tools_hash_changes: int | None = None
    outcome: str | None = None
    finish_reason: str | None = None
    error_class: str | None = None
    kpi_score: float | None = None
    kpi_components: str | None = None
    flagged: int = 0
    judged: int = 0

    @property
    def total_tokens(self) -> int:
        return (self.prompt_tokens or 0) + (self.completion_tokens or 0)

    @property
    def context_occupancy(self) -> float | None:
        if not self.context_window or self.context_peak is None:
            return None
        return min(1.0, self.context_peak / self.context_window)

    def to_row(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> Turn:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in row.items() if k in known})


@dataclass
class ScopeRecord:
    """One scoping pass over one task. Field groups map 1:1 to the KPI pairs -- each
    quality counter sits beside the cost counter that stops it being gamed."""

    record_id: str
    session_id: str
    created_at: int
    host: str
    tier: str
    verdict: str
    turn_id: str | None = None
    cwd: str | None = None
    git_sha: str | None = None
    # coverage (KPI 1) vs ceremony cost (KPI 2)
    found_total: int = 0
    committed: int = 0
    non_goals: int = 0
    assumptions: int = 0
    late_discovered: int = 0
    ceremony_ms: int | None = None
    ceremony_tokens: int | None = None
    # clarification value (KPI 5) vs question count (KPI 6)
    questions_asked: int = 0
    questions_valuable: int = 0
    # rework (KPI 3) vs turns (KPI 4)
    files_edited: int = 0
    rework_files: int = 0
    turns_to_first_edit: int | None = None
    turns_to_done: int | None = None
    # human signal (KPI 7, 8)
    corrections: int = 0
    correction_families: dict[str, int] = field(default_factory=dict)
    assumptions_overridden: int = 0
    # divergence generator yield (KPI 9)
    divergence_flagged: int = 0
    divergence_kept: int = 0
    # provenance: never compare rows across a key change
    provenance: dict[str, Any] = field(default_factory=dict)
    prev_hash: str | None = None
    row_hash: str | None = None

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["correction_families"] = json.dumps(self.correction_families, separators=(",", ":"))
        row["provenance"] = json.dumps(self.provenance, separators=(",", ":"))
        return row


@dataclass
class Event:
    turn_id: str
    ts: int
    kind: str
    name: str | None = None
    duration_ms: int | None = None
    ok: int | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["payload"] = json.dumps(self.payload, separators=(",", ":")) if self.payload else None
        return row

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> Event:
        payload = row.get("payload")
        return cls(
            turn_id=row["turn_id"],
            ts=row["ts"],
            kind=row["kind"],
            name=row.get("name"),
            duration_ms=row.get("duration_ms"),
            ok=row.get("ok"),
            payload=json.loads(payload) if payload else {},
        )


@dataclass
class TextBlob:
    turn_id: str
    kind: str
    seq: int
    body: str
    expires_at: int

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Judgment:
    turn_id: str
    judge_model: str
    verdict: str
    created_at: int
    rubric: dict[str, Any] = field(default_factory=dict)
    notes: str | None = None
    lesson: str | None = None

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["rubric"] = json.dumps(self.rubric, separators=(",", ":")) if self.rubric else None
        return row


@dataclass
class GovernorDecision:
    turn_id: str
    domain: str
    chosen: str
    shadow: int
    alternatives: dict[str, float] = field(default_factory=dict)
    features: dict[str, Any] = field(default_factory=dict)
    weights_version: str | None = None
    # Propensity. `alternatives` records what each arm scored; these record how the pick was
    # actually drawn, which is the part an off-policy estimator needs and cannot infer from
    # scores alone: an argmax arm and an explored arm can be the same arm at different P(a|x).
    explored: int = 0
    reason: str = ""
    epsilon: float | None = None

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["alternatives"] = json.dumps(self.alternatives, separators=(",", ":"))
        row["features"] = json.dumps(self.features, separators=(",", ":"))
        return row


@dataclass
class Probe:
    ts: int
    host: str
    kind: str
    ok: int
    total: int
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def healthy(self) -> bool:
        return self.total > 0 and self.ok == self.total

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["detail"] = json.dumps(self.detail, separators=(",", ":")) if self.detail else None
        return row


@dataclass
class TuningChange:
    change_id: str
    applied_at: int
    domain: str
    path: str
    summary: str
    commit_sha: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    kpi_before: float | None = None
    kpi_after: float | None = None
    reverted_at: int | None = None
    revert_reason: str | None = None

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        evidence = self.evidence
        row["evidence"] = json.dumps(evidence, separators=(",", ":")) if evidence else None
        return row
