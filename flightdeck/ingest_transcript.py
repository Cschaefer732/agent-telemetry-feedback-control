"""Backfill claude-code turns from the Claude Code transcript JSONL the hook collector never
reads.

collect_claude.py's module docstring is explicit about why `model_ms`, `prompt_tokens`,
`completion_tokens`, `model`, and (for a Stop the hook never saw) `outcome` are NULL on almost
every claude-code turn: a hook payload genuinely never carries them, and that collector correctly
refuses to guess. The data exists anyway, one layer down, in `~/.claude/projects/<project
slug>/<session-uuid>.jsonl` -- the transcript Claude Code itself writes, which the hooks were
never wired to read. This module is a one-way READ from that transcript and WRITE to `turns`
only; it never touches `events` or `texts`, and it never invents a number the transcript doesn't
state outright.

JOIN KEY, verified empirically before writing a line of attribution logic: a transcript file's
name (`<sessionId>.jsonl`) is the join key, not the `session_id` field carried inside individual
records. On the live corpus (2026-09-05, ~/.claude/projects, 3056 turns / 50 distinct claude-code
session_ids) filename-matching a turn's `turns.session_id` against `sessionId`/filename recovered
2414 of 2415 claude-code turns (99.96%); the one miss was `verify-test-1`, a test fixture id that
was never a real session. The inner `session_id` field is NOT safe to join on: a resumed/continued
session can carry the ORIGINAL session's id in that field on carried-over records while the
filename (and `sessionId`) reflect the session actually running now -- confirmed on a real
"[Request interrupted by user]" record whose `session_id` pointed at a different session than the
file it lived in.

DEDUPE, the second thing that would have silently 2-3x'd every token count: Claude Code splits one
assistant API response into one JSONL row PER CONTENT BLOCK (thinking / tool_use / text), and
EVERY block for that response repeats the SAME `message.usage` object verbatim. Measured on one
real transcript: 535 usage-bearing rows collapsed to 202 unique `message.id`s (up to 11 rows per
id). Summing `usage.input_tokens` per ROW rather than per unique `message.id` overcounts by the
number of content blocks in the response. Every aggregate here is computed per unique
`message.id` ("one request"), never per row.

UNITS: transcript `timestamp` is an ISO-8601 STRING; `turns.started_at`/`ended_at` and every other
timestamp already in this store are epoch MILLISECONDS (see store.now_ms()). `_iso_to_ms` is the
one conversion point; get it wrong and every window comparison silently no-ops (a seconds-scale
turn boundary is ~1000x smaller than a real ms-scale transcript timestamp, so nothing ever falls
inside it) rather than raising, which is exactly the failure mode that motivated writing a test
for it (see test_ingest_transcript.py).

WHY NOT `ingest_cursor`: that table is a byte-offset watermark for OUR OWN JSONL wire format
(`{"_kind": ..., ...}` records with per-table dedupe keys already built into the INSERT), replayed
by `Store.ingest_jsonl`. A Claude Code transcript is a different format entirely (no `_kind`
envelope), and — more fundamentally — a byte offset is the wrong idempotency primitive for what
this module computes. A turn's aggregate is a function of every row in its window, and a turn's
window can still be OPEN (no Stop seen yet, `hi=None`) when this runs; resuming from a byte offset
would mean re-deriving a partial aggregate from a partial re-read, which cannot be done correctly
without re-reading the whole file to start with. Idempotency here comes from a cheaper, stronger
property instead: every write is guarded by "the column is currently NULL" (see `backfill`), so a
turn already backfilled is a no-op on every subsequent run regardless of what's re-read — running
this twice does not just avoid double-counting, it makes zero further writes at all.

WHAT NEVER GETS WRITTEN:
  - `ttft_ms` -- always left alone. A transcript timestamp marks when a full message (or, worse,
    one content block of one) finished arriving, never when the first token did. Writing ttft_ms
    from a full-message timestamp would silently conflate "time to first token" with "time to
    finish generating", which is precisely the prefill/generation distinction ttft_ms exists to
    keep apart (see tiers.py's docstring on cache-warmth-driven tier choice). Not derivable here,
    so left NULL.
  - `outcome='ok'` -- never fabricated for a turn whose Stop was never observed. That rule is
    collect_claude.py's `_close_orphan_turn` (see its docstring, ~line 168): a turn with no Stop
    has an UNKNOWN outcome, not a successful one. This module only ever writes `outcome` when the
    transcript itself states unambiguous negative evidence -- an API error record
    (`isApiErrorMessage: true`) or Claude Code's own literal `"[Request interrupted by user]"`
    marker -- never a guess that the turn probably finished fine because the transcript trails off
    quietly.
"""

from __future__ import annotations

import json
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from flightdeck.models import Turn
from flightdeck.store import Store

DEFAULT_ROOT = Path("~/.claude/projects").expanduser()

# The literal model id Claude Code writes on a synthetic error notice (rate limit, etc.) in place
# of a real model. Its `usage` is all zeros. Folding one of these into normal aggregation would
# attribute a fake "model" and a zero-token "request" to whatever turn it lands in -- it is
# excluded from the usage-group machinery entirely and handled only as an outcome signal.
_SYNTHETIC_MODEL = "<synthetic>"

# Claude Code's own literal text for an Escape-key interrupt (confirmed on real transcripts;
# also listed as noise in scope_baseline.LITERAL_NOISE, independently, for a different purpose).
# There is no other transcript shape for "the human hit interrupt" to derive outcome from.
_INTERRUPT_MARKERS = (
    "[Request interrupted by user]",
    "[Request interrupted by user for tool use]",
)

# The largest gap between two transcript rows that is plausibly ONE model round-trip, not a
# human stepping away. Measured across the whole live corpus (2026-09-05, 23,133 per-request
# gap samples over every session): p99.9 is 209,735ms, p99.99 is 603,593ms, and exactly ONE
# sample in the entire corpus exceeds 15 minutes at all -- 63,053,536ms (~17.5 hours), which
# turned out to be a session resumed the next day into the SAME transcript file whose last
# claude-code turn had no `ended_at` (Stop never seen) and therefore an unbounded window. That
# one sample alone inflated one turn's `model_ms` to ~63M and its (model_ms+tool_ms)/wall_ms
# ratio to 110x. 15 minutes cleanly separates the one real anomaly from everything legitimate,
# including "xhigh" extended-thinking responses, which this corpus never pushed past ~10 minutes.
_MAX_PLAUSIBLE_MODEL_MS = 15 * 60 * 1000


def _iso_to_ms(ts: str) -> int | None:
    """Transcript ISO-8601 timestamp -> epoch milliseconds, or None if it doesn't parse.

    `round()`, not `int()`: `datetime.timestamp()` returns a float, and truncating a value like
    1788552718.405 loses the sub-ms fraction in a way that occasionally rounds an identical
    instant down by 1ms relative to the ms already stored elsewhere -- harmless for windows this
    coarse, but rounding is the honest operation here, truncation is not.
    """
    if not ts:
        return None
    try:
        return round(datetime.fromisoformat(ts).timestamp() * 1000)
    except (ValueError, TypeError):
        return None


@dataclass
class Window:
    """One turn's attribution window: transcript rows with `lo <= ts < hi` belong to it.

    `hi=None` means unbounded -- either this is the session's last known turn and it has no
    `ended_at` yet (Stop never seen, still possibly in progress), or by construction it's the
    open interval up to whatever bound the caller supplied. A row before the first turn's `lo` or
    inside a real idle gap (a closed turn's `ended_at` short of the next turn's `started_at`)
    belongs to no window and is simply not attributed -- that is correct, not a bug: nothing in
    this system claims a turn for time when no turn was open.
    """

    turn_id: str
    lo: int
    hi: int | None


def windows_for_turns(turns: Sequence[Turn]) -> list[Window]:
    """One session's claude-code turns -> their attribution windows, per requirement: a turn
    with `ended_at=None` borrows the NEXT turn's `started_at` as its upper bound; the last turn
    in the session (or one with a real `ended_at`) gets that value, or None if there is no next
    turn and no `ended_at` yet."""
    ordered = sorted(turns, key=lambda t: t.started_at)
    windows: list[Window] = []
    for i, turn in enumerate(ordered):
        if turn.ended_at is not None:
            hi = turn.ended_at
        elif i + 1 < len(ordered):
            hi = ordered[i + 1].started_at
        else:
            hi = None
        windows.append(Window(turn_id=turn.turn_id, lo=turn.started_at, hi=hi))
    return windows


@dataclass
class Attribution:
    """What one turn accumulates from the transcript. `requests` counts unique `message.id`s,
    never JSONL rows -- see the module docstring on why a row count would be wrong by the
    content-block duplication factor."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    requests: int = 0
    model: str | None = None
    model_ms: int = 0
    has_model_ms: bool = False
    outcome: str | None = None
    error_class: str | None = None


def _find_window_index(los: list[int], ts: int) -> int | None:
    i = bisect_right(los, ts) - 1
    return i if i >= 0 else None


def _is_interrupt_row(record: dict[str, Any]) -> bool:
    if record.get("type") != "user":
        return False
    message = record.get("message")
    if not isinstance(message, dict) or message.get("role") != "user":
        return False
    content = message.get("content")
    if isinstance(content, str):
        texts = [content]
    elif isinstance(content, list):
        texts = [
            block.get("text") or ""
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
    else:
        return False
    return any(text.strip() in _INTERRUPT_MARKERS for text in texts)


def attribute_transcript(path: Path, windows: Sequence[Window]) -> dict[str, Attribution]:
    """Single streaming pass over one transcript file, attributing usage/model/outcome evidence
    to whichever turn's window contains each row's timestamp.

    Grouping by `message.id`: a content-block run sharing one `message.id` is ONE API request.
    Its `model_ms` is `(timestamp of its LAST block) - (timestamp of the row immediately BEFORE
    its FIRST block)` -- the full round-trip for that request, deliberately excluding tool
    execution time. A tool's own duration lives between a tool_use block and its tool_result row,
    neither of which opens or extends a usage group, so it is never folded into `model_ms`; the
    next request's group instead starts counting from the tool_result's own timestamp, i.e. from
    when the tool finished, not when it started. This is what makes `model_ms` additive with
    `events.duration_ms` (tool time) rather than double-counting it.

    A `message.id` reappearing after its group has already been closed (should never happen in an
    append-only transcript; guarded anyway) is dropped rather than reopened, so a transcript
    anomaly can only ever cause an undercount, never a double count.

    An OPEN-ENDED window (`hi=None`, the session's last known claude-code turn -- Stop never
    seen) is closed EARLY, the moment a gap between two consecutive rows exceeds
    `_MAX_PLAUSIBLE_MODEL_MS`. Without this, a session resumed hours or days later into the SAME
    transcript file -- a real event this corpus contains exactly once -- has nowhere else to go
    (no new turn_id was ever created for it) and silently attributes an unrelated later
    conversation's tokens and a many-hour "model latency" to the stale turn. Once such a gap is
    crossed, everything from that point on is simply unattributed rather than misattributed --
    losing that data is the correct outcome, not a compromise. A single implausible SAMPLE within
    an otherwise-bounded window (should not happen, since a real Stop bounds the window before
    any such gap could occur) is dropped the same way as a belt-and-suspenders measure: excluded
    from `model_ms`, never clamped to the cap, which would fabricate a data point instead of
    omitting one.

    Reads the file line by line -- transcripts can run to tens of thousands of lines and this
    holds only the (tiny) per-turn accumulator dict in memory, never the file.
    """
    ordered = sorted(windows, key=lambda w: w.lo)
    los = [w.lo for w in ordered]
    his: list[int | None] = [w.hi for w in ordered]
    attrs: dict[str, Attribution] = {}

    def attribution_for(ts: int) -> Attribution | None:
        i = _find_window_index(los, ts)
        if i is None:
            return None
        hi = his[i]
        if hi is not None and ts >= hi:
            return None
        return attrs.setdefault(ordered[i].turn_id, Attribution())

    prev_ts: int | None = None
    current_id: str | None = None
    closed_ids: set[str] = set()
    group_start: int | None = None
    group_last: int | None = None
    group_model: str | None = None
    group_usage: dict[str, Any] | None = None

    def flush() -> None:
        nonlocal current_id, group_start, group_last, group_model, group_usage
        if current_id is not None and group_usage is not None and group_last is not None:
            target = attribution_for(group_last)
            if target is not None:
                target.requests += 1
                target.prompt_tokens += int(group_usage.get("input_tokens") or 0)
                target.completion_tokens += int(group_usage.get("output_tokens") or 0)
                target.cached_tokens += int(group_usage.get("cache_read_input_tokens") or 0)
                target.model = group_model  # last-wins: groups are flushed in chronological order
                if group_start is not None:
                    elapsed = group_last - group_start
                    if 0 <= elapsed <= _MAX_PLAUSIBLE_MODEL_MS:
                        target.model_ms += elapsed
                        target.has_model_ms = True
            closed_ids.add(current_id)
        current_id = None
        group_start = group_last = group_model = None
        group_usage = None

    try:
        handle = path.open(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    with handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue  # torn/corrupt line -- same tolerance as JsonlLog.read() elsewhere

            ts = _iso_to_ms(record.get("timestamp") or "")
            if ts is None:
                continue

            # An implausible gap since the previous row closes an OPEN-ENDED window early, right
            # after the last row seen before the gap (`+ 1` so that row itself stays included;
            # the exclusion test below is `ts >= hi`). A window with a real `hi` (a Stop was
            # seen, or a later turn exists) is never touched here -- it already can't reach a
            # gap this size, because it stops admitting rows at its own real boundary first.
            if prev_ts is not None and ts - prev_ts > _MAX_PLAUSIBLE_MODEL_MS:
                prev_i = _find_window_index(los, prev_ts)
                if prev_i is not None and his[prev_i] is None:
                    his[prev_i] = prev_ts + 1

            # Outcome evidence is checked on every row, independent of the usage-group state
            # machine below -- an error or interrupt marker can arrive on a row that never
            # participates in a usage group at all.
            target = attribution_for(ts)
            if target is not None and target.outcome is None:
                if record.get("isApiErrorMessage"):
                    target.outcome = "error"
                    target.error_class = record.get("error") or target.error_class
                elif _is_interrupt_row(record):
                    target.outcome = "interrupted"

            message = record.get("message")
            usage = message.get("usage") if isinstance(message, dict) else None
            model = message.get("model") if isinstance(message, dict) else None
            mid = message.get("id") if isinstance(message, dict) else None
            is_group_row = (
                record.get("type") == "assistant"
                and isinstance(usage, dict)
                and bool(mid)
                and model != _SYNTHETIC_MODEL
                and not record.get("isApiErrorMessage")
            )

            if is_group_row:
                if mid == current_id:
                    group_last = ts  # another content block of the same request
                elif mid in closed_ids:
                    pass  # anomalous reappearance of a closed id -- drop, don't double count
                else:
                    flush()
                    current_id = mid
                    group_start = prev_ts  # None on the file's very first group -- see below
                    group_last = ts
                    group_model = model
                    group_usage = usage
            elif current_id is not None:
                flush()

            prev_ts = ts

    flush()
    return attrs


def _claude_code_turns(store: Store, session_id: str) -> list[Turn]:
    rows = store.conn.execute(
        "SELECT * FROM turns WHERE session_id=? AND source='claude-code' ORDER BY started_at",
        (session_id,),
    ).fetchall()
    return [Turn.from_row(dict(row)) for row in rows]


def find_sessions_needing_backfill(store: Store, *, since_ms: int | None = None) -> list[str]:
    """Distinct claude-code session_ids with at least one turn still missing something this
    module can fill. Cheap pre-filter so a rerun does no work for sessions this module has
    already finished (every guarded write below is itself idempotent, but there is no reason to
    re-open and re-scan a transcript file to learn that again)."""
    clauses = [
        "source='claude-code'",
        "(prompt_tokens IS NULL OR model IS NULL OR model_ms IS NULL OR outcome IS NULL)",
    ]
    params: list[Any] = []
    if since_ms is not None:
        clauses.append("started_at >= ?")
        params.append(since_ms)
    rows = store.conn.execute(
        f"SELECT DISTINCT session_id FROM turns WHERE {' AND '.join(clauses)}", params
    ).fetchall()
    return [row[0] for row in rows]


def find_transcript(session_id: str, root: Path | str = DEFAULT_ROOT) -> Path | None:
    """Locate `session_id`'s transcript by FILENAME, not by any field inside it -- see the
    module docstring for why the inner `session_id` field is not safe to join on.

    Tries the main-session layout (`<project>/<session_id>.jsonl`, one level under root, matching
    scope_snapshot.SESSION_GLOB) first since that covers the entire live corpus; falls back to a
    recursive search for the rare case of a session whose transcript lives nested (e.g. under a
    `subagents/` directory) rather than paying the recursive glob's cost on every lookup.
    """
    base = Path(root).expanduser()
    direct = sorted(base.glob(f"*/{session_id}.jsonl"))
    if direct:
        return direct[0]
    nested = sorted(base.rglob(f"{session_id}.jsonl"))
    return nested[0] if nested else None


@dataclass
class BackfillReport:
    sessions_scanned: int = 0
    sessions_matched: int = 0
    sessions_unmatched: int = 0
    unmatched_session_ids: list[str] = field(default_factory=list)
    turns_scanned: int = 0
    turns_updated_tokens: int = 0
    turns_updated_model: int = 0
    turns_updated_model_ms: int = 0
    turns_updated_outcome: int = 0
    dry_run: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def backfill(
    store: Store,
    *,
    root: Path | str = DEFAULT_ROOT,
    since_ms: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Fill in what collect_claude.py structurally cannot observe, for every claude-code turn
    that still needs it, from its Claude Code transcript.

    Per-field write guard is "the column is currently NULL" -- this is both the never-clobber
    rule (requirement: a turn that already has real data, from any source, is left exactly
    alone) and the whole idempotency story (requirement: running this twice does not double
    anything). `model_ms` is the one field this module derives rather than recovers verbatim, so
    it is the only one that sets `estimated=1` -- `prompt_tokens`/`completion_tokens`/
    `cached_tokens`/`requests`/`model` are the literal values the transcript's `message.usage`
    and `message.model` already stated, recovered from a different place, not estimated.

    Each turn is written through `store.upsert_turn(turn, present={...only the touched
    columns...})`, never a bare `upsert_turn(turn)`. This store is genuinely live (the hook
    collector can be closing THIS SAME turn via its own Stop/PostToolUse handlers while this
    runs) and `present`-scoping is the store's documented mechanism for a partial update that
    cannot clobber a column it didn't mean to touch (see Store.upsert_turn's docstring) -- a bare
    upsert here would round-trip this module's slightly-stale in-memory `ended_at`/`wall_ms`/
    `outcome` back over whatever the hook just wrote concurrently.
    """
    report = BackfillReport(dry_run=dry_run)
    session_ids = find_sessions_needing_backfill(store, since_ms=since_ms)
    report.sessions_scanned = len(session_ids)

    for session_id in session_ids:
        turns = _claude_code_turns(store, session_id)
        if not turns:
            continue
        path = find_transcript(session_id, root)
        if path is None:
            report.sessions_unmatched += 1
            report.unmatched_session_ids.append(session_id)
            continue
        report.sessions_matched += 1

        windows = windows_for_turns(turns)
        attrs = attribute_transcript(path, windows)
        by_id = {turn.turn_id: turn for turn in turns}

        for turn_id, attribution in attrs.items():
            turn = by_id.get(turn_id)
            if turn is None:
                continue
            report.turns_scanned += 1
            present: set[str] = set()

            if turn.prompt_tokens is None and attribution.requests > 0:
                turn.prompt_tokens = attribution.prompt_tokens
                turn.completion_tokens = attribution.completion_tokens
                turn.cached_tokens = attribution.cached_tokens
                turn.requests = attribution.requests
                present |= {"prompt_tokens", "completion_tokens", "cached_tokens", "requests"}
                report.turns_updated_tokens += 1

            if turn.model is None and attribution.model is not None:
                turn.model = attribution.model
                present.add("model")
                report.turns_updated_model += 1

            if turn.model_ms is None and attribution.has_model_ms:
                turn.model_ms = attribution.model_ms
                turn.estimated = 1
                present |= {"model_ms", "estimated"}
                report.turns_updated_model_ms += 1

            if turn.outcome is None and attribution.outcome is not None:
                turn.outcome = attribution.outcome
                present.add("outcome")
                if attribution.error_class and turn.error_class is None:
                    turn.error_class = attribution.error_class
                    present.add("error_class")
                report.turns_updated_outcome += 1

            if present and not dry_run:
                store.upsert_turn(turn, present=present)

    return report.to_dict()
