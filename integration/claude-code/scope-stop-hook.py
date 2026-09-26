#!/usr/bin/env python3
"""Stop hook: close whatever ScopePass is active for this session, honestly.

Must never fail or delay a Claude Code turn -- always exits 0, whatever stdin holds or
breaks.

The Stop event payload (session_id, cwd, transcript_path, stop_hook_active) carries NO
signal about whether the turn's work was actually correct -- only that the turn ended.
Synthesizing "pass" or "fail" from that would be exactly the fabrication
flightdeck/scope_ingest.py refuses to do when a gate decision alone gets ingested: it
insists a ScopeRecord's verdict be earned by an observation, not manufactured from
adjacent facts. The same refusal applies here. So every pass closed by this hook is closed
with verdict "silent" -- not because nothing happened, but because nothing observable at
Stop time says whether it went well. A closed "silent" record is still strictly better than
an open pass no one ever closes: it lets the KPI layer see that scoping fired without
lying about the outcome.

Before this hook existed, ScopePass.open/.add/.close had zero production callers. This
closes the loop the hook opened -- it does not, and cannot, close the honesty gap that
requires a real evidence source (test runs, diff review, human correction) to fill.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from flightdeck.scope_pass import ScopePass  # noqa: E402
from flightdeck.store import DEFAULT_DIR, Store  # noqa: E402

STOP_LOG = Path(DEFAULT_DIR).expanduser() / "scope" / "stop-log.jsonl"

#: The only verdict a Stop event can honestly support. See module docstring.
UNOBSERVED = "silent"


def _log(record: dict) -> None:
    """Best effort. A logging failure must never cost the user their turn -- but every
    exception path in this hook calls this, so failures are never silent, only unlogged
    in the one case where even logging itself is broken."""
    try:
        STOP_LOG.parent.mkdir(parents=True, exist_ok=True)
        with STOP_LOG.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
    except OSError:
        pass


def _record_id(pass_: ScopePass) -> str:
    """The SAME id `scope_ingest.record_id_for` derives for this turn's gate-log line.

    Two writers reach scope_records for one decision: this hook, and `scope_ingest` reading
    the gate log. Left on separate id schemes they both insert, and a single turn is counted
    twice in every rate the KPI layer computes. The pass carries the gate's own ts and
    prompt_sha1 for exactly this reason, so whichever writer arrives second is skipped by
    the existing-id check rather than duplicating the first.

    A pass opened before that plumbing existed has no gate_ts; those fall back to the old
    per-pass seed, which is stable and non-colliding but will not dedupe against ingest.
    """
    if pass_.gate_ts:
        seed = f"{pass_.session_id}|{pass_.gate_ts}|{pass_.prompt_sha1}"
        return "gate-" + hashlib.sha1(seed.encode()).hexdigest()[:16]
    seed = f"stop|{pass_.session_id}|{pass_.created_at}"
    return "stop-" + hashlib.sha1(seed.encode()).hexdigest()[:16]


def main() -> int:
    ts = datetime.now(UTC).isoformat()
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
    except Exception as exc:
        _log({"ts": ts, "closed": False, "reason": f"bad payload: {exc!r}"})
        return 0
    if not isinstance(payload, dict):
        _log({"ts": ts, "closed": False, "reason": "payload is not an object"})
        return 0
    if payload.get("hook_event_name") not in (None, "Stop"):
        return 0

    session_id = payload.get("session_id") or "unknown"
    cwd = payload.get("cwd") or ""

    try:
        active = ScopePass.load(session_id)
    except Exception as exc:
        _log(
            {
                "ts": ts,
                "session_id": session_id,
                "cwd": cwd,
                "closed": False,
                "reason": f"load raised: {exc!r}",
            }
        )
        return 0

    if active is None:
        _log(
            {
                "ts": ts,
                "session_id": session_id,
                "cwd": cwd,
                "closed": False,
                "reason": "no active scope pass for this session",
            }
        )
        return 0

    stale = active.stale_reason(cwd)
    if stale is not None:
        _log(
            {
                "ts": ts,
                "session_id": session_id,
                "cwd": cwd,
                "closed": False,
                "reason": f"stale, left for its own session to close or expire: {stale}",
            }
        )
        return 0

    record_id = _record_id(active)
    try:
        with Store(DEFAULT_DIR) as store:
            record = active.close(store, record_id, verdict=UNOBSERVED)
    except Exception as exc:
        _log(
            {
                "ts": ts,
                "session_id": session_id,
                "cwd": cwd,
                "closed": False,
                "reason": f"close raised: {exc!r}",
            }
        )
        return 0

    # Closed: remove the state file so a later Stop (or a stray retry) does not try to
    # close the same pass twice. Best effort -- an orphaned file is caught by the existing
    # staleness guard on the next load, not a correctness problem.
    with contextlib.suppress(OSError):
        active.path().unlink(missing_ok=True)

    _log(
        {
            "ts": ts,
            "session_id": session_id,
            "cwd": cwd,
            "closed": True,
            "record_id": record.record_id,
            "tier": record.tier,
            "verdict": record.verdict,
        }
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
