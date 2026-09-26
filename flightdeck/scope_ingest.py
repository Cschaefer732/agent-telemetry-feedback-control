"""Turn gate decisions into scope_records, so the KPI layer has something to read.

Until this module existed the subsystem was open at its widest point. The hook wrote every
decision to gate-log.jsonl and nothing read it back; `ScopePass.close()` was the only writer
of scope_records and nothing in the live path constructed a pass. Measured on the live
store: scope_records held 0 rows while the gate had made 27 decisions. Every KPI was
computed over an empty table, returned "silent", and looked correct doing it.

A gate decision is NOT a scoping pass, and this module refuses to pretend otherwise. An
ingested row carries verdict "silent": the gate fired, and nothing observed whether the
scoping that followed was any good. Writing "pass" here would manufacture the exact
measurement the subsystem is supposed to earn. When a real ScopePass is closed for the same
turn it supersedes the thin row -- same record_id, richer content.

Suppressed decisions (tier None -- the input filter rejected the text as harness-generated)
are counted but never ingested: they are traffic the gate correctly declined to judge, and
mixing them into the tier distribution would put the filter's work into the gate's numbers.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from flightdeck.models import ScopeRecord
from flightdeck.store import Store, hostname

#: The gate decided; no pass was closed. This is the whole point of a three-valued verdict.
UNOBSERVED = "silent"


def gate_log_path(directory: Path | str) -> Path:
    return Path(directory).expanduser() / "scope" / "gate-log.jsonl"


def read_gate_log(path: Path | str) -> Iterator[dict[str, Any]]:
    target = Path(path).expanduser()
    if not target.exists():
        return
    for line in target.read_text().splitlines():
        if not line.strip():
            continue
        try:
            yield json.loads(line)
        except ValueError:
            continue          # a truncated tail must not lose the rows before it


def record_id_for(row: dict[str, Any]) -> str:
    """Stable across reruns: the same decision must not ingest twice under a new id."""
    seed = f"{row.get('session_id')}|{row.get('ts')}|{row.get('prompt_sha1')}"
    return "gate-" + hashlib.sha1(seed.encode()).hexdigest()[:16]


def _epoch(stamp: str) -> int | None:
    try:
        return int(dt.datetime.fromisoformat(stamp).timestamp())
    except (TypeError, ValueError):
        return None


def to_record(row: dict[str, Any], *, host: str) -> ScopeRecord | None:
    created = _epoch(row.get("ts", ""))
    if created is None or not row.get("tier"):
        return None
    return ScopeRecord(
        record_id=record_id_for(row),
        session_id=row.get("session_id") or "unknown",
        created_at=created,
        host=host,
        cwd=row.get("cwd"),
        tier=row["tier"],
        verdict=UNOBSERVED,
        provenance={
            "source": "gate-log",
            "thin": True,
            "gate_reasons": row.get("reasons", []),
            "has_active_scope": row.get("has_active_scope"),
            "prompt_sha1": row.get("prompt_sha1"),
            "prompt_len": row.get("prompt_len"),
        },
    )


def ingest(
    store: Store, *, directory: Path | str | None = None, since: int | None = None
) -> dict[str, Any]:
    """Write one thin scope_record per gate decision not already stored.

    Existing ids are SKIPPED rather than replaced: `add_scope_record` chains each row's hash
    over the previous one for its host, so rewriting a row silently invalidates every hash
    after it. Idempotence here means "do nothing", not "write again".
    """
    path = gate_log_path(directory if directory is not None else store.directory)
    host = hostname()
    known = {r["record_id"] for r in store.scope_records()}
    written = skipped = suppressed = malformed = 0

    for row in read_gate_log(path):
        if row.get("suppressed") or row.get("tier") is None:
            suppressed += 1
            continue
        record = to_record(row, host=host)
        if record is None:
            malformed += 1
            continue
        if since is not None and record.created_at < since:
            continue
        if record.record_id in known:
            skipped += 1
            continue
        store.add_scope_record(record)
        known.add(record.record_id)
        written += 1

    return {
        "path": str(path),
        "written": written,
        "already_present": skipped,
        "suppressed_not_ingested": suppressed,
        "malformed": malformed,
        "rows_now": len(store.scope_records()),
    }


def verify_chain(store: Store) -> dict[str, Any]:
    """Walk the hash chain per host and report the first break.

    The chain was written from the start and never checked anywhere, which made it
    decoration rather than a guarantee -- a deleted or edited row was undetectable.

    Walked in INSERTION order (rowid), not timestamp order. `add_scope_record` links each
    row to whichever row currently has the greatest created_at, and created_at is whole
    seconds, so several rows written inside one second are ordered by rowid at write time
    and are indistinguishable by timestamp afterwards. Verifying in (created_at, record_id)
    order reported 7 false breaks on a 27-row chain that was in fact intact.

    That also bounds what the chain proves: it detects edits and deletions in an
    append-in-order history. It does NOT survive a backfill -- a row inserted with an older
    created_at than rows already stored links to the newest row, not to its own neighbour,
    and every later verification sees a genuine break. Backfilled rows need their own host
    partition or the chain has to be rebuilt.
    """
    import hashlib as _h

    rows = store.conn.execute(
        "SELECT rowid AS _rowid, * FROM scope_records ORDER BY _rowid"
    ).fetchall()

    broken: list[dict[str, Any]] = []
    last_hash: dict[str, str | None] = {}
    hosts: set[str] = set()

    for row in rows:
        data = dict(row)
        data.pop("_rowid")
        host = data["host"]
        hosts.add(host)
        stored_hash = data.pop("row_hash")
        payload = json.dumps(
            {k: data[k] for k in sorted(data)}, separators=(",", ":"), sort_keys=True
        )
        recomputed = _h.sha256(payload.encode()).hexdigest()
        if data["prev_hash"] != last_hash.get(host):
            broken.append(
                {"host": host, "record_id": data["record_id"], "problem": "prev_hash mismatch"}
            )
        if stored_hash != recomputed:
            broken.append(
                {"host": host, "record_id": data["record_id"], "problem": "row_hash mismatch"}
            )
        last_hash[host] = stored_hash

    return {
        "checked": len(rows),
        "hosts": sorted(hosts),
        "broken": broken,
        "verdict": "silent" if not rows else ("fail" if broken else "pass"),
    }
