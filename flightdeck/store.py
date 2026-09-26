"""Local append-only store: sqlite for queries, JSONL for durability.

Two writes per record is deliberate. sqlite can be locked by the nightly rollup, corrupted, or
mid-migration; the JSONL append is a single `write()` to an O_APPEND handle and is what guarantees
a turn is never lost. sqlite is a derived index that can always be rebuilt by replaying JSONL.

This store is never `crush.db`. A crush migration, a session delete, or a db reset must not be
able to destroy telemetry history.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import sqlite3
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from flightdeck.models import (
    Event,
    GovernorDecision,
    Judgment,
    Probe,
    ScopeRecord,
    TextBlob,
    TuningChange,
    Turn,
)
from flightdeck.schema import migrate
from flightdeck.tiers import derive_tier

DEFAULT_DIR = Path(
    os.environ.get("SPARKY_TURNLOG_DIR", "~/.local/state/sparky/turnlog")
).expanduser()
DEFAULT_RETENTION_DAYS = 14

_PERCENTILE_COLUMNS = {"wall_ms", "model_ms", "kpi_score", "prompt_tokens", "completion_tokens"}
# total_tokens has no column of its own (Turn.total_tokens is a Python property); this is the
# SQL equivalent so Store.percentile can rank it without pulling full rows into Python.
_PERCENTILE_EXPRESSIONS = {
    "total_tokens": "(COALESCE(prompt_tokens, 0) + COALESCE(completion_tokens, 0))",
}


def now_ms() -> int:
    return int(time.time() * 1000)


def hostname() -> str:
    return os.environ.get("SPARKY_HOST_LABEL") or socket.gethostname().split(".")[0]


class JsonlLog:
    """Day-partitioned append-only log. One JSON object per line, flushed on every write."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def path_for(self, ts_ms: int | None = None) -> Path:
        stamp = time.strftime("%Y-%m-%d", time.gmtime((ts_ms or now_ms()) / 1000))
        return self.directory / f"events-{stamp}.jsonl"

    def append(self, kind: str, record: dict[str, Any], ts_ms: int | None = None) -> None:
        line = json.dumps({"_kind": kind, **record}, separators=(",", ":"), default=str)
        path = self.path_for(ts_ms)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def read(self, path: Path) -> Iterator[tuple[str, dict[str, Any]]]:
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    # A torn final line from a killed process. Everything before it is intact.
                    continue
                kind = record.pop("_kind", None)
                if kind:
                    yield kind, record

    def files(self) -> list[Path]:
        return sorted(self.directory.glob("events-*.jsonl"))

    def read_from(
        self, path: Path, offset: int = 0
    ) -> tuple[list[tuple[str, dict[str, Any]]], int]:
        """Records appended after `offset` bytes, plus the byte offset to resume from next time.
        Backs Store.ingest_jsonl's watermark: a rerun seeks straight to the last-known offset
        instead of re-parsing the whole file. A trailing line with no newline yet (writer still
        mid-flush) is left unread and `offset` stops before it, so a future call picks it up once
        it's complete — same tolerance `read()` has for a torn final line."""
        if not path.exists():
            return [], offset
        records: list[tuple[str, dict[str, Any]]] = []
        pos = offset
        with path.open("rb") as handle:
            handle.seek(offset)
            for raw_line in handle:
                if not raw_line.endswith(b"\n"):
                    break
                pos += len(raw_line)
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                kind = record.pop("_kind", None)
                if kind:
                    records.append((kind, record))
        return records, pos


class Store:
    def __init__(self, directory: Path | str = DEFAULT_DIR, db_name: str = "turnlog.db") -> None:
        self.directory = Path(directory).expanduser()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.db_path = self.directory / db_name
        self.jsonl = JsonlLog(self.directory)
        self.conn = sqlite3.connect(self.db_path, timeout=15.0, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA busy_timeout=15000")
        self.version = migrate(self.conn)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---------- writes ----------

    def upsert_turn(
        self, turn: Turn, *, mirror: bool = True, present: set[str] | None = None
    ) -> None:
        """Upsert a turn. `present` names the columns the SOURCE RECORD actually carried.

        A turn is written twice -- an opening record when the prompt arrives and a completing
        record when it ends -- and the two carry different fields. Rebuilding a `Turn` from a
        partial record fills every absent field with the dataclass default, so updating every
        column unconditionally made the second write ERASE what the first one knew. That was
        not hypothetical: every completed opencode turn had its `cwd` nulled, 63 of 63 turns
        that kept a cwd were the ones that never completed, `agent_name` survived only on
        those same unfinished turns, and `is_subagent` was reset to its 0 default on every
        turn in the database -- 2,190 of them, not one marked a subagent.

        COALESCE is not enough here: `is_subagent` and `estimated` are non-optional ints whose
        default is 0, so an absent field is indistinguishable from a real 0 once the dataclass
        has been built. The absent/present distinction only survives upstream, in the record's
        own key set, which is why it is passed in rather than inferred.

        Callers with a fully-populated Turn pass nothing and get the previous behaviour.
        """
        row = turn.to_row()
        cols = ", ".join(row)
        placeholders = ", ".join(f":{c}" for c in row)
        updatable = [c for c in row if c != "turn_id" and (present is None or c in present)]
        if updatable:
            updates = ", ".join(f"{c}=excluded.{c}" for c in updatable)
            conflict = f"DO UPDATE SET {updates}"
        else:
            # A record that carries nothing but the key must not be an error, and must not
            # rewrite the row with defaults either.
            conflict = "DO NOTHING"
        self.conn.execute(
            f"INSERT INTO turns ({cols}) VALUES ({placeholders}) ON CONFLICT(turn_id) {conflict}",
            row,
        )
        if mirror:
            self.jsonl.append("turn", row, turn.started_at)

    def add_events(self, events: list[Event], *, mirror: bool = True) -> int:
        if not events:
            return 0
        written = 0
        for event in events:
            row = event.to_row()
            # dedupe_id distinguishes genuinely concurrent same-ms events (see idx_events_dedupe);
            # derived from payload rather than stored in JSONL so a replayed record recomputes the
            # identical value and still dedupes against the original.
            params = {**row, "dedupe_id": str(event.payload.get("tool_use_id") or "")}
            cur = self.conn.execute(
                "INSERT OR IGNORE INTO events "
                "(turn_id, ts, kind, name, duration_ms, ok, payload, dedupe_id) "
                "VALUES (:turn_id, :ts, :kind, :name, :duration_ms, :ok, :payload, :dedupe_id)",
                params,
            )
            written += cur.rowcount or 0
            if mirror:
                self.jsonl.append("event", row, event.ts)
        return written

    def add_texts(self, texts: list[TextBlob], *, mirror: bool = True) -> None:
        for text in texts:
            row = text.to_row()
            self.conn.execute(
                "INSERT OR REPLACE INTO texts (turn_id, kind, seq, body, expires_at) "
                "VALUES (:turn_id, :kind, :seq, :body, :expires_at)",
                row,
            )
            if mirror:
                self.jsonl.append("text", row)

    def add_judgment(self, judgment: Judgment, *, mirror: bool = True) -> None:
        row = judgment.to_row()
        self.conn.execute(
            "INSERT OR REPLACE INTO judgments "
            "(turn_id, judge_model, verdict, rubric, notes, lesson, created_at) VALUES "
            "(:turn_id, :judge_model, :verdict, :rubric, :notes, :lesson, :created_at)",
            row,
        )
        self.conn.execute("UPDATE turns SET judged=1 WHERE turn_id=?", (judgment.turn_id,))
        if mirror:
            self.jsonl.append("judgment", row, judgment.created_at)

    def add_scope_record(self, record: ScopeRecord, *, mirror: bool = True) -> None:
        """Write one scoping pass. Chains row_hash over the previous row for this host so a
        rewritten history is detectable, same discipline as the *_trend.jsonl files.

        Synthetic rows are refused here, at the store, and not only in the generator that
        makes them. A guard that lives in the writer protects the writer; a synthetic row
        that reached this table by any other path would be read back as measurement by
        everything downstream and would be indistinguishable from one.
        """
        if record.provenance.get("synthetic") and (
            Path(self.directory).resolve() == Path(DEFAULT_DIR).expanduser().resolve()
        ):
            raise ValueError(
                "refusing to write a synthetic scope_record into the live store "
                f"({self.directory}); point the Store at a scratch directory"
            )
        row = record.to_row()
        prev = self.conn.execute(
            "SELECT row_hash FROM scope_records WHERE host=? "
            "ORDER BY created_at DESC, rowid DESC LIMIT 1",
            (record.host,),
        ).fetchone()
        row["prev_hash"] = prev["row_hash"] if prev else None
        payload = json.dumps(
            {k: row[k] for k in sorted(row) if k not in ("row_hash",)},
            separators=(",", ":"),
            sort_keys=True,
        )
        row["row_hash"] = hashlib.sha256(payload.encode()).hexdigest()
        cols = [k for k in row]
        self.conn.execute(
            f"INSERT OR REPLACE INTO scope_records ({', '.join(cols)}) "
            f"VALUES ({', '.join(':' + c for c in cols)})",
            row,
        )
        if mirror:
            # ScopeRecord.created_at is whole SECONDS (scope_pass.py builds it from
            # int(time.time())); JsonlLog.path_for takes MILLISECONDS. Passing it raw put
            # every scope_record ever written into events-1970-01-21.jsonl -- 27 rows, all
            # of them, in a segment 56 years from the day they happened. The mirror is what
            # replay reads, so a day-partitioned log that partitions by the wrong epoch is
            # not a cosmetic filename bug.
            self.jsonl.append("scope_record", row, record.created_at * 1000)

    def scope_records(self, *, since: int | None = None) -> list[sqlite3.Row]:
        """Scope records, optionally newer than `since`.

        `since` is whole SECONDS, not milliseconds -- scope_records.created_at is the one
        timestamp in this store that is not epoch-ms (every other model stamps itself with
        now_ms()). A caller that passes now_ms() here gets an empty list and no error.
        """
        sql = "SELECT * FROM scope_records"
        args: tuple[Any, ...] = ()
        if since is not None:
            sql += " WHERE created_at >= ?"
            args = (since,)
        return list(self.conn.execute(sql + " ORDER BY created_at", args))

    def add_decision(self, decision: GovernorDecision, *, mirror: bool = True) -> None:
        row = decision.to_row()
        self.conn.execute(
            "INSERT OR REPLACE INTO governor_decisions "
            "(turn_id, domain, chosen, alternatives, features, shadow, weights_version, "
            "explored, reason, epsilon) VALUES "
            "(:turn_id, :domain, :chosen, :alternatives, :features, :shadow, :weights_version, "
            ":explored, :reason, :epsilon)",
            row,
        )
        if mirror:
            self.jsonl.append("decision", row)

    def add_probe(self, probe: Probe, *, mirror: bool = True) -> None:
        """Dedup relies on UNIQUE(host, kind, ts): a probe run always mints a fresh ts, so a
        replayed record for a host/kind/ts already seen (e.g. from re-ingesting a JSONL log) is a
        no-op, same pattern as events/host_metrics."""
        row = probe.to_row()
        self.conn.execute(
            "INSERT OR IGNORE INTO probes (ts, host, kind, ok, total, detail) "
            "VALUES (:ts, :host, :kind, :ok, :total, :detail)",
            row,
        )
        if mirror:
            self.jsonl.append("probe", row, probe.ts)

    def add_host_metrics(self, record: dict[str, Any], *, mirror: bool = True) -> int:
        """Point-in-time CPU/RAM/GPU/disk snapshot. Dedup relies on UNIQUE(host, ts), same as
        every other write path — a replayed record for a host/ts already seen is a no-op."""
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO host_metrics "
            "(host, ts, cpu_pct, mem_used_mb, mem_total_mb, gpu_pct, gpu_mem_used_mb, "
            "disk_used_gb, disk_total_gb) VALUES "
            "(:host, :ts, :cpu_pct, :mem_used_mb, :mem_total_mb, :gpu_pct, :gpu_mem_used_mb, "
            ":disk_used_gb, :disk_total_gb)",
            record,
        )
        if mirror:
            self.jsonl.append("host_metrics", record)
        return cur.rowcount or 0

    def add_tuning_change(self, change: TuningChange, *, mirror: bool = True) -> None:
        row = change.to_row()
        cols = ", ".join(row)
        placeholders = ", ".join(f":{c}" for c in row)
        updates = ", ".join(f"{c}=excluded.{c}" for c in row if c != "change_id")
        self.conn.execute(
            f"INSERT INTO tuning_changes ({cols}) VALUES ({placeholders}) "
            f"ON CONFLICT(change_id) DO UPDATE SET {updates}",
            row,
        )
        if mirror:
            self.jsonl.append("tuning_change", row, change.applied_at)

    # ---------- reads ----------

    def get_turn(self, turn_id: str) -> Turn | None:
        row = self.conn.execute("SELECT * FROM turns WHERE turn_id=?", (turn_id,)).fetchone()
        return Turn.from_row(dict(row)) if row else None

    def iter_turns(
        self,
        *,
        since_ms: int | None = None,
        until_ms: int | None = None,
        source: str | None = None,
        host: str | None = None,
        limit: int | None = None,
    ) -> Iterator[Turn]:
        clauses, params = ["1=1"], []
        if since_ms is not None:
            clauses.append("started_at >= ?")
            params.append(since_ms)
        if until_ms is not None:
            clauses.append("started_at < ?")
            params.append(until_ms)
        if source:
            clauses.append("source = ?")
            params.append(source)
        if host:
            clauses.append("host = ?")
            params.append(host)
        sql = f"SELECT * FROM turns WHERE {' AND '.join(clauses)} ORDER BY started_at"
        if limit:
            sql += f" LIMIT {int(limit)}"
        for row in self.conn.execute(sql, params):
            yield Turn.from_row(dict(row))

    def events_for(self, turn_id: str, kind: str | None = None) -> list[Event]:
        if kind:
            rows = self.conn.execute(
                "SELECT * FROM events WHERE turn_id=? AND kind=? ORDER BY ts, event_id",
                (turn_id, kind),
            )
        else:
            rows = self.conn.execute(
                "SELECT * FROM events WHERE turn_id=? ORDER BY ts, event_id", (turn_id,)
            )
        return [Event.from_row(dict(r)) for r in rows]

    def texts_for(self, turn_id: str, kind: str | None = None) -> list[TextBlob]:
        if kind:
            rows = self.conn.execute(
                "SELECT * FROM texts WHERE turn_id=? AND kind=? ORDER BY seq", (turn_id, kind)
            )
        else:
            rows = self.conn.execute(
                "SELECT * FROM texts WHERE turn_id=? ORDER BY kind, seq", (turn_id,)
            )
        return [TextBlob(**dict(r)) for r in rows]

    def judgment_for(self, turn_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM judgments WHERE turn_id=?", (turn_id,)).fetchone()
        return dict(row) if row else None

    def decisions_for(self, turn_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM governor_decisions WHERE turn_id=?", (turn_id,))
        return [dict(r) for r in rows]

    def pending_judgments(self, limit: int = 20) -> list[Turn]:
        rows = self.conn.execute(
            "SELECT * FROM turns WHERE flagged=1 AND judged=0 ORDER BY started_at DESC LIMIT ?",
            (limit,),
        )
        return [Turn.from_row(dict(r)) for r in rows]

    def percentile(
        self,
        column: str,
        pct: float,
        *,
        host: str | None = None,
        tier: str | None = None,
        source: str | None = None,
        since_ms: int | None = None,
        until_ms: int | None = None,
        exclude_turn_id: str | None = None,
    ) -> float | None:
        """Percentile of a numeric turn column, or `total_tokens` (prompt_tokens +
        completion_tokens, computed in SQL since it has no column of its own). Used for the
        wall-time flag threshold and the KPI efficiency baseline."""
        if column in _PERCENTILE_EXPRESSIONS:
            expr = _PERCENTILE_EXPRESSIONS[column]
            not_null_clause = f"{expr} > 0"
        elif column in _PERCENTILE_COLUMNS:
            expr = column
            not_null_clause = f"{column} IS NOT NULL"
        else:
            raise ValueError(f"percentile not allowed on column {column!r}")
        clauses, params = [not_null_clause], []
        if host:
            clauses.append("host = ?")
            params.append(host)
        if tier:
            clauses.append("tier = ?")
            params.append(tier)
        if source:
            clauses.append("source = ?")
            params.append(source)
        if since_ms is not None:
            clauses.append("started_at >= ?")
            params.append(since_ms)
        if until_ms is not None:
            clauses.append("started_at < ?")
            params.append(until_ms)
        if exclude_turn_id is not None:
            clauses.append("turn_id != ?")
            params.append(exclude_turn_id)
        rows = self.conn.execute(
            f"SELECT {expr} FROM turns WHERE {' AND '.join(clauses)} ORDER BY {expr}", params
        ).fetchall()
        if not rows:
            return None
        idx = min(len(rows) - 1, max(0, int(round(pct * (len(rows) - 1)))))
        return float(rows[idx][0])

    def scored_decisions(self, domain: str, *, since_ms: int) -> list[dict[str, Any]]:
        """Governor decisions for `domain` whose turn has a KPI score, bounded by `since_ms`.
        One indexed join, in place of the N+1 scan (iter_turns + decisions_for per turn,
        filtering domain in Python) success_table used to do over every turn ever recorded, and
        graduation_report was refactored to use for the same reason. tier/mode are included so a
        caller can compare the decision's chosen arm against what actually ran, per domain."""
        rows = self.conn.execute(
            "SELECT d.chosen AS chosen, d.features AS features, "
            "t.kpi_score AS kpi_score, t.started_at AS started_at, "
            "t.tier AS tier, t.mode AS mode "
            "FROM governor_decisions d JOIN turns t USING (turn_id) "
            "WHERE d.domain = ? AND t.kpi_score IS NOT NULL AND t.started_at >= ?",
            (domain, since_ms),
        )
        return [dict(r) for r in rows]

    def turns_pending_decision(self, domain: str, *, since_ms: int) -> list[Turn]:
        """Scored turns since `since_ms` that don't yet have a governor_decisions row for
        `domain` — the counterpart to iter_turns' kpi_score IS NULL filter cmd_rollup uses to
        find unscored turns. Backs Governor.record_shadow_decisions: only a scored turn can
        contribute evidence to scored_decisions/success_table, so there is no point deciding for
        one that hasn't been scored yet, and a turn that already has a decision for this domain
        is skipped rather than overwritten on every rerun."""
        rows = self.conn.execute(
            "SELECT t.* FROM turns t WHERE t.kpi_score IS NOT NULL AND t.started_at >= ? "
            "AND NOT EXISTS ("
            "  SELECT 1 FROM governor_decisions d WHERE d.turn_id = t.turn_id AND d.domain = ?"
            ") ORDER BY t.started_at",
            (since_ms, domain),
        )
        return [Turn.from_row(dict(r)) for r in rows]

    # ---------- maintenance ----------

    def expire_texts(self, now: int | None = None) -> int:
        cur = self.conn.execute("DELETE FROM texts WHERE expires_at <= ?", (now or now_ms(),))
        return cur.rowcount or 0

    def ingest_jsonl(self, path: Path) -> dict[str, int]:
        """Replay newly appended JSONL records into sqlite, resuming from the persisted cursor
        (`ingest_cursor`) so a rerun only reads bytes written since the last call instead of the
        file's entire history — cmd_rollup used to replay every events-*.jsonl file in full on
        every run, which was O(total history) per rollup. Still idempotent even if the cursor is
        reset or a file is replayed from scratch: every row type has its own dedupe key too
        (INSERT OR IGNORE / an upsert on a natural key)."""
        counts: dict[str, int] = {}
        cursor_key = str(path)
        cursor_row = self.conn.execute(
            "SELECT offset FROM ingest_cursor WHERE path=?", (cursor_key,)
        ).fetchone()
        offset = cursor_row[0] if cursor_row else 0
        records, new_offset = self.jsonl.read_from(path, offset)
        for kind, record in records:
            counts[kind] = counts.get(kind, 0) + 1
            if kind == "turn":
                turn = Turn.from_row(record)
                if turn.tier is None:
                    # The Go/TS emitters write `model` but not `tier` (no wire point sets it
                    # yet); the claude-code collector sets tier itself at write time (see
                    # collect_claude.py). This is the one place every other source's turn record
                    # passes through sqlite, so it's the ingest-time fallback for the rest.
                    turn.tier = derive_tier(turn.source, turn.model)
                # Only the columns this record actually carried; see upsert_turn. A key
                # present with a null value is NOT carried: the enriched closing row the
                # rollup re-appends spells out every column and writes `cwd: null` for the
                # ones it never knew, and treating that as information nulled the cwd of 377
                # of one week's 425 opencode turns -- after the absent-key fix had landed.
                carried = {k for k, v in record.items() if v is not None}
                self.upsert_turn(turn, mirror=False, present=carried | {"tier"})
            elif kind == "event":
                self.add_events([Event.from_row(record)], mirror=False)
            elif kind == "text":
                # A rebuilt/behind-cursor ingest can replay a text whose row was already purged
                # by an earlier --expire run (expire_texts deletes from sqlite, never from the
                # JSONL durability log). Re-inserting it would resurrect already-expired text
                # with nothing scheduled to re-purge it on a plain `rollup` (no --expire). Don't
                # resurrect: an already-expired record from the log stays absent from sqlite.
                if record.get("expires_at", 0) > now_ms():
                    self.add_texts([TextBlob(**record)], mirror=False)
            elif kind == "judgment":
                record["rubric"] = json.loads(record["rubric"]) if record.get("rubric") else {}
                self.add_judgment(Judgment(**record), mirror=False)
            elif kind == "decision":
                record["alternatives"] = json.loads(record.get("alternatives") or "{}")
                record["features"] = json.loads(record.get("features") or "{}")
                self.add_decision(GovernorDecision(**record), mirror=False)
            elif kind == "probe":
                record["detail"] = json.loads(record["detail"]) if record.get("detail") else {}
                self.add_probe(Probe(**record), mirror=False)
            elif kind == "tuning_change":
                record["evidence"] = (
                    json.loads(record["evidence"]) if record.get("evidence") else {}
                )
                self.add_tuning_change(TuningChange(**record), mirror=False)
            elif kind == "host_metrics":
                self.add_host_metrics(record, mirror=False)
        if new_offset != offset:
            self.conn.execute(
                "INSERT INTO ingest_cursor (path, offset) VALUES (:path, :offset) "
                "ON CONFLICT(path) DO UPDATE SET offset=excluded.offset",
                {"path": cursor_key, "offset": new_offset},
            )
        return counts

    def _common_columns(self, table: str) -> list[str]:
        """Columns present in both this store's schema and the attached `src` schema, in this
        store's declared order. A peer box a migration behind is missing later columns; a bare
        `SELECT *` INSERT then selects a different column count/order than dest expects and
        throws, aborting the whole merge instead of just skipping the columns the peer lacks."""
        dest_cols = [row[1] for row in self.conn.execute(f"PRAGMA main.table_info({table})")]
        src_cols = {row[1] for row in self.conn.execute(f"PRAGMA src.table_info({table})")}
        return [c for c in dest_cols if c in src_cols]

    # Autoincrement PKs that must never be copied across a merge — dest assigns its own so two
    # boxes' independently-assigned ids don't collide or overwrite unrelated rows.
    _MERGE_PK_EXCLUDE = {"events": "event_id", "probes": "probe_id"}

    # Tables where a row, once written, never changes. An event never gets rewritten, a
    # judgment is one verdict per turn_id, a scope_record's id is freshly generated per
    # scoping pass (add_scope_record never revises an existing record_id) — so the only two
    # cases at a given primary key are "already have it" and "new", and IGNORE is correct.
    _MERGE_APPEND_ONLY = (
        "turns",
        "events",
        "texts",
        "judgments",
        "governor_decisions",
        "probes",
        "tuning_changes",
        "host_metrics",
        "scope_records",
        # amend_plan/set_plan_status are CAS on plans.revision, never a rewrite of an existing
        # plan_revisions row -- each accepted change mints a fresh (todo_id, revision) pair.
        "plan_revisions",
    )

    # Tables where the SAME primary key can be revised on either machine after it first
    # appears (a todo's status, a session's heartbeat). Keyed by (pk columns, recency
    # column): a peer row overwrites the local row only if it is strictly newer, so a stale
    # peer can never clobber a fresher local write and INSERT OR IGNORE's silent-drop-on-
    # conflict is not an option — that is the bug being fixed here.
    _MERGE_MUTABLE: dict[str, tuple[tuple[str, ...], str]] = {
        "todos": (("todo_id",), "updated_at"),
        "agent_sessions": (("session_id", "host"), "heartbeat_at"),
        # A plan revised on one box (amend_plan/set_plan_status bumps revision + updated_at)
        # must win over a stale copy elsewhere -- same recency rule as todos/agent_sessions.
        "plans": (("todo_id",), "updated_at"),
    }

    def _merge_append_only(self, table: str) -> int:
        cols = self._common_columns(table)
        exclude = self._MERGE_PK_EXCLUDE.get(table)
        if exclude:
            cols = [c for c in cols if c != exclude]
        if not cols:
            # A peer before the migration that created this table entirely (PRAGMA
            # table_info on a nonexistent src table returns no rows) has nothing to
            # contribute — skip it rather than emit an empty column list.
            return 0
        col_list = ", ".join(cols)
        sql = f"INSERT OR IGNORE INTO {table} ({col_list}) SELECT {col_list} FROM src.{table}"
        return self.conn.execute(sql).rowcount or 0

    def _merge_mutable(self, table: str, pk_cols: tuple[str, ...], recency_col: str) -> int:
        cols = self._common_columns(table)
        if not cols:
            return 0  # peer predates the table (same schema-skew case as append-only)
        if recency_col not in cols or any(pk not in cols for pk in pk_cols):
            # A peer on a schema before recency_col existed can't prove which row is
            # newer — fall back to the append-only behavior rather than guessing or
            # raising. This also covers a peer with columns but a different PK shape.
            return self._merge_append_only(table)
        col_list = ", ".join(cols)
        pk_list = ", ".join(pk_cols)
        update_cols = [c for c in cols if c not in pk_cols]
        set_clause = ", ".join(f"{c}=excluded.{c}" for c in update_cols)
        sql = (
            f"INSERT INTO {table} ({col_list}) SELECT {col_list} FROM src.{table} "
            # sqlite requires a WHERE on the SELECT when it's followed by ON CONFLICT, or
            # the parser can't disambiguate the grammar — this WHERE is that requirement,
            # not a filter (it matches every row).
            "WHERE 1=1 "
            f"ON CONFLICT({pk_list}) DO UPDATE SET {set_clause} "
            f"WHERE excluded.{recency_col} > {table}.{recency_col}"
        )
        return self.conn.execute(sql).rowcount or 0

    def merge_from(self, other_db: Path) -> dict[str, int]:
        """Merge another box's store into this one. Used by the pre-review sync on spark."""
        counts: dict[str, int] = {}
        self.conn.execute("ATTACH DATABASE ? AS src", (str(other_db),))
        try:
            for table in self._MERGE_APPEND_ONLY:
                counts[table] = self._merge_append_only(table)
            for table, (pk_cols, recency_col) in self._MERGE_MUTABLE.items():
                counts[table] = self._merge_mutable(table, pk_cols, recency_col)
        finally:
            self.conn.execute("DETACH DATABASE src")
        return counts
