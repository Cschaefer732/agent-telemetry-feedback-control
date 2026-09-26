from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from flightdeck.models import Event, Turn
from flightdeck.schema import MIGRATIONS
from flightdeck.store import Store, now_ms


@pytest.fixture
def store(tmp_path: Path) -> Store:
    s = Store(tmp_path / "store")
    yield s
    s.close()


def _turn(turn_id: str, *, started_at: int | None = None) -> Turn:
    return Turn(
        turn_id=turn_id,
        session_id="sess-1",
        source="claude-code",
        host="testhost",
        started_at=started_at if started_at is not None else now_ms(),
    )


# ---------- events dedupe (idx_events_dedupe) ----------


def test_add_events_keeps_distinct_same_ms_tool_calls(store: Store) -> None:
    """Two real tool_call events completing in the same turn/ms with different tool_use_id
    values must both persist — the old (turn_id, ts, kind, name) key collided on this and
    INSERT OR IGNORE silently dropped the second, undercounting tool_reliability."""
    store.upsert_turn(_turn("t1"))
    ts = now_ms()
    written = store.add_events(
        [
            Event(
                turn_id="t1",
                ts=ts,
                kind="tool_call",
                name="bash",
                ok=1,
                payload={"tool_use_id": "call-1"},
            ),
            Event(
                turn_id="t1",
                ts=ts,
                kind="tool_call",
                name="bash",
                ok=1,
                payload={"tool_use_id": "call-2"},
            ),
        ]
    )
    assert written == 2

    rows = store.events_for("t1", kind="tool_call")
    assert len(rows) == 2
    assert {e.payload["tool_use_id"] for e in rows} == {"call-1", "call-2"}


def test_add_events_still_dedupes_exact_replay(store: Store) -> None:
    """A record replayed byte-for-byte (same turn/ts/kind/name/tool_use_id) — e.g. a fleet sync
    merge or a JSONL replay — must still collapse to one row."""
    store.upsert_turn(_turn("t1"))
    ts = now_ms()
    event = Event(
        turn_id="t1", ts=ts, kind="tool_call", name="bash", ok=1, payload={"tool_use_id": "call-1"}
    )

    first = store.add_events([event])
    second = store.add_events([event])  # identical row, simulates a replay

    assert first == 1
    assert second == 0
    assert len(store.events_for("t1", kind="tool_call")) == 1


def test_add_events_dedupes_same_ms_events_with_no_tool_use_id(store: Store) -> None:
    """Sources that can't supply tool_use_id keep the old collision-prone behavior — this just
    documents that the fallback ('') still collapses on exact key match, unchanged from before."""
    store.upsert_turn(_turn("t1"))
    ts = now_ms()
    event = Event(turn_id="t1", ts=ts, kind="hook", name="PreToolUse", ok=1)

    first = store.add_events([event])
    second = store.add_events([event])

    assert first == 1
    assert second == 0


# ---------- merge_from schema skew ----------


def _old_schema_db(path: Path, *, up_to_version: int) -> Path:
    """A peer store frozen at an earlier migration — the fixture for merge_from's schema-skew
    tolerance. Built directly with the raw MIGRATIONS list rather than Store, since Store always
    migrates to SCHEMA_VERSION on open."""
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE schema_version (version INTEGER NOT NULL)")
    for version, sql in MIGRATIONS:
        if version > up_to_version:
            continue
        conn.executescript(sql)
        conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
    conn.commit()
    conn.close()
    return path


def test_merge_from_tolerates_turns_missing_later_columns(tmp_path: Path) -> None:
    """A peer box a migration behind (here: before migration 4 added ttft_ms/tools_hash_changes)
    must not abort the whole sync — merge_from used to do `INSERT OR IGNORE INTO turns
    SELECT * FROM src.turns`, which throws on a column-count mismatch the moment schemas skew."""
    src_path = _old_schema_db(tmp_path / "src.db", up_to_version=3)
    src_conn = sqlite3.connect(src_path)
    src_conn.execute(
        "INSERT INTO turns (turn_id, session_id, source, host, started_at, kpi_score) "
        "VALUES ('t1', 's1', 'claude-code', 'oldbox', 1000, 0.5)"
    )
    src_conn.commit()
    src_conn.close()

    dest = Store(tmp_path / "dest")
    try:
        counts = dest.merge_from(src_path)  # must not raise
        assert counts["turns"] == 1

        merged = dest.get_turn("t1")
        assert merged is not None
        assert merged.kpi_score == pytest.approx(0.5)
        assert merged.ttft_ms is None  # column src never had; dest fills its default (NULL)
    finally:
        dest.close()


def test_merge_from_tolerates_events_missing_dedupe_id_column(tmp_path: Path) -> None:
    """A peer box before migration 9 (dedupe_id) has a 7-column events table; the merge must
    still copy every row it can and let dest default dedupe_id to ''."""
    src_path = _old_schema_db(tmp_path / "src.db", up_to_version=8)
    src_conn = sqlite3.connect(src_path)
    src_conn.execute(
        "INSERT INTO turns (turn_id, session_id, source, host, started_at) "
        "VALUES ('t1', 's1', 'claude-code', 'oldbox', 1000)"
    )
    src_conn.execute(
        "INSERT INTO events (turn_id, ts, kind, name, duration_ms, ok, payload) "
        "VALUES ('t1', 1000, 'tool_call', 'bash', 5, 1, NULL)"
    )
    src_conn.commit()
    src_conn.close()

    dest = Store(tmp_path / "dest")
    try:
        counts = dest.merge_from(src_path)  # must not raise
        assert counts["events"] == 1

        events = dest.events_for("t1")
        assert len(events) == 1
        assert events[0].name == "bash"
    finally:
        dest.close()


# ---------- percentile covering index ----------


def test_percentile_query_plan_uses_index_not_full_scan(store: Store) -> None:
    """trailing_baseline's only call pattern (source + tier + started_at range) had no matching
    index, so every KPI scoring call did a full scan of `turns`. This mirrors the exact WHERE
    clause Store.percentile builds for that call — EXPLAIN QUERY PLAN must show the covering
    index in use, not a table scan."""
    plan = store.conn.execute(
        "EXPLAIN QUERY PLAN SELECT wall_ms FROM turns "
        "WHERE wall_ms IS NOT NULL AND tier = ? AND source = ? "
        "AND started_at >= ? AND started_at < ? AND turn_id != ? "
        "ORDER BY wall_ms",
        ("fast", "claude-code", 0, 1000, "x"),
    ).fetchall()
    plan_text = " ".join(str(tuple(row)) for row in plan)
    assert "idx_turns_percentile" in plan_text
    assert "SCAN turns" not in plan_text


def test_a_null_in_the_record_is_not_a_value_to_carry(store: Store, tmp_path: Path) -> None:
    """The rollup re-appends an enriched closing row that spells out every column and writes
    `cwd: null` for the ones it never knew. Ingesting that row must keep the opening record's
    cwd: a key present with null is absence, not information. This nulled 377 of one week's
    425 opencode turns after the absent-key guard had already landed."""
    opening = {
        "turn_id": "t-null",
        "session_id": "s",
        "source": "opencode",
        "host": "h",
        "started_at": 1,
        "cwd": "/work/here",
    }
    enriched = {
        "turn_id": "t-null",
        "session_id": "s",
        "source": "opencode",
        "host": "h",
        "started_at": 1,
        "ended_at": 2,
        "cwd": None,
        "agent_name": None,
    }
    log = tmp_path / "events-null.jsonl"
    with log.open("w", encoding="utf-8") as fh:
        for rec in (opening, enriched):
            fh.write(json.dumps({"_kind": "turn", **rec}) + "\n")
    store.ingest_jsonl(log)
    row = store.conn.execute("select cwd, ended_at from turns where turn_id='t-null'").fetchone()
    assert row["cwd"] == "/work/here" and row["ended_at"] == 2


def test_completing_record_does_not_erase_what_the_opening_record_knew(store: Store) -> None:
    """The bug this guards against emptied real columns across the whole database.

    A turn is written twice. The opening record carries cwd/agent_name/is_subagent; the
    completing record carries outcome/ended_at and does NOT repeat them. Rebuilding a Turn
    from the second record fills those with dataclass defaults, and updating every column
    unconditionally wrote the defaults over the real values.
    """
    opening = {
        "turn_id": "t-merge",
        "session_id": "sess-1",
        "source": "opencode",
        "host": "testhost",
        "started_at": now_ms(),
        "cwd": "/tmp/fixture-dir",
        "agent_name": "build",
        "is_subagent": 1,
    }
    store.upsert_turn(Turn.from_row(opening), mirror=False, present=set(opening))

    completing = {
        "turn_id": "t-merge",
        "session_id": "sess-1",
        "source": "opencode",
        "host": "testhost",
        "started_at": opening["started_at"],
        "ended_at": now_ms(),
        "outcome": "ok",
    }
    store.upsert_turn(Turn.from_row(completing), mirror=False, present=set(completing))

    row = store.conn.execute(
        "SELECT cwd, agent_name, is_subagent, outcome FROM turns WHERE turn_id='t-merge'"
    ).fetchone()
    assert row["outcome"] == "ok", "the completing record must still apply what it carries"
    assert row["cwd"] == "/tmp/fixture-dir", "cwd was erased by the completing record"
    assert row["agent_name"] == "build", "agent_name was erased by the completing record"
    # A non-optional int whose default is 0: COALESCE alone would not have saved this one.
    assert row["is_subagent"] == 1, "is_subagent was reset to its dataclass default"


def test_a_record_carrying_only_the_key_leaves_the_row_untouched(store: Store) -> None:
    opening = {
        "turn_id": "t-bare",
        "session_id": "sess-1",
        "source": "opencode",
        "host": "testhost",
        "started_at": now_ms(),
        "cwd": "/tmp/keep-me",
    }
    store.upsert_turn(Turn.from_row(opening), mirror=False, present=set(opening))
    store.upsert_turn(Turn.from_row(opening), mirror=False, present={"turn_id"})
    row = store.conn.execute("SELECT cwd FROM turns WHERE turn_id='t-bare'").fetchone()
    assert row["cwd"] == "/tmp/keep-me"


# ---------- merge_from: append-only vs mutable tables ----------
#
# merge_from used one strategy (INSERT OR IGNORE) for every table. That is correct for a
# table where a row, once written, never changes -- an old peer row and a missing local row
# are the only two cases. It is wrong for a table where the SAME primary key can be revised
# on either machine (a todo's status, a session's heartbeat): IGNORE means the update is
# silently dropped the moment the row already exists locally, no error, no count. These
# tests pin both strategies and the peer-schema fallback for the mutable one.


def _insert_judgment(
    conn: sqlite3.Connection, *, turn_id: str, verdict: str, created_at: int
) -> None:
    conn.execute(
        "INSERT INTO judgments (turn_id, judge_model, verdict, created_at) VALUES (?, ?, ?, ?)",
        (turn_id, "judge-1", verdict, created_at),
    )


def _insert_todo(
    conn: sqlite3.Connection,
    *,
    todo_id: str,
    status: str,
    updated_at: int,
    text: str = "do the thing",
    done_at: int | None = None,
) -> None:
    conn.execute(
        "INSERT INTO todos "
        "(todo_id, scope, repo, text, status, source, created_at, updated_at, done_at) "
        "VALUES (?, 'global', NULL, ?, ?, 'human', ?, ?, ?)",
        (todo_id, text, status, updated_at, updated_at, done_at),
    )


def _insert_agent_session(
    conn: sqlite3.Connection, *, session_id: str, host: str, task: str, heartbeat_at: int
) -> None:
    conn.execute(
        "INSERT INTO agent_sessions "
        "(session_id, host, task, agent, status, started_at, heartbeat_at) "
        "VALUES (?, ?, ?, 'claude-code', 'working', ?, ?)",
        (session_id, host, task, heartbeat_at, heartbeat_at),
    )


def test_merge_from_append_only_table_does_not_overwrite_existing_row(tmp_path: Path) -> None:
    """judgments is append-only (one verdict per turn_id, never revised). A peer's row for a
    turn_id dest already has must be ignored, not used to replace dest's verdict."""
    dest = Store(tmp_path / "dest")
    src = Store(tmp_path / "src")
    try:
        _insert_judgment(dest.conn, turn_id="t1", verdict="fail", created_at=1000)
        _insert_judgment(src.conn, turn_id="t1", verdict="pass", created_at=2000)
        src.close()

        counts = dest.merge_from(src.db_path)

        assert counts["judgments"] == 0
        row = dest.conn.execute("SELECT verdict FROM judgments WHERE turn_id='t1'").fetchone()
        assert row["verdict"] == "fail"
    finally:
        dest.close()


def test_merge_from_todos_newer_peer_row_lands_locally(tmp_path: Path) -> None:
    """A todo filed on spark and closed there (open -> done, newer updated_at) must overwrite
    the still-open local row -- this is now the authoritative cross-machine todo store."""
    dest = Store(tmp_path / "dest")
    src = Store(tmp_path / "src")
    try:
        _insert_todo(dest.conn, todo_id="td1", status="open", updated_at=1000)
        _insert_todo(src.conn, todo_id="td1", status="done", updated_at=2000, done_at=2000)
        src.close()

        counts = dest.merge_from(src.db_path)

        assert counts["todos"] == 1
        row = dest.conn.execute(
            "SELECT status, updated_at, done_at FROM todos WHERE todo_id='td1'"
        ).fetchone()
        assert row["status"] == "done"
        assert row["updated_at"] == 2000
        assert row["done_at"] == 2000
    finally:
        dest.close()


def test_merge_from_todos_older_peer_row_does_not_clobber_newer_local_row(tmp_path: Path) -> None:
    """The reverse of the above: a stale peer replaying an old open/updated_at=1000 row must
    not undo a local close that already happened at updated_at=5000."""
    dest = Store(tmp_path / "dest")
    src = Store(tmp_path / "src")
    try:
        _insert_todo(dest.conn, todo_id="td2", status="done", updated_at=5000, done_at=5000)
        _insert_todo(src.conn, todo_id="td2", status="open", updated_at=1000)
        src.close()

        counts = dest.merge_from(src.db_path)

        assert counts["todos"] == 0
        row = dest.conn.execute(
            "SELECT status, updated_at FROM todos WHERE todo_id='td2'"
        ).fetchone()
        assert row["status"] == "done"
        assert row["updated_at"] == 5000
    finally:
        dest.close()


def test_merge_from_todos_new_peer_row_is_inserted(tmp_path: Path) -> None:
    """A todo that only exists on the peer (no local conflict) is a plain insert."""
    dest = Store(tmp_path / "dest")
    src = Store(tmp_path / "src")
    try:
        _insert_todo(src.conn, todo_id="td3", status="open", updated_at=1000, text="new from spark")
        src.close()

        counts = dest.merge_from(src.db_path)

        assert counts["todos"] == 1
        row = dest.conn.execute("SELECT text, status FROM todos WHERE todo_id='td3'").fetchone()
        assert row["text"] == "new from spark"
        assert row["status"] == "open"
    finally:
        dest.close()


def test_merge_from_agent_sessions_newer_heartbeat_wins(tmp_path: Path) -> None:
    """Same recency rule as todos, keyed on (session_id, host) with heartbeat_at as the clock."""
    dest = Store(tmp_path / "dest")
    src = Store(tmp_path / "src")
    try:
        _insert_agent_session(
            dest.conn, session_id="sess-a", host="h1", task="old task", heartbeat_at=1000
        )
        _insert_agent_session(
            src.conn, session_id="sess-a", host="h1", task="new task", heartbeat_at=2000
        )
        src.close()

        counts = dest.merge_from(src.db_path)

        assert counts["agent_sessions"] == 1
        row = dest.conn.execute(
            "SELECT task, heartbeat_at FROM agent_sessions "
            "WHERE session_id='sess-a' AND host='h1'"
        ).fetchone()
        assert row["task"] == "new task"
        assert row["heartbeat_at"] == 2000
    finally:
        dest.close()


def test_merge_from_agent_sessions_older_heartbeat_does_not_clobber(tmp_path: Path) -> None:
    """A stale peer heartbeat must not stomp a local session that reported in more recently."""
    dest = Store(tmp_path / "dest")
    src = Store(tmp_path / "src")
    try:
        _insert_agent_session(
            dest.conn, session_id="sess-b", host="spark", task="current task", heartbeat_at=9000
        )
        _insert_agent_session(
            src.conn, session_id="sess-b", host="spark", task="stale task", heartbeat_at=1000
        )
        src.close()

        counts = dest.merge_from(src.db_path)

        assert counts["agent_sessions"] == 0
        row = dest.conn.execute(
            "SELECT task, heartbeat_at FROM agent_sessions "
            "WHERE session_id='sess-b' AND host='spark'"
        ).fetchone()
        assert row["task"] == "current task"
        assert row["heartbeat_at"] == 9000
    finally:
        dest.close()


def test_merge_from_peer_missing_mutable_tables_entirely_merges_cleanly(tmp_path: Path) -> None:
    """A peer frozen before migration 13 (which added both todos and agent_sessions) has
    neither table. merge_from must not raise and must report 0, same as the existing
    schema-skew tolerance for append-only tables."""
    src_path = _old_schema_db(tmp_path / "src.db", up_to_version=12)

    dest = Store(tmp_path / "dest")
    try:
        counts = dest.merge_from(src_path)  # must not raise

        assert counts["todos"] == 0
        assert counts["agent_sessions"] == 0
    finally:
        dest.close()


def test_merge_from_now_includes_scope_records(tmp_path: Path) -> None:
    """scope_records was silently absent from the old hardcoded table list -- a scoping pass
    on one box never reached the other. It is append-only (add_scope_record generates a fresh
    record_id per pass; no code path revises an existing row), so a plain new row from the
    peer must land, same as any other append-only table."""
    dest = Store(tmp_path / "dest")
    src = Store(tmp_path / "src")
    try:
        src.conn.execute(
            "INSERT INTO scope_records (record_id, session_id, created_at, host, tier, verdict) "
            "VALUES ('r1', 'sess-1', 1000, 'spark', 'mini', 'pass')"
        )
        src.close()

        counts = dest.merge_from(src.db_path)

        assert counts["scope_records"] == 1
        row = dest.conn.execute("SELECT verdict FROM scope_records WHERE record_id='r1'").fetchone()
        assert row["verdict"] == "pass"
    finally:
        dest.close()


# ---------- merge_from: plans (mutable) / plan_revisions (append-only) ----------


def _insert_plan(
    conn: sqlite3.Connection,
    *,
    todo_id: str,
    status: str,
    revision: int,
    updated_at: int,
    goal: str = "g",
    superseded_by: str | None = None,
) -> None:
    conn.execute(
        "INSERT INTO plans "
        "(todo_id, goal, stages, non_goals, status, superseded_by, revision, "
        " created_at, updated_at) "
        "VALUES (?, ?, '[]', '[]', ?, ?, ?, ?, ?)",
        (todo_id, goal, status, superseded_by, revision, updated_at, updated_at),
    )


def _insert_plan_revision(
    conn: sqlite3.Connection, *, todo_id: str, revision: int, changed_at: int, note: str = ""
) -> None:
    conn.execute(
        "INSERT INTO plan_revisions (todo_id, revision, snapshot, changed_by, changed_at, note) "
        "VALUES (?, ?, '{}', 'test', ?, ?)",
        (todo_id, revision, changed_at, note),
    )


def test_merge_from_plans_newer_peer_row_lands_locally(tmp_path: Path) -> None:
    """Same recency rule as todos/agent_sessions, keyed on todo_id with updated_at as the
    clock -- a plan amended on spark must overwrite a stale local copy."""
    dest = Store(tmp_path / "dest")
    src = Store(tmp_path / "src")
    try:
        _insert_plan(dest.conn, todo_id="p1", status="draft", revision=1, updated_at=1000)
        _insert_plan(src.conn, todo_id="p1", status="active", revision=2, updated_at=2000)
        src.close()

        counts = dest.merge_from(src.db_path)

        assert counts["plans"] == 1
        row = dest.conn.execute(
            "SELECT status, revision, updated_at FROM plans WHERE todo_id='p1'"
        ).fetchone()
        assert row["status"] == "active"
        assert row["revision"] == 2
        assert row["updated_at"] == 2000
    finally:
        dest.close()


def test_merge_from_plans_older_peer_row_does_not_clobber_newer_local_row(tmp_path: Path) -> None:
    dest = Store(tmp_path / "dest")
    src = Store(tmp_path / "src")
    try:
        _insert_plan(dest.conn, todo_id="p2", status="done", revision=5, updated_at=9000)
        _insert_plan(src.conn, todo_id="p2", status="draft", revision=1, updated_at=1000)
        src.close()

        counts = dest.merge_from(src.db_path)

        assert counts["plans"] == 0
        row = dest.conn.execute("SELECT status, revision FROM plans WHERE todo_id='p2'").fetchone()
        assert row["status"] == "done"
        assert row["revision"] == 5
    finally:
        dest.close()


def test_merge_from_plan_revisions_is_append_only(tmp_path: Path) -> None:
    """plan_revisions never gets rewritten -- a peer's row for a (todo_id, revision) dest
    already has must be ignored, and a genuinely new revision from the peer must land."""
    dest = Store(tmp_path / "dest")
    src = Store(tmp_path / "src")
    try:
        _insert_plan_revision(dest.conn, todo_id="p3", revision=1, changed_at=1000, note="local")
        _insert_plan_revision(src.conn, todo_id="p3", revision=1, changed_at=1000, note="peer-dupe")
        _insert_plan_revision(src.conn, todo_id="p3", revision=2, changed_at=2000, note="peer-new")
        src.close()

        counts = dest.merge_from(src.db_path)

        assert counts["plan_revisions"] == 1  # only the new revision, not the duplicate
        rows = dest.conn.execute(
            "SELECT revision, note FROM plan_revisions WHERE todo_id='p3' ORDER BY revision"
        ).fetchall()
        assert [(r["revision"], r["note"]) for r in rows] == [(1, "local"), (2, "peer-new")]
    finally:
        dest.close()


def test_merge_from_peer_missing_plans_tables_entirely_merges_cleanly(tmp_path: Path) -> None:
    """A peer frozen before migration 16 has neither table -- same schema-skew tolerance as
    the pre-13 todos/agent_sessions case."""
    src_path = _old_schema_db(tmp_path / "src.db", up_to_version=15)

    dest = Store(tmp_path / "dest")
    try:
        counts = dest.merge_from(src_path)  # must not raise

        assert counts["plans"] == 0
        assert counts["plan_revisions"] == 0
    finally:
        dest.close()
