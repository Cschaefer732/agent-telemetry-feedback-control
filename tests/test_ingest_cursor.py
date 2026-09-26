"""Regression coverage for two coupled bugs in the rollup ingest path:

1. `add_probe` used to INSERT unconditionally, so any replay of an already-seen probe record
   (JSONL replay, a fleet sync merge) duplicated it. Fixed with a UNIQUE(host, kind, ts) index
   and INSERT OR IGNORE, same pattern as events/host_metrics.
2. `Store.ingest_jsonl` used to replay a JSONL file's entire contents on every call, so
   `cmd_rollup` reprocessed the whole history every run — O(total history) per rollup, and the
   direct cause of #1's unbounded growth in practice. Fixed with a persisted per-file byte-offset
   cursor (`ingest_cursor`) so a rerun only reads what was appended since the last call.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from flightdeck.models import Probe
from flightdeck.schema import MIGRATIONS, SCHEMA_VERSION, migrate
from flightdeck.store import Store

# ---------- probes dedupe ----------


def test_add_probe_is_idempotent_on_replay(tmp_path: Path) -> None:
    store = Store(tmp_path)
    try:
        probe = Probe(ts=100, host="h", kind="judge_queue", ok=1, total=1, detail={})
        store.add_probe(probe, mirror=False)
        store.add_probe(probe, mirror=False)  # simulates replaying the same JSONL record twice

        count = store.conn.execute(
            "SELECT COUNT(*) FROM probes WHERE host='h' AND kind='judge_queue' AND ts=100"
        ).fetchone()[0]
        assert count == 1
    finally:
        store.close()


def test_add_probe_distinguishes_kind_and_host(tmp_path: Path) -> None:
    """The dedupe key is (host, kind, ts) together — two legitimately different probes that
    happen to share a timestamp must both survive."""
    store = Store(tmp_path)
    try:
        store.add_probe(Probe(ts=100, host="h", kind="judge_queue", ok=1, total=1, detail={}))
        store.add_probe(Probe(ts=100, host="h", kind="hook_liveness", ok=1, total=1, detail={}))
        store.add_probe(Probe(ts=100, host="other", kind="judge_queue", ok=1, total=1, detail={}))

        count = store.conn.execute("SELECT COUNT(*) FROM probes").fetchone()[0]
        assert count == 3
    finally:
        store.close()


def test_probes_dedupe_migration_collapses_existing_duplicates(tmp_path: Path) -> None:
    """A box that ran the pre-fix code for a while can already have exact-duplicate probe rows
    (same host/kind/ts) sitting in `probes` by the time this migration ships. Creating a UNIQUE
    index on a table that still has duplicates raises sqlite3.IntegrityError — that must not
    happen; the migration has to collapse them first."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE schema_version (version INTEGER NOT NULL)")
    for version, sql in MIGRATIONS:
        if version >= 7:
            continue
        conn.executescript(sql)
        conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
    conn.commit()

    # The bug this migration fixes, reproduced directly: two exact-duplicate rows.
    for _ in range(2):
        conn.execute(
            "INSERT INTO probes (ts, host, kind, ok, total, detail) "
            "VALUES (100, 'h', 'judge_queue', 1, 1, NULL)"
        )
    conn.commit()

    version = migrate(conn)  # must not raise

    assert version == SCHEMA_VERSION
    count = conn.execute(
        "SELECT COUNT(*) FROM probes WHERE host='h' AND kind='judge_queue' AND ts=100"
    ).fetchone()[0]
    assert count == 1
    indexes = {row[1] for row in conn.execute("PRAGMA index_list(probes)")}
    assert "idx_probes_dedupe" in indexes


# ---------- ingest watermark ----------


def _write(path: Path, *lines: str) -> None:
    with path.open("a", encoding="utf-8") as fh:
        for line in lines:
            fh.write(line + "\n")


def test_ingest_jsonl_only_reads_new_bytes_on_rerun(tmp_path: Path) -> None:
    store = Store(tmp_path)
    try:
        log = tmp_path / "events-2026-08-15.jsonl"
        _write(
            log,
            '{"_kind":"turn","turn_id":"t1","session_id":"s1","source":"claude-code",'
            '"host":"h1","started_at":1000}',
        )
        first = store.ingest_jsonl(log)
        assert first == {"turn": 1}

        # Nothing new appended: a rerun must not re-read the file's existing content.
        second = store.ingest_jsonl(log)
        assert second == {}

        _write(
            log,
            '{"_kind":"turn","turn_id":"t2","session_id":"s1","source":"claude-code",'
            '"host":"h1","started_at":2000}',
        )
        third = store.ingest_jsonl(log)
        assert third == {"turn": 1}  # only the newly appended record, not t1 again

        turn_ids = {r[0] for r in store.conn.execute("SELECT turn_id FROM turns")}
        assert turn_ids == {"t1", "t2"}
    finally:
        store.close()


# ---------- tier derivation on ingest ----------


def test_ingest_jsonl_derives_tier_from_model_when_missing(tmp_path: Path) -> None:
    """The Go/TS emitters write `model` but no wire point sets `tier` yet — ingest_jsonl is the
    one place every crush/opencode turn record passes through sqlite, so it's the fallback."""
    store = Store(tmp_path)
    try:
        log = tmp_path / "events-2026-08-21.jsonl"
        _write(
            log,
            '{"_kind":"turn","turn_id":"t1","session_id":"s1","source":"crush",'
            '"host":"spark","started_at":1000,"model":"qwen3.8:27b"}',
        )
        store.ingest_jsonl(log)
        turn = store.get_turn("t1")
        assert turn is not None
        assert turn.tier == "fast"
    finally:
        store.close()


def test_ingest_jsonl_leaves_explicit_tier_alone(tmp_path: Path) -> None:
    """A record that already carries a tier (a future emitter that sets it, or a replayed
    correction) must not be overwritten by the model-based guess."""
    store = Store(tmp_path)
    try:
        log = tmp_path / "events-2026-08-21.jsonl"
        _write(
            log,
            '{"_kind":"turn","turn_id":"t1","session_id":"s1","source":"crush",'
            '"host":"spark","started_at":1000,"model":"qwen3.8:27b","tier":"frontier"}',
        )
        store.ingest_jsonl(log)
        turn = store.get_turn("t1")
        assert turn is not None
        assert turn.tier == "frontier"
    finally:
        store.close()


def test_ingest_jsonl_leaves_tier_null_for_an_unmapped_model(tmp_path: Path) -> None:
    store = Store(tmp_path)
    try:
        log = tmp_path / "events-2026-08-21.jsonl"
        _write(
            log,
            '{"_kind":"turn","turn_id":"t1","session_id":"s1","source":"crush",'
            '"host":"spark","started_at":1000,"model":"some-unmapped-model"}',
        )
        store.ingest_jsonl(log)
        turn = store.get_turn("t1")
        assert turn is not None
        assert turn.tier is None
    finally:
        store.close()


def test_ingest_jsonl_persists_cursor_across_store_instances(tmp_path: Path) -> None:
    log = tmp_path / "events-2026-08-15.jsonl"
    _write(
        log,
        '{"_kind":"turn","turn_id":"t1","session_id":"s1","source":"claude-code",'
        '"host":"h1","started_at":1000}',
    )
    store1 = Store(tmp_path)
    store1.ingest_jsonl(log)
    store1.close()

    # A fresh Store opening the same directory (e.g. the next rollup run's process) must resume
    # from the persisted cursor, not from byte 0.
    store2 = Store(tmp_path)
    try:
        counts = store2.ingest_jsonl(log)
        assert counts == {}
    finally:
        store2.close()


def test_ingest_jsonl_does_not_advance_past_an_incomplete_trailing_line(tmp_path: Path) -> None:
    """A writer's line is only safe to count once the trailing newline lands — otherwise a
    rollup racing an in-flight append could read a torn line and then never revisit it once the
    write completes, because a byte-offset cursor by definition never looks backward."""
    store = Store(tmp_path)
    try:
        log = tmp_path / "events-2026-08-15.jsonl"
        _write(
            log,
            '{"_kind":"turn","turn_id":"t1","session_id":"s1","source":"claude-code",'
            '"host":"h1","started_at":1000}',
        )
        # Append a line with no trailing newline yet — simulates a writer mid-flush.
        with log.open("a", encoding="utf-8") as fh:
            fh.write('{"_kind":"turn","turn_id":"t2","session_id":"s1","source":"claude-code"')

        counts = store.ingest_jsonl(log)
        assert counts == {"turn": 1}  # only t1; t2's line isn't complete yet
        assert store.get_turn("t2") is None

        # The writer finishes the line.
        with log.open("a", encoding="utf-8") as fh:
            fh.write(',"host":"h1","started_at":2000}\n')

        counts = store.ingest_jsonl(log)
        assert counts == {"turn": 1}  # now t2 is picked up
        assert store.get_turn("t2") is not None
    finally:
        store.close()
