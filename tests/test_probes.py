from __future__ import annotations

import datetime as dt
import json
import sqlite3
from pathlib import Path

import pytest

from flightdeck import probes
from flightdeck.models import Event, ScopeRecord, Turn
from flightdeck.scope_ingest import gate_log_path
from flightdeck.store import Store, now_ms


@pytest.fixture
def store(tmp_path: Path) -> Store:
    s = Store(tmp_path / "store")
    yield s
    s.close()


def _turn(turn_id: str, source: str, started_at: int, *, flagged: int = 0, judged: int = 0) -> Turn:
    return Turn(
        turn_id=turn_id,
        session_id="sess-1",
        source=source,
        host="testhost",
        started_at=started_at,
        flagged=flagged,
        judged=judged,
    )


# ---------- symlink probe ----------


def test_probe_symlinks_ok(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("hi")
    link = tmp_path / "link"
    link.symlink_to(target)

    probe = probes.probe_symlinks({link: target}, host="h")

    assert probe.kind == "symlink"
    assert probe.ok == 1
    assert probe.total == 1
    assert probe.detail[str(link)]["status"] == "ok"


def test_probe_symlinks_wrong_target(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("hi")
    other = tmp_path / "other.txt"
    other.write_text("bye")
    link = tmp_path / "link"
    link.symlink_to(other)

    probe = probes.probe_symlinks({link: target}, host="h")

    assert probe.ok == 0
    assert probe.detail[str(link)]["status"] == "wrong_target"


def test_probe_symlinks_broken(tmp_path: Path) -> None:
    target = tmp_path / "missing_target.txt"
    link = tmp_path / "link"
    link.symlink_to(target)

    probe = probes.probe_symlinks({link: target}, host="h")

    assert probe.ok == 0
    assert probe.detail[str(link)]["status"] == "broken"


def test_probe_symlinks_not_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("hi")
    link = tmp_path / "link"
    link.write_text("i am a stale real file, not a symlink")

    probe = probes.probe_symlinks({link: target}, host="h")

    assert probe.ok == 0
    assert probe.detail[str(link)]["status"] == "not_symlink"


def test_probe_symlinks_missing(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("hi")
    link = tmp_path / "does_not_exist"

    probe = probes.probe_symlinks({link: target}, host="h")

    assert probe.ok == 0
    assert probe.detail[str(link)]["status"] == "missing"


def test_probe_symlinks_symlinked_parent_false_positive(tmp_path: Path) -> None:
    """macOS-style /tmp vs /private/tmp: a real dir, a symlink to it, and a link *into* the
    symlinked path must still classify ok once both sides are resolve()d."""
    real_dir = tmp_path / "real_dir"
    real_dir.mkdir()
    target = real_dir / "target.txt"
    target.write_text("hi")

    parent_link = tmp_path / "parent_link"
    parent_link.symlink_to(real_dir)

    link = tmp_path / "link"
    link.symlink_to(parent_link / "target.txt")  # resolves through the symlinked parent

    # expected target given via the symlinked-parent path, same as a config file would specify
    expected_target = parent_link / "target.txt"

    probe = probes.probe_symlinks({link: expected_target}, host="h")

    assert probe.detail[str(link)]["status"] == "ok"
    assert probe.ok == 1


def test_probe_symlinks_multiple_mixed(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("hi")
    ok_link = tmp_path / "ok_link"
    ok_link.symlink_to(target)
    missing_link = tmp_path / "missing_link"

    probe = probes.probe_symlinks({ok_link: target, missing_link: target}, host="h")

    assert probe.total == 2
    assert probe.ok == 1


# ---------- migration probe ----------


def _make_schema_version_db(path: Path, version: int) -> None:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE schema_version (version INTEGER NOT NULL)")
    conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
    conn.commit()
    conn.close()


def _make_goose_db(path: Path, version_id: int) -> None:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE goose_db_version (id INTEGER PRIMARY KEY, version_id INTEGER)")
    conn.execute("INSERT INTO goose_db_version (version_id) VALUES (?)", (version_id,))
    conn.commit()
    conn.close()


def test_probe_migrations_schema_version_table(tmp_path: Path) -> None:
    db = tmp_path / "a.db"
    _make_schema_version_db(db, 3)

    probe = probes.probe_migrations({"a": db}, {"a": 3}, host="h")

    assert probe.ok == 1
    assert probe.total == 1
    assert probe.detail["a"]["status"] == "ok"
    assert probe.detail["a"]["version"] == 3


def test_probe_migrations_goose_db_version_table(tmp_path: Path) -> None:
    db = tmp_path / "b.db"
    _make_goose_db(db, 5)

    probe = probes.probe_migrations({"b": db}, {"b": 5}, host="h")

    assert probe.ok == 1
    assert probe.detail["b"]["version"] == 5


def test_probe_migrations_stale_version(tmp_path: Path) -> None:
    db = tmp_path / "c.db"
    _make_schema_version_db(db, 1)

    probe = probes.probe_migrations({"c": db}, {"c": 3}, host="h")

    assert probe.ok == 0
    assert probe.detail["c"]["status"] == "stale"


def test_probe_migrations_absent_db(tmp_path: Path) -> None:
    db = tmp_path / "does_not_exist.db"

    probe = probes.probe_migrations({"d": db}, {"d": 1}, host="h")

    assert probe.ok == 0
    assert probe.total == 1
    assert probe.detail["d"]["status"] == "absent"


def test_probe_migrations_multiple(tmp_path: Path) -> None:
    good_db = tmp_path / "good.db"
    _make_schema_version_db(good_db, 3)
    missing_db = tmp_path / "missing.db"

    probe = probes.probe_migrations(
        {"good": good_db, "missing": missing_db}, {"good": 3, "missing": 1}, host="h"
    )

    assert probe.total == 2
    assert probe.ok == 1
    assert probe.detail["missing"]["status"] == "absent"


# ---------- hook liveness ----------


def test_probe_hook_liveness_present_and_missing(store: Store) -> None:
    turn = _turn("t1", "claude-code", now_ms())
    store.upsert_turn(turn)
    store.add_events([Event(turn_id="t1", ts=now_ms(), kind="hook", name="PreToolUse", ok=1)])

    probe = probes.probe_hook_liveness(
        store, window_hours=24, expected_hooks=["PreToolUse", "PostToolUse"], host="h"
    )

    assert probe.total == 2
    assert probe.ok == 1
    assert probe.detail["PreToolUse"]["status"] == "ok"
    assert probe.detail["PostToolUse"]["status"] == "silent"


def test_probe_hook_liveness_outside_window_counts_as_silent(store: Store) -> None:
    turn = _turn("t1", "claude-code", now_ms())
    store.upsert_turn(turn)
    stale_ts = now_ms() - 48 * 3600_000
    store.add_events([Event(turn_id="t1", ts=stale_ts, kind="hook", name="PreToolUse", ok=1)])

    probe = probes.probe_hook_liveness(
        store, window_hours=24, expected_hooks=["PreToolUse"], host="h"
    )

    assert probe.ok == 0
    assert probe.detail["PreToolUse"]["status"] == "silent"


def test_probe_hook_liveness_claude_code_collector_evidence(store: Store) -> None:
    """collect_claude.py never emits kind="hook" events (see its handle() dispatch) — only the
    Go emitter does. A healthy Claude-Code-only box must still read green, using the evidence
    that collector actually writes: a turn with a tool_call event and captured text."""
    from flightdeck.models import TextBlob

    turn = _turn("t1", "claude-code", now_ms())
    store.upsert_turn(turn)
    store.add_events([Event(turn_id="t1", ts=now_ms(), kind="tool_call", ok=1)])
    store.add_texts(
        [TextBlob(turn_id="t1", kind="prompt", seq=0, body="hi", expires_at=now_ms() + 1000)]
    )

    probe = probes.probe_hook_liveness(
        store,
        window_hours=24,
        expected_hooks=["UserPromptSubmit", "PreToolUse", "PostToolUse", "Stop"],
        host="h",
    )

    assert probe.ok == probe.total == 4
    assert probe.healthy
    for hook in ("UserPromptSubmit", "PreToolUse", "PostToolUse", "Stop"):
        assert probe.detail[hook]["status"] == "ok"
        assert probe.detail[hook]["evidence"] == "claude_code_collector"


def test_probe_hook_liveness_claude_code_without_tool_call_stays_silent(store: Store) -> None:
    """A turn alone isn't liveness evidence — collect_claude.py writes a turn row for nearly
    every hook including ones that don't prove the pipeline is doing anything (e.g. a bare
    UserPromptSubmit). Requiring a tool_call + text keeps this probe meaningful."""
    store.upsert_turn(_turn("t1", "claude-code", now_ms()))

    probe = probes.probe_hook_liveness(
        store, window_hours=24, expected_hooks=["UserPromptSubmit"], host="h"
    )

    assert probe.ok == 0
    assert probe.detail["UserPromptSubmit"]["status"] == "silent"
    assert probe.detail["UserPromptSubmit"]["evidence"] is None


def test_probe_hook_liveness_go_emitter_still_works(store: Store) -> None:
    """The fix must not regress the Go emitter's existing kind='hook' evidence path."""
    turn = _turn("t1", "crush", now_ms())
    store.upsert_turn(turn)
    store.add_events([Event(turn_id="t1", ts=now_ms(), kind="hook", name="PreToolUse", ok=1)])

    probe = probes.probe_hook_liveness(
        store, window_hours=24, expected_hooks=["PreToolUse"], host="h"
    )

    assert probe.ok == probe.total == 1
    assert probe.detail["PreToolUse"]["evidence"] == "hook_event"


# ---------- collector heartbeat ----------


def test_probe_collector_heartbeat_silent_source(store: Store) -> None:
    store.upsert_turn(_turn("t1", "claude-code", now_ms()))

    probe = probes.probe_collector_heartbeat(
        store, window_hours=24, expected_sources=["claude-code", "crush"], host="h"
    )

    assert probe.total == 2
    assert probe.ok == 1
    assert probe.detail["claude-code"]["status"] == "ok"
    assert probe.detail["crush"]["status"] == "silent"


# ---------- judge queue ----------


def test_probe_judge_queue_under_cap(store: Store) -> None:
    for i in range(3):
        store.upsert_turn(_turn(f"t{i}", "claude-code", now_ms(), flagged=1, judged=0))

    probe = probes.probe_judge_queue(store, window_hours=24, host="h", cap=20)

    assert probe.ok == 1
    assert probe.total == 1
    assert probe.detail["backlog"] == 3
    assert probe.detail["cap"] == 20


def test_probe_judge_queue_over_cap(store: Store) -> None:
    for i in range(5):
        store.upsert_turn(_turn(f"t{i}", "claude-code", now_ms(), flagged=1, judged=0))

    probe = probes.probe_judge_queue(store, window_hours=24, host="h", cap=3)

    assert probe.ok == 0
    assert probe.detail["backlog"] == 5


def test_probe_judge_queue_counts_drops(store: Store) -> None:
    store.upsert_turn(_turn("t1", "claude-code", now_ms(), flagged=1, judged=0))
    store.add_events([Event(turn_id="t1", ts=now_ms(), kind="queue", name="judge", ok=0)])

    probe = probes.probe_judge_queue(store, window_hours=24, host="h", cap=20)

    assert probe.detail["dropped"] == 1


def test_probe_judge_queue_ignores_judged(store: Store) -> None:
    store.upsert_turn(_turn("t1", "claude-code", now_ms(), flagged=1, judged=1))

    probe = probes.probe_judge_queue(store, window_hours=24, host="h", cap=20)

    assert probe.detail["backlog"] == 0
    assert probe.ok == 1


# ---------- never raises ----------


def test_probe_symlinks_never_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def boom(self: Path) -> Path:
        raise OSError("simulated resolve failure")

    monkeypatch.setattr(Path, "resolve", boom)
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "target")

    probe = probes.probe_symlinks({link: tmp_path / "target"}, host="h")

    assert probe.ok == 0
    assert probe.total == 0
    assert "OSError" in probe.detail.get("error", "")


def test_probe_hook_liveness_never_raises(monkeypatch: pytest.MonkeyPatch, store: Store) -> None:
    class _BoomConn:
        def execute(self, *args: object, **kwargs: object) -> None:
            raise sqlite3.OperationalError("simulated db failure")

        def close(self) -> None:
            pass  # store fixture teardown calls this before monkeypatch restores the real conn

    # sqlite3.Connection is an immutable C type and can't be attribute-patched; swap the whole
    # connection object on the Store instance instead (monkeypatch restores the real one after).
    monkeypatch.setattr(store, "conn", _BoomConn())

    probe = probes.probe_hook_liveness(store, expected_hooks=["PreToolUse"], host="h")

    assert probe.ok == 0
    assert probe.total == 0
    assert probe.detail["error"] == "OperationalError"


def test_probe_migrations_never_raises_on_corrupt_db(tmp_path: Path) -> None:
    db = tmp_path / "corrupt.db"
    db.write_bytes(b"not a sqlite file at all")

    probe = probes.probe_migrations({"c": db}, {"c": 1}, host="h")

    # per-db failures are caught inside the loop, not by the outer guard
    assert probe.ok == 0
    assert probe.total == 1
    assert probe.detail["c"]["status"] == "error"


# ---------- default_config ----------


def test_default_config_does_not_raise() -> None:
    config = probes.default_config()
    assert config.expected_sources
    assert config.expected_hooks
    assert config.judge_backlog_cap == 20


def test_default_config_expected_hooks_matches_collect_claude_dispatch() -> None:
    """collect_claude.handle() dispatches on exactly these hook_event_name values (see its
    if/elif chain). SessionStart/SubagentStart are received but no-op there, so a hook_liveness
    probe expecting them would be permanently "silent" regardless of whether hooks are wired."""
    config = probes.default_config()

    assert set(config.expected_hooks) == {
        "UserPromptSubmit",
        "PreToolUse",
        "PostToolUse",
        "PostToolUseFailure",
        "Stop",
    }
    assert "SessionStart" not in config.expected_hooks
    assert "SubagentStart" not in config.expected_hooks


def test_default_config_expected_sources_matches_models_sources() -> None:
    from flightdeck.models import SOURCES

    assert probes.default_config().expected_sources == list(SOURCES)


# ---------- run_all ----------


def test_run_all_writes_every_probe(store: Store, tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("hi")
    link = tmp_path / "link"
    link.symlink_to(target)

    db = tmp_path / "db.db"
    _make_schema_version_db(db, 3)

    store.upsert_turn(_turn("t1", "claude-code", now_ms()))
    store.add_events([Event(turn_id="t1", ts=now_ms(), kind="hook", name="PreToolUse", ok=1)])

    config = probes.ProbeConfig(
        symlinks={link: target},
        databases={"db": db},
        expected_versions={"db": 3},
        expected_hooks=["PreToolUse"],
        expected_sources=["claude-code"],
        judge_backlog_cap=20,
    )

    result = probes.run_all(store, config)

    assert len(result) == 6
    assert {p.kind for p in result} == {
        "symlink",
        "migration",
        "hook_liveness",
        "collector_heartbeat",
        "judge_queue",
        "scope_liveness",
    }

    rows = store.conn.execute("SELECT kind FROM probes ORDER BY probe_id").fetchall()
    assert len(rows) == 6
    assert {r[0] for r in rows} == {p.kind for p in result}


# ---------- scope liveness probe ----------


def _write_gate_log(directory: Path, entries: list[dict]) -> None:
    path = gate_log_path(directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for entry in entries:
            fh.write(json.dumps(entry) + "\n")


def _gate_entry(ts: dt.datetime, *, tier: str = "mini") -> dict:
    return {
        "ts": ts.isoformat(),
        "session_id": "sess-1",
        "tier": tier,
        "prompt_sha1": "deadbeef",
        "prompt_len": 42,
    }


def _scope_record(created_at: int, *, verdict: str = "silent") -> ScopeRecord:
    return ScopeRecord(
        record_id=f"gate-{created_at}",
        session_id="sess-1",
        created_at=created_at,
        host="testhost",
        tier="mini",
        verdict=verdict,
    )


def test_probe_scope_liveness_healthy_when_advancing_and_ingesting(store: Store) -> None:
    now = dt.datetime.now(dt.UTC)
    _write_gate_log(store.directory, [_gate_entry(now)])
    store.add_scope_record(_scope_record(int(now.timestamp())))

    probe = probes.probe_scope_liveness(store, host="h")

    assert probe.kind == "scope_liveness"
    assert probe.ok == 2
    assert probe.total == 2
    assert probe.healthy
    assert probe.detail["checks"]["gate_log"]["status"] == "ok"
    assert probe.detail["checks"]["ingestion"]["status"] == "ok"


def test_probe_scope_liveness_ingestion_broken(store: Store) -> None:
    """Gate log fresh, nothing landed in scope_records: the hook is fine, ingest is not."""
    now = dt.datetime.now(dt.UTC)
    _write_gate_log(store.directory, [_gate_entry(now)])

    probe = probes.probe_scope_liveness(store, host="h")

    assert probe.detail["checks"]["gate_log"]["status"] == "ok"
    assert probe.detail["checks"]["ingestion"]["status"] == "fail"
    assert not probe.healthy


def test_probe_scope_liveness_hook_broken_when_machine_active_but_gate_silent(
    store: Store,
) -> None:
    """Turns are being recorded (the machine is in active use) but the gate log has not
    moved: this is the critical case -- a broken hook, and this is the one signal available
    that actually distinguishes it from an idle box."""
    store.upsert_turn(_turn("t1", "claude-code", now_ms()))

    probe = probes.probe_scope_liveness(store, host="h")

    assert probe.detail["checks"]["gate_log"]["status"] == "fail"
    assert probe.detail["checks"]["ingestion"]["status"] == "not_applicable"
    assert not probe.healthy


def test_probe_scope_liveness_inconclusive_when_idle_does_not_fail(store: Store) -> None:
    """No gate decisions AND no turns: cannot tell a broken hook from a quiet week with the
    data this store has. Must report the uncertainty honestly but must NOT fail the check --
    that would page someone every idle weekend."""
    probe = probes.probe_scope_liveness(store, host="h")

    assert probe.detail["checks"]["gate_log"]["status"] == "inconclusive"
    assert probe.detail["checks"]["ingestion"]["status"] == "not_applicable"
    assert probe.healthy
    assert "cannot" in probe.detail["limitation"].lower()


def test_probe_scope_liveness_stale_gate_log_outside_window_with_activity_fails(
    store: Store,
) -> None:
    """A gate log that exists but stopped advancing days ago, on a box that is still being
    used, is exactly the 'weeks-broken hook' shape this probe exists to catch."""
    stale = dt.datetime.now(dt.UTC) - dt.timedelta(hours=48)
    _write_gate_log(store.directory, [_gate_entry(stale)])
    store.upsert_turn(_turn("t1", "claude-code", now_ms()))

    probe = probes.probe_scope_liveness(store, window_hours=24, host="h")

    assert probe.detail["checks"]["gate_log"]["status"] == "fail"
    assert not probe.healthy


def test_probe_scope_liveness_never_raises(monkeypatch: pytest.MonkeyPatch, store: Store) -> None:
    def boom(*args, **kwargs):
        raise RuntimeError("gate log parse blew up")

    monkeypatch.setattr(probes, "_last_gate_entry_epoch", boom)

    probe = probes.probe_scope_liveness(store, host="h")

    assert probe.ok == 0
    assert probe.total == 0
    assert probe.detail["error"] == "RuntimeError"
