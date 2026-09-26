from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from flightdeck import ledger
from flightdeck.store import Store, now_ms

_HOOK = Path(__file__).resolve().parents[1] / "integration" / "claude-code" / "session-heartbeat.py"


@pytest.fixture
def store(tmp_path: Path) -> Store:
    s = Store(tmp_path / "store")
    yield s
    s.close()


# ---------- beat() upsert ----------


def test_beat_upserts_not_duplicates(store: Store) -> None:
    ledger.beat(store, session_id="s1", cwd="/tmp", host="h1", now=1_000)
    ledger.beat(store, session_id="s1", cwd="/tmp", host="h1", now=2_000)
    rows = store.conn.execute("SELECT * FROM agent_sessions").fetchall()
    assert len(rows) == 1
    assert rows[0]["heartbeat_at"] == 2_000
    assert rows[0]["started_at"] == 1_000  # started_at is sticky across re-beats


def test_beat_keyed_by_session_and_host(store: Store) -> None:
    ledger.beat(store, session_id="s1", cwd="/tmp", host="h1", now=1_000)
    ledger.beat(store, session_id="s1", cwd="/tmp", host="h2", now=1_000)
    rows = store.conn.execute("SELECT * FROM agent_sessions").fetchall()
    assert len(rows) == 2


def test_beat_preserves_goal_task_when_omitted(store: Store) -> None:
    ledger.beat(store, session_id="s1", cwd="/tmp", host="h1", goal="ship ledger", now=1_000)
    ledger.beat(store, session_id="s1", cwd="/tmp", host="h1", now=2_000)
    row = store.conn.execute("SELECT goal FROM agent_sessions").fetchone()
    assert row["goal"] == "ship ledger"


def test_beat_derives_git_context(store: Store, tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "checkout", "-q", "-b", "main"], cwd=repo, check=True)
    subprocess.run(
        ["git", "remote", "add", "origin", "git@github.com:acme/widgets.git"],
        cwd=repo,
        check=True,
    )
    ledger.beat(store, session_id="s1", cwd=str(repo), host="h1", now=1_000)
    row = store.conn.execute("SELECT * FROM agent_sessions").fetchone()
    assert row["repo"] == str(repo.resolve())
    assert row["branch"] == "main"
    assert row["github_remote"] == "acme/widgets"


def test_beat_outside_git_leaves_context_null(store: Store, tmp_path: Path) -> None:
    ledger.beat(store, session_id="s1", cwd=str(tmp_path), host="h1", now=1_000)
    row = store.conn.execute("SELECT * FROM agent_sessions").fetchone()
    assert row["repo"] is None
    assert row["github_remote"] is None


# ---------- liveness vs status ----------


def test_stale_session_excluded_from_active(store: Store) -> None:
    now = now_ms()
    old = now - (ledger.DEFAULT_STALE_SECONDS + 60) * 1000
    ledger.beat(store, session_id="s1", cwd="/tmp", host="h1", now=old)
    active = ledger.active_sessions(store, now=now)
    assert active == []


def test_working_status_with_old_heartbeat_is_not_active(store: Store) -> None:
    """The core design rule: status is intent, liveness is time. A 'working' row does not get
    to claim it is active just because it says so."""
    now = now_ms()
    old = now - (ledger.DEFAULT_STALE_SECONDS + 60) * 1000
    ledger.beat(store, session_id="s1", cwd="/tmp", host="h1", status="working", now=old)
    active = ledger.active_sessions(store, now=now)
    assert active == []
    all_rows = store.conn.execute("SELECT status FROM agent_sessions").fetchall()
    assert all_rows[0]["status"] == "working"


def test_fresh_session_is_active_with_age(store: Store) -> None:
    now = now_ms()
    ledger.beat(store, session_id="s1", cwd="/tmp", host="h1", now=now - 5_000)
    active = ledger.active_sessions(store, now=now)
    assert len(active) == 1
    assert active[0]["heartbeat_age_seconds"] == 5.0


# ---------- render() ----------


def test_render_empty_store_is_well_formed(store: Store) -> None:
    view = ledger.render(store)
    assert view["sessions"] == []
    assert view["total_sessions_ever"] == 0
    assert view["data_present"] is False
    assert view["errors"] == []
    assert view["todos"] == {"local": [], "global": []}


def test_render_distinguishes_no_data_from_no_active_sessions(store: Store) -> None:
    now = now_ms()
    old = now - (ledger.DEFAULT_STALE_SECONDS + 60) * 1000
    ledger.beat(store, session_id="s1", cwd="/tmp", host="h1", now=old)
    view = ledger.render(store, now=now)
    assert view["sessions"] == []
    assert view["total_sessions_ever"] == 1
    assert view["data_present"] is True  # there IS data, it's just all stale


def test_render_includes_active_session_and_repo(store: Store) -> None:
    now = now_ms()
    ledger.beat(
        store,
        session_id="s1",
        cwd="/tmp",
        host="h1",
        goal="g",
        task="t",
        status="working",
        now=now,
    )
    view = ledger.render(store, now=now)
    assert len(view["sessions"]) == 1
    assert view["sessions"][0]["goal"] == "g"


def test_render_pulls_recent_events(store: Store) -> None:
    from flightdeck.models import Event, Turn

    now = now_ms()
    store.upsert_turn(
        Turn(turn_id="t1", session_id="s1", source="claude-code", host="h1", started_at=now)
    )
    store.add_events([Event(turn_id="t1", ts=now, kind="tool_call", name="Read")])
    view = ledger.render(store, now=now)
    assert len(view["recent_events"]) == 1
    assert view["recent_events"][0]["name"] == "Read"


def test_render_never_raises_on_broken_store(tmp_path: Path) -> None:
    s = Store(tmp_path / "store2")
    s.conn.execute("DROP TABLE todos")
    view = ledger.render(s)
    assert view["todos"] == {"local": [], "global": []}
    assert any("todos" in e for e in view["errors"])
    s.close()


# ---------- todos ----------


def test_todo_add_list_done(store: Store) -> None:
    tid = ledger.add_todo(store, text="fix the thing", scope="local", repo="/repo", source="s1")
    open_local = ledger.list_todos(store, scope="local")
    assert len(open_local) == 1
    assert open_local[0]["todo_id"] == tid
    assert ledger.done_todo(store, tid) is True
    assert ledger.list_todos(store, scope="local") == []
    assert ledger.list_todos(store, scope="local", status="done")[0]["todo_id"] == tid


def test_todo_done_unknown_id_is_noop(store: Store) -> None:
    assert ledger.done_todo(store, "does-not-exist") is False


def test_todo_global_ignores_repo(store: Store) -> None:
    ledger.add_todo(store, text="global thing", scope="global", repo="/repo")
    row = ledger.list_todos(store, scope="global")[0]
    assert row["repo"] is None


def test_todo_invalid_scope_raises(store: Store) -> None:
    with pytest.raises(ValueError):
        ledger.add_todo(store, text="x", scope="bogus")


# ---------- todo: key idempotency (migration 16) ----------


def test_todo_add_idempotent_key_returns_existing_id(store: Store) -> None:
    """A probe that re-files the same failure every session start must not create
    duplicates -- (source, key) is the dedupe key, not a full text match."""
    id1 = ledger.add_todo(store, text="first filing", source="probe:x", key="dup")
    id2 = ledger.add_todo(store, text="second filing, different text", source="probe:x", key="dup")
    assert id1 == id2
    rows = store.conn.execute("SELECT COUNT(*) FROM todos WHERE source='probe:x'").fetchone()
    assert rows[0] == 1


def test_todo_add_without_key_never_dedupes(store: Store) -> None:
    id1 = ledger.add_todo(store, text="a", source="human")
    id2 = ledger.add_todo(store, text="b", source="human")
    assert id1 != id2


def test_todo_add_keyed_reopens_done_row(store: Store) -> None:
    """A key names a condition. Closed once, recurring later, it must come back as the same
    open todo -- the unique index has no status clause, so a plain DO NOTHING would leave a
    recurring probe failure unfilable forever after its first recovery."""
    id1 = ledger.add_todo(store, text="vikunja down", source="probe", key="vikunja", now=1000)
    assert ledger.done_todo(store, id1, now=2000)
    id2 = ledger.add_todo(store, text="vikunja down again", source="probe", key="vikunja", now=3000)
    assert id2 == id1
    row = store.conn.execute("SELECT status, text, done_at, updated_at FROM todos").fetchone()
    assert (row["status"], row["text"], row["done_at"], row["updated_at"]) == (
        "open",
        "vikunja down again",
        None,
        3000,
    )


def test_todo_add_keyed_open_row_untouched(store: Store) -> None:
    id1 = ledger.add_todo(store, text="first", source="probe", key="k", now=1000)
    ledger.add_todo(store, text="second", source="probe", key="k", now=2000)
    row = store.conn.execute(
        "SELECT text, updated_at FROM todos WHERE todo_id=?", (id1,)
    ).fetchone()
    assert (row["text"], row["updated_at"]) == ("first", 1000)


def test_todo_add_same_key_different_source_not_deduped(store: Store) -> None:
    id1 = ledger.add_todo(store, text="a", source="probe:x", key="k")
    id2 = ledger.add_todo(store, text="b", source="probe:y", key="k")
    assert id1 != id2


# ---------- hook: fail loud, never block ----------


def test_hook_exits_0_on_malformed_payload(tmp_path: Path) -> None:
    env = {"SPARKY_TURNLOG_DIR": str(tmp_path / "turnlog")}
    result = subprocess.run(
        [sys.executable, str(_HOOK)],
        input="not json",
        capture_output=True,
        text=True,
        env={**__import__("os").environ, **env},
    )
    assert result.returncode == 0


def test_hook_exits_0_on_empty_stdin(tmp_path: Path) -> None:
    env = {"SPARKY_TURNLOG_DIR": str(tmp_path / "turnlog")}
    result = subprocess.run(
        [sys.executable, str(_HOOK)],
        input="",
        capture_output=True,
        text=True,
        env={**__import__("os").environ, **env},
    )
    assert result.returncode == 0


def test_hook_writes_heartbeat_row_and_error_log_on_bad_event(tmp_path: Path) -> None:
    turnlog_dir = tmp_path / "turnlog"
    payload = {
        "hook_event_name": "UserPromptSubmit",
        "session_id": "hooksess",
        "cwd": str(tmp_path),
        "prompt": "build the ledger please",
    }
    result = subprocess.run(
        [sys.executable, str(_HOOK)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "SPARKY_TURNLOG_DIR": str(turnlog_dir)},
    )
    assert result.returncode == 0
    s = Store(turnlog_dir)
    rows = s.conn.execute("SELECT * FROM agent_sessions").fetchall()
    assert len(rows) == 1
    assert rows[0]["session_id"] == "hooksess"
    assert rows[0]["goal"] == "build the ledger please"
    s.close()


def test_hook_missing_session_id_fails_loud_not_silent(tmp_path: Path) -> None:
    turnlog_dir = tmp_path / "turnlog"
    payload = {"hook_event_name": "UserPromptSubmit", "cwd": str(tmp_path)}
    result = subprocess.run(
        [sys.executable, str(_HOOK)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "SPARKY_TURNLOG_DIR": str(turnlog_dir)},
    )
    assert result.returncode == 0
    error_log = turnlog_dir / "hook-errors.log"
    assert error_log.exists()
    assert "missing session_id" in error_log.read_text()
