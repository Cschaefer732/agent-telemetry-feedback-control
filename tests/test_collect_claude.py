from __future__ import annotations

import json
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from flightdeck import collect_claude
from flightdeck.collect_claude import (
    SOURCE,
    close_turn,
    current_turn_id,
    handle,
    open_turn,
    record_tool,
    set_current_turn,
)
from flightdeck.store import Store, now_ms

REPO_ROOT = Path(__file__).resolve().parents[1]
HOOK_SCRIPT = REPO_ROOT / "integration" / "claude-code" / "turnlog-hook.py"


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Store:
    # Turn correlation (current_turn_id/set_current_turn) resolves its directory from
    # SPARKY_TURNLOG_DIR independently of the Store instance a test constructs, so both must
    # point at the same tmp dir for a test to see consistent behavior.
    monkeypatch.setenv("SPARKY_TURNLOG_DIR", str(tmp_path))
    s = Store(tmp_path)
    yield s
    s.close()


def _prompt_payload(session_id: str = "sess-1", **overrides: object) -> dict:
    payload = {
        "hook_event_name": "UserPromptSubmit",
        "session_id": session_id,
        "cwd": "/repo",
        "prompt": "please fix the bug",
    }
    payload.update(overrides)
    return payload


def _pre_tool_payload(
    session_id: str = "sess-1", tool_use_id: str = "tu-1", **overrides: object
) -> dict:
    payload = {
        "hook_event_name": "PreToolUse",
        "session_id": session_id,
        "cwd": "/repo",
        "tool_name": "Bash",
        "tool_input": {"command": "ls"},
        "tool_use_id": tool_use_id,
    }
    payload.update(overrides)
    return payload


def _post_tool_payload(
    session_id: str = "sess-1", tool_use_id: str = "tu-1", **overrides: object
) -> dict:
    payload = {
        "hook_event_name": "PostToolUse",
        "session_id": session_id,
        "cwd": "/repo",
        "tool_name": "Bash",
        "tool_input": {"command": "ls"},
        "tool_use_id": tool_use_id,
        "tool_response": {"stdout": "file.txt", "stderr": "", "interrupted": False},
    }
    payload.update(overrides)
    return payload


def _stop_payload(session_id: str = "sess-1", **overrides: object) -> dict:
    payload = {
        "hook_event_name": "Stop",
        "session_id": session_id,
        "cwd": "/repo",
        "last_assistant_message": "done, fixed it",
        "turn_number": 1,
    }
    payload.update(overrides)
    return payload


# ---------- open_turn ----------


def test_open_turn_creates_row(store: Store) -> None:
    turn_id = open_turn(store, _prompt_payload())
    turn = store.get_turn(turn_id)
    assert turn is not None
    assert turn.session_id == "sess-1"
    assert turn.source == SOURCE
    assert turn.cwd == "/repo"
    assert turn.host
    assert turn.started_at is not None


def test_open_turn_sets_tier_frontier(store: Store) -> None:
    """claude-code always runs Anthropic Claude; hook payloads never carry model, so tier has to
    come from source alone (flightdeck.tiers.derive_tier)."""
    turn_id = open_turn(store, _prompt_payload())
    turn = store.get_turn(turn_id)
    assert turn is not None
    assert turn.tier == "frontier"


def test_open_turn_sets_current_turn(store: Store) -> None:
    turn_id = open_turn(store, _prompt_payload(session_id="sess-x"))
    assert current_turn_id("sess-x") == turn_id


def test_open_turn_captures_redacted_prompt(store: Store) -> None:
    secret = "sk-ant-" + "A1b2C3d4E5f6G7h8I9j0"
    turn_id = open_turn(store, _prompt_payload(prompt=f"my key is {secret}"))
    texts = store.texts_for(turn_id, "prompt")
    assert len(texts) == 1
    assert secret not in texts[0].body
    assert "[REDACTED:anthropic_key]" in texts[0].body


def test_open_turn_expiry_14_days_out(store: Store) -> None:
    now = now_ms()
    turn_id = open_turn(store, _prompt_payload(), now=now)
    texts = store.texts_for(turn_id, "prompt")
    expected = now + 14 * 24 * 3600 * 1000
    assert texts[0].expires_at == expected


def test_open_turn_no_prompt_text_when_empty(store: Store) -> None:
    turn_id = open_turn(store, _prompt_payload(prompt=""))
    assert store.texts_for(turn_id, "prompt") == []


def test_open_turn_captures_prompt_from_prompt_field(store: Store) -> None:
    # Claude Code sends the text under "prompt" — capturing it is the whole point.
    payload = {"hook_event_name": "UserPromptSubmit", "session_id": "s", "prompt": "hello there"}
    turn_id = open_turn(store, payload)
    texts = store.texts_for(turn_id, kind="prompt")
    assert len(texts) == 1 and texts[0].body == "hello there"


def test_open_turn_prompt_falls_back_to_user_input(store: Store) -> None:
    # Defensive: a non-standard emitter that only sets user_input still gets captured.
    payload = {"hook_event_name": "UserPromptSubmit", "session_id": "s", "user_input": "legacy"}
    turn_id = open_turn(store, payload)
    texts = store.texts_for(turn_id, kind="prompt")
    assert len(texts) == 1 and texts[0].body == "legacy"


def test_write_state_survives_concurrent_writers(store: Store) -> None:
    # Reproduces the fixed-tmp race: many threads (standing in for the parallel hook processes
    # a session fires) hammering the same session's state file must not raise and must leave a
    # readable JSON state. Before the mkstemp fix this raised FileNotFoundError in replace().
    import threading

    from flightdeck.collect_claude import _note_tool_start, _read_state

    errors: list[BaseException] = []
    barrier = threading.Barrier(8)

    def worker(n: int) -> None:
        try:
            barrier.wait()
            for i in range(40):
                _note_tool_start("race-sess", f"tool-{n}-{i}", 1000 + i)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent writers raised: {errors[:3]}"
    assert isinstance(_read_state("race-sess").get("pending_tools"), dict)


def test_open_turn_closes_prior_orphan_turn(store: Store) -> None:
    # A turn opens, records a tool, but its Stop never fires; the next prompt must finalize it.
    first = open_turn(store, _prompt_payload(), now=1000)
    record_tool(store, _post_tool_payload(), ok=True, now=1600)
    open_turn(store, _prompt_payload(), now=2000)  # next turn, no intervening Stop
    orphan = store.get_turn(first)
    assert orphan.ended_at == 1600  # backfilled from the last event, not the new prompt ts
    assert orphan.wall_ms == 600
    assert orphan.outcome is None  # never saw a Stop → success unknown, not fabricated "ok"


def test_open_turn_orphan_with_no_events_ends_at_start(store: Store) -> None:
    first = open_turn(store, _prompt_payload(), now=1000)
    open_turn(store, _prompt_payload(), now=2000)
    orphan = store.get_turn(first)
    assert orphan.ended_at == 1000  # no events → falls back to started_at
    assert orphan.wall_ms == 0


def test_open_turn_does_not_touch_cleanly_closed_prior_turn(store: Store) -> None:
    first = open_turn(store, _prompt_payload(), now=1000)
    close_turn(store, _stop_payload(), now=1500)  # clean Stop
    open_turn(store, _prompt_payload(), now=2000)
    closed = store.get_turn(first)
    assert closed.ended_at == 1500  # unchanged
    assert closed.outcome == "ok"  # clean close preserved


# ---------- record_tool ----------


def test_record_tool_creates_event(store: Store) -> None:
    turn_id = open_turn(store, _prompt_payload())
    record_tool(store, _post_tool_payload(), ok=True)
    events = store.events_for(turn_id, "tool_call")
    assert len(events) == 1
    assert events[0].name == "Bash"
    assert events[0].ok == 1


def test_record_tool_failure_ok_false_and_error_in_payload(store: Store) -> None:
    open_turn(store, _prompt_payload())
    record_tool(
        store,
        _post_tool_payload(tool_response={"stdout": "", "stderr": "boom"}, error="command failed"),
        ok=False,
    )
    events = store.events_for(current_turn_id("sess-1"), "tool_call")
    assert events[0].ok == 0
    assert events[0].payload.get("error") == "command failed"


def test_record_tool_duration_from_payload(store: Store) -> None:
    # PostToolUse carries duration_ms directly — use it, not the racy state-file pairing.
    open_turn(store, _prompt_payload())
    record_tool(store, _post_tool_payload(duration_ms=137), ok=True)
    events = store.events_for(current_turn_id("sess-1"), "tool_call")
    assert events[0].duration_ms == 137


def test_record_tool_duration_from_pre_post_pairing(store: Store) -> None:
    open_turn(store, _prompt_payload())
    start = now_ms()
    handle(_pre_tool_payload(), store=store)
    end = start + 250
    record_tool(store, _post_tool_payload(), ok=True, now=end)
    turn_id = current_turn_id("sess-1")
    events = store.events_for(turn_id, "tool_call")
    assert events[0].duration_ms is not None
    assert events[0].duration_ms >= 0


def test_record_tool_no_matching_pre_tool_duration_is_none(store: Store) -> None:
    open_turn(store, _prompt_payload())
    record_tool(store, _post_tool_payload(tool_use_id="never-started"), ok=True)
    turn_id = current_turn_id("sess-1")
    events = store.events_for(turn_id, "tool_call")
    assert events[0].duration_ms is None


def test_record_tool_with_no_open_turn_creates_minimal_turn(store: Store) -> None:
    # No UserPromptSubmit ever seen for this session (e.g. hooks wired mid-session).
    record_tool(store, _post_tool_payload(session_id="orphan-session"), ok=True)
    turn_id = current_turn_id("orphan-session")
    assert turn_id is not None
    turn = store.get_turn(turn_id)
    assert turn is not None
    assert turn.source == SOURCE
    assert turn.tier == "frontier"
    events = store.events_for(turn_id, "tool_call")
    assert len(events) == 1


# ---------- close_turn ----------


def test_close_turn_sets_end_fields_and_outcome(store: Store) -> None:
    start = now_ms()
    turn_id = open_turn(store, _prompt_payload(), now=start)
    close_turn(store, _stop_payload(), now=start + 500)
    turn = store.get_turn(turn_id)
    assert turn.ended_at == start + 500
    assert turn.wall_ms == 500
    assert turn.outcome == "ok"


def test_close_turn_captures_redacted_response(store: Store) -> None:
    secret = "sk-ant-" + "A1b2C3d4E5f6G7h8I9j0"
    turn_id = open_turn(store, _prompt_payload())
    close_turn(store, _stop_payload(last_assistant_message=f"done, key was {secret}"))
    texts = store.texts_for(turn_id, "response")
    assert len(texts) == 1
    assert secret not in texts[0].body


def test_stop_with_no_open_turn(store: Store) -> None:
    """A Stop with no matching open turn is still evidence, not a dropped record."""
    assert current_turn_id("never-opened") is None
    close_turn(store, _stop_payload(session_id="never-opened"))
    # close_turn clears session state on completion, so re-reading after close returns None;
    # assert instead that a turn row was actually written for this session.
    assert current_turn_id("never-opened") is None
    rows = list(store.iter_turns(source=SOURCE))
    assert len(rows) == 1
    assert rows[0].outcome == "ok"
    assert rows[0].session_id == "never-opened"


def test_close_turn_clears_session_state(store: Store) -> None:
    open_turn(store, _prompt_payload(session_id="sess-clear"))
    assert current_turn_id("sess-clear") is not None
    close_turn(store, _stop_payload(session_id="sess-clear"))
    assert current_turn_id("sess-clear") is None


# ---------- NULL not zero ----------


def test_unobservable_fields_are_none_not_zero(store: Store) -> None:
    start = now_ms()
    turn_id = open_turn(store, _prompt_payload(), now=start)
    handle(_pre_tool_payload(), store=store)
    record_tool(store, _post_tool_payload(), ok=True, now=start + 100)
    close_turn(store, _stop_payload(), now=start + 200)

    turn = store.get_turn(turn_id)
    assert turn.model_ms is None
    assert turn.retries is None
    assert turn.cached_tokens is None
    assert turn.context_peak is None
    assert turn.context_window is None
    assert turn.prompt_tokens is None
    assert turn.completion_tokens is None
    assert turn.estimated == 0


# ---------- turn correlation across interleaved sessions ----------


def test_correlation_across_interleaved_sessions(store: Store) -> None:
    turn_a = open_turn(store, _prompt_payload(session_id="A"))
    turn_b = open_turn(store, _prompt_payload(session_id="B"))
    assert turn_a != turn_b

    handle(_pre_tool_payload(session_id="A", tool_use_id="a-1"), store=store)
    handle(_pre_tool_payload(session_id="B", tool_use_id="b-1"), store=store)

    record_tool(store, _post_tool_payload(session_id="B", tool_use_id="b-1"), ok=True)
    record_tool(store, _post_tool_payload(session_id="A", tool_use_id="a-1"), ok=False)

    events_a = store.events_for(turn_a, "tool_call")
    events_b = store.events_for(turn_b, "tool_call")
    assert len(events_a) == 1 and events_a[0].ok == 0
    assert len(events_b) == 1 and events_b[0].ok == 1

    close_turn(store, _stop_payload(session_id="A"))
    close_turn(store, _stop_payload(session_id="B"))
    assert current_turn_id("A") is None
    assert current_turn_id("B") is None


# ---------- handle() dispatch ----------


def test_handle_user_prompt_submit(store: Store) -> None:
    rc = handle(_prompt_payload(), store=store)
    assert rc == 0
    assert current_turn_id("sess-1") is not None


def test_handle_pre_tool_use_then_post_tool_use(store: Store) -> None:
    handle(_prompt_payload(), store=store)
    assert handle(_pre_tool_payload(), store=store) == 0
    assert handle(_post_tool_payload(), store=store) == 0
    turn_id = current_turn_id("sess-1")
    events = store.events_for(turn_id, "tool_call")
    assert len(events) == 1


def test_handle_post_tool_use_failure(store: Store) -> None:
    handle(_prompt_payload(), store=store)
    rc = handle(
        {
            "hook_event_name": "PostToolUseFailure",
            "session_id": "sess-1",
            "cwd": "/repo",
            "tool_name": "Bash",
            "tool_use_id": "tu-9",
            "tool_response": {"stdout": "", "stderr": "boom"},
            "error": "non-zero exit",
        },
        store=store,
    )
    assert rc == 0
    turn_id = current_turn_id("sess-1")
    events = store.events_for(turn_id, "tool_call")
    assert events[0].ok == 0


def test_handle_stop(store: Store) -> None:
    handle(_prompt_payload(), store=store)
    rc = handle(_stop_payload(), store=store)
    assert rc == 0
    assert current_turn_id("sess-1") is None


def test_handle_stop_failure_marks_outcome_error(store: Store) -> None:
    # StopFailure (rate limit / API error) must close the turn as an error, not be dropped —
    # else it never closes and later scores as a success.
    handle(_prompt_payload(), store=store)
    rc = handle(
        {
            "hook_event_name": "StopFailure",
            "session_id": "sess-1",
            "error": "rate_limit_error",
            "last_assistant_message": "Overloaded",
        },
        store=store,
    )
    assert rc == 0
    turn = next(iter(store.iter_turns()))
    assert turn.outcome == "error"
    assert turn.ended_at is not None
    assert turn.error_class == "rate_limit_error"


def test_handle_unknown_event_is_noop_and_returns_0(store: Store) -> None:
    rc = handle({"hook_event_name": "SomethingElse", "session_id": "sess-1"}, store=store)
    assert rc == 0
    assert list(store.iter_turns()) == []


# ---------- PreToolUse fast path (no sqlite store) ----------


def test_handle_pre_tool_use_does_not_create_the_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PreToolUse only writes the per-session JSON state file (_note_tool_start); it must never
    pay for a sqlite connect + WAL pragma + migrate() on this highest-frequency hook event."""
    monkeypatch.setenv("SPARKY_TURNLOG_DIR", str(tmp_path))

    rc = handle(_pre_tool_payload())

    assert rc == 0
    assert not (tmp_path / "turnlog.db").exists()
    # the state-file write itself still happened -- this is a fast path, not a no-op
    state_files = list((tmp_path / "claude_code_state").glob("*.json"))
    assert len(state_files) == 1


def test_handle_pre_tool_use_never_constructs_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SPARKY_TURNLOG_DIR", str(tmp_path))
    construct_count = {"n": 0}

    class _ExplodingStore:
        def __init__(self, *args: object, **kwargs: object) -> None:
            construct_count["n"] += 1
            raise AssertionError("Store must not be constructed on the PreToolUse fast path")

    monkeypatch.setattr(collect_claude, "Store", _ExplodingStore)

    rc = handle(_pre_tool_payload())

    assert rc == 0
    assert construct_count["n"] == 0


# ---------- never-fails contract ----------


def test_handle_malformed_payload_not_a_dict_returns_0(store: Store) -> None:
    assert handle(None, store=store) == 0  # type: ignore[arg-type]
    assert handle([1, 2, 3], store=store) == 0  # type: ignore[arg-type]


def test_handle_empty_payload_returns_0(store: Store) -> None:
    assert handle({}, store=store) == 0


def test_handle_missing_session_id_returns_0_and_still_records(store: Store) -> None:
    rc = handle({"hook_event_name": "UserPromptSubmit", "prompt": "hi"}, store=store)
    assert rc == 0
    rows = list(store.iter_turns())
    assert len(rows) == 1
    assert rows[0].session_id == "unknown"


def test_handle_exception_inside_dispatch_returns_0(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args: object, **kwargs: object) -> str:
        raise RuntimeError("simulated collector bug")

    monkeypatch.setattr(collect_claude, "open_turn", boom)
    assert handle(_prompt_payload(), store=store) == 0


def test_handle_unwritable_store_dir_returns_0(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    locked_parent = tmp_path / "locked"
    locked_parent.mkdir()
    locked_parent.chmod(0o500)  # read+execute, not writable: mkdir of a child must fail
    bad_dir = locked_parent / "turnlog"
    monkeypatch.setenv("SPARKY_TURNLOG_DIR", str(bad_dir))
    try:
        rc = handle(
            _prompt_payload()
        )  # no store passed: handle() must construct its own and fail closed
        assert rc == 0
    finally:
        locked_parent.chmod(0o700)  # allow tmp_path cleanup to remove it


def test_handle_writes_debug_log_on_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SPARKY_TURNLOG_DIR", str(tmp_path))

    def boom(*args: object, **kwargs: object) -> str:
        raise RuntimeError("simulated collector bug")

    monkeypatch.setattr(collect_claude, "open_turn", boom)
    store = Store(tmp_path)
    try:
        assert handle(_prompt_payload(), store=store) == 0
    finally:
        store.close()
    debug_log = tmp_path / "claude_code_debug.log"
    assert debug_log.exists()
    assert "RuntimeError" in debug_log.read_text()


# ---------- current_turn_id / set_current_turn ----------


def test_set_and_get_current_turn(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SPARKY_TURNLOG_DIR", str(tmp_path))
    assert current_turn_id("brand-new-session") is None
    set_current_turn("brand-new-session", "turn-abc")
    assert current_turn_id("brand-new-session") == "turn-abc"


def test_current_turn_id_survives_corrupt_state_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SPARKY_TURNLOG_DIR", str(tmp_path))
    state_dir = tmp_path / "claude_code_state"
    state_dir.mkdir(parents=True)
    (state_dir / "sess-corrupt.json").write_text("{not valid json")
    assert current_turn_id("sess-corrupt") is None


# ---------- entrypoint script ----------


def test_entrypoint_exits_0_on_malformed_stdin(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(HOOK_SCRIPT)],
        input="not json at all {{{",
        capture_output=True,
        text=True,
        env={"SPARKY_TURNLOG_DIR": str(tmp_path), "PATH": "/usr/bin:/bin"},
        timeout=30,
    )
    assert result.returncode == 0


def test_entrypoint_exits_0_on_empty_stdin(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(HOOK_SCRIPT)],
        input="",
        capture_output=True,
        text=True,
        env={"SPARKY_TURNLOG_DIR": str(tmp_path), "PATH": "/usr/bin:/bin"},
        timeout=30,
    )
    assert result.returncode == 0


def test_entrypoint_is_executable() -> None:
    mode = HOOK_SCRIPT.stat().st_mode
    assert mode & stat.S_IXUSR


def test_entrypoint_resolves_repo_root_from_any_cwd(tmp_path: Path) -> None:
    payload = json.dumps(_prompt_payload())
    result = subprocess.run(
        [sys.executable, str(HOOK_SCRIPT)],
        input=payload,
        capture_output=True,
        text=True,
        cwd=str(tmp_path),  # a cwd with no relationship to the repo and no PYTHONPATH set
        env={"SPARKY_TURNLOG_DIR": str(tmp_path / "turnlog"), "PATH": "/usr/bin:/bin"},
        timeout=30,
    )
    assert result.returncode == 0

    store = Store(tmp_path / "turnlog")
    try:
        rows = list(store.iter_turns(source=SOURCE))
        assert len(rows) == 1
    finally:
        store.close()


# --- argument signature (content-free tool-call identity) --------------------------------


def test_arg_signature_stores_only_allowlist_members():
    """The promise is structural, not careful: the field can only hold a known constant.

    The previous version of this test was a tautology -- it put the secret at argument
    position 3 of a single-segment command, which `_stem_of` can never reach, so it passed
    whether or not a leak existed. An audit then found 325 of 1600 live rows (20.3%) holding
    raw command-line fragments. This version puts the secret in the SEGMENT-LEADING position
    that actually reaches the stem, across every splitting behaviour, and asserts on what
    `record_tool` would really write.
    """
    from flightdeck.collect_claude import _VERIFY_STEMS, _arg_signature
    from flightdeck.redact import redact_payload

    secret = "Tr0ub4dor-and-3-horses"  # noqa: S105 -- literal under test, not a credential
    hostile = [
        f"git commit -m 'rotate creds; {secret} retired'",   # quote-blind connective split
        f"ssh spark bash -lc '\n  {secret} | vault login -\n'",  # newline inside a quoted script
        f"true && {secret} --login",                          # leading a real segment
        f"echo hi; {secret}",                                 # after a semicolon
        f"python3 -c 'print({secret})'",                      # interpreter-skipped position
        f"{secret} --run",                                    # the very first word
    ]
    for command in hostile:
        sig = redact_payload(_arg_signature("Bash", {"command": command}))
        assert secret.lower() not in repr(sig).lower(), command
        for stem in sig.get("stems", []):
            assert stem in _VERIFY_STEMS, f"non-allowlist stem {stem!r} from {command!r}"


def test_scanned_distinguishes_no_match_from_never_looked():
    """A missing `stems` must not mean two different things.

    `scanned` is how a consumer tells "this ran and matched no verification tool" (which is
    EXEC) from "no signature was recorded at all" (which is unknowable, and must stay SILENT).
    """
    from flightdeck.collect_claude import _arg_signature

    matched = _arg_signature("Bash", {"command": "cd /repo && pytest -q"})
    assert matched["stems"] == ["pytest"] and matched["scanned"] is True

    no_match = _arg_signature("Bash", {"command": "git status"})
    assert "stems" not in no_match and no_match["scanned"] is True

    not_a_command = _arg_signature("Read", {"file_path": "/repo/a.py"})
    assert "scanned" not in not_a_command


def test_fingerprint_is_salted_against_a_dictionary_attack():
    """An unsalted truncated digest over a guessable domain is a lookup key, not a seal.

    An audit recovered 27 of 128 real path fingerprints in one second by hashing the local
    filesystem. Equality within a turn is the only property any consumer needs, and HMAC
    keeps it.
    """
    import hashlib

    from flightdeck.collect_claude import _arg_signature

    path = "/repo/a.py"
    fp = _arg_signature("Read", {"file_path": path})["arg_fp"]
    assert fp != hashlib.sha256(path.encode()).hexdigest()[:12]
    assert _arg_signature("Edit", {"file_path": path})["arg_fp"] == fp   # still comparable
    assert _arg_signature("Read", {"file_path": "/repo/b.py"})["arg_fp"] != fp


def test_arg_signature_is_empty_rather_than_guessing():
    from flightdeck.collect_claude import _arg_signature

    assert _arg_signature("Bash", None) == {}
    assert _arg_signature("Bash", {}) == {}
    assert _arg_signature("Bash", {"command": "   "}) == {}
    assert _arg_signature("Read", {"file_path": ""}) == {}


def test_heredoc_body_never_becomes_a_stem():
    """Everything after `<<` is data, not commands.

    Splitting a heredoc on newlines walks into its body and records the body's own source
    lines as stems. That is meaningless, and it is a content leak: a heredoc can carry
    anything, including a credential. Stem extraction stops at the marker; the fingerprint
    still covers the whole command.
    """
    from flightdeck.collect_claude import _arg_signature

    secret = "sk-ant-DO-NOT-STORE-999"  # noqa: S105 -- literal under test
    sig = _arg_signature("Bash", {"command": f"python3 - <<'PY'\nimport os\nk = '{secret}'\nPY"})
    assert secret not in repr(sig)
    assert sig.get("stems", []) == []
    assert sig["arg_fp"]  # identity is still recorded


def test_fingerprint_discriminates_search_and_chunk_arguments():
    """A path alone does not identify the call.

    Two greps of one directory for different patterns, or two disjoint chunk-reads of one
    file, hashed identically -- so a duplicate-read detector scored the second as waste. The
    discriminating fields are hashed into the digest, never stored, so this adds no exposure.
    """
    from flightdeck.collect_claude import _arg_signature

    grep_a = _arg_signature("Grep", {"pattern": "FOO", "path": "/r"})["arg_fp"]
    grep_b = _arg_signature("Grep", {"pattern": "BAR", "path": "/r"})["arg_fp"]
    assert grep_a != grep_b

    head = _arg_signature("Read", {"file_path": "/r/a.py", "offset": 1, "limit": 50})["arg_fp"]
    tail = _arg_signature("Read", {"file_path": "/r/a.py", "offset": 500, "limit": 50})["arg_fp"]
    assert head != tail

    # A whole-file read and an edit of it must still agree, or invalidation breaks.
    assert (
        _arg_signature("Read", {"file_path": "/r/a.py"})["arg_fp"]
        == _arg_signature("Edit", {"file_path": "/r/a.py"})["arg_fp"]
    )

    secret = "Tr0ub4dor-pattern"  # noqa: S105 -- literal under test
    assert secret not in repr(_arg_signature("Grep", {"pattern": secret, "path": "/r"}))
