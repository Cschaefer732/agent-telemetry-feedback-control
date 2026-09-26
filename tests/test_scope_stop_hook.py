from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from flightdeck.scope_gate import classify
from flightdeck.scope_pass import ScopePass
from flightdeck.store import Store

HOOK = Path(__file__).resolve().parents[1] / "integration" / "claude-code" / "scope-stop-hook.py"


def _load_stop_hook():
    """The hook is a hyphenated script, not an importable module name."""
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    path = root / "integration" / "claude-code" / "scope-stop-hook.py"
    spec = importlib.util.spec_from_file_location("scope_stop_hook", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_stop_record_id = _load_stop_hook()._record_id


def run(payload, state_dir, raw: str | None = None):
    env = dict(os.environ, SPARKY_TURNLOG_DIR=str(state_dir))
    text = raw if raw is not None else json.dumps(payload)
    return subprocess.run(
        [sys.executable, str(HOOK)], input=text, capture_output=True, text=True, env=env
    )


def _payload(**kw):
    base = {"hook_event_name": "Stop", "session_id": "t1", "cwd": "/repo"}
    base.update(kw)
    return base


def _open_pass(state_dir, **kw):
    base = dict(session_id="t1", cwd="/repo", goal="add a save feature")
    base.update(kw)
    decision = classify("add a save feature")
    sp = ScopePass.open(decision=decision, **base)
    sp.save(state_dir)
    return sp


# ------------------------------------------------------------------ never break a turn


def test_hook_always_exits_zero_on_garbage(tmp_path):
    for raw in ("", "   ", "not json", "[]", "null"):
        assert run(None, tmp_path, raw=raw).returncode == 0


def test_other_hook_events_are_ignored(tmp_path):
    _open_pass(tmp_path)
    result = run(_payload(hook_event_name="PreToolUse"), tmp_path)
    assert result.returncode == 0
    # untouched: no close attempted
    assert ScopePass.load("t1", tmp_path) is not None


# ------------------------------------------------------------------ nothing to close


def test_no_active_pass_closes_nothing_and_logs_why(tmp_path):
    result = run(_payload(), tmp_path)
    assert result.returncode == 0
    log = (tmp_path / "scope" / "stop-log.jsonl").read_text().splitlines()
    row = json.loads(log[-1])
    assert row["closed"] is False
    assert "no active scope pass" in row["reason"]


def test_a_stale_pass_is_left_untouched_and_logged(tmp_path):
    _open_pass(tmp_path, cwd="/somewhere/else")
    result = run(_payload(cwd="/repo"), tmp_path)
    assert result.returncode == 0
    row = json.loads((tmp_path / "scope" / "stop-log.jsonl").read_text().splitlines()[-1])
    assert row["closed"] is False
    assert "stale" in row["reason"]
    # the pass file itself is not deleted -- another session/turn may still resolve it
    assert ScopePass.load("t1", tmp_path) is not None


# ------------------------------------------------------------------ the closure path itself


def test_closing_an_active_pass_writes_a_silent_scope_record(tmp_path):
    """The core claim of this hook: it gives ScopePass.close() a real caller. It must
    NEVER fabricate 'pass' from the mere fact that a turn ended -- a Stop payload carries
    no correctness signal, so the verdict must be exactly 'silent'."""
    _open_pass(tmp_path)
    result = run(_payload(), tmp_path)
    assert result.returncode == 0

    with Store(tmp_path) as store:
        rows = store.scope_records()
    assert len(rows) == 1
    assert rows[0]["verdict"] == "silent"
    assert rows[0]["tier"] == "mini"


def test_an_unobservable_turn_never_yields_pass_even_when_the_pass_is_valid(tmp_path):
    """Even a ScopePass that is fully valid (committed items, criteria, the works) must
    close 'silent' here -- validity is a property of the scoping content, not evidence
    that the Stop hook can observe about what happened afterward."""
    sp = _open_pass(tmp_path)
    sp.add("save writes to disk", "committed", "file exists afterwards")
    sp.save(tmp_path)
    assert sp.valid  # sanity: this pass WOULD emit "pass" if to_record() decided verdicts

    run(_payload(), tmp_path)

    with Store(tmp_path) as store:
        rows = store.scope_records()
    assert rows[0]["verdict"] == "silent"
    assert rows[0]["verdict"] != "pass"


def test_closing_removes_the_state_file_so_it_is_not_double_closed(tmp_path):
    _open_pass(tmp_path)
    run(_payload(), tmp_path)
    assert ScopePass.load("t1", tmp_path) is None

    # a second Stop for the same (now-closed) session must not error or duplicate rows
    result = run(_payload(), tmp_path)
    assert result.returncode == 0
    with Store(tmp_path) as store:
        rows = store.scope_records()
    assert len(rows) == 1


def test_a_broken_state_file_does_not_break_the_turn(tmp_path):
    scope_dir = tmp_path / "scope"
    scope_dir.mkdir(parents=True)
    (scope_dir / "t1.json").write_text("not json at all")
    result = run(_payload(), tmp_path)
    assert result.returncode == 0
    row = json.loads((scope_dir / "stop-log.jsonl").read_text().splitlines()[-1])
    assert row["closed"] is False
    assert "load raised" in row["reason"]


def test_an_unwritable_log_still_does_not_break_the_turn(tmp_path):
    scope_dir = tmp_path / "scope"
    scope_dir.mkdir(parents=True)
    (scope_dir / "stop-log.jsonl").mkdir()  # a directory where the log file should be
    _open_pass(tmp_path)
    result = run(_payload(), tmp_path)
    assert result.returncode == 0
    # closure itself still succeeded despite the log being unwritable
    with Store(tmp_path) as store:
        assert len(store.scope_records()) == 1


def test_stop_hook_and_ingest_agree_on_the_record_id_for_one_decision():
    """Two writers reach scope_records for a single turn: this hook, and scope_ingest
    reading the same gate-log line. On separate id schemes both insert, and the turn is
    counted twice in every rate the KPI layer computes over that denominator."""
    from flightdeck.scope_ingest import record_id_for
    from flightdeck.scope_pass import ScopePass

    gate_row = {
        "session_id": "s-1",
        "ts": "2026-09-05T12:00:00+00:00",
        "prompt_sha1": "abc123def456",
        "tier": "mini",
    }
    opened = ScopePass(
        session_id=gate_row["session_id"], cwd="/tmp/x", tier="mini", goal="g",
        gate_ts=gate_row["ts"], prompt_sha1=gate_row["prompt_sha1"],
    )
    assert _stop_record_id(opened) == record_id_for(gate_row)


def test_a_pass_predating_the_gate_identity_still_gets_a_stable_id():
    """Passes opened before gate_ts existed have no seed to share; they must still mint a
    stable, non-colliding id rather than crashing or reusing another turn's."""
    from flightdeck.scope_pass import ScopePass

    old = ScopePass(session_id="s-2", cwd="/tmp/x", tier="full", goal="g", created_at=1700000000)
    first = _stop_record_id(old)
    assert first.startswith("stop-")
    assert first == _stop_record_id(old)
