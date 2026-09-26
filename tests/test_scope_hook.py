from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from flightdeck.scope_gate import classify
from flightdeck.scope_pass import ScopePass

REPO_ROOT = Path(__file__).resolve().parents[1]
HOOK = REPO_ROOT / "integration" / "claude-code" / "scope-hook.py"


def run(payload, state_dir, raw: str | None = None):
    env = dict(os.environ, SPARKY_TURNLOG_DIR=str(state_dir))
    text = raw if raw is not None else json.dumps(payload)
    return subprocess.run(
        [sys.executable, str(HOOK)], input=text, capture_output=True, text=True, env=env
    )


def _payload(prompt, **kw):
    base = {
        "hook_event_name": "UserPromptSubmit",
        "prompt": prompt,
        "session_id": "t1",
        "cwd": "/repo",
    }
    base.update(kw)
    return base


# ------------------------------------------------------------------ never break a turn


@pytest.mark.parametrize("raw", ["", "   ", "not json", "[]", "null", '{"prompt": null}'])
def test_hook_always_exits_zero(raw, tmp_path):
    result = run(None, tmp_path, raw=raw)
    assert result.returncode == 0


def test_garbage_produces_no_injection(tmp_path):
    assert run(None, tmp_path, raw="not json").stdout == ""


def test_other_hook_events_are_ignored(tmp_path):
    result = run(_payload("add a save feature", hook_event_name="PreToolUse"), tmp_path)
    assert result.stdout == ""


# ------------------------------------------------------------------ silence is the default


def test_a_self_specifying_request_gets_no_injection(tmp_path):
    """59% of task-start turns take this path. Speaking on every turn is the ceremony tax."""
    result = run(_payload("fix the typo in flightdeck/store.py"), tmp_path)
    assert result.stdout == ""
    assert result.returncode == 0


def test_a_refinement_of_an_active_scope_is_silent(tmp_path):
    sp = ScopePass.open("t1", "/repo", "add a save feature", classify("add a save feature"))
    sp.add("save writes to disk", "committed", "file exists afterwards")
    sp.save(tmp_path)
    assert run(_payload("make it 2px smaller"), tmp_path).stdout == ""


def test_a_stale_scope_does_not_suppress_the_gate(tmp_path):
    """A scope made in another directory must not silence scoping here -- that is the
    four-day-old-spec bleed in its other direction."""
    sp = ScopePass.open("t1", "/somewhere/else", "old goal", classify("add a save feature"))
    sp.add("x", "committed", "y")
    sp.save(tmp_path)
    assert "[scope:" in run(_payload("add a save feature"), tmp_path).stdout


# ------------------------------------------------------------------ the injection itself


def test_an_underspecified_request_is_injected(tmp_path):
    out = run(_payload("add a save feature"), tmp_path).stdout
    assert "[scope:" in out
    assert "deliberately NOT doing" in out


def test_an_open_ended_request_gets_the_wider_pass(tmp_path):
    prompt = "research the options then rewrite all the prompts everywhere"
    out = run(_payload(prompt), tmp_path).stdout
    assert "competent engineer would assume" in out or "competent engineer" in out
    assert "THREE questions" in out


def test_every_injection_ends_with_the_instruction_to_build(tmp_path):
    """The measured failure: a loud 'do X before the task' instruction with no stated
    continuation ends the turn with the scope written and nothing built."""
    prompts = (
        "add a save feature",
        "research the options then rewrite everything across both machines",
    )
    for prompt in prompts:
        out = run(_payload(prompt), tmp_path).stdout.strip()
        assert out.endswith("Then build it in this same turn.")


def test_the_wider_pass_tells_the_model_to_persist_one_plan(tmp_path):
    """FULL tier carries the plan-add command with this session's id and its repo, so the
    plan lands keyed to the session (the delta header prefers a session match) and
    `plan_header`'s repo fallback can show it to a sibling session in the same checkout."""
    prompt = "research the options then rewrite all the prompts everywhere"
    out = run(_payload(prompt, cwd=str(REPO_ROOT)), tmp_path).stdout
    assert "plan add <<'EOF'" in out
    assert '"session_id": "t1"' in out
    assert f'"scope": "local", "repo": {json.dumps(str(REPO_ROOT))}' in out
    assert '"status": "active"' in out
    assert out.count("{{") == 0  # every format brace resolved
    assert out.strip().endswith("Then build it in this same turn.")


def test_outside_a_repo_the_plan_is_global(tmp_path):
    prompt = "research the options then rewrite all the prompts everywhere"
    out = run(_payload(prompt, cwd=str(tmp_path)), tmp_path).stdout
    assert '"scope": "global"' in out
    assert '"repo"' not in out


def test_the_mini_pass_carries_no_plan_ceremony(tmp_path):
    out = run(_payload("add a save feature"), tmp_path).stdout
    assert "plan add" not in out


# ------------------------------------------------------------------ the gate log


def test_every_decision_is_logged(tmp_path):
    run(_payload("add a save feature"), tmp_path)
    run(_payload("fix the typo in flightdeck/store.py"), tmp_path)
    lines = (tmp_path / "scope" / "gate-log.jsonl").read_text().splitlines()
    assert len(lines) == 2
    tiers = [json.loads(x)["tier"] for x in lines]
    assert tiers == ["mini", "none"]


def test_the_log_records_a_hash_not_the_prompt(tmp_path):
    """Decisions are auditable without keeping a copy of everything the user typed."""
    run(_payload("add a save feature for the secret project"), tmp_path)
    body = (tmp_path / "scope" / "gate-log.jsonl").read_text()
    assert "secret project" not in body
    row = json.loads(body.splitlines()[0])
    assert len(row["prompt_sha1"]) == 12
    assert row["prompt_len"] > 0


def test_log_entries_carry_a_timestamp_and_reasons(tmp_path):
    run(_payload("add a save feature"), tmp_path)
    row = json.loads((tmp_path / "scope" / "gate-log.jsonl").read_text().splitlines()[0])
    assert row["ts"].startswith("20") and row["reasons"]
    assert row["session_id"] == "t1"


def test_an_unwritable_log_still_does_not_break_the_turn(tmp_path):
    blocked = tmp_path / "scope"
    blocked.write_text("i am a file, not a directory")
    result = run(_payload("add a save feature"), tmp_path)
    assert result.returncode == 0
    assert "[scope:" in result.stdout


# ------------------------------------------------------------------ opening a real pass


def test_a_scoped_decision_opens_a_pass(tmp_path):
    """Before this, ScopePass.open/.add/.close had zero production callers -- the hook
    only ever consulted ScopePass.load(). A non-NONE decision must now persist one."""
    run(_payload("add a save feature"), tmp_path)
    opened = ScopePass.load("t1", tmp_path)
    assert opened is not None
    assert opened.tier == "mini"
    assert opened.session_id == "t1" and opened.cwd == "/repo"


def test_a_none_tier_decision_opens_no_pass(tmp_path):
    run(_payload("fix the typo in flightdeck/store.py"), tmp_path)
    assert ScopePass.load("t1", tmp_path) is None


def test_a_fresh_active_pass_is_reused_not_clobbered(tmp_path):
    """A refinement turn must not wipe out found items already recorded on the active
    pass -- reopening from scratch would silently discard scoping work."""
    sp = ScopePass.open("t1", "/repo", "add a save feature", classify("add a save feature"))
    sp.add("save writes to disk", "committed", "file exists afterwards")
    sp.save(tmp_path)
    run(_payload("make it 2px smaller"), tmp_path)  # silent refinement path
    still_there = ScopePass.load("t1", tmp_path)
    assert len(still_there.found) == 1


def test_a_stale_pass_is_replaced_by_a_fresh_one(tmp_path):
    sp = ScopePass.open("t1", "/somewhere/else", "old goal", classify("add a save feature"))
    sp.add("x", "committed", "y")
    sp.save(tmp_path)
    run(_payload("add a save feature"), tmp_path)
    fresh = ScopePass.load("t1", tmp_path)
    assert fresh.cwd == "/repo"
    assert fresh.found == []  # the stale one's items are not carried over


def test_gate_raises_still_exits_zero_and_logs(tmp_path):
    """Corrupt the on-disk pass so ScopePass.load() raises inside the hook's try block."""
    scope_dir = tmp_path / "scope"
    scope_dir.mkdir(parents=True)
    (scope_dir / "t1.json").write_text("not json at all")
    result = run(_payload("add a save feature"), tmp_path)
    assert result.returncode == 0
    log = (scope_dir / "gate-log.jsonl").read_text().splitlines()
    assert any(json.loads(line).get("error") for line in log)
