from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from flightdeck import ledger
from flightdeck.store import Store, hostname, now_ms

HOOK = Path(__file__).resolve().parents[1] / "integration" / "claude-code" / "delta-hook.py"

BLOCKING_THRESHOLD_S = 30 * 60  # keep in sync with delta-hook.py's BLOCKING_AGE_THRESHOLD_SECONDS


def run(payload, store_dir, jobs_dir=None, raw: str | None = None):
    env = dict(os.environ, SPARKY_TURNLOG_DIR=str(store_dir))
    if jobs_dir is not None:
        env["SPARKY_CLAUDE_JOBS_DIR"] = str(jobs_dir)
    else:
        # An empty, guaranteed-nonexistent dir -- tests must never read the real
        # ~/.claude/jobs on the machine running them.
        env["SPARKY_CLAUDE_JOBS_DIR"] = str(store_dir / "no-such-jobs-dir")
    text = raw if raw is not None else json.dumps(payload)
    return subprocess.run(
        [sys.executable, str(HOOK)], input=text, capture_output=True, text=True, env=env
    )


def _payload(session_id="t1", cwd="/nonexistent-repo", **kw):
    base = {
        "hook_event_name": "UserPromptSubmit",
        "prompt": "anything",
        "session_id": session_id,
        "cwd": cwd,
    }
    base.update(kw)
    return base


def _write_job(
    jobs_dir: Path,
    job_id: str,
    *,
    state: str = "blocked",
    needs: str | None = "approve X or reject it",
    name: str = "test job",
    age_seconds: float = 3600,
):
    job_dir = jobs_dir / job_id
    job_dir.mkdir(parents=True)
    updated_at = (datetime.now(UTC) - timedelta(seconds=age_seconds)).isoformat()
    data = {"state": state, "name": name, "updatedAt": updated_at}
    if needs is not None:
        data["needs"] = needs
    (job_dir / "state.json").write_text(json.dumps(data))


@pytest.fixture
def store_dir(tmp_path: Path) -> Path:
    return tmp_path / "store"


# ------------------------------------------------------------------ never break a turn


@pytest.mark.parametrize("raw", ["", "   ", "not json", "[]", "null", '{"prompt": null}'])
def test_hook_always_exits_zero(raw, store_dir):
    result = run(None, store_dir, raw=raw)
    assert result.returncode == 0


def test_malformed_stdin_produces_no_stdout(store_dir):
    result = run(None, store_dir, raw="not json")
    assert result.returncode == 0
    assert result.stdout == ""


def test_other_hook_events_are_ignored(store_dir):
    result = run(_payload(hook_event_name="PreToolUse"), store_dir)
    assert result.stdout == ""
    assert result.returncode == 0


def test_missing_session_id_is_silent(store_dir):
    payload = _payload()
    del payload["session_id"]
    result = run(payload, store_dir)
    assert result.stdout == ""
    assert result.returncode == 0


# ------------------------------------------------------------------ silence is the default


def test_empty_ledger_emits_nothing(store_dir):
    result = run(_payload(), store_dir)
    assert result.stdout == ""
    assert result.returncode == 0


def test_a_repeat_turn_with_no_new_delta_is_silent(store_dir):
    with Store(store_dir) as store:
        ledger.add_todo(store, text="ship the header", scope="global")
    first = run(_payload(), store_dir)
    assert first.stdout != ""  # sanity: the first turn does see the todo
    second = run(_payload(), store_dir)
    assert second.stdout == ""
    assert second.returncode == 0


# ------------------------------------------------------------------ line tier


def test_a_new_todo_gets_the_one_line_tier(store_dir):
    with Store(store_dir) as store:
        ledger.add_todo(store, text="ship the header", scope="global")
    result = run(_payload(), store_dir)
    out = result.stdout.strip()
    assert out.startswith("[ledger]")
    assert "\n" not in out
    assert "1 open" in out
    assert result.returncode == 0


def test_a_new_repo_in_play_joins_the_one_line_tier(tmp_path, store_dir):
    other_repo = tmp_path / "other-repo"
    other_repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=other_repo, check=True)
    subprocess.run(["git", "checkout", "-q", "-b", "main"], cwd=other_repo, check=True)
    with Store(store_dir) as store:
        ledger.add_todo(store, text="a", scope="global")
        ledger.beat(store, session_id="other-session", cwd=str(other_repo), now=now_ms())
    out = run(_payload(session_id="t1"), store_dir).stdout.strip()
    assert "repo" in out and "moved" in out
    assert "1 open" in out


def test_an_other_active_session_joins_the_one_line_tier(store_dir):
    with Store(store_dir) as store:
        ledger.beat(store, session_id="other-session", cwd="/elsewhere", now=now_ms())
    out = run(_payload(session_id="t1"), store_dir).stdout.strip()
    assert "session" in out


def test_own_session_is_never_reported_as_new(store_dir):
    with Store(store_dir) as store:
        ledger.beat(store, session_id="t1", host=hostname(), cwd="/repo", now=now_ms())
    result = run(_payload(session_id="t1", cwd="/repo"), store_dir)
    assert result.stdout == ""


# ------------------------------------------------------------------ block tier: blocking decisions


def test_a_blocked_item_older_than_the_threshold_gets_the_block_tier(tmp_path, store_dir):
    jobs_dir = tmp_path / "jobs"
    _write_job(
        jobs_dir,
        "abc123",
        needs="approve the release or ask for changes",
        age_seconds=BLOCKING_THRESHOLD_S + 3600,
    )
    result = run(_payload(), store_dir, jobs_dir=jobs_dir)
    out = result.stdout.strip()
    assert out.startswith("[ledger] blocking:")
    assert "approve the release" in out
    assert result.returncode == 0


def test_a_blocked_item_younger_than_the_threshold_is_silent(tmp_path, store_dir):
    jobs_dir = tmp_path / "jobs"
    _write_job(jobs_dir, "abc123", age_seconds=BLOCKING_THRESHOLD_S - 60)
    result = run(_payload(), store_dir, jobs_dir=jobs_dir)
    assert result.stdout == ""
    assert result.returncode == 0


def test_blocking_wins_over_the_line_tier(tmp_path, store_dir):
    """Any blocking item at all escalates straight past the one-liner, per the spec's
    tier ordering, even with other non-blocking deltas also pending."""
    jobs_dir = tmp_path / "jobs"
    _write_job(jobs_dir, "abc123", age_seconds=BLOCKING_THRESHOLD_S + 60)
    with Store(store_dir) as store:
        ledger.add_todo(store, text="also new", scope="global")
    out = run(_payload(), store_dir, jobs_dir=jobs_dir).stdout.strip()
    assert out.startswith("[ledger] blocking:")


def test_a_missing_jobs_dir_is_handled_defensively(store_dir, tmp_path):
    result = run(_payload(), store_dir, jobs_dir=tmp_path / "does-not-exist")
    assert result.returncode == 0
    assert result.stdout == ""


def test_a_corrupt_job_state_file_is_skipped_not_fatal(tmp_path, store_dir):
    jobs_dir = tmp_path / "jobs"
    bad = jobs_dir / "broken"
    bad.mkdir(parents=True)
    (bad / "state.json").write_text("not json at all")
    with Store(store_dir) as store:
        ledger.add_todo(store, text="a", scope="global")
    result = run(_payload(), store_dir, jobs_dir=jobs_dir)
    assert result.returncode == 0
    assert "1 open" in result.stdout


# ------------------------------------------------------------------ hard byte cap


def test_output_never_exceeds_900_bytes_given_an_absurd_ledger(tmp_path, store_dir):
    jobs_dir = tmp_path / "jobs"
    long_needs = "please make a decision about this thing " * 20
    for i in range(50):
        _write_job(
            jobs_dir,
            f"job{i}",
            needs=long_needs,
            name=f"a very long descriptive job name number {i}",
            age_seconds=BLOCKING_THRESHOLD_S + 3600,
        )
    with Store(store_dir) as store:
        for i in range(200):
            ledger.add_todo(
                store, text=f"todo number {i} with a fairly long description", scope="global"
            )
        for i in range(200):
            ledger.beat(
                store,
                session_id=f"session-{i}",
                cwd=f"/repo-{i}",
                host=f"host-{i}",
                goal=f"a fairly long goal description for session {i}",
                now=now_ms(),
            )
    result = run(_payload(session_id="t1"), store_dir, jobs_dir=jobs_dir)
    assert result.returncode == 0
    assert len(result.stdout.encode()) <= 900
    assert result.stdout.startswith("[ledger] blocking:")


# ------------------------------------------------------------------ per-session cursor


def test_corrupt_cursor_file_is_treated_as_unseen_not_fatal(store_dir):
    with Store(store_dir) as store:
        ledger.add_todo(store, text="ship the header", scope="global")

    first = run(_payload(), store_dir)
    assert "1 open" in first.stdout

    second = run(_payload(), store_dir)
    assert second.stdout == ""  # already shown, silent as expected

    cursor_path = store_dir / "delta" / "t1.json"
    assert cursor_path.exists()
    cursor_path.write_text("not json at all")

    third = run(_payload(), store_dir)
    assert third.returncode == 0
    assert "1 open" in third.stdout  # corrupt cursor -> fails toward showing, not silence


def test_a_shown_item_is_not_repeated_across_turns_but_a_new_one_still_surfaces(store_dir):
    with Store(store_dir) as store:
        ledger.add_todo(store, text="first", scope="global")
    first = run(_payload(), store_dir)
    assert "1 open" in first.stdout

    second = run(_payload(), store_dir)
    assert second.stdout == ""

    with Store(store_dir) as store:
        ledger.add_todo(store, text="second", scope="global")
    third = run(_payload(), store_dir)
    assert "1 open" in third.stdout  # only the new one, not a re-announcement of the first


# ------------------------------------------------------------------ plan line


def _plan(store, *, goal="fix merge_from", session_id="t1", status="active"):
    return ledger.add_plan(
        store,
        goal=goal,
        stages=[
            {"name": "upsert guard", "expected_check": "test_store green"},
            {"name": "wire hook", "expected_check": "hook rc=0"},
        ],
        non_goals=["vikunja"],
        scope="global",
        session_id=session_id,
        status=status,
    )


def test_an_active_plan_leads_the_header_once(store_dir):
    with Store(store_dir) as store:
        _plan(store)
    first = run(_payload(), store_dir)
    lines = first.stdout.splitlines()
    assert lines[0].startswith('[plan] fix merge_from · stage 1/2 "upsert guard"')
    assert lines[1].startswith("[ledger]")  # the plan is a todo, so it counts as one
    second = run(_payload(), store_dir)
    assert second.stdout == ""


def test_a_draft_plan_is_not_shown(store_dir):
    with Store(store_dir) as store:
        _plan(store, status="draft")
    out = run(_payload(), store_dir).stdout
    assert "[plan]" not in out


def test_a_revision_reshows_the_plan_and_nothing_else(store_dir):
    with Store(store_dir) as store:
        todo_id = _plan(store)
    run(_payload(), store_dir)
    with Store(store_dir) as store:
        plan = ledger.get_plan(store, todo_id)
        stages = plan["stages"]
        stages[0].update(status="done", evidence="812 passed")
        ledger.amend_plan(
            store, todo_id, expected_revision=plan["revision"], changed_by="t1", stages=stages
        )
    out = run(_payload(), store_dir).stdout
    assert out.strip() == '[plan] fix merge_from · stage 2/2 "wire hook"'
    assert run(_payload(), store_dir).stdout == ""


def test_plan_line_fits_inside_the_cap_with_a_blocking_block(tmp_path, store_dir):
    jobs_dir = tmp_path / "jobs"
    for i in range(50):
        _write_job(
            jobs_dir,
            f"job{i}",
            needs="please make a decision about this thing " * 20,
            age_seconds=BLOCKING_THRESHOLD_S + 3600,
        )
    with Store(store_dir) as store:
        _plan(store, goal="g" * 400)
    result = run(_payload(), store_dir, jobs_dir=jobs_dir)
    assert result.returncode == 0
    assert len(result.stdout.encode()) <= 900
    lines = result.stdout.splitlines()
    assert lines[0].startswith("[plan] ")
    assert lines[1] == "[ledger] blocking:"
