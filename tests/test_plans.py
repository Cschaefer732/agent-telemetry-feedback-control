"""Plans: one plan = one todos row, stages are fields on it, never separate todos.

Covers migration 16 (todos.key idempotency index, plans, plan_revisions), the ledger.py
plan API (add_plan/get_plan/list_plans/amend_plan/set_plan_status/plan_problems/plan_header),
and the `todo`/`plan` CLI subcommands.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from flightdeck import __main__ as cli
from flightdeck import ledger
from flightdeck.schema import MIGRATIONS, SCHEMA_VERSION
from flightdeck.store import Store


@pytest.fixture
def store(tmp_path: Path) -> Store:
    s = Store(tmp_path / "store")
    yield s
    s.close()


def _stage(name: str = "stage one", check: str = "tests pass", **kw: object) -> dict:
    return {"name": name, "expected_check": check, **kw}


def _add_plan(store: Store, **overrides: object) -> str:
    kwargs: dict[str, object] = {
        "goal": "fix merge_from",
        "stages": [_stage("stage one", "pytest -q green"), _stage("stage two", "ruff clean")],
        "non_goals": ["rewrite the merge algorithm"],
    }
    kwargs.update(overrides)
    return ledger.add_plan(store, **kwargs)


# ---------- migration 16 applies cleanly ----------


def test_add_plan_can_start_active(store: Store) -> None:
    todo_id = ledger.add_plan(
        store,
        goal="g",
        stages=[{"name": "s", "expected_check": "c"}],
        non_goals=[],
        status="active",
    )
    assert ledger.get_plan(store, todo_id)["status"] == "active"
    assert ledger.active_plan(store) is not None
    with pytest.raises(ValueError):
        ledger.add_plan(
            store,
            goal="g",
            stages=[{"name": "s", "expected_check": "c"}],
            non_goals=[],
            status="done",
        )


def test_migration_16_is_current_schema_version() -> None:
    assert SCHEMA_VERSION == 16
    assert MIGRATIONS[-1][0] == 16


def test_migration_16_applies_to_a_fresh_store(tmp_path: Path) -> None:
    s = Store(tmp_path / "fresh")
    try:
        assert s.version == 16
        cols = {row[1] for row in s.conn.execute("PRAGMA table_info(todos)")}
        assert "key" in cols
        plan_cols = {row[1] for row in s.conn.execute("PRAGMA table_info(plans)")}
        assert {"todo_id", "goal", "stages", "non_goals", "status", "revision"} <= plan_cols
        rev_cols = {row[1] for row in s.conn.execute("PRAGMA table_info(plan_revisions)")}
        assert {"todo_id", "revision", "snapshot", "changed_at"} <= rev_cols
    finally:
        s.close()


def test_migration_16_applied_to_an_older_schema_copy_preserves_rows(tmp_path: Path) -> None:
    """Build a v15 store with real rows, apply migrate() again (as Store.__init__ does), and
    confirm the pre-existing data is untouched -- this is the additive-only guarantee the
    CRITICAL section requires before a MIGRATIONS entry may exist at all."""
    path = tmp_path / "older.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE schema_version (version INTEGER NOT NULL)")
    for version, sql in MIGRATIONS:
        if version > 15:
            continue
        conn.executescript(sql)
        conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
    conn.execute(
        "INSERT INTO turns (turn_id, session_id, source, host, started_at) "
        "VALUES ('t1', 's1', 'claude-code', 'oldbox', 1000)"
    )
    conn.commit()
    conn.close()

    s = Store(tmp_path / "unused")  # not used; keeps Store importable side effects consistent
    s.close()

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    from flightdeck.schema import migrate

    version = migrate(conn)
    assert version == 16
    row = conn.execute("SELECT * FROM turns WHERE turn_id='t1'").fetchone()
    assert row is not None
    assert row["session_id"] == "s1"
    conn.close()


# ---------- add_plan ----------

# add_todo's `key` idempotency is a direct extension of the existing add/list/done coverage
# in tests/test_ledger.py -- see test_add_todo_idempotent_key_returns_existing_id there.


def test_add_plan_creates_todo_plan_and_revision_one(store: Store) -> None:
    todo_id = _add_plan(store)
    todo_row = store.conn.execute("SELECT * FROM todos WHERE todo_id=?", (todo_id,)).fetchone()
    assert todo_row is not None
    assert todo_row["text"] == "fix merge_from"
    assert todo_row["status"] == "open"

    plan_row = store.conn.execute("SELECT * FROM plans WHERE todo_id=?", (todo_id,)).fetchone()
    assert plan_row is not None
    assert plan_row["status"] == "draft"
    assert plan_row["revision"] == 1
    assert json.loads(plan_row["non_goals"]) == ["rewrite the merge algorithm"]

    revs = store.conn.execute("SELECT * FROM plan_revisions WHERE todo_id=?", (todo_id,)).fetchall()
    assert len(revs) == 1
    assert revs[0]["revision"] == 1
    snapshot = json.loads(revs[0]["snapshot"])
    assert snapshot["goal"] == "fix merge_from"


def test_add_plan_normalizes_stage_defaults(store: Store) -> None:
    todo_id = _add_plan(store, stages=[{"name": "only stage", "expected_check": "green"}])
    plan = ledger.get_plan(store, todo_id)
    assert plan["stages"] == [
        {"name": "only stage", "expected_check": "green", "evidence": None, "status": "pending"}
    ]


def test_add_plan_rejects_empty_goal(store: Store) -> None:
    with pytest.raises(ValueError):
        _add_plan(store, goal="   ")


def test_add_plan_rejects_empty_stages(store: Store) -> None:
    with pytest.raises(ValueError):
        _add_plan(store, stages=[])


def test_add_plan_rejects_stage_missing_expected_check(store: Store) -> None:
    with pytest.raises(ValueError):
        _add_plan(store, stages=[{"name": "no check"}])


def test_add_plan_rejects_stage_missing_name(store: Store) -> None:
    with pytest.raises(ValueError):
        _add_plan(store, stages=[{"expected_check": "green"}])


def test_add_plan_rejects_non_list_non_goals(store: Store) -> None:
    with pytest.raises(ValueError):
        _add_plan(store, non_goals="not a list")


# ---------- get_plan / list_plans ----------


def test_get_plan_missing_returns_none(store: Store) -> None:
    assert ledger.get_plan(store, "does-not-exist") is None


def test_get_plan_decodes_json_fields(store: Store) -> None:
    todo_id = _add_plan(store, risks=["scope creep"], assumptions=None)
    plan = ledger.get_plan(store, todo_id)
    assert plan["risks"] == ["scope creep"]
    assert plan["assumptions"] is None
    assert isinstance(plan["stages"], list)


def test_list_plans_filters_by_status(store: Store) -> None:
    t1 = _add_plan(store, goal="plan one")
    _add_plan(store, goal="plan two")
    ledger.set_plan_status(store, t1, "active", expected_revision=1, changed_by="human")

    active = ledger.list_plans(store, status="active")
    assert [p["todo_id"] for p in active] == [t1]

    drafts = ledger.list_plans(store, status="draft")
    assert len(drafts) == 1
    assert drafts[0]["goal"] == "plan two"


# ---------- amend_plan: CAS ----------


def test_amend_plan_success_bumps_revision_and_writes_snapshot(store: Store) -> None:
    todo_id = _add_plan(store)
    new_rev = ledger.amend_plan(
        store,
        todo_id,
        expected_revision=1,
        changed_by="human",
        note="tighten check",
        goal="fix merge_from cleanly",
    )
    assert new_rev == 2
    plan = ledger.get_plan(store, todo_id)
    assert plan["goal"] == "fix merge_from cleanly"
    assert plan["revision"] == 2

    revs = store.conn.execute(
        "SELECT revision, note FROM plan_revisions WHERE todo_id=? ORDER BY revision", (todo_id,)
    ).fetchall()
    assert [r["revision"] for r in revs] == [1, 2]
    assert revs[1]["note"] == "tighten check"


def test_amend_plan_stale_revision_raises_plan_conflict(store: Store) -> None:
    todo_id = _add_plan(store)
    ledger.amend_plan(store, todo_id, expected_revision=1, changed_by="a", goal="v2")

    with pytest.raises(ledger.PlanConflict) as exc_info:
        ledger.amend_plan(store, todo_id, expected_revision=1, changed_by="b", goal="v3-stale")

    assert exc_info.value.expected == 1
    assert exc_info.value.actual == 2
    # the stale amend must not have landed
    assert ledger.get_plan(store, todo_id)["goal"] == "v2"


def test_amend_plan_unknown_field_raises(store: Store) -> None:
    todo_id = _add_plan(store)
    with pytest.raises(ValueError):
        ledger.amend_plan(store, todo_id, expected_revision=1, changed_by="a", bogus_field=1)


def test_amend_plan_missing_todo_raises(store: Store) -> None:
    with pytest.raises(ValueError):
        ledger.amend_plan(store, "nope", expected_revision=1, changed_by="a", goal="x")


def test_amend_plan_rejects_invalid_stages(store: Store) -> None:
    todo_id = _add_plan(store)
    with pytest.raises(ValueError):
        ledger.amend_plan(
            store, todo_id, expected_revision=1, changed_by="a", stages=[{"name": "no check"}]
        )


# ---------- set_plan_status ----------


def test_set_plan_status_done_requires_evidence_on_every_stage(store: Store) -> None:
    todo_id = _add_plan(store)
    with pytest.raises(ValueError):
        ledger.set_plan_status(store, todo_id, "done", expected_revision=1, changed_by="human")


def test_set_plan_status_done_succeeds_with_evidence(store: Store) -> None:
    todo_id = _add_plan(
        store,
        stages=[
            _stage("stage one", "pytest -q green", evidence="ran, exit 0"),
            _stage("stage two", "ruff clean", evidence="ruff check exit 0"),
        ],
    )
    new_rev = ledger.set_plan_status(
        store, todo_id, "done", expected_revision=1, changed_by="human"
    )
    assert new_rev == 2
    plan = ledger.get_plan(store, todo_id)
    assert plan["status"] == "done"
    # marking a plan done also closes its todo
    todo_row = store.conn.execute("SELECT status FROM todos WHERE todo_id=?", (todo_id,)).fetchone()
    assert todo_row["status"] == "done"


def test_set_plan_status_done_error_names_stages_lacking_evidence(store: Store) -> None:
    todo_id = _add_plan(
        store,
        stages=[
            _stage("has evidence", "a", evidence="proof"),
            _stage("missing evidence", "b"),
        ],
    )
    with pytest.raises(ValueError, match="missing evidence"):
        ledger.set_plan_status(store, todo_id, "done", expected_revision=1, changed_by="human")


def test_set_plan_status_superseded_requires_superseded_by(store: Store) -> None:
    todo_id = _add_plan(store)
    with pytest.raises(ValueError):
        ledger.set_plan_status(
            store, todo_id, "superseded", expected_revision=1, changed_by="human"
        )


def test_set_plan_status_stale_revision_raises_conflict(store: Store) -> None:
    todo_id = _add_plan(store)
    ledger.set_plan_status(store, todo_id, "active", expected_revision=1, changed_by="a")
    with pytest.raises(ledger.PlanConflict):
        ledger.set_plan_status(store, todo_id, "abandoned", expected_revision=1, changed_by="b")


def test_supersede_chain_preserves_old_row(store: Store) -> None:
    """Superseding a plan must not delete or rewrite the old plan's history -- it stays
    readable at its final status, pointing at its replacement."""
    old_id = _add_plan(store, goal="old approach")
    new_id = _add_plan(store, goal="new approach")

    ledger.set_plan_status(
        store,
        old_id,
        "superseded",
        expected_revision=1,
        changed_by="human",
        superseded_by=new_id,
    )

    old_plan = ledger.get_plan(store, old_id)
    assert old_plan is not None
    assert old_plan["status"] == "superseded"
    assert old_plan["superseded_by"] == new_id
    assert old_plan["goal"] == "old approach"  # untouched

    new_plan = ledger.get_plan(store, new_id)
    assert new_plan["status"] == "draft"


# ---------- plan_problems (pure function) ----------


def _base_plan(**overrides: object) -> dict:
    plan = {
        "todo_id": "t1",
        "goal": "g",
        "stages": [_stage("s1", "check one")],
        "non_goals": [],
        "status": "draft",
        "superseded_by": None,
        "revision": 1,
    }
    plan.update(overrides)
    return plan


def test_plan_problems_clean_plan_has_none() -> None:
    assert ledger.plan_problems(_base_plan()) == []


def test_plan_problems_stage_without_expected_check() -> None:
    plan = _base_plan(stages=[{"name": "s1", "expected_check": "", "status": "pending"}])
    problems = ledger.plan_problems(plan)
    assert any("expected_check" in p for p in problems)


def test_plan_problems_stage_done_without_evidence() -> None:
    plan = _base_plan(
        stages=[{"name": "s1", "expected_check": "c", "status": "done", "evidence": None}]
    )
    problems = ledger.plan_problems(plan)
    assert any("evidence" in p for p in problems)


def test_plan_problems_active_plan_zero_stages() -> None:
    plan = _base_plan(stages=[], status="active")
    problems = ledger.plan_problems(plan)
    assert any("zero stages" in p for p in problems)


def test_plan_problems_superseded_without_superseded_by() -> None:
    plan = _base_plan(status="superseded", superseded_by=None)
    problems = ledger.plan_problems(plan)
    assert any("superseded" in p for p in problems)


def test_plan_problems_done_status_but_stage_missing_evidence() -> None:
    plan = _base_plan(
        status="done",
        stages=[{"name": "s1", "expected_check": "c", "status": "done", "evidence": None}],
    )
    problems = ledger.plan_problems(plan)
    assert any("evidence" in p for p in problems)


def test_plan_problems_bad_revision() -> None:
    plan = _base_plan(revision=0)
    problems = ledger.plan_problems(plan)
    assert any("revision" in p for p in problems)


# ---------- plan_header ----------


def test_plan_header_empty_when_no_active_plan(store: Store) -> None:
    _add_plan(store)  # draft, not active
    assert ledger.plan_header(store) == ""


def test_plan_header_reports_active_plan(store: Store) -> None:
    todo_id = _add_plan(
        store,
        stages=[
            _stage("upsert guard", "pytest -q", status="done", evidence="ran"),
            _stage("dedupe check", "ruff clean"),
        ],
    )
    ledger.set_plan_status(store, todo_id, "active", expected_revision=1, changed_by="human")
    header = ledger.plan_header(store)
    assert header != ""
    assert header.startswith("[plan]")
    assert "fix merge_from" in header


def test_plan_header_respects_max_bytes(store: Store) -> None:
    todo_id = _add_plan(store, goal="x" * 2000)
    ledger.set_plan_status(store, todo_id, "active", expected_revision=1, changed_by="human")
    header = ledger.plan_header(store, max_bytes=80)
    assert len(header.encode("utf-8")) <= 80


def test_plan_header_prefers_session_match(store: Store) -> None:
    t_session = _add_plan(store, goal="session plan", session_id="sess-1")
    ledger.set_plan_status(store, t_session, "active", expected_revision=1, changed_by="human")
    t_other = _add_plan(store, goal="other active plan")
    ledger.set_plan_status(store, t_other, "active", expected_revision=1, changed_by="human")

    header = ledger.plan_header(store, session_id="sess-1")
    assert "session plan" in header
    assert "other active plan" not in header


def test_plan_header_falls_back_to_repo_match(store: Store) -> None:
    t_repo = _add_plan(store, goal="repo plan", repo="/repo/a")
    ledger.set_plan_status(store, t_repo, "active", expected_revision=1, changed_by="human")

    header = ledger.plan_header(store, session_id="no-such-session", repo="/repo/a")
    assert "repo plan" in header


# ---------- CLI smoke ----------


def test_cli_todo_add_list_done(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    rc = cli.main(["--dir", str(tmp_path), "todo", "add", "--text", "ship it", "--scope", "global"])
    assert rc == 0
    todo_id = capsys.readouterr().out.strip()
    assert todo_id

    rc = cli.main(["--dir", str(tmp_path), "--json", "todo", "list", "--scope", "global"])
    assert rc == 0
    rows = json.loads(capsys.readouterr().out)
    assert any(r["todo_id"] == todo_id for r in rows)

    rc = cli.main(["--dir", str(tmp_path), "--json", "todo", "done", todo_id])
    assert rc == 0
    result = json.loads(capsys.readouterr().out)
    assert result["done"] is True


def test_cli_todo_add_with_key_is_idempotent(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    args = [
        "--dir",
        str(tmp_path),
        "todo",
        "add",
        "--text",
        "recurring probe finding",
        "--scope",
        "global",
        "--source",
        "probe:disk",
        "--key",
        "disk-full",
    ]
    cli.main(args)
    id1 = capsys.readouterr().out.strip()
    cli.main(args)
    id2 = capsys.readouterr().out.strip()
    assert id1 == id2


def test_cli_plan_add_show_check_header(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    payload = {
        "goal": "fix the flaky test",
        "stages": [
            {"name": "reproduce", "expected_check": "fails locally"},
            {"name": "fix", "expected_check": "pytest -q green"},
        ],
        "non_goals": ["rewrite the suite"],
    }
    payload_path = tmp_path / "plan.json"
    payload_path.write_text(json.dumps(payload))

    rc = cli.main(["--dir", str(tmp_path / "store"), "plan", "add", "--file", str(payload_path)])
    assert rc == 0
    todo_id = capsys.readouterr().out.strip()
    assert todo_id

    rc = cli.main(["--dir", str(tmp_path / "store"), "plan", "show", todo_id])
    assert rc == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["goal"] == "fix the flaky test"

    # check: draft plan with unfinished stages is not yet "done"-worthy, but check() only
    # reports plan_problems(), and a draft with a filled-in expected_check has none.
    rc = cli.main(["--dir", str(tmp_path / "store"), "--json", "plan", "check", todo_id])
    assert rc == 0
    problems = json.loads(capsys.readouterr().out)
    assert problems == []

    rc = cli.main(
        [
            "--dir",
            str(tmp_path / "store"),
            "plan",
            "status",
            todo_id,
            "active",
            "--revision",
            "1",
            "--by",
            "test",
        ]
    )
    assert rc == 0
    capsys.readouterr()

    rc = cli.main(["--dir", str(tmp_path / "store"), "plan", "header"])
    assert rc == 0
    header = capsys.readouterr().out.strip()
    assert "fix the flaky test" in header


def test_cli_plan_amend_conflict_exits_nonzero(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    payload = {
        "goal": "g",
        "stages": [{"name": "s", "expected_check": "c"}],
        "non_goals": [],
    }
    payload_path = tmp_path / "plan.json"
    payload_path.write_text(json.dumps(payload))
    store_dir = tmp_path / "store"

    cli.main(["--dir", str(store_dir), "plan", "add", "--file", str(payload_path)])
    todo_id = capsys.readouterr().out.strip()

    rc = cli.main(
        [
            "--dir",
            str(store_dir),
            "plan",
            "amend",
            todo_id,
            "--revision",
            "99",
            "--by",
            "test",
            "--set",
            'goal="new"',
        ]
    )
    assert rc != 0
