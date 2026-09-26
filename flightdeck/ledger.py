"""The active ledger: what is being worked on RIGHT NOW, by which agent session, in which repo,
toward what goal.

Presence is derived from heartbeat RECENCY, never from `status`. A killed session, a crashed
process, or a sleeping laptop never gets a chance to deregister itself, so a session counts as
"active" only while its `heartbeat_at` is newer than `DEFAULT_STALE_SECONDS`. `status` records
*intent* ('working' / 'blocked' / 'done'); it is written by whoever last called `beat()` and is
never trusted as liveness. A 'working' row with a heartbeat from an hour ago is a dead session
that never got to say so, not a live one.

TODO boundary: this table is the authoritative store (decided 2026-09-10). It was previously
documented as a mirror of Vikunja; that inverted where the truth lives and left the table empty
while the upstream it deferred to was unreachable. Todos are local sqlite, in the same store as
`agent_sessions`, `turns` and `events`, so a session can read them with no network hop and no
gateway between it and its own work queue. `source` names who filed a row — an agent session id,
'human', or a producer name ('blocked-job', 'probe', 'subagent'). Vikunja, if it returns, is a
downstream mirror of this table and never the other way around.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from flightdeck.ids import ulid
from flightdeck.store import Store, hostname, now_ms

# How long a heartbeat stays "live" before the session is presumed dead. Chosen against this
# hook's own cadence (fires on UserPromptSubmit / PreToolUse / Stop — i.e. on every turn, not on
# a timer), so a session idle mid-thought for a couple minutes doesn't flicker dead. Exposed as a
# parameter everywhere rather than only a module constant so callers can tune it without patching.
DEFAULT_STALE_SECONDS = 180

# Default lookback window and row cap for the "recent events" section of the ledger view.
DEFAULT_EVENTS_WINDOW_SECONDS = 3600
DEFAULT_EVENTS_LIMIT = 25
DEFAULT_TODOS_LIMIT = 50


def _git(cwd: str, *args: str) -> str | None:
    """Best-effort read-only git call. Returns None on anything short of success — no repo,
    no git binary, detached weirdness, timeout. Never raises."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    out = result.stdout.strip()
    return out or None


def _github_remote(cwd: str) -> str | None:
    """owner/name parsed out of `origin`, for both SSH and HTTPS remote forms. None if there is
    no origin, or it isn't a github.com remote we can parse."""
    url = _git(cwd, "remote", "get-url", "origin")
    if not url:
        return None
    url = url.removesuffix(".git")
    if "github.com" not in url:
        return None
    # git@github.com:owner/name  or  https://github.com/owner/name
    tail = url.split("github.com", 1)[1].lstrip(":/")
    parts = [p for p in tail.split("/") if p]
    if len(parts) < 2:
        return None
    return f"{parts[0]}/{parts[1]}"


def derive_git_context(cwd: str | None) -> dict[str, str | None]:
    """repo/branch/github_remote for a cwd, all None when cwd isn't inside a git repo (or is
    missing/unreadable). Read-only git commands only."""
    if not cwd or not Path(cwd).expanduser().exists():
        return {"repo": None, "branch": None, "github_remote": None}
    toplevel = _git(cwd, "rev-parse", "--show-toplevel")
    if not toplevel:
        return {"repo": None, "branch": None, "github_remote": None}
    # symbolic-ref works on a fresh repo with zero commits (rev-parse --abbrev-ref HEAD does
    # not: HEAD is ambiguous until something is committed); fall back to rev-parse for a
    # detached HEAD, where symbolic-ref fails by design.
    branch = _git(cwd, "symbolic-ref", "--short", "HEAD") or _git(
        cwd, "rev-parse", "--abbrev-ref", "HEAD"
    )
    return {
        "repo": toplevel,
        "branch": branch,
        "github_remote": _github_remote(cwd),
    }


def beat(
    store: Store,
    *,
    session_id: str,
    cwd: str,
    host: str | None = None,
    goal: str | None = None,
    task: str | None = None,
    agent: str = "claude-code",
    status: str = "working",
    now: int | None = None,
) -> None:
    """Upsert this session's heartbeat row. Idempotent: calling it repeatedly for the same
    (session_id, host) updates one row, never inserts a duplicate.

    `goal`/`task` are sticky when omitted: a heartbeat call that doesn't know the goal (most of
    them — see the hook) must not null out a goal set earlier in the session.
    """
    host = host or hostname()
    ts = now if now is not None else now_ms()
    git_ctx = derive_git_context(cwd)

    existing = store.conn.execute(
        "SELECT goal, task, started_at FROM agent_sessions WHERE session_id=? AND host=?",
        (session_id, host),
    ).fetchone()
    if existing is not None:
        goal = goal if goal is not None else existing["goal"]
        task = task if task is not None else existing["task"]
        started_at = existing["started_at"]
    else:
        started_at = ts

    store.conn.execute(
        """
        INSERT INTO agent_sessions
            (session_id, host, cwd, repo, github_remote, branch, goal, task, agent, status,
             started_at, heartbeat_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(session_id, host) DO UPDATE SET
            cwd=excluded.cwd,
            repo=excluded.repo,
            github_remote=excluded.github_remote,
            branch=excluded.branch,
            goal=excluded.goal,
            task=excluded.task,
            agent=excluded.agent,
            status=excluded.status,
            heartbeat_at=excluded.heartbeat_at
        """,
        (
            session_id,
            host,
            cwd,
            git_ctx["repo"],
            git_ctx["github_remote"],
            git_ctx["branch"],
            goal,
            task,
            agent,
            status,
            started_at,
            ts,
        ),
    )


def active_sessions(
    store: Store,
    *,
    within_seconds: int = DEFAULT_STALE_SECONDS,
    now: int | None = None,
) -> list[dict[str, Any]]:
    """Sessions whose heartbeat is newer than `within_seconds`. This is the ONLY definition of
    "active" in this module — never `status`. Each row carries `heartbeat_age_seconds` so a
    caller (or a human) can see how fresh "active" actually is."""
    ts = now if now is not None else now_ms()
    cutoff = ts - within_seconds * 1000
    rows = store.conn.execute(
        "SELECT * FROM agent_sessions WHERE heartbeat_at >= ? ORDER BY heartbeat_at DESC",
        (cutoff,),
    ).fetchall()
    out = []
    for row in rows:
        d = dict(row)
        d["heartbeat_age_seconds"] = round((ts - d["heartbeat_at"]) / 1000, 1)
        out.append(d)
    return out


def add_todo(
    store: Store,
    *,
    text: str,
    scope: str = "local",
    repo: str | None = None,
    source: str | None = None,
    key: str | None = None,
    now: int | None = None,
) -> str:
    """`scope='local'` requires a repo; `scope='global'` ignores it. Returns the new todo_id.

    `key`, paired with `source`, lets a producer file idempotently: a probe that re-files the
    same failure every session start gets the SAME todo_id back rather than a fresh duplicate
    row. The upsert targets the migration-16 partial unique index directly (atomic -- no
    select-then-insert race), and the follow-up SELECT recovers the winning id when this call
    lost the race. A key names a *condition*, not a filing: if the row it hits was already
    closed, the condition has recurred and the row is reopened with the fresh text -- the
    index has no status predicate, so without this a condition could only ever be filed once.
    `key=None` (the common case) skips the dedupe entirely: two ad-hoc todos with the same
    text from the same source are two different things.
    """
    if scope not in ("local", "global"):
        raise ValueError(f"scope must be 'local' or 'global', got {scope!r}")
    ts = now if now is not None else now_ms()
    todo_id = ulid(ts)
    store.conn.execute(
        """
        INSERT INTO todos (todo_id, scope, repo, text, status, source, key, created_at, updated_at)
        VALUES (?, ?, ?, ?, 'open', ?, ?, ?, ?)
        ON CONFLICT(source, key) WHERE key IS NOT NULL DO UPDATE SET
            status='open', text=excluded.text, updated_at=excluded.updated_at, done_at=NULL
        WHERE todos.status='done'
        """,
        (todo_id, scope, repo if scope == "local" else None, text, source, key, ts, ts),
    )
    if key is not None:
        existing = store.conn.execute(
            "SELECT todo_id FROM todos WHERE source IS ? AND key = ?", (source, key)
        ).fetchone()
        if existing is not None:
            return existing["todo_id"]
    return todo_id


def list_todos(
    store: Store,
    *,
    scope: str | None = None,
    repo: str | None = None,
    status: str | None = "open",
) -> list[dict[str, Any]]:
    clauses, params = ["1=1"], []
    if scope is not None:
        clauses.append("scope = ?")
        params.append(scope)
    if repo is not None:
        clauses.append("repo = ?")
        params.append(repo)
    if status is not None:
        clauses.append("status = ?")
        params.append(status)
    sql = f"SELECT * FROM todos WHERE {' AND '.join(clauses)} ORDER BY created_at DESC"
    return [dict(r) for r in store.conn.execute(sql, params)]


def done_todo(store: Store, todo_id: str, *, now: int | None = None) -> bool:
    """Marks a todo done. Returns False (no-op) if the id doesn't exist."""
    ts = now if now is not None else now_ms()
    cur = store.conn.execute(
        "UPDATE todos SET status='done', updated_at=?, done_at=? WHERE todo_id=?",
        (ts, ts, todo_id),
    )
    return cur.rowcount > 0


def render(
    store: Store,
    *,
    within_seconds: int = DEFAULT_STALE_SECONDS,
    events_window_seconds: int = DEFAULT_EVENTS_WINDOW_SECONDS,
    events_limit: int = DEFAULT_EVENTS_LIMIT,
    todos_limit: int = DEFAULT_TODOS_LIMIT,
    now: int | None = None,
) -> dict[str, Any]:
    """Assemble the at-a-glance ledger view. Never raises: an empty or partially-migrated store
    renders a well-formed empty ledger rather than blowing up the caller.

    Distinguishes "no sessions" (data model is fine, nobody is active) from "no data at all"
    (the table itself has zero rows ever, which on a store that should be receiving heartbeats
    is itself a finding) via `total_sessions_ever` and `data_present`.
    """
    ts = now if now is not None else now_ms()
    result: dict[str, Any] = {
        "rendered_at": ts,
        "stale_after_seconds": within_seconds,
        "data_present": False,
        "sessions": [],
        "total_sessions_ever": 0,
        "repos_in_play": [],
        "recent_events": [],
        "todos": {"local": [], "global": []},
        "errors": [],
    }

    try:
        result["total_sessions_ever"] = store.conn.execute(
            "SELECT COUNT(*) FROM agent_sessions"
        ).fetchone()[0]
    except Exception as exc:  # noqa: BLE001 - render must never raise
        result["errors"].append(f"agent_sessions count failed: {exc!r}")

    try:
        sessions = active_sessions(store, within_seconds=within_seconds, now=ts)
        result["sessions"] = sessions
        result["data_present"] = result["total_sessions_ever"] > 0
        repos = {
            (s.get("repo"), s.get("github_remote"))
            for s in sessions
            if s.get("repo") or s.get("github_remote")
        }
        result["repos_in_play"] = [
            {"repo": repo, "github_remote": remote} for repo, remote in sorted(repos)
        ]
    except Exception as exc:  # noqa: BLE001
        result["errors"].append(f"active_sessions failed: {exc!r}")

    try:
        since = ts - events_window_seconds * 1000
        rows = store.conn.execute(
            """
            SELECT event_id, turn_id, ts, kind, name, duration_ms, ok
            FROM events
            WHERE ts >= ?
            ORDER BY ts DESC
            LIMIT ?
            """,
            (since, events_limit),
        ).fetchall()
        result["recent_events"] = [dict(r) for r in rows]
        result["events_window_seconds"] = events_window_seconds
    except Exception as exc:  # noqa: BLE001
        result["errors"].append(f"recent_events failed: {exc!r}")

    try:
        local = list_todos(store, scope="local", status="open")[:todos_limit]
        glob = list_todos(store, scope="global", status="open")[:todos_limit]
        result["todos"] = {"local": local, "global": glob}
    except Exception as exc:  # noqa: BLE001
        result["errors"].append(f"todos failed: {exc!r}")

    return result


# ============================================================================================
# Plans: one plan = one todos row. Stages are fields of the plan, never separate todos --
# scope/repo live on the todo row only, so a plan is filterable the same way any other todo
# is, and `ledger show` doesn't sprout five entries for a four-stage plan.
# ============================================================================================

#: Body fields that hold JSON-or-null. `goal`/`stages`/`non_goals` are handled separately
#: (goal is a plain string; stages/non_goals are NOT NULL, never absent).
_PLAN_OPTIONAL_JSON_FIELDS = (
    "assumptions",
    "alternatives",
    "risks",
    "gotchas",
    "rollback",
    "observability",
    "open_questions",
    "reasoning",
)

#: Fields amend_plan may change. Deliberately excludes status/superseded_by/revision --
#: status transitions go through set_plan_status, which enforces the done-needs-evidence and
#: superseded-needs-superseded_by rules amend_plan has no business re-implementing.
PLAN_CONTENT_FIELDS = ("goal", "stages", "non_goals", *_PLAN_OPTIONAL_JSON_FIELDS)

PLAN_STATUSES = ("draft", "active", "done", "abandoned", "superseded")

_STAGE_REQUIRED = ("name", "expected_check")


class PlanConflict(Exception):
    """Raised by amend_plan/set_plan_status when `expected_revision` no longer matches the
    store -- someone else's amend landed first. Carries the actual revision so a caller can
    decide whether to re-fetch and retry. There is deliberately no lock: two amenders race on
    the UPDATE and the loser gets this, not a wait."""

    def __init__(self, todo_id: str, expected: int, actual: int) -> None:
        self.todo_id = todo_id
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"plan {todo_id} revision conflict: expected {expected}, store has {actual}"
        )


def _normalize_stages(stages: Any) -> list[dict[str, Any]]:
    """Validates shape and fills evidence/status defaults, so every stage stored ever carries
    the same four keys regardless of how much the caller supplied."""
    if not isinstance(stages, list) or not stages:
        raise ValueError("stages must be a non-empty list")
    normalized = []
    for i, stage in enumerate(stages):
        if not isinstance(stage, dict):
            raise ValueError(f"stage {i} must be an object")
        missing = [k for k in _STAGE_REQUIRED if not stage.get(k)]
        if missing:
            raise ValueError(f"stage {i} ({stage.get('name')!r}) missing {missing}")
        normalized.append(
            {
                "name": stage["name"],
                "expected_check": stage["expected_check"],
                "evidence": stage.get("evidence"),
                "status": stage.get("status", "pending"),
            }
        )
    return normalized


def _validate_non_goals(non_goals: Any) -> None:
    if not isinstance(non_goals, list):
        raise ValueError("non_goals must be a list")


def _encode_plan_field(name: str, value: Any) -> Any:
    """goal is stored as plain text; every other content field is JSON-or-null."""
    if name == "goal":
        return value
    return json.dumps(value) if value is not None else None


def _decode_plan_row(row: dict[str, Any]) -> dict[str, Any]:
    """Mutates and returns `row`: json.loads every JSON-bearing column present in it. Works on
    both a plans-only row and the todo+plan join (the latter carries extra todo_* columns this
    never touches)."""
    row["stages"] = json.loads(row["stages"])
    row["non_goals"] = json.loads(row["non_goals"])
    for field_name in _PLAN_OPTIONAL_JSON_FIELDS:
        if row.get(field_name) is not None:
            row[field_name] = json.loads(row[field_name])
    return row


def _plan_snapshot(store: Store, todo_id: str) -> dict[str, Any]:
    """The plan row alone (no todo fields), decoded -- what plan_revisions.snapshot records."""
    row = store.conn.execute("SELECT * FROM plans WHERE todo_id=?", (todo_id,)).fetchone()
    return _decode_plan_row(dict(row))


def _write_revision(
    store: Store,
    *,
    todo_id: str,
    revision: int,
    snapshot: dict[str, Any],
    changed_by: str | None,
    note: str | None,
    now: int,
) -> None:
    store.conn.execute(
        "INSERT INTO plan_revisions (todo_id, revision, snapshot, changed_by, changed_at, note) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (todo_id, revision, json.dumps(snapshot, default=str), changed_by, now, note),
    )


def add_plan(
    store: Store,
    *,
    goal: str,
    stages: list[dict[str, Any]],
    non_goals: list[str],
    scope: str = "local",
    repo: str | None = None,
    session_id: str | None = None,
    host: str | None = None,
    source: str | None = None,
    assumptions: Any = None,
    alternatives: Any = None,
    risks: Any = None,
    gotchas: Any = None,
    rollback: Any = None,
    observability: Any = None,
    open_questions: Any = None,
    reasoning: Any = None,
    status: str = "draft",
    now: int | None = None,
) -> str:
    """Create a plan: one todos row (text=goal, status='open'), one plans row (status='draft'
    unless 'active' is asked for -- a plan written at the top of the turn that executes it
    starts active, no second call), and a plan_revisions snapshot for revision 1. Returns
    the new todo_id.

    Validates goal is non-empty, stages is a non-empty list where every stage has a name and
    an expected_check, and non_goals is a list -- a plan that skipped scoping (no items, no
    check to verify against) is refused rather than silently persisted, same discipline as
    ScopePass.problems()'s "no items found" check.
    """
    if not goal.strip():
        raise ValueError("goal must not be empty")
    if status not in ("draft", "active"):
        raise ValueError(f"a new plan is 'draft' or 'active', got {status!r}")
    stages_norm = _normalize_stages(stages)
    _validate_non_goals(non_goals)

    ts = now if now is not None else now_ms()
    host = host or hostname()
    todo_id = add_todo(store, text=goal, scope=scope, repo=repo, source=source, now=ts)

    fields: dict[str, Any] = {
        "assumptions": assumptions,
        "alternatives": alternatives,
        "risks": risks,
        "gotchas": gotchas,
        "rollback": rollback,
        "observability": observability,
        "open_questions": open_questions,
        "reasoning": reasoning,
    }
    row = {
        "todo_id": todo_id,
        "goal": goal,
        "stages": json.dumps(stages_norm),
        "non_goals": json.dumps(non_goals),
        **{k: _encode_plan_field(k, v) for k, v in fields.items()},
        "status": status,
        "superseded_by": None,
        "revision": 1,
        "session_id": session_id,
        "host": host,
        "created_at": ts,
        "updated_at": ts,
    }
    store.conn.execute(
        "INSERT INTO plans "
        "(todo_id, goal, stages, non_goals, assumptions, alternatives, risks, gotchas, "
        " rollback, observability, open_questions, reasoning, status, superseded_by, "
        " revision, session_id, host, created_at, updated_at) "
        "VALUES "
        "(:todo_id, :goal, :stages, :non_goals, :assumptions, :alternatives, :risks, :gotchas, "
        " :rollback, :observability, :open_questions, :reasoning, :status, :superseded_by, "
        " :revision, :session_id, :host, :created_at, :updated_at)",
        row,
    )
    _write_revision(
        store,
        todo_id=todo_id,
        revision=1,
        snapshot=_plan_snapshot(store, todo_id),
        changed_by=source or session_id or "system",
        note="created",
        now=ts,
    )
    return todo_id


def get_plan(store: Store, todo_id: str) -> dict[str, Any] | None:
    """The plan joined with its todo row (scope/repo/todo status live there), JSON fields
    decoded. None if there is no plan for this todo_id."""
    row = store.conn.execute(
        """
        SELECT t.todo_id, t.scope, t.repo, t.status AS todo_status, t.source, t.key,
               t.created_at AS todo_created_at, t.updated_at AS todo_updated_at, t.done_at,
               p.*
        FROM todos t JOIN plans p ON p.todo_id = t.todo_id
        WHERE t.todo_id = ?
        """,
        (todo_id,),
    ).fetchone()
    if row is None:
        return None
    return _decode_plan_row(dict(row))


def list_plans(
    store: Store,
    *,
    status: str | None = None,
    scope: str | None = None,
    repo: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    clauses, params = ["1=1"], []
    if status is not None:
        clauses.append("p.status = ?")
        params.append(status)
    if scope is not None:
        clauses.append("t.scope = ?")
        params.append(scope)
    if repo is not None:
        clauses.append("t.repo = ?")
        params.append(repo)
    sql = f"""
        SELECT t.todo_id, t.scope, t.repo, t.status AS todo_status, t.source, t.key,
               t.created_at AS todo_created_at, t.updated_at AS todo_updated_at, t.done_at,
               p.*
        FROM plans p JOIN todos t ON t.todo_id = p.todo_id
        WHERE {" AND ".join(clauses)}
        ORDER BY p.updated_at DESC
        LIMIT ?
    """
    params.append(limit)
    rows = store.conn.execute(sql, params).fetchall()
    return [_decode_plan_row(dict(r)) for r in rows]


def amend_plan(
    store: Store,
    todo_id: str,
    *,
    expected_revision: int,
    changed_by: str,
    note: str | None = None,
    now: int | None = None,
    **fields: Any,
) -> int:
    """Compare-and-swap edit of a plan's content fields. `UPDATE ... WHERE revision =
    expected_revision`; zero rows updated means someone else's amend landed first, and that is
    reported as PlanConflict (carrying the actual revision) rather than silently overwritten.
    Never a lock -- concurrent amenders race on the UPDATE, the loser gets the conflict. On
    success, bumps revision and writes a plan_revisions snapshot. Returns the new revision.
    """
    if not fields:
        raise ValueError("amend_plan called with no fields to change")
    unknown = set(fields) - set(PLAN_CONTENT_FIELDS)
    if unknown:
        raise ValueError(f"unknown plan field(s): {sorted(unknown)}")

    current = get_plan(store, todo_id)
    if current is None:
        raise ValueError(f"no plan for todo_id {todo_id!r}")

    if "stages" in fields:
        fields["stages"] = _normalize_stages(fields["stages"])
    if "non_goals" in fields:
        _validate_non_goals(fields["non_goals"])
    if "goal" in fields and not str(fields["goal"]).strip():
        raise ValueError("goal must not be empty")

    ts = now if now is not None else now_ms()
    new_revision = expected_revision + 1
    set_cols = [f"{name} = :{name}" for name in fields]
    set_cols += ["revision = :new_revision", "updated_at = :updated_at"]
    params: dict[str, Any] = {
        "todo_id": todo_id,
        "expected_revision": expected_revision,
        "new_revision": new_revision,
        "updated_at": ts,
        **{name: _encode_plan_field(name, value) for name, value in fields.items()},
    }
    cur = store.conn.execute(
        f"UPDATE plans SET {', '.join(set_cols)} "
        "WHERE todo_id = :todo_id AND revision = :expected_revision",
        params,
    )
    if cur.rowcount == 0:
        actual = store.conn.execute(
            "SELECT revision FROM plans WHERE todo_id = ?", (todo_id,)
        ).fetchone()
        raise PlanConflict(todo_id, expected_revision, actual["revision"] if actual else -1)

    _write_revision(
        store,
        todo_id=todo_id,
        revision=new_revision,
        snapshot=_plan_snapshot(store, todo_id),
        changed_by=changed_by,
        note=note,
        now=ts,
    )
    return new_revision


def set_plan_status(
    store: Store,
    todo_id: str,
    status: str,
    *,
    expected_revision: int,
    changed_by: str,
    superseded_by: str | None = None,
    note: str | None = None,
    now: int | None = None,
) -> int:
    """CAS status transition, same conflict semantics as amend_plan. `done` requires every
    stage to carry non-empty evidence -- a plan marked done on vibes is the exact failure mode
    the stage/evidence split exists to catch. `superseded` requires `superseded_by`: a
    superseded plan with nothing pointing at its replacement is a dead end, not a chain.
    Marking done also closes the underlying todo (done_todo) -- a done plan that still shows
    up as an open todo is the same "ghost" failure mode `agent_sessions` liveness guards
    against, just on the todo table instead.
    """
    if status not in PLAN_STATUSES:
        raise ValueError(f"status must be one of {PLAN_STATUSES}, got {status!r}")
    plan = get_plan(store, todo_id)
    if plan is None:
        raise ValueError(f"no plan for todo_id {todo_id!r}")
    if plan["revision"] != expected_revision:
        raise PlanConflict(todo_id, expected_revision, plan["revision"])

    if status == "done":
        missing = [s["name"] for s in plan["stages"] if not (s.get("evidence") or "").strip()]
        if missing:
            raise ValueError(f"cannot mark done: stages missing evidence: {missing}")
    if status == "superseded" and not superseded_by:
        raise ValueError("status 'superseded' requires superseded_by")

    ts = now if now is not None else now_ms()
    new_revision = expected_revision + 1
    cur = store.conn.execute(
        "UPDATE plans SET status=:status, superseded_by=:superseded_by, revision=:new_revision, "
        "updated_at=:updated_at WHERE todo_id=:todo_id AND revision=:expected_revision",
        {
            "status": status,
            "superseded_by": superseded_by,
            "new_revision": new_revision,
            "updated_at": ts,
            "todo_id": todo_id,
            "expected_revision": expected_revision,
        },
    )
    if cur.rowcount == 0:
        actual = store.conn.execute(
            "SELECT revision FROM plans WHERE todo_id = ?", (todo_id,)
        ).fetchone()
        raise PlanConflict(todo_id, expected_revision, actual["revision"] if actual else -1)

    _write_revision(
        store,
        todo_id=todo_id,
        revision=new_revision,
        snapshot=_plan_snapshot(store, todo_id),
        changed_by=changed_by,
        note=note,
        now=ts,
    )
    if status == "done":
        done_todo(store, todo_id, now=ts)
    return new_revision


def plan_problems(plan: dict[str, Any]) -> list[str]:
    """Everything wrong with a plan dict (as returned by get_plan/list_plans), all at once --
    same shape as ScopePass.problems(): return every issue, not just the first, so a caller
    doesn't have to fix-and-rerun to discover the second one. Pure function, no store access,
    so it can run against a plan a caller already has in hand.
    """
    issues: list[str] = []
    stages = plan.get("stages") or []
    status = plan.get("status")

    if status == "active" and not stages:
        issues.append("active plan has zero stages")

    for i, stage in enumerate(stages):
        label = stage.get("name") or f"stage {i}"
        if not stage.get("expected_check"):
            issues.append(f"stage {label!r} has no expected_check")
        if stage.get("status") == "done" and not (stage.get("evidence") or "").strip():
            issues.append(f"stage {label!r} marked done without evidence")

    if status == "done":
        missing = [
            stage.get("name") or f"stage {i}"
            for i, stage in enumerate(stages)
            if not (stage.get("evidence") or "").strip()
        ]
        if missing:
            issues.append(f"plan marked done but stages lack evidence: {missing}")

    if status == "superseded" and not plan.get("superseded_by"):
        issues.append("plan marked superseded with no superseded_by")

    revision = plan.get("revision")
    if revision is None or revision < 1:
        issues.append(f"revision must be >= 1, got {revision!r}")

    return issues


def active_plan(
    store: Store,
    *,
    session_id: str | None = None,
    repo: str | None = None,
) -> dict[str, Any] | None:
    """The active plan most relevant to this session/repo -- prefers a session match, falls
    back to a repo match, then the newest active plan overall; None when there is none.
    Split out of `plan_header` so a caller can key on (todo_id, revision) -- "have I shown
    this revision yet" -- using the same selection the header line is built from."""
    plan = None
    if session_id is not None:
        row = store.conn.execute(
            "SELECT todo_id FROM plans WHERE status='active' AND session_id=? "
            "ORDER BY updated_at DESC LIMIT 1",
            (session_id,),
        ).fetchone()
        if row:
            plan = get_plan(store, row["todo_id"])
    if plan is None and repo is not None:
        row = store.conn.execute(
            "SELECT p.todo_id FROM plans p JOIN todos t ON t.todo_id = p.todo_id "
            "WHERE p.status='active' AND t.repo=? ORDER BY p.updated_at DESC LIMIT 1",
            (repo,),
        ).fetchone()
        if row:
            plan = get_plan(store, row["todo_id"])
    if plan is None:
        row = store.conn.execute(
            "SELECT todo_id FROM plans WHERE status='active' ORDER BY updated_at DESC LIMIT 1"
        ).fetchone()
        if row:
            plan = get_plan(store, row["todo_id"])
    return plan


def plan_header(
    store: Store,
    *,
    session_id: str | None = None,
    repo: str | None = None,
    max_bytes: int = 500,
    plan: dict[str, Any] | None = None,
) -> str:
    """One derived line for `plan`, or for `active_plan(...)` when none is passed; empty
    string when there is none. Hard-capped at max_bytes: this is what the delta header
    embeds, so it earns its bytes by naming what's active and what's blocking it, never by
    restating the plan body.
    """
    if plan is None:
        plan = active_plan(store, session_id=session_id, repo=repo)
    if plan is None:
        return ""

    stages = plan.get("stages") or []
    total = len(stages)
    done = sum(1 for s in stages if s.get("status") == "done")
    current = next((s for s in stages if s.get("status") != "done"), None)
    if current is not None:
        stage_desc = f'stage {done + 1}/{total} "{current["name"]}"'
    else:
        stage_desc = f"{done}/{total} done"

    problem_count = len(plan_problems(plan))
    problem_desc = ""
    if problem_count:
        noun = "issue" if problem_count == 1 else "issues"
        problem_desc = f" · {problem_count} {noun}"

    line = f"[plan] {plan['goal']} · {stage_desc}{problem_desc}"
    encoded = line.encode("utf-8")
    if len(encoded) > max_bytes:
        encoded = encoded[:max_bytes]
        # Don't cut mid-codepoint: back off while we're inside a UTF-8 continuation byte.
        while encoded and (encoded[-1] & 0xC0) == 0x80:
            encoded = encoded[:-1]
        line = encoded.decode("utf-8", errors="ignore")
    return line
