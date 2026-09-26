#!/usr/bin/env python3
"""UserPromptSubmit hook: a compact "what changed" header, shown once per new fact.

Must never fail a Claude Code turn -- always exits 0, whatever stdin holds or breaks.

Silence is the default, not a fallback. Per-turn injections land after the prompt-cache
prefix and bill as uncached tokens every single turn they fire, and this fleet has already
measured what an unconditional injection costs: see scope-hook.py's docstring (an
always-on ceremony step measurably dropped file delivery from 3-4/5 to 0-2/5). This hook
speaks only when the ledger holds something this session has not already been shown --
never on the strength of the prompt alone.

Output is plain text on stdout, matching what scope-hook.py actually does (verified by
reading it): Claude Code folds a UserPromptSubmit hook's stdout into context on exit 0.
There is no JSON envelope to build or parse.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from flightdeck import ledger  # noqa: E402
from flightdeck.ledger_schedule import stale_or_failing  # noqa: E402
from flightdeck.store import DEFAULT_DIR, Store, hostname  # noqa: E402

ERROR_LOG = Path(DEFAULT_DIR).expanduser() / "hook-errors.log"
STATE_SUBDIR = "delta"
CURSOR_BUCKETS = (
    "todo_ids",
    "session_keys",
    "repo_keys",
    "blocking_keys",
    "failed_job_keys",
    "plan_keys",
)

# The hard cap on emitted context, per the spec. Generous enough for a handful of named
# items, small enough that a runaway ledger can't turn this into the very ceremony tax the
# module docstring describes. print() adds one more '\n' byte after the text this hook
# builds, so builders target CAP - 1 and the printed line still lands at or under CAP.
BYTE_CAP = 900
_BUILD_CAP = BYTE_CAP - 1

# A blocked job younger than this is very likely mid-flight background work catching its
# breath -- most jobs on this fleet clear a blocked state on their own within minutes.
# 30 minutes is long enough to filter that noise and short enough that a real
# waiting-on-you case still surfaces inside the same work session it started in, not a
# day later. This is the "27h yes, 4 minutes no" line from the spec, turned into a number.
BLOCKING_AGE_THRESHOLD_SECONDS = 30 * 60

# Caps how many individual blocking items earn their own line, so an "absurdly large
# ledger" can't turn the block tier into an unbounded list before the byte cap even gets a
# turn to trim it. Anything past this collapses into one "+N more" line.
MAX_BLOCKING_LINES = 5

# Any single rendered line is clipped here, at a word boundary, before the byte-fit pass --
# one job's `needs` text (the real fleet's ran 190 chars, a rate-limit message) must not be
# able to crowd out every other category on its own.
MAX_LINE_CHARS = 160

CLAUDE_JOBS_DIR = Path(os.environ.get("SPARKY_CLAUDE_JOBS_DIR", "~/.claude/jobs")).expanduser()


def _log_error(context: str, exc: BaseException) -> None:
    """Best effort. A logging failure must never cost the user their turn."""
    try:
        ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
        with ERROR_LOG.open("a") as handle:
            handle.write(
                json.dumps(
                    {
                        "ts": datetime.now(UTC).isoformat(),
                        "hook": "delta-hook",
                        "context": context,
                        "error": repr(exc),
                    }
                )
                + "\n"
            )
    except OSError:
        pass


# ------------------------------------------------------------------ per-session cursor


def _cursor_path(session_id: str, directory: Path | str = DEFAULT_DIR) -> Path:
    return Path(directory).expanduser() / STATE_SUBDIR / f"{session_id}.json"


def _load_cursor(session_id: str, directory: Path | str = DEFAULT_DIR) -> dict[str, list[str]]:
    """Keys this session has already been shown, per category. Missing or corrupt reads as
    every key unseen -- deliberately: the failure to fail toward here is silence about a
    live blocker, not one repeated line."""
    empty = {bucket: [] for bucket in CURSOR_BUCKETS}
    path = _cursor_path(session_id, directory)
    if not path.exists():
        return empty
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return empty
    if not isinstance(raw, dict):
        return empty
    for bucket in CURSOR_BUCKETS:
        value = raw.get(bucket)
        if isinstance(value, list):
            empty[bucket] = [str(v) for v in value]
    return empty


def _save_cursor(
    session_id: str, cursor: dict[str, list[str]], directory: Path | str = DEFAULT_DIR
) -> None:
    path = _cursor_path(session_id, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cursor))


def _advance_cursor(
    cursor: dict[str, list[str]], shown: dict[str, set[str]]
) -> dict[str, list[str]]:
    """Only keys that actually made it into the printed text are marked seen. A category
    dropped by the byte-fit pass stays unseen, so it is still eligible -- and still at the
    front of the queue, since it is already the oldest -- on a future turn."""
    merged = {bucket: set(cursor.get(bucket, [])) for bucket in CURSOR_BUCKETS}
    for bucket, keys in shown.items():
        merged[bucket].update(keys)
    return {bucket: sorted(merged[bucket]) for bucket in CURSOR_BUCKETS}


# ------------------------------------------------------------------ text shaping


def _clip(text: str, max_chars: int = MAX_LINE_CHARS) -> str:
    """Trim at the last whitespace before max_chars -- never mid-word. Collapses embedded
    whitespace runs first so a multi-line `needs` string still reads as one line."""
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    head = text[:max_chars]
    cut = head.rfind(" ")
    return (head[:cut] if cut > 0 else head) + "…"


def _fit_lines(
    header: str, items: list[dict[str, Any]], cap: int = _BUILD_CAP
) -> tuple[str, dict[str, set[str]]]:
    """Add already priority-ordered `items` (each {"line", "keys", "bucket"}) to `header`,
    one whole line at a time, stopping before the byte cap is crossed. Whole lines are
    dropped, never sliced -- that is what keeps truncation deterministic and free of
    mid-token garbage."""
    text = header
    shown: dict[str, set[str]] = {}
    for item in items:
        candidate = text + "\n" + item["line"]
        if len(candidate.encode()) > cap:
            break
        text = candidate
        shown.setdefault(item["bucket"], set()).update(item["keys"])
    return text, shown


def _fit_parts(
    prefix: str, items: list[dict[str, Any]], cap: int = _BUILD_CAP
) -> tuple[str, dict[str, set[str]]]:
    """Same contract as `_fit_lines`, joining priority-ordered {"part", "keys", "bucket"}
    items onto one line with a middot separator instead of a newline."""
    text = prefix
    shown: dict[str, set[str]] = {}
    for item in items:
        sep = " " if text == prefix else " · "
        candidate = text + sep + item["part"]
        if len(candidate.encode()) > cap:
            break
        text = candidate
        shown.setdefault(item["bucket"], set()).update(item["keys"])
    return text, shown


# ------------------------------------------------------------------ blocking jobs (~/.claude/jobs)


def _job_age_seconds(data: dict[str, Any], now: datetime) -> float:
    """Age since the job last updated. A missing or unparseable timestamp fails toward
    showing (treated as infinitely old) -- silence about a real blocker is the worse
    failure of the two here."""
    stamp = data.get("updatedAt") or data.get("createdAt")
    if not stamp:
        return float("inf")
    try:
        when = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
    except ValueError:
        return float("inf")
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return (now - when).total_seconds()


def _blocking_items(jobs_dir: Path, now: datetime) -> list[dict[str, Any]]:
    """Blocked background jobs from ~/.claude/jobs, aged past the threshold. Defensive by
    design: this directory belongs to the CLI, not this hook, so a missing dir or one
    corrupt state.json is a skip, never a crash."""
    items: list[dict[str, Any]] = []
    if not jobs_dir.is_dir():
        return items
    for job_dir in sorted(jobs_dir.iterdir()):
        state_path = job_dir / "state.json"
        if not state_path.is_file():
            continue
        try:
            data = json.loads(state_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict) or data.get("state") != "blocked":
            continue
        age_s = _job_age_seconds(data, now)
        if age_s < BLOCKING_AGE_THRESHOLD_SECONDS:
            continue
        needs = str(data.get("needs") or data.get("detail") or "waiting on a decision")
        name = str(data.get("name") or job_dir.name)
        content_key = hashlib.sha1(needs.encode()).hexdigest()[:8]
        items.append(
            {
                "key": f"{job_dir.name}:{content_key}",
                "name": name,
                "needs": needs,
                "age_hours": age_s / 3600 if age_s != float("inf") else 999.0,
            }
        )
    return items


def _failed_job_key(finding: dict[str, Any]) -> str:
    ident = (
        finding.get("job_id")
        or f"{finding.get('source')}|{finding.get('host')}|{finding.get('name')}"
    )
    reasons_sig = hashlib.sha1("|".join(finding.get("reasons") or []).encode()).hexdigest()[:8]
    return f"{ident}:{reasons_sig}"


# ------------------------------------------------------------------ delta computation


def _new(rows: list[dict[str, Any]], key_fn: Any, seen: set[str]) -> list[dict[str, Any]]:
    return [row for row in rows if key_fn(row) not in seen]


def _block_candidates(
    new_blocking: list[dict[str, Any]],
    new_todos: list[dict[str, Any]],
    new_sessions: list[dict[str, Any]],
    new_repos: list[dict[str, Any]],
    new_failed: list[tuple[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Content priority when trimming to fit: blocking decisions > new todos > other live
    agent sessions > repos that moved > failed scheduled jobs -- callers hand items in that
    order and `_fit_lines` keeps it by constructions (it only ever drops from the tail)."""
    items: list[dict[str, Any]] = []

    shown_blocking = new_blocking[:MAX_BLOCKING_LINES]
    overflow = new_blocking[MAX_BLOCKING_LINES:]
    for b in shown_blocking:
        line = _clip(f"- {b['name']}: {b['needs']} (waiting {b['age_hours']:.0f}h)")
        items.append({"line": line, "keys": [b["key"]], "bucket": "blocking_keys"})
    if overflow:
        items.append(
            {
                "line": f"- +{len(overflow)} more blocking",
                "keys": [b["key"] for b in overflow],
                "bucket": "blocking_keys",
            }
        )

    if new_todos:
        sample = ", ".join(_clip(t["text"], 40) for t in new_todos[:2])
        n = len(new_todos)
        line = f"- {n} new todo{'s' if n != 1 else ''}" + (f": {sample}" if sample else "")
        items.append(
            {"line": _clip(line), "keys": [t["todo_id"] for t in new_todos], "bucket": "todo_ids"}
        )

    if new_sessions:
        names = ", ".join(
            _clip(s.get("goal") or s.get("repo") or s.get("cwd") or "?", 40)
            for s in new_sessions[:2]
        )
        n = len(new_sessions)
        line = f"- {n} other session{'s' if n != 1 else ''} active" + (
            f": {names}" if names else ""
        )
        items.append(
            {
                "line": _clip(line),
                "keys": [f"{s['session_id']}|{s['host']}" for s in new_sessions],
                "bucket": "session_keys",
            }
        )

    if new_repos:
        names = ", ".join(
            _clip(Path(r["repo"]).name if r.get("repo") else (r.get("github_remote") or "?"))
            for r in new_repos[:3]
        )
        n = len(new_repos)
        line = _clip(f"- {n} repo{'s' if n != 1 else ''} moved: {names}")
        items.append(
            {
                "line": line,
                "keys": [f"{r['repo']}|{r['github_remote']}" for r in new_repos],
                "bucket": "repo_keys",
            }
        )

    if new_failed:
        names = ", ".join(
            _clip(f.get("name") or f.get("source") or "?", 30) for _, f in new_failed[:3]
        )
        n = len(new_failed)
        line = _clip(f"- {n} scheduled job{'s' if n != 1 else ''} failing: {names}")
        items.append(
            {"line": line, "keys": [k for k, _ in new_failed], "bucket": "failed_job_keys"}
        )

    return items


def _line_candidates(
    new_todos: list[dict[str, Any]],
    new_sessions: list[dict[str, Any]],
    new_repos: list[dict[str, Any]],
    new_failed: list[tuple[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Same priority order as `_block_candidates`, minus blocking (this tier is only ever
    reached when there is none) -- counts only, no examples, to keep the single line short."""
    items: list[dict[str, Any]] = []
    if new_todos:
        n = len(new_todos)
        items.append(
            {"part": f"{n} open", "keys": [t["todo_id"] for t in new_todos], "bucket": "todo_ids"}
        )
    if new_sessions:
        n = len(new_sessions)
        items.append(
            {
                "part": f"{n} session{'s' if n != 1 else ''}",
                "keys": [f"{s['session_id']}|{s['host']}" for s in new_sessions],
                "bucket": "session_keys",
            }
        )
    if new_repos:
        n = len(new_repos)
        items.append(
            {
                "part": f"{n} repo{'s' if n != 1 else ''} moved",
                "keys": [f"{r['repo']}|{r['github_remote']}" for r in new_repos],
                "bucket": "repo_keys",
            }
        )
    if new_failed:
        n = len(new_failed)
        items.append(
            {
                "part": f"{n} job{'s' if n != 1 else ''} failing",
                "keys": [k for k, _ in new_failed],
                "bucket": "failed_job_keys",
            }
        )
    return items


def _delta(
    store: Store,
    rendered: dict[str, Any],
    session_id: str,
    cwd: str,
    now: datetime,
    cursor: dict[str, list[str]],
) -> tuple[str | None, dict[str, set[str]]]:
    own_host = hostname()
    git_ctx = ledger.derive_git_context(cwd)
    own_repo_key = (
        f"{git_ctx['repo']}|{git_ctx.get('github_remote')}" if git_ctx.get("repo") else None
    )

    new_blocking = [
        b
        for b in _blocking_items(CLAUDE_JOBS_DIR, now)
        if b["key"] not in set(cursor["blocking_keys"])
    ]

    todos = rendered["todos"]["local"] + rendered["todos"]["global"]
    new_todos = _new(todos, lambda t: t["todo_id"], set(cursor["todo_ids"]))

    other_sessions = [
        s
        for s in rendered["sessions"]
        if not (s["session_id"] == session_id and s["host"] == own_host)
    ]
    new_sessions = _new(
        other_sessions, lambda s: f"{s['session_id']}|{s['host']}", set(cursor["session_keys"])
    )

    other_repos = [
        r for r in rendered["repos_in_play"] if f"{r['repo']}|{r['github_remote']}" != own_repo_key
    ]
    new_repos = _new(
        other_repos, lambda r: f"{r['repo']}|{r['github_remote']}", set(cursor["repo_keys"])
    )

    failed = [(_failed_job_key(f), f) for f in stale_or_failing(store)]
    seen_failed = set(cursor["failed_job_keys"])
    new_failed = [(k, f) for k, f in failed if k not in seen_failed]

    # The active plan is keyed on (todo_id, revision): it is re-shown when a stage lands or
    # the plan is amended, and silent in between. Reciting it every turn is the ceremony
    # tax the module docstring warns about; reciting it on every revision follows progress
    # instead of the clock, which is the drift check that actually pays for its bytes.
    plan = ledger.active_plan(store, session_id=session_id, repo=git_ctx.get("repo"))
    plan_key = f"{plan['todo_id']}:{plan['revision']}" if plan else None
    new_plan = plan_key is not None and plan_key not in set(cursor["plan_keys"])

    if not (new_plan or new_blocking or new_todos or new_sessions or new_repos or new_failed):
        return None, {}

    plan_line = ledger.plan_header(store, plan=plan) if new_plan else ""
    cap = _BUILD_CAP - (len(plan_line.encode()) + 1 if plan_line else 0)

    if new_blocking:
        candidates = _block_candidates(new_blocking, new_todos, new_sessions, new_repos, new_failed)
        text, shown = _fit_lines("[ledger] blocking:", candidates, cap=cap)
    elif new_todos or new_sessions or new_repos or new_failed:
        candidates = _line_candidates(new_todos, new_sessions, new_repos, new_failed)
        text, shown = _fit_parts("[ledger]", candidates, cap=cap)
    else:
        text, shown = "", {}

    if plan_line:
        text = plan_line + ("\n" + text if text else "")
        shown["plan_keys"] = {plan_key}

    return text, shown


def _emit(session_id: str, cwd: str) -> str | None:
    now = datetime.now(UTC)
    cursor = _load_cursor(session_id)
    with Store() as store:
        rendered = ledger.render(store)
        text, shown = _delta(store, rendered, session_id, cwd, now, cursor)
    if text is None:
        return None
    _save_cursor(session_id, _advance_cursor(cursor, shown))
    return text


def main() -> int:
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
    except Exception:
        return 0
    if not isinstance(payload, dict):
        return 0
    if payload.get("hook_event_name") not in (None, "UserPromptSubmit"):
        return 0

    session_id = payload.get("session_id")
    if not session_id:
        return 0
    cwd = payload.get("cwd") or ""

    try:
        text = _emit(str(session_id), cwd)
    except Exception as exc:
        _log_error(f"emit for session={session_id!r}", exc)
        return 0

    if text:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
