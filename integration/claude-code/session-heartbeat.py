#!/usr/bin/env python3
"""Heartbeat hook: keeps the active ledger's `agent_sessions` row for this Claude Code session
current. Must never fail a Claude Code turn -- always exits 0, whatever stdin holds or breaks.

Bound to UserPromptSubmit, PreToolUse, and Stop:
- UserPromptSubmit fires once per user turn and is also the only event that carries the actual
  prompt text, so it is the one place this hook can make a best-effort stab at `goal` -- see
  below.
- PreToolUse fires many times per turn during active work, which is what keeps the heartbeat
  fresh enough for `ledger.DEFAULT_STALE_SECONDS` to mean something while an agent is mid-tool-use
  and not about to touch UserPromptSubmit again for a while.
- Stop marks the end of a turn; heartbeating there means a session that goes quiet between turns
  doesn't read as *more* stale than it is the instant the turn actually ends.
PostToolUse/PostToolUseFailure are deliberately skipped -- PreToolUse already heartbeats once per
tool call, and doubling that adds writes without changing what `active_sessions()` reports (it
only cares about recency, not count).

Honest limitation on goal/task: neither is in ANY Claude Code hook payload. There is no field to
read them from. What this hook actually does:
  - `goal`: only set on a session's FIRST heartbeat (i.e. when no row exists yet for
    (session_id, host)), taken as the first ~120 chars of the first UserPromptSubmit prompt.
    That is a guess about intent, not a real goal -- it is often right for a fresh session and
    almost always wrong once the conversation has moved on. It is never overwritten afterward.
  - `task`: left NULL by this hook, always. There is no signal in any of these payloads that
    describes "what is happening right now" better than the tool name PreToolUse already gives
    `flightdeck.collect_claude`, and guessing from an arbitrary tool call would be noise more
    than signal.
  - A real `goal`/`task` has to come from somewhere that actually knows them: a human editing the
    row directly (`python -m flightdeck ledger set-task ...`, once that CLI verb exists), or a
    future integration point that has explicit access to plan/task state (e.g. a TodoWrite
    payload, if one is ever exposed to hooks). This hook does not invent one.
"""

from __future__ import annotations

import json
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from flightdeck import ledger  # noqa: E402
from flightdeck.store import DEFAULT_DIR, Store  # noqa: E402

ERROR_LOG = Path(DEFAULT_DIR).expanduser() / "hook-errors.log"

_HANDLED_EVENTS = {"UserPromptSubmit", "PreToolUse", "Stop"}


def _fail_loud(context: str, exc: BaseException) -> None:
    """The whole point of this function: an exception here must be visible somewhere, not just
    swallowed into a silent exit 0. This codebase has shipped hooks before that passed every
    check while doing nothing -- see flightdeck/probes.py's collector_heartbeat probe."""
    try:
        ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
        with ERROR_LOG.open("a") as handle:
            handle.write(
                json.dumps(
                    {
                        "ts": datetime.now(UTC).isoformat(),
                        "hook": "session-heartbeat",
                        "context": context,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                + "\n"
            )
    except OSError:
        # Even the error log can't be trusted to exist. There is nothing left to do but not
        # crash the user's turn over it.
        pass


def main() -> int:
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
    except Exception as exc:
        _fail_loud("stdin parse", exc)
        return 0
    if not isinstance(payload, dict):
        _fail_loud("payload not a dict", TypeError(f"got {type(payload).__name__}"))
        return 0

    event = payload.get("hook_event_name")
    if event not in _HANDLED_EVENTS:
        return 0

    session_id = payload.get("session_id")
    cwd = payload.get("cwd")
    if not session_id or not cwd:
        _fail_loud("missing session_id/cwd", ValueError(f"session_id={session_id!r} cwd={cwd!r}"))
        return 0

    status = "done" if event == "Stop" else "working"

    try:
        with Store() as store:
            existing = store.conn.execute(
                "SELECT 1 FROM agent_sessions WHERE session_id=? AND host=?",
                (session_id, ledger.hostname()),
            ).fetchone()
            goal = None
            if existing is None and event == "UserPromptSubmit":
                prompt = (payload.get("prompt") or "").strip()
                if prompt:
                    goal = prompt[:120]
            ledger.beat(
                store,
                session_id=session_id,
                cwd=cwd,
                agent="claude-code",
                status=status,
                goal=goal,
            )
    except Exception as exc:
        _fail_loud(f"beat() on event={event}", exc)
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
