#!/usr/bin/env python3
"""UserPromptSubmit hook: decide whether this request earns a scoping pass, and say so.

Must never fail a Claude Code turn -- always exits 0, whatever stdin holds or breaks.

Silence is the common case and the point. 59% of task-start turns and 85% of mid-task
turns get no injection at all, because ceremony is not free: injecting a spec step into a
one-shot run measurably dropped file creation from 3-4/5 to 0-2/5 on this fleet. A hook
that spoke on every turn would reproduce exactly that.

The instruction to BUILD is the last line of every injection. An instruction of the form
"do X before the task" needs its continuation stated at the end, or the model treats the
loud first instruction as the whole job and the turn ends with the scope written and
nothing built.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from flightdeck.ledger import derive_git_context  # noqa: E402
from flightdeck.scope_gate import MINI, NONE, classify  # noqa: E402
from flightdeck.scope_input import synthetic_reason  # noqa: E402
from flightdeck.scope_pass import ScopePass  # noqa: E402
from flightdeck.store import DEFAULT_DIR  # noqa: E402

# The CLI is not installed as a console script anywhere on the fleet; this is the one
# invocation that works from any cwd, so it is the one the guidance below spells out.
PLAN_CMD = f'PYTHONPATH="{_REPO_ROOT}" python3 -m flightdeck'

GATE_LOG = Path(DEFAULT_DIR).expanduser() / "scope" / "gate-log.jsonl"

MINI_GUIDANCE = """[scope: {reasons}]
This request does not fully specify itself. Before building, state briefly:
- the goal, one sentence
- what you will build, each with how it will be checked
- what you are deliberately NOT doing
- any default you had to pick because the request did not say
Ask at most ONE question, and only if a different answer changes what you build.
Then build it in this same turn."""

FULL_GUIDANCE = """[scope: {reasons}]
This request is open-ended and spans more than one surface. Before building, state:
- the goal, one sentence
- what you will build, each with how it will be checked
- what you are deliberately NOT doing, and why
- any default you had to pick because the request did not say
Then widen once, briefly: what would a competent engineer assume is included that
was not said -- error paths, empty and failure states, permissions, migration,
rollback, observability? Add what belongs; list the rest as explicit non-goals.
Ask at most THREE questions, and only where a different answer changes the work.
Persist that as ONE plan -- the plan is the todo; its stages are not todos -- in one command:
  {plan_cmd} plan add <<'EOF'
  {{"goal": "...", "status": "active", "session_id": "{session_id}", {scope_json},
   "stages": [{{"name": "...", "expected_check": "..."}}], "non_goals": ["..."],
   "assumptions": [], "risks": [], "rollback": null, "open_questions": []}}
  EOF
As each stage lands, amend with its evidence (the next prompt shows where you are):
  {plan_cmd} plan amend <id> --revision N --by {session_id} --set 'stages=[...]'
Then build it in this same turn."""


def _log(record: dict) -> None:
    """Best effort. A logging failure must never cost the user their turn."""
    try:
        GATE_LOG.parent.mkdir(parents=True, exist_ok=True)
        with GATE_LOG.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
    except OSError:
        pass


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

    prompt = payload.get("prompt") or ""
    session_id = payload.get("session_id") or "unknown"
    cwd = payload.get("cwd") or ""
    if not prompt.strip():
        return 0

    # Not everything arriving on UserPromptSubmit was typed by a human. Task notifications,
    # system reminders, slash commands and this hook's own injections all land here in the
    # same shape. Measured on the first live log: 20 of 27 decisions (74%) were made on
    # such text, and the ceremonial rate read 89% against a design target near 40%.
    # Suppression is LOGGED, not silent -- a filter that drops traffic invisibly cannot be
    # told apart from a filter that is broken.
    suppressed = synthetic_reason(prompt)
    if suppressed:
        _log(
            {
                "ts": datetime.now(UTC).isoformat(),
                "session_id": session_id,
                "cwd": cwd,
                "tier": None,
                "suppressed": suppressed,
                "prompt_sha1": hashlib.sha1(prompt.encode()).hexdigest()[:12],
                "prompt_len": len(prompt),
            }
        )
        return 0

    try:
        active = ScopePass.load(session_id)
        has_active = bool(active and active.is_fresh_for(cwd))
        decision = classify(prompt, has_active_scope=has_active)
    except Exception as exc:
        # Fail-open (never block the turn) but NOT fail-silent: a swallowed exception here
        # previously left no trace at all, indistinguishable from "nothing happened".
        _log(
            {
                "ts": datetime.now(UTC).isoformat(),
                "session_id": session_id,
                "cwd": cwd,
                "tier": None,
                "error": f"gate raised: {exc!r}",
                "prompt_sha1": hashlib.sha1(prompt.encode()).hexdigest()[:12],
                "prompt_len": len(prompt),
            }
        )
        return 0

    # One ts and one sha for BOTH the log line and the pass: scope_ingest seeds its record
    # id from the logged values, so a pass carrying a freshly-generated timestamp would hash
    # to a different id and duplicate the row it is supposed to be.
    decision_ts = datetime.now(UTC).isoformat()
    prompt_sha1 = hashlib.sha1(prompt.encode()).hexdigest()[:12]
    _log(
        {
            "ts": decision_ts,
            "session_id": session_id,
            "cwd": cwd,
            "tier": decision.tier,
            "reasons": decision.reasons,
            "has_active_scope": has_active,
            "prompt_sha1": prompt_sha1,
            "prompt_len": len(prompt),
        }
    )

    if decision.tier == NONE:
        return 0  # say nothing; this is the majority path

    # A non-NONE decision earns scoping ceremony -- and gives the KPI layer something real
    # to close later. Reuse a still-fresh pass rather than clobbering in-progress found
    # items; open a new one otherwise. Best-effort: a save failure must not cost the turn,
    # but it is logged, not swallowed.
    if not has_active:
        try:
            ScopePass.open(
                session_id, cwd, prompt.strip()[:2000], decision,
                gate_ts=decision_ts, prompt_sha1=prompt_sha1,
            ).save()
        except Exception as exc:
            _log(
                {
                    "ts": datetime.now(UTC).isoformat(),
                    "session_id": session_id,
                    "cwd": cwd,
                    "tier": decision.tier,
                    "error": f"open/save raised: {exc!r}",
                }
            )

    reasons = "; ".join(decision.reasons)
    if decision.tier == MINI:
        print(MINI_GUIDANCE.format(reasons=reasons))
        return 0
    repo = derive_git_context(cwd).get("repo")
    scope_json = f'"scope": "local", "repo": {json.dumps(repo)}' if repo else '"scope": "global"'
    print(
        FULL_GUIDANCE.format(
            reasons=reasons, plan_cmd=PLAN_CMD, session_id=session_id, scope_json=scope_json
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
