"""Claude Code hook collector: turns Claude Code hook events into the same schema the Go crush
emitter writes, so the nightly review can compare local-model turns against frontier turns on
equivalent task shapes.

Each hook fires as its own short-lived process (confirmed against the current Claude Code hooks
docs: no timestamp field and no model id in any payload), so two things this collector needs —
"what turn is this tool call part of" and "how long did this tool call take" — cannot live in
memory. Both are threaded through a small on-disk state file per session, keyed by session_id for
`current_turn_id`/`set_current_turn` and by the payload's `tool_use_id` for tool-call timing.

Fields this collector genuinely cannot observe (model_ms, retries, cached_tokens, context_peak,
context_window — all of these require seeing the actual provider request, which hooks never see)
are left as the Turn dataclass default of None. A zero would read as a real measurement and
corrupt the cross-source KPI comparison this collector exists to feed. `estimated` stays 0 to
match: this collector never derives token counts from anything (no chars/4 guess), it only ever
records what a hook payload states outright, so there is nothing to flag as estimated.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import os
import re
import secrets
import tempfile
import traceback
from pathlib import Path, PurePosixPath
from typing import Any

from flightdeck.ids import ulid
from flightdeck.models import Event, TextBlob, Turn
from flightdeck.redact import redact_or_drop, redact_payload
from flightdeck.store import DEFAULT_DIR, DEFAULT_RETENTION_DAYS, Store, hostname, now_ms
from flightdeck.tiers import derive_tier

SOURCE = "claude-code"

_RETENTION_MS = DEFAULT_RETENTION_DAYS * 24 * 3600 * 1000
_STATE_SUBDIR = "claude_code_state"
_DEBUG_LOG_NAME = "claude_code_debug.log"
_SAFE_SESSION = re.compile(r"[^A-Za-z0-9_.-]")


# ---------- directory + per-session state ----------
#
# Resolved fresh on every call (not cached at import time like Store.DEFAULT_DIR is) so a test
# can monkeypatch SPARKY_TURNLOG_DIR per-case and both the Store this module opens and the state
# file it reads/writes land in the same place.


def _turnlog_dir() -> Path:
    return Path(os.environ.get("SPARKY_TURNLOG_DIR", str(DEFAULT_DIR))).expanduser()


def _state_path(session_id: str) -> Path:
    safe = _SAFE_SESSION.sub("_", session_id) or "unknown"
    directory = _turnlog_dir() / _STATE_SUBDIR
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{safe}.json"


def _read_state(session_id: str) -> dict[str, Any]:
    try:
        return json.loads(_state_path(session_id).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _write_state(session_id: str, state: dict[str, Any]) -> None:
    path = _state_path(session_id)
    # Unique tmp per writer. Every hook event is its own process, and parallel tool calls plus
    # subagents fire concurrently — a fixed "<session>.tmp" made two writers share one tmp: the
    # first's replace() moved it, the second's replace() then hit FileNotFoundError and crashed
    # the hook (losing that tool's timing/turn state). mkstemp gives each writer a private tmp.
    fd, tmpname = tempfile.mkstemp(dir=str(path.parent), prefix=f"{path.stem}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(state))
        os.replace(tmpname, path)  # atomic: a concurrent reader never sees a half-written file
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmpname)
        raise


def _clear_session_state(session_id: str) -> None:
    with contextlib.suppress(OSError):
        _state_path(session_id).unlink(missing_ok=True)


def current_turn_id(session_id: str) -> str | None:
    return _read_state(session_id).get("turn_id")


def set_current_turn(session_id: str, turn_id: str) -> None:
    state = _read_state(session_id)
    state["turn_id"] = turn_id
    _write_state(session_id, state)


def _note_tool_start(session_id: str, tool_use_id: str, ts: int) -> None:
    state = _read_state(session_id)
    state.setdefault("pending_tools", {})[tool_use_id] = ts
    _write_state(session_id, state)


def _pop_tool_start(session_id: str, tool_use_id: str) -> int | None:
    state = _read_state(session_id)
    pending = state.get("pending_tools", {})
    start = pending.pop(tool_use_id, None)
    if start is not None:
        _write_state(session_id, state)
    return start


# ---------- shared helpers ----------


def _capture_text(store: Store, turn_id: str, kind: str, body: str, *, started_at: int) -> None:
    scrubbed = redact_or_drop(body)
    if not scrubbed:
        return  # redaction failed or body was empty after scrubbing: drop rather than store raw
    seq = len(store.texts_for(turn_id, kind))
    store.add_texts(
        [
            TextBlob(
                turn_id=turn_id,
                kind=kind,
                seq=seq,
                body=scrubbed,
                expires_at=started_at + _RETENTION_MS,
            )
        ]
    )


def _get_or_create_turn(store: Store, session_id: str, *, now: int, cwd: str | None) -> Turn:
    """Look up the turn this session has open; if there is none — hooks wired mid-session, or a
    Stop that never saw a matching UserPromptSubmit — mint a minimal row instead of dropping the
    event. A turn we only half-saw is still evidence."""
    turn_id = current_turn_id(session_id)
    if turn_id is not None:
        existing = store.get_turn(turn_id)
        if existing is not None:
            return existing
    new_id = turn_id or ulid(now)
    turn = Turn(
        turn_id=new_id,
        session_id=session_id,
        source=SOURCE,
        host=hostname(),
        started_at=now,
        cwd=cwd,
        tier=derive_tier(SOURCE, None),
    )
    store.upsert_turn(turn)
    set_current_turn(session_id, new_id)
    return turn


# ---------- public collector API ----------


def _close_orphan_turn(store: Store, session_id: str, *, now: int) -> None:
    """A prior turn that opened but never saw its Stop (compaction, interrupt, Stop-hook
    timeout) would sit with ended_at=None forever. When the next turn opens, finalize it:
    backfill ended_at from its last event so wall_ms reflects real work, and leave outcome
    None — we never saw a Stop, so we must not fabricate 'ok'. None is neutral to kpi/judge
    (they only flag a non-null, non-'ok' outcome), which is correct: the turn's success is
    unknown, not failed."""
    prev_id = current_turn_id(session_id)
    if prev_id is None:
        return
    prev = store.get_turn(prev_id)
    if prev is None or prev.ended_at is not None:
        return
    last_event = max((e.ts for e in store.events_for(prev_id)), default=prev.started_at)
    prev.ended_at = last_event
    prev.wall_ms = max(0, last_event - prev.started_at)
    store.upsert_turn(prev)


def open_turn(store: Store, payload: dict, *, now: int | None = None) -> str:
    """UserPromptSubmit: create the turn row, capture the redacted prompt, return turn_id."""
    session_id = payload.get("session_id") or "unknown"
    started_at = now if now is not None else now_ms()
    _close_orphan_turn(store, session_id, now=started_at)
    turn_id = ulid(started_at)

    turn = Turn(
        turn_id=turn_id,
        session_id=session_id,
        source=SOURCE,
        host=hostname(),
        started_at=started_at,
        cwd=payload.get("cwd"),
        model=payload.get(
            "model"
        ),  # never present in current hook payloads; kept for forward-compat
        tier=derive_tier(SOURCE, payload.get("model")),
    )
    store.upsert_turn(turn)
    set_current_turn(session_id, turn_id)

    # Claude Code's UserPromptSubmit payload carries the text under "prompt" (confirmed against
    # the hooks input schema); "user_input" was never a real field, so prompts were silently
    # never captured. Keep user_input as a defensive fallback for any non-standard emitter.
    prompt = payload.get("prompt") or payload.get("user_input")
    if prompt:
        _capture_text(store, turn_id, "prompt", prompt, started_at=started_at)

    return turn_id


# Shell builtins that carry no information about what the call actually did. `cd repo &&
# pytest` must read as a pytest run, not as navigation.
_SHELL_PREAMBLE = frozenset(
    {"cd", "set", "export", "unset", "source", ".", "umask", "shopt", "alias", "trap",
     "local", "if", "then", "else", "elif", "fi", "do", "done", "while", "for", "eval"}
)

# Shell connectives. Splitting on these is what makes `stems` a list rather than a lie.
_SEGMENT_SPLIT = re.compile(r"&&|\|\||[;|]|\n")

# Command stems that are worth telling apart from ordinary execution. Kept deliberately
# coarse: the point is to separate "this call was a verification run" from everything else,
# not to build a taxonomy of shell tools.
_INTERPRETERS = frozenset(
    # "run"/"exec" only ever reach here after a real interpreter was already skipped:
    # `cargo run` yields "cargo" first, because cargo is not itself an interpreter.
    # fmt: off
    {"python", "python3", "uv", "uvx", "npx", "bunx",
     "poetry", "env", "time", "sudo", "run", "exec"}
    # fmt: on
)


# The ONLY thing any consumer asks of a recorded stem is whether it names a verification
# run (`any(s in VERIFY_STEMS ...)` in the effort subsystem). So only allowlist members are
# ever stored. That is a structural guarantee rather than a careful one: an arbitrary
# argument cannot leak through a field that can only hold one of these 31 constants.
#
# Measured before this narrowing: 325 of 1600 live stem rows (20.3%) held raw command-line
# fragments -- values like "print(sklearn.__version__,numpy.__version__" -- and four separate
# splitting behaviours could put an unredacted secret in one. Quote-blindness was the worst:
# `git commit -m 'rotate creds; hunter2 retired'` stored "hunter2".
#
# Keep in lock-step with VERIFY_STEMS in the harness's training/effort.py by hand; the two
# repos cannot import from each other.
_VERIFY_STEMS: frozenset[str] = frozenset({
    "pytest", "unittest", "tox", "nox", "ruff", "mypy", "pyright", "flake8", "black",
    "bun", "jest", "vitest", "mocha", "npm", "pnpm", "yarn", "tsc", "eslint", "prettier",
    "go", "cargo", "make", "just", "gradle", "mvn", "ctest", "cmake",
    "shellcheck", "shfmt", "bats", "check.sh", "doctor.sh", "property-tests.sh",
})

_SALT_PATH = Path.home() / ".config" / "sparky" / "telemetry-salt"
_salt_cache: bytes | None = None


def _salt() -> bytes:
    """Per-install secret for the argument fingerprint, created once, kept out of telemetry.

    An unsalted truncated digest over a guessable domain is a lookup key, not a seal: an
    audit recovered 27 of 128 real path fingerprints in one second by hashing the local
    filesystem. Salting keeps the only property any consumer needs -- equality of two calls
    within one turn -- while removing the offline dictionary attack.

    Failure to read or create the salt falls back to a process-local random value: telemetry
    must never take down a tool call, and a fingerprint that is only comparable within this
    process still satisfies every in-turn comparison.
    """
    global _salt_cache
    if _salt_cache is not None:
        return _salt_cache
    try:
        if _SALT_PATH.exists():
            _salt_cache = _SALT_PATH.read_bytes().strip()
        if not _salt_cache:
            _SALT_PATH.parent.mkdir(parents=True, exist_ok=True)
            value = secrets.token_hex(32).encode()
            _SALT_PATH.write_text(value.decode(), encoding="utf-8")
            _SALT_PATH.chmod(0o600)
            _salt_cache = value
    except OSError:
        _salt_cache = secrets.token_hex(32).encode()
    return _salt_cache


def _fingerprint(text: str) -> str:
    return hmac.new(_salt(), text.encode("utf-8", "replace"), hashlib.sha256).hexdigest()[:12]


def _stem_of(segment: str) -> str | None:
    """First substantive word of one shell segment, or None if it has none.

    Interpreters and preamble builtins are handled differently on purpose. An interpreter
    is a transparent wrapper -- `python3 -m pytest` is a pytest run, so scanning continues
    past it. A preamble builtin makes the WHOLE segment uninformative: `cd /repo` says
    nothing about the work, and its argument is a directory, not a command. Skipping only
    the word would record `repo` as the stem.
    """
    for raw in segment.strip().split():
        if raw.startswith("-"):
            continue  # a flag, never the stem
        if "=" in raw and raw.split("=", 1)[0].isidentifier():
            continue  # leading environment assignment
        word = PurePosixPath(raw.strip("'\"`()")).name.lower()
        if not word:
            continue
        if word in _SHELL_PREAMBLE:
            return None  # the entire segment is preamble
        if word in _INTERPRETERS:
            continue  # transparent wrapper: keep looking
        return word
    return None


def _arg_signature(tool_name: str, tool_input: object) -> dict[str, Any]:
    """Content-free identity for one tool call: command stems plus a stable fingerprint.

    This records NO argument text, by construction and on purpose. Two facts are all the
    effort subsystem needs from a tool's arguments -- whether two calls were the same call
    (`arg_fp`), and what kind of work a shell call did (`stems`) -- and both survive
    hashing. Storing the command itself would put every secret that ever appeared on a
    command line into the `events` table, which has NO retention at all (only `texts`
    expires), behind a redaction pass that would have to be right every single time.

    `stems` is a LIST, and splitting on shell connectives is the whole point. A real
    command is `cd repo && pytest -q`, whose first word is `cd`; recording that would file
    every test run under shell navigation and quietly destroy the verify/exec split this
    exists to produce. Environment assignments, flags, generic interpreters and shell
    preamble builtins are all skipped, so `cd repo && FOO=1 python3 -m pytest` yields
    `["pytest"]`. Capped at four to bound the row.

    Classification is deliberately NOT done here: the collector records what ran, and the
    consumer decides what counts as verification.
    """
    if not isinstance(tool_input, dict):
        return {}
    out: dict[str, Any] = {}
    command = tool_input.get("command")
    if isinstance(command, str) and command.strip():
        # Everything from the first heredoc marker on is DATA, not commands. Splitting it
        # on newlines walks into the body and records its source lines as stems -- which is
        # both meaningless and a content leak, since a heredoc can carry anything. The
        # fingerprint below still covers the whole command; only stem extraction is cut.
        head = command.split("<<", 1)[0]
        found: list[str] = []
        for segment in _SEGMENT_SPLIT.split(head):
            stem = _stem_of(segment)
            # ONLY allowlist members are retained. Anything else -- a filename, a python
            # fragment, a password that happened to lead a quoted segment -- is counted and
            # discarded. The consumer asks one question of this field ("was this a
            # verification run"), and answering it needs no arbitrary text.
            if stem and stem in _VERIFY_STEMS and stem not in found:
                found.append(stem)
        out["scanned"] = True  # distinguishes "looked, matched nothing" from "never looked"
        if found:
            out["stems"] = sorted(found)
        out["arg_fp"] = _fingerprint(command)
        return out
    target = (
        tool_input.get("file_path")
        or tool_input.get("path")
        or tool_input.get("notebook_path")
    )
    if isinstance(target, str) and target:
        # The path alone does not identify the call. Two greps of one directory for
        # different patterns, or two disjoint chunk-reads of one file, hashed identically --
        # so a duplicate-read detector scored the second as waste. The discriminating
        # fields join the digest; they are hashed, never stored, so this adds no exposure.
        parts = [target]
        for field in ("pattern", "glob", "offset", "limit"):
            value = tool_input.get(field)
            if value not in (None, ""):
                parts.append(f"{field}={value}")
        out["arg_fp"] = _fingerprint("\x00".join(parts))
    return out


def record_tool(store: Store, payload: dict, *, ok: bool, now: int | None = None) -> None:
    """PostToolUse / PostToolUseFailure: record one tool_call event.

    Duration comes straight from the payload's `duration_ms` (both Post events carry it — tool
    execution time excluding permission prompts and PreToolUse hooks). That is authoritative and
    avoids pairing our own PreToolUse/PostToolUse hook invocations through the per-session state
    file, whose concurrent read-modify-write loses ~85% of entries under parallel tool calls. The
    state-file pairing is kept only as a fallback for a (non-standard) emitter that omits it.
    """
    session_id = payload.get("session_id") or "unknown"
    end_ts = now if now is not None else now_ms()
    tool_name = payload.get("tool_name") or "unknown"
    tool_use_id = payload.get("tool_use_id")

    turn = _get_or_create_turn(store, session_id, now=end_ts, cwd=payload.get("cwd"))

    duration_ms = payload.get("duration_ms")
    if duration_ms is None and tool_use_id:
        start_ts = _pop_tool_start(session_id, tool_use_id)
        duration_ms = (end_ts - start_ts) if start_ts is not None else None

    # Tool output is under "tool_response" (NOT "tool_result" — never a real Claude Code field).
    # Its shape is tool-specific and carries no exit_code, so success/failure is conveyed by `ok`
    # (PostToolUse vs PostToolUseFailure); we keep only the correlation id + any error text.
    event_payload: dict[str, Any] = {}
    if tool_use_id:
        event_payload["tool_use_id"] = tool_use_id
    event_payload.update(_arg_signature(tool_name, payload.get("tool_input")))
    if not ok and payload.get("error"):
        event_payload["error"] = payload.get("error")

    store.add_events(
        [
            Event(
                turn_id=turn.turn_id,
                ts=end_ts,
                kind="tool_call",
                name=tool_name,
                duration_ms=duration_ms,
                ok=1 if ok else 0,
                payload=redact_payload(event_payload) if event_payload else {},
            )
        ]
    )


def close_turn(store: Store, payload: dict, *, now: int | None = None, outcome: str = "ok") -> None:
    """Stop / StopFailure: finish the turn row and capture the redacted response.

    `Stop` fires on a normal finish → outcome "ok". `StopFailure` fires on an API/model error
    (rate limit, etc.) → outcome "error"; without handling it the turn never closes here and, once
    orphan-backfilled with outcome None, a genuinely failed turn scores as if it succeeded (kpi
    only zeroes completion on "error"/"cancelled"), corrupting the cross-source comparison.
    """
    session_id = payload.get("session_id") or "unknown"
    end_ts = now if now is not None else now_ms()

    turn = _get_or_create_turn(store, session_id, now=end_ts, cwd=payload.get("cwd"))
    turn.ended_at = end_ts
    turn.wall_ms = max(0, end_ts - turn.started_at)
    turn.outcome = outcome
    if outcome != "ok":
        turn.error_class = payload.get("error") or payload.get("error_details") or "stop_failure"
    store.upsert_turn(turn)

    # On StopFailure last_assistant_message holds the error text, not a real response — still worth
    # capturing (redacted) for the review; the outcome/error_class already mark it as a failure.
    response = payload.get("last_assistant_message")
    if response:
        _capture_text(store, turn.turn_id, "response", response, started_at=turn.started_at)

    _clear_session_state(session_id)


def _log_exception(payload: Any) -> None:
    """Best-effort crash log. Never logs payload values (they may hold prompt/tool text that
    hasn't been redacted yet) — only keys, so this can't become a second leak path."""
    try:
        debug_dir = _turnlog_dir()
        debug_dir.mkdir(parents=True, exist_ok=True)
        with (debug_dir / _DEBUG_LOG_NAME).open("a", encoding="utf-8") as fh:
            keys = list(payload.keys()) if isinstance(payload, dict) else type(payload).__name__
            fh.write(f"--- {now_ms()} payload_keys={keys} ---\n")
            fh.write(traceback.format_exc())
            fh.write("\n")
    except Exception:
        pass  # logging must never be the reason this hook fails a turn


def handle(payload: dict, *, store: Store | None = None) -> int:
    """Dispatch on payload['hook_event_name']; returns a process exit code. ALWAYS 0.

    Every exception — a malformed payload, an unwritable store dir, a bug below — is caught here
    and turned into a best-effort debug-log entry. A hook must never fail a Claude Code turn.

    PreToolUse is dispatched before a Store is opened at all: it only writes the small per-session
    JSON state file (`_note_tool_start`), never touches sqlite, and fires on every single tool
    call — the highest-frequency hook event. Paying for a sqlite connect + WAL pragma + migrate()
    check it never uses on that path was pure overhead.
    """
    try:
        if not isinstance(payload, dict):
            payload = {}
        event = payload.get("hook_event_name")

        if event == "PreToolUse":
            session_id = payload.get("session_id") or "unknown"
            tool_use_id = payload.get("tool_use_id")
            if tool_use_id:
                _note_tool_start(session_id, tool_use_id, now_ms())
            return 0

        st = store
        owns_store = False
        try:
            if st is None:
                st = Store(_turnlog_dir())
                owns_store = True

            if event == "UserPromptSubmit":
                open_turn(st, payload)
            elif event == "PostToolUse":
                record_tool(st, payload, ok=True)
            elif event == "PostToolUseFailure":
                record_tool(st, payload, ok=False)
            elif event == "Stop":
                close_turn(st, payload)
            elif event == "StopFailure":
                close_turn(st, payload, outcome="error")
            # any other hook_event_name: nothing this collector records, not an error
            return 0
        finally:
            if owns_store and st is not None:
                # A failure to close is not worth failing a turn over — the record is written.
                with contextlib.suppress(Exception):
                    st.close()
    except Exception:
        _log_exception(payload)
        return 0
