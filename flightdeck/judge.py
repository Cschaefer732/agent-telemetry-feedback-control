"""Semantic verdicts for turns that tripped the flag rule (see kpi.should_flag).

This runs DETACHED from every turn's critical path — a nightly-batch style consumer of
`pending_judgments`, never called synchronously from a collector or hook. The judge model is
expected to be pinned resident on your inference host (e.g. via an ollama keep-alive) alongside
the rest of the warm set, so judging costs zero extra memory — never point this at a cold model.
This module is still written to fail closed rather than compete for memory: concurrency 1, a
hard backlog cap, and a single JudgeUnavailable halts the whole batch instead of retrying into a
box that is down. The judge is advisory only — it writes rows to `judgments` and nothing else;
no file the running system reads is ever touched here.

Endpoint and model are configurable via FLIGHTDECK_JUDGE_ENDPOINT / FLIGHTDECK_JUDGE_MODEL so
this points at your own inference host rather than a hardcoded one; the defaults assume a local
ollama on the box running this process.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from flightdeck.kpi import DEFAULT_THRESHOLDS
from flightdeck.models import Event, Judgment, TextBlob, Turn
from flightdeck.store import now_ms

if TYPE_CHECKING:
    from flightdeck.store import Store

DEFAULT_ENDPOINT = os.environ.get("FLIGHTDECK_JUDGE_ENDPOINT", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("FLIGHTDECK_JUDGE_MODEL", "qwen3.8:27b")

# Marks a Judgment produced by the requirement-3 fallback (model output didn't parse) rather
# than a genuine semantic verdict, so run_queue can bucket it as "failed" instead of "judged".
_PARSE_FAILURE_PREFIX = "parse failure: "

_RUBRIC_INSTRUCTIONS = (
    "You are judging a single coding-agent turn that tripped an automated flag rule "
    "(slow, failed, low KPI, edit/revert loop, or similar). Judge only what is shown below.\n\n"
    "Answer these five questions:\n"
    "1. Did the turn accomplish what the prompt asked? (pass/partial/fail)\n"
    "2. Was the model tier appropriate -- overkill, right, or under-powered?\n"
    "3. Were the loaded skills actually used, and did any missing skill cause the failure?\n"
    "4. Was the failure the model's or the harness's (tool error, permission, missing context)?\n"
    "5. One-line lesson, or NONE.\n\n"
    "Respond in EXACTLY this format, one field per line, no extra commentary:\n\n"
    "VERDICT: pass|partial|fail\n"
    "TIER: overkill|right|underpowered\n"
    "SKILLS: used|unused|missing:<name>\n"
    "BLAME: model|harness|user|none\n"
    "LESSON: <one line, or NONE>\n"
    "NOTES: <at most two sentences>"
)

_RUBRIC_KEYS = ("VERDICT", "TIER", "SKILLS", "BLAME", "LESSON", "NOTES")
_VALID_VERDICTS = {"pass", "partial", "fail"}

# TextBlob's primary key is (turn_id, kind, seq) -- seq is only ordered within a kind, so kinds
# can't be interleaved chronologically. Fixed block order instead: what was asked, what came
# back, any summary.
_CONVERSATION_KINDS = ("prompt", "response", "summary")


class JudgeUnavailable(Exception):
    """The judge endpoint could not be reached or returned something unusable."""


@dataclass
class JudgeConfig:
    endpoint: str
    model: str
    timeout_s: float = 90.0
    max_backlog: int = 20
    max_output_tokens: int = 1024
    max_context_chars: int = 12000
    enabled: bool = True

    @classmethod
    def from_env(cls) -> JudgeConfig:
        return cls(
            endpoint=os.environ.get("SPARKY_JUDGE_ENDPOINT", DEFAULT_ENDPOINT),
            model=os.environ.get("SPARKY_JUDGE_MODEL", DEFAULT_MODEL),
            enabled=os.environ.get("SPARKY_JUDGE", "1") != "0",
        )


def _turn_summary(turn: Turn) -> str:
    """kpi_score is deliberately NOT in this summary. training/outcome_score.py weights the judge
    verdict at 2.0 specifically to get a signal that does NOT inherit kpi_score's bias (its own
    docstring: R^2 -0.87 against a mean predictor), and training/reward_model.py calls the verdict
    an *external* calibration set. Printing the score here made both false: the judge read the
    number it was being used to validate, so agreement measured echo, not independence."""
    return (
        "TURN: "
        f"turn_id={turn.turn_id} source={turn.source} outcome={turn.outcome} "
        f"tier={turn.tier} model={turn.model} wall_ms={turn.wall_ms}"
    )


def _flag_reasons(turn: Turn, events: list[Event]) -> list[str]:
    """Mirrors kpi.should_flag's reason set. Reimplemented (not called) because should_flag needs
    a Components object and a trailing p95 that build_prompt, a pure function of turn/events/texts,
    has no store access to recompute; turn.kpi_score already IS that composite, so we read it
    straight off the row instead."""
    reasons: list[str] = []
    if turn.outcome is not None and turn.outcome != "ok":
        reasons.append("outcome_not_ok")
    if any(e.kind == "critic" and e.payload.get("verdict") == "fail" for e in events):
        reasons.append("critic_fail")
    tool_calls = [e for e in events if e.kind == "tool_call"]
    if tool_calls:
        failed = sum(1 for e in tool_calls if e.ok == 0)
        if failed / len(tool_calls) > DEFAULT_THRESHOLDS["tool_error_rate"]:
            reasons.append("tool_error_rate")
    if any(e.kind == "compaction" for e in events):
        reasons.append("midturn_compaction")
    if any(e.kind == "revert" for e in events):
        reasons.append("edit_revert")
    if any(e.kind == "interrupt" for e in events):
        reasons.append("user_interrupt")
    if turn.kpi_score is not None and turn.kpi_score < DEFAULT_THRESHOLDS["low_kpi"]:
        reasons.append("low_kpi")
    return reasons


def _tool_failures(events: list[Event]) -> list[str]:
    lines = []
    for event in events:
        if event.kind == "tool_call" and event.ok == 0:
            detail = event.payload.get("error") or event.payload.get("message") or ""
            lines.append(f"- {event.name or 'tool'}: {detail}".strip())
    return lines


def _conversation_text(texts: list[TextBlob]) -> str:
    blocks = []
    for kind in _CONVERSATION_KINDS:
        for text in sorted((t for t in texts if t.kind == kind), key=lambda t: t.seq):
            blocks.append(f"[{kind}#{text.seq}] {text.body}")
    return "\n\n".join(blocks)


def build_prompt(
    turn: Turn, events: list[Event], texts: list[TextBlob], *, max_context_chars: int = 12000
) -> str:
    """Assembles the judge prompt, truncated to max_context_chars. Truncation drops from the head
    of the conversation first — the turn summary, flag reasons, and tool failures are small and
    are always kept whole; the failure is usually near the end of the conversation, so that's the
    part worth spending the remaining budget on."""
    header_lines = [_turn_summary(turn)]
    reasons = _flag_reasons(turn, events)
    header_lines.append("FLAG_REASONS: " + (", ".join(reasons) if reasons else "none"))
    failures = _tool_failures(events)
    if failures:
        header_lines.append("TOOL_FAILURES:")
        header_lines.extend(failures)
    else:
        header_lines.append("TOOL_FAILURES: none")
    header = "\n".join(header_lines)

    convo = _conversation_text(texts)
    context_block = header
    if convo:
        separator = "\n\nCONVERSATION (tail):\n"
        marker = "...[truncated]...\n"
        available = max_context_chars - len(header) - len(separator)
        if available > 0:
            if len(convo) > available:
                body_budget = max(0, available - len(marker))
                kept = marker + convo[-body_budget:] if body_budget > 0 else marker[-available:]
            else:
                kept = convo
            context_block = header + separator + kept

    return _RUBRIC_INSTRUCTIONS + "\n\n" + context_block


def parse_verdict(raw: str) -> dict[str, Any]:
    """Parses the fixed-format rubric response. Raises ValueError on anything that doesn't carry
    a recognizable VERDICT line -- callers must treat that as a failed judgment, not a crash."""
    fields: dict[str, str] = {}
    for key in _RUBRIC_KEYS:
        match = re.search(rf"^{key}:\s*(.+)$", raw, re.IGNORECASE | re.MULTILINE)
        if match:
            fields[key.lower()] = match.group(1).strip()

    verdict = fields.get("verdict", "").lower()
    if verdict not in _VALID_VERDICTS:
        raise ValueError(f"no valid VERDICT line in judge output: {raw[:200]!r}")

    lesson = fields.get("lesson")
    if lesson is not None and lesson.strip().upper() == "NONE":
        lesson = None

    return {
        "verdict": verdict,
        "tier": fields.get("tier"),
        "skills": fields.get("skills"),
        "blame": fields.get("blame"),
        "lesson": lesson,
        "notes": fields.get("notes"),
    }


def ollama_client(prompt: str, config: JudgeConfig) -> str:
    """The real HTTP call, on ollama's NATIVE /api/chat. The OpenAI-compat path cannot disable
    thinking ("think" is silently ignored there — measured 2026-08-23), and qwen3 burned the
    whole output budget inside reasoning, returning empty/truncated content on every long judge
    prompt. Native "think": false hard-disables it; the /no_think prompt switch was advisory
    and long prompts re-triggered thinking anyway."""
    url = config.endpoint.rstrip("/") + "/api/chat"
    body: dict[str, Any] = {
        "model": config.model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0, "num_predict": config.max_output_tokens},
    }
    # Only thinking-capable models accept the flag; sending it to others errors the call.
    if config.model.startswith("qwen3"):
        body["think"] = False
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=config.timeout_s) as response:
            raw_body = response.read().decode("utf-8", errors="replace")
    except OSError as exc:
        # Covers connection-refused, timeout, and non-200 (urllib.error.HTTPError/URLError are
        # OSError subclasses) -- a box that's down must not be retried into.
        raise JudgeUnavailable(f"judge endpoint unreachable: {exc}") from exc

    try:
        payload = json.loads(raw_body)
        return payload["message"]["content"]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        raise JudgeUnavailable(f"malformed judge response: {exc}") from exc


def judge_turn(
    store: Store,
    turn: Turn,
    config: JudgeConfig,
    *,
    client: Callable[[str, JudgeConfig], str] | None = None,
) -> Judgment | None:
    if not config.enabled:
        return None
    client = client or ollama_client

    events = store.events_for(turn.turn_id)
    texts = store.texts_for(turn.turn_id)
    prompt = build_prompt(turn, events, texts, max_context_chars=config.max_context_chars)

    # JudgeUnavailable propagates deliberately: run_queue needs it to halt the batch rather than
    # having judge_turn swallow it and keep hammering a box that's down.
    raw = client(prompt, config)

    try:
        parsed = parse_verdict(raw)
    except ValueError as exc:
        # Requirement: a turn the model chokes on must still be marked judged, or it blocks the
        # queue behind it forever on every subsequent run.
        judgment = Judgment(
            turn_id=turn.turn_id,
            judge_model=config.model,
            verdict="fail",
            created_at=now_ms(),
            rubric={},
            notes=f"{_PARSE_FAILURE_PREFIX}{exc}",
            lesson=None,
        )
        store.add_judgment(judgment)
        return judgment

    judgment = Judgment(
        turn_id=turn.turn_id,
        judge_model=config.model,
        verdict=parsed["verdict"],
        created_at=now_ms(),
        rubric=parsed,
        notes=parsed.get("notes"),
        lesson=parsed.get("lesson"),
    )
    store.add_judgment(judgment)
    return judgment


def _log_backlog_drops(store: Store, turn_ids: list[str], cap: int) -> None:
    """A silently dropped backlog reads as 'everything was judged' when it wasn't -- write one
    ok=0 queue event per dropped turn so probe_judge_queue's dropped count reflects reality."""
    events = [
        Event(
            turn_id=turn_id,
            ts=now_ms(),
            kind="queue",
            name="judge_backlog_drop",
            ok=0,
            payload={"reason": "backlog_cap_exceeded", "cap": cap},
        )
        for turn_id in turn_ids
    ]
    store.add_events(events)


def run_queue(
    store: Store,
    config: JudgeConfig | None = None,
    *,
    limit: int | None = None,
    client: Callable[[str, JudgeConfig], str] | None = None,
) -> dict[str, int]:
    config = config or JudgeConfig.from_env()
    result = {"judged": 0, "skipped": 0, "failed": 0, "backlog": 0}
    if not config.enabled:
        return result

    total_backlog = store.conn.execute(
        "SELECT COUNT(*) FROM turns WHERE flagged=1 AND judged=0"
    ).fetchone()[0]
    result["backlog"] = total_backlog

    overflow = max(0, total_backlog - config.max_backlog)
    if overflow:
        # pending_judgments returns the newest max_backlog rows (ORDER BY started_at DESC LIMIT
        # ?); the ones it leaves out -- fetched here via OFFSET -- are exactly the oldest excess.
        rows = store.conn.execute(
            "SELECT turn_id FROM turns WHERE flagged=1 AND judged=0 "
            "ORDER BY started_at DESC LIMIT -1 OFFSET ?",
            (config.max_backlog,),
        ).fetchall()
        overflow_ids = [row[0] for row in rows]
        _log_backlog_drops(store, overflow_ids, config.max_backlog)
        result["skipped"] += len(overflow_ids)

    pending = store.pending_judgments(limit=config.max_backlog)
    to_process = pending[:limit] if limit is not None else pending

    for idx, turn in enumerate(to_process):
        try:
            judgment = judge_turn(store, turn, config, client=client)
        except JudgeUnavailable:
            # Stop hammering a box that's down. Everything left in this run's slice, including
            # the turn that just failed, is counted skipped rather than failed -- it was never
            # actually judged.
            result["skipped"] += len(to_process) - idx
            break

        if judgment is None:
            continue
        if judgment.notes and judgment.notes.startswith(_PARSE_FAILURE_PREFIX):
            result["failed"] += 1
        else:
            result["judged"] += 1

    return result
