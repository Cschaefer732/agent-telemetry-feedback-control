"""Baseline scoping KPIs recovered from Claude Code transcripts.

Measured over 90 sessions / 945 human-typed turns (2026-08-02 .. 2026-09-04, the whole
retention window -- older sessions are gone, so this is the longest baseline obtainable):

    rework rate, main-thread      23.7%  (278 / 1175 files)
    rework rate, incl. subagents  30.4%  (889 / 2921 files)
    correction rate / turn         5.2%  (49 / 945)   <- LOWER BOUND, see below

Two denominator traps this module exists to encode, both of which silently produced
wrong numbers before they were caught:

1. `<root>/*/*.jsonl` is 90 session files; `rglob` is 1,779 because 1,689 subagent
   transcripts live under `<session-uuid>/subagents/`. Human-turn metrics want the
   former, edit metrics want both -- subagent edits are real edits and moved the rework
   rate by 6.7 points once attributed back to the parent session.

2. Only ~54% of non-tool-result `role:user` records were typed by the human. Skill
   bodies, compaction summaries, Stop-hook feedback and cross-session messages all
   arrive as user records. `learning/SKILL.md` contains the literal string "I told you
   already", so an unfiltered regex scored a correction every time a skill loaded:
   precision was 24%. Structural filtering (promptSource / isMeta / isCompactSummary)
   took the SAME patterns to 84% precision (21/25 hand-audited).

Correction counts are a floor, not an estimate. Hand-audit of 25 unflagged turns found
5-6 real corrections phrased as symptoms -- "the page appears blank, no elements are
loading", "im still getting these responses", "you never moved the todo calendar" --
which carry no correction keyword and which no regex will reach. True rate by hand-audit
is nearer 20-26% of typed turns. Treat this metric as a trend line on a fixed method,
never as an absolute, and see the module TODO about a kappa-validated judge.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from flightdeck.models import CORRECTION_FAMILIES

ANALYZER_VERSION = "1"
DEFAULT_ROOT = Path.home() / ".claude" / "projects"
EDIT_TOOLS = frozenset({"Edit", "Write", "MultiEdit", "NotebookEdit"})

#: promptSource values that mean "a human typed this". Anything else -- notably
#: "system" -- is an injection wearing a user record's clothes.
HUMAN_PROMPT_SOURCES = frozenset({"typed", "queued", "suggestion_accepted"})

#: Exact-match noise that survives the structural filters.
LITERAL_NOISE = frozenset(
    {
        "[Request interrupted by user]",
        "[Request interrupted by user for tool use]",
        "continue",
        "Continue",
        "API Error: Request was aborted.",
    }
)

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_WRAPPERS = tuple(
    re.compile(p, re.S)
    for p in (
        r"<system-reminder>.*?</system-reminder>",
        r"<local-command-caveat>.*?</local-command-caveat>",
        r"<local-command-stdout>.*?</local-command-stdout>",
        r"<local-command-stderr>.*?</local-command-stderr>",
        r"<command-name>.*?</command-name>",
        r"<command-message>.*?</command-message>",
        r"<command-args>.*?</command-args>",
        r"<task-notification>.*?</task-notification>",
    )
)

#: Keys MUST equal models.CORRECTION_FAMILIES -- test_scope asserts it. `redirect` is
#: matched against the head of the message only; the rest match anywhere.
CORRECTION_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "negate": tuple(
        re.compile(p, re.I)
        for p in (
            r"\bno,", r"^no\b", r"\bnope\b", r"that'?s not (what|right|it)",
            r"not what i (asked|meant|said|wanted)", r"i did ?n'?t ask",
            r"i did ?n'?t say",
            r"\bthat'?s wrong\b", r"\brevert\b", r"\bundo (that|it)\b",
        )
    ),
    "missed": tuple(
        re.compile(p, re.I)
        for p in (
            r"you forgot", r"forgot to", r"you did ?n'?t (do|add|include|create|update|run)",
            r"you (also )?need to", r"\bis missing\b", r"you missed", r"what about the",
            r"still (need|needs|missing)", r"you never (moved|added|did|made|updated)",
        )
    ),
    "repeat": tuple(
        re.compile(p, re.I)
        for p in (
            r"i (already )?told you", r"as i said", r"like i said", r"i said (to|that)",
            r"\bagain,", r"you keep ",
        )
    ),
    "redirect": tuple(
        re.compile(p, re.I)
        for p in (
            r"^actually\b", r"\bactually,? i", r"\binstead of\b", r"i meant\b", r"rather than",
        )
    ),
}
_HEAD_ONLY = frozenset({"redirect"})
_HEAD_CHARS = 200


def clean_text(text: str) -> str:
    """Strip ANSI and wrapper blocks. ANSI first: a digit inside `\\x1b[0m` has already
    fooled one grader into reporting a fabricated value."""
    text = _ANSI.sub("", text)
    for wrapper in _WRAPPERS:
        text = wrapper.sub("", text)
    return text.strip()


def is_human_prompt(record: dict[str, Any]) -> bool:
    """Structural test only -- never a prefix match on the body. Prefix lists miss the
    spelling nobody enumerated; these fields are set by the writer."""
    if record.get("type") != "user" or record.get("isSidechain"):
        return False
    if record.get("isMeta") or record.get("isCompactSummary"):
        return False
    source = record.get("promptSource")
    if source is not None and source not in HUMAN_PROMPT_SOURCES:
        return False
    message = record.get("message")
    return isinstance(message, dict) and message.get("role") == "user"


def human_prompt_text(record: dict[str, Any]) -> str | None:
    """The human's words, or None if this record is a tool result / wrapper / noise."""
    if not is_human_prompt(record):
        return None
    content = record["message"].get("content")
    if isinstance(content, str):
        text = clean_text(content)
    elif isinstance(content, list):
        if any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content):
            return None
        text = clean_text(
            "\n".join(
                b.get("text") or ""
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
        )
    else:
        return None
    if not text or text in LITERAL_NOISE or text.startswith("[Request interrupted"):
        return None
    return text


def classify_correction(text: str) -> list[str]:
    """Families matched, in registry order. Empty list means 'no keyword', which is NOT
    the same as 'not a correction' -- symptom-phrased corrections match nothing."""
    head = text[:_HEAD_CHARS]
    hits: list[str] = []
    for family in CORRECTION_FAMILIES:
        target = head if family in _HEAD_ONLY else text
        if any(p.search(target) for p in CORRECTION_PATTERNS[family]):
            hits.append(family)
    return hits


def edit_episodes(edits: Iterable[tuple[str, str]]) -> dict[str, int]:
    """Count contiguous runs per file over a (timestamp, path) stream. A file edited
    twice in a row is one episode; returning to it after touching another file is the
    second, and that is what 'rework' means here."""
    episodes: dict[str, int] = defaultdict(int)
    previous: str | None = None
    for _, path in sorted(edits):
        if path != previous:
            episodes[path] += 1
        previous = path
    return dict(episodes)


def _iter_records(path: Path) -> Iterator[dict[str, Any]]:
    try:
        handle = path.open(encoding="utf-8", errors="replace")
    except OSError:
        return
    with handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except ValueError:
                continue


def _edits_in(path: Path) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for record in _iter_records(path):
        message = record.get("message")
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            if block.get("name") not in EDIT_TOOLS:
                continue
            args = block.get("input") or {}
            target = args.get("file_path") or args.get("notebook_path")
            if target:
                found.append((record.get("timestamp") or "", target))
    return found


def collect_subagent_edits(root: Path) -> dict[str, list[tuple[str, str]]]:
    """Subagent edits keyed by the PARENT session uuid. They live in separate files;
    counting only the session file undercounts rework by ~7 points."""
    by_parent: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for path in root.rglob("*.jsonl"):
        parts = path.relative_to(root).parts
        if "subagents" not in parts or len(parts) < 2:
            continue
        by_parent[parts[1]].extend(_edits_in(path))
    return dict(by_parent)


def analyze_session(
    path: Path, subagent_edits: list[tuple[str, str]] | None = None
) -> dict[str, Any]:
    human_turns = 0
    correction_turns = 0
    families: Counter[str] = Counter()
    main_edits: list[tuple[str, str]] = []
    first_ts: str | None = None
    last_ts: str | None = None
    cwd: str | None = None
    branch: str | None = None
    version: str | None = None

    for record in _iter_records(path):
        timestamp = record.get("timestamp")
        if timestamp:
            first_ts = first_ts or timestamp
            last_ts = timestamp
        cwd = cwd or record.get("cwd")
        branch = branch or record.get("gitBranch")
        version = version or record.get("version")

        text = human_prompt_text(record)
        if text is not None:
            human_turns += 1
            hits = classify_correction(text)
            if hits:
                correction_turns += 1
                families.update(hits)
            continue

        message = record.get("message")
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        if record.get("isSidechain"):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            if block.get("name") not in EDIT_TOOLS:
                continue
            args = block.get("input") or {}
            target = args.get("file_path") or args.get("notebook_path")
            if target:
                main_edits.append((timestamp or "", target))

    main = edit_episodes(main_edits)
    combined = edit_episodes(main_edits + list(subagent_edits or []))
    return {
        "session": path.stem,
        "project": path.parent.name,
        "cwd": cwd,
        "git_branch": branch,
        "cc_version": version,
        "ts_first": first_ts,
        "ts_last": last_ts,
        "human_turns": human_turns,
        "correction_turns": correction_turns,
        "families": dict(families),
        "files_edited": len(main),
        "rework_files": sum(1 for n in main.values() if n >= 2),
        "files_edited_incl_sub": len(combined),
        "rework_files_incl_sub": sum(1 for n in combined.values() if n >= 2),
        "sub_edit_ops": len(subagent_edits or []),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Denominators are explicit: a single-turn session cannot contain a correction, and
    a session that edited nothing cannot contain rework. Averaging over all sessions
    instead would dilute both toward zero and read as improvement."""
    conversational = [r for r in rows if r["human_turns"] >= 2]
    editing = [r for r in rows if r["files_edited"] >= 1]
    editing_sub = [r for r in rows if r["files_edited_incl_sub"] >= 1]
    turns = sum(r["human_turns"] for r in conversational)
    corrections = sum(r["correction_turns"] for r in conversational)
    files = sum(r["files_edited"] for r in editing)
    rework = sum(r["rework_files"] for r in editing)
    files_sub = sum(r["files_edited_incl_sub"] for r in editing_sub)
    rework_sub = sum(r["rework_files_incl_sub"] for r in editing_sub)
    families: Counter[str] = Counter()
    for row in conversational:
        families.update(row["families"])
    return {
        "sessions_total": len(rows),
        "sessions_conversational": len(conversational),
        "sessions_with_edits": len(editing),
        "human_turns": turns,
        "correction_turns": corrections,
        "correction_rate_per_turn": round(corrections / turns, 4) if turns else None,
        "session_correction_rate": (
            round(
                sum(1 for r in conversational if r["correction_turns"]) / len(conversational), 4
            )
            if conversational
            else None
        ),
        "files_edited": files,
        "rework_files": rework,
        "rework_rate": round(rework / files, 4) if files else None,
        "files_edited_incl_sub": files_sub,
        "rework_files_incl_sub": rework_sub,
        "rework_rate_incl_sub": round(rework_sub / files_sub, 4) if files_sub else None,
        "families": dict(families.most_common()),
        "analyzer_version": ANALYZER_VERSION,
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
    }


def analyze(root: Path = DEFAULT_ROOT) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    subagents = collect_subagent_edits(root)
    rows = [analyze_session(p, subagents.get(p.stem)) for p in sorted(root.glob("*/*.jsonl"))]
    return rows, summarize(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--sessions-out", type=Path, help="write per-session rows as JSONL")
    args = parser.parse_args(argv)

    rows, summary = analyze(args.root)
    if args.sessions_out:
        args.sessions_out.parent.mkdir(parents=True, exist_ok=True)
        with args.sessions_out.open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
