"""The filter that decides what counts as a turn a human typed.

Every number in the scope subsystem is computed over whatever survives this file, so its
two error directions are not symmetric and are tested separately: a leak costs ceremony on
every notification for the rest of a session, a false drop costs one un-scoped turn.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from flightdeck.scope_input import (
    MIN_RESIDUE_CHARS,
    human_residue,
    is_human_prompt,
    synthetic_reason,
)

REPO = Path(__file__).resolve().parents[1]
HOOK = REPO / "integration" / "claude-code" / "scope-hook.py"

HUMAN = [
    "fix the typo in store.py",
    "can you improve the whole thing everywhere",
    "so what should we do what are our options",
    "1",
    "research agent scoping. I want you to explore the main solutions",
    "yes",
    # A short markdown heading is not, by itself, a skill-body dump -- the doc-shape
    # check requires a later "##" subsection AND enough length to be a document.
    "# quick question\n\ncan you fix the typo in store.py",
]

#: A skill/reference-doc body long enough to trip the doc-shape check (heading + a later
#: "##" subsection + > MIN_DOC_CHARS), used below to prove the structural check fires on
#: SHAPE and not on any hardcoded skill title.
_FAKE_SKILL_BODY = (
    "# Some Brand New Skill Nobody Has Shipped Yet\n\n"
    + ("This reference doc explains a workflow in careful, padded detail. " * 6)
    + "\n\n## A subsection\n\nMore reference material that pads this out past the "
    "minimum length threshold so the shape check actually has something to match."
)

SYNTHETIC = [
    "[SYSTEM NOTIFICATION - NOT USER INPUT]\n<task-notification>done</task-notification>",
    "This is an automated background-task event, NOT a message from the user.",
    "Another Claude session sent a message: <cross-session-message from='x'>hi</>",
    "This session is being continued from a previous conversation that ran out of context.",
    "Stop hook feedback: [do a full review]",
    "Base directory for this skill: /Users/x/.claude/skills/learning",
    "Continue from where you left off.",
    "[Cross-session delivery notice] Your message was held",
    "<command-name>/compact</command-name><command-message>compact</command-message>",
    "[Image: original 2200x1400, displayed at 2000x1273]",
    "[scope: unbounded verb]\nThis request does not fully specify itself.",
    'A session-scoped Stop hook is now active with condition: "keep going". Briefly '
    "acknowledge the goal, then immediately start working toward it.",
    _FAKE_SKILL_BODY,
]


@pytest.mark.parametrize("text", HUMAN)
def test_human_prompts_survive(text: str) -> None:
    assert is_human_prompt(text), f"real prompt dropped: {synthetic_reason(text)}"


@pytest.mark.parametrize("text", SYNTHETIC)
def test_synthetic_prompts_rejected(text: str) -> None:
    assert not is_human_prompt(text)


def test_rejection_names_the_marker() -> None:
    """A filter that drops traffic without saying why cannot be told apart from a broken
    one. The reason is the audit trail, so it must name what fired."""
    reason = synthetic_reason("<task-notification>x</task-notification>")
    assert reason is not None
    assert "task-notification" in reason


def test_appended_system_reminder_does_not_drop_the_real_prompt() -> None:
    """Claude Code appends reminders to messages the user really typed. Rejecting on
    substring measured a 30% false-drop rate over 1,584 real turns; strip, then judge."""
    text = "fix the parser\n<system-reminder>remember to be nice</system-reminder>"
    assert is_human_prompt(text)
    assert human_residue(text).strip() == "fix the parser"


def test_reminder_only_payload_is_rejected() -> None:
    """The same wrapper with nothing else in it is not a turn."""
    assert not is_human_prompt("<system-reminder>only this</system-reminder>")


def test_single_character_answer_survives() -> None:
    """'1' and '2' are real answers to a numbered question. A residue floor above one
    character silently ate them."""
    assert MIN_RESIDUE_CHARS <= 1
    assert is_human_prompt("2")


def test_hook_injection_is_never_re_scored() -> None:
    """The gate reading its own output back builds a loop out of its own exhaust."""
    assert not is_human_prompt("[scope: names a quality]\nstate the goal")


def _run_hook(payload: dict, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(HOOK)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "SPARKY_TURNLOG_DIR": str(tmp_path), "HOME": str(tmp_path)},
        check=False,
    )


def test_hook_stays_silent_on_a_notification(tmp_path: Path) -> None:
    payload = {
        "hook_event_name": "UserPromptSubmit",
        "prompt": (
            "[SYSTEM NOTIFICATION - NOT USER INPUT]\n<task-notification>x</task-notification>"
        ),
        "session_id": "s",
        "cwd": "/tmp",
    }
    result = _run_hook(payload, tmp_path)
    assert result.returncode == 0
    assert result.stdout.strip() == "", "hook injected scoping into a background notification"


def test_hook_logs_the_suppression(tmp_path: Path) -> None:
    """Suppressed turns are logged with tier=None so the drop rate stays measurable.
    Silence that leaves no record is how a dead capture layer reads green."""
    payload = {
        "hook_event_name": "UserPromptSubmit",
        "prompt": "<task-notification>x</task-notification>",
        "session_id": "s",
        "cwd": "/tmp",
    }
    _run_hook(payload, tmp_path)
    rows = [
        json.loads(line)
        for line in (tmp_path / "scope" / "gate-log.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(rows) == 1
    assert rows[0]["tier"] is None
    assert "task-notification" in rows[0]["suppressed"]
    assert "prompt" not in rows[0], "the prompt itself must never be written to the log"


def test_hook_still_fires_on_a_real_prompt(tmp_path: Path) -> None:
    payload = {
        "hook_event_name": "UserPromptSubmit",
        "prompt": "can you improve the whole thing everywhere and then clean it up",
        "session_id": "s",
        "cwd": "/tmp",
    }
    result = _run_hook(payload, tmp_path)
    assert result.returncode == 0
    assert "[scope:" in result.stdout
    assert result.stdout.rstrip().endswith("Then build it in this same turn.")
