"""The Go emitter and the Python store share a JSONL contract and no code.

This is the test that catches a drift between them. It runs the real Go emitter, then replays its
output through the real Python ingest path — a hand-written fixture would pass forever while the
two implementations quietly diverged.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from flightdeck.store import Store

REPO_ROOT = Path(__file__).resolve().parents[1]
GO_DIR = REPO_ROOT / "go"

pytestmark = pytest.mark.skipif(
    shutil.which("go") is None, reason="Go toolchain not installed on this box"
)


@pytest.fixture(scope="module")
def emitted(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str]:
    out_dir = tmp_path_factory.mktemp("turnlog")
    result = subprocess.run(
        ["go", "run", "./cmd/emit-fixture", str(out_dir)],
        cwd=GO_DIR,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if result.returncode != 0:
        pytest.fail(f"emit-fixture failed: {result.stderr}")
    return out_dir, result.stdout.strip()


def test_go_output_replays_into_python_store(emitted: tuple[Path, str], tmp_path: Path) -> None:
    out_dir, turn_id = emitted
    logs = sorted(out_dir.glob("events-*.jsonl"))
    assert logs, "Go emitter wrote no JSONL"

    store = Store(tmp_path)
    counts = store.ingest_jsonl(logs[0])

    assert counts.get("turn") == 1, f"turn record not ingested: {counts}"
    assert counts.get("event", 0) >= 15, f"events missing: {counts}"
    assert counts.get("text", 0) >= 2, f"texts missing: {counts}"

    turn = store.get_turn(turn_id)
    assert turn is not None, "turn id from Go did not survive the round trip"
    # Spot-check one field from every category the emitter accumulates, so a dropped column in
    # either language fails here rather than showing up as a silently empty KPI months later.
    assert turn.session_id == "fixture-session"
    assert turn.source == "crush"
    assert turn.host == "fixture-host"
    assert turn.is_subagent == 1
    assert turn.agent_name == "code-reviewer"
    assert turn.tier == "fast"
    assert turn.prompt_tokens == 8000
    assert turn.completion_tokens == 700
    assert turn.cached_tokens == 200
    assert turn.model_ms == 1200
    assert turn.retries == 1
    assert turn.context_peak == 21000
    assert turn.context_window == 32000
    assert turn.outcome == "error"
    assert turn.error_class == "TestFailure"
    # Prefill and prompt-cache signals, merged in from the fork's own timing work.
    assert turn.ttft_ms == 85
    assert turn.tools_hash_changes == 0
    assert turn.wall_ms is not None and turn.wall_ms >= 0


def test_every_event_kind_survives(emitted: tuple[Path, str], tmp_path: Path) -> None:
    out_dir, turn_id = emitted
    store = Store(tmp_path)
    store.ingest_jsonl(sorted(out_dir.glob("events-*.jsonl"))[0])

    kinds = {event.kind for event in store.events_for(turn_id)}
    expected = {
        "tool_call",
        "model_req",
        "compaction",
        "mode_change",
        "skill_load",
        "skill_use",
        "mcp_call",
        "lsp_event",
        "hook",
        "permission",
        "recall",
        "critic",
        "queue",
        "todo",
        "edit",
        "revert",
        "interrupt",
        "worktree",
        "delegate",
    }
    assert expected <= kinds, f"event kinds lost in translation: {sorted(expected - kinds)}"


def test_event_payloads_decode(emitted: tuple[Path, str], tmp_path: Path) -> None:
    out_dir, turn_id = emitted
    store = Store(tmp_path)
    store.ingest_jsonl(sorted(out_dir.glob("events-*.jsonl"))[0])

    # models.Event.from_row json.loads() the payload; if Go emitted an object instead of a JSON
    # string this raises rather than silently producing an unusable dict.
    tool_calls = store.events_for(turn_id, kind="tool_call")
    assert len(tool_calls) == 2
    paths = {call.payload.get("path") for call in tool_calls}
    assert "/repo/parser.go" in paths
    failed = [call for call in tool_calls if call.ok == 0]
    assert len(failed) == 1 and failed[0].name == "bash"


def test_texts_carry_ttl_and_are_scrubbed(emitted: tuple[Path, str], tmp_path: Path) -> None:
    out_dir, turn_id = emitted
    store = Store(tmp_path)
    store.ingest_jsonl(sorted(out_dir.glob("events-*.jsonl"))[0])

    texts = store.texts_for(turn_id)
    assert {t.kind for t in texts} == {"prompt", "response"}
    for text in texts:
        assert text.expires_at > 0
    # Expiry must actually delete: retention is a promise about what stays on disk.
    assert store.expire_texts(now=max(t.expires_at for t in texts) + 1) == len(texts)
    assert store.texts_for(turn_id) == []


def test_replay_is_idempotent(emitted: tuple[Path, str], tmp_path: Path) -> None:
    out_dir, turn_id = emitted
    log = sorted(out_dir.glob("events-*.jsonl"))[0]
    store = Store(tmp_path)
    store.ingest_jsonl(log)
    first = len(store.events_for(turn_id))
    store.ingest_jsonl(log)
    assert len(store.events_for(turn_id)) == first, "re-ingesting a log double-counted events"


def test_no_unknown_record_kinds(emitted: tuple[Path, str]) -> None:
    out_dir, _ = emitted
    log = sorted(out_dir.glob("events-*.jsonl"))[0]
    kinds = set()
    for line in log.read_text().splitlines():
        if line.strip():
            kinds.add(json.loads(line)["_kind"])
    # A record kind the Python side does not handle is dropped silently by ingest_jsonl, which is
    # exactly the class of bug this project exists to eliminate.
    assert kinds <= {"turn", "event", "text", "probe"}, f"unhandled record kinds: {kinds}"


def test_emitter_reports_its_own_losses(emitted: tuple[Path, str], tmp_path: Path) -> None:
    """The emitter may drop records under load; it may not drop them silently."""
    out_dir, _ = emitted
    store = Store(tmp_path)
    store.ingest_jsonl(sorted(out_dir.glob("events-*.jsonl"))[0])

    rows = store.conn.execute(
        "SELECT ok, total, detail FROM probes WHERE kind='collector_heartbeat'"
    ).fetchall()
    assert rows, "emitter wrote no heartbeat probe on shutdown"
    detail = json.loads(rows[0]["detail"])
    assert detail["dropped"] == 0 and detail["write_errors"] == 0
    assert rows[0]["ok"] == rows[0]["total"], "heartbeat reports losses the fixture did not incur"


def test_context_snapshots_are_not_compactions(emitted: tuple[Path, str], tmp_path: Path) -> None:
    """These shared an event kind once, which silently zeroed the context_health component."""
    out_dir, turn_id = emitted
    store = Store(tmp_path)
    store.ingest_jsonl(sorted(out_dir.glob("events-*.jsonl"))[0])

    assert len(store.events_for(turn_id, kind="context_snapshot")) == 1
    compactions = store.events_for(turn_id, kind="compaction")
    assert len(compactions) == 1 and compactions[0].name == "auto"
