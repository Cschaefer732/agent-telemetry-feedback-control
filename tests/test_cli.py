from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from flightdeck import __main__ as cli
from flightdeck import probes as probes_mod
from flightdeck.models import GOVERNOR_DOMAINS, Probe, TextBlob, TuningChange, Turn
from flightdeck.review import DAY_MS
from flightdeck.schema import SCHEMA_VERSION
from flightdeck.store import Store, now_ms


def _turn(
    turn_id: str,
    *,
    started_at: int,
    session_id: str = "sess-1",
    source: str = "claude-code",
    host: str = "testhost",
    tier: str | None = None,
    kpi_score: float | None = None,
    outcome: str | None = "ok",
    flagged: int = 0,
    judged: int = 0,
) -> Turn:
    return Turn(
        turn_id=turn_id,
        session_id=session_id,
        source=source,
        host=host,
        started_at=started_at,
        tier=tier,
        kpi_score=kpi_score,
        outcome=outcome,
        flagged=flagged,
        judged=judged,
    )


def _healthy_probes() -> list[Probe]:
    ts = now_ms()
    return [
        Probe(ts=ts, host="h", kind=kind, ok=1, total=1, detail={})
        for kind in ("symlink", "migration", "hook_liveness", "collector_heartbeat", "judge_queue")
    ]


# ---------- parse_duration ----------


@pytest.mark.parametrize(
    "value,expected_ms",
    [("30s", 30_000), ("5m", 300_000), ("24h", 86_400_000), ("7d", 604_800_000)],
)
def test_parse_duration_accepts(value: str, expected_ms: int) -> None:
    assert cli.parse_duration(value) == expected_ms


@pytest.mark.parametrize("value", ["24", "h", "", "1w"])
def test_parse_duration_rejects(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        cli.parse_duration(value)


# ---------- init ----------


def test_init_creates_store_and_reports_schema(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    exit_code = cli.main(["--dir", str(tmp_path), "--json", "init"])
    assert exit_code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["schema"] == SCHEMA_VERSION
    assert (tmp_path / "turnlog.db").exists()


# ---------- doctor ----------


def test_doctor_unhealthy_on_empty_window(
    tmp_path: Path, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(probes_mod, "default_config", lambda: object())
    monkeypatch.setattr(probes_mod, "run_all", lambda store, config: _healthy_probes())

    exit_code = cli.main(["--dir", str(tmp_path), "--json", "doctor"])

    assert exit_code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["healthy"] is False
    assert any("no turns recorded" in finding for finding in out["findings"])


def test_doctor_healthy_with_recent_turns_and_stubbed_probes(
    tmp_path: Path, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(probes_mod, "default_config", lambda: object())
    monkeypatch.setattr(probes_mod, "run_all", lambda store, config: _healthy_probes())

    store = Store(tmp_path)
    store.upsert_turn(_turn("t1", started_at=now_ms() - 60_000))
    store.close()

    exit_code = cli.main(["--dir", str(tmp_path), "--json", "doctor"])

    assert exit_code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["healthy"] is True
    assert out["findings"] == []


# ---------- collect ----------


def test_collect_populates_host_metrics_row(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    exit_code = cli.main(["--dir", str(tmp_path), "--json", "collect"])
    assert exit_code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["host"]

    store = Store(tmp_path)
    row = store.conn.execute("SELECT COUNT(*) FROM host_metrics WHERE host=?", (out["host"],))
    count = row.fetchone()[0]
    store.close()
    assert count == 1


# ---------- rollup ----------


def test_rollup_scores_unscored_turns(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    store = Store(tmp_path)
    store.upsert_turn(_turn("t1", started_at=now_ms() - 60_000, kpi_score=None))
    store.close()

    exit_code = cli.main(["--dir", str(tmp_path), "--json", "rollup"])
    assert exit_code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["scored"] == 1

    store2 = Store(tmp_path)
    turn = store2.get_turn("t1")
    store2.close()
    assert turn is not None
    assert turn.kpi_score is not None


def test_rollup_records_governor_shadow_decisions(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """Scoring alone never touches the governor — nothing else in the codebase calls
    Governor.choose()/.record() per turn, so rollup is the only place governor_decisions gets
    populated at all. This is what makes `governor_decisions` end up non-empty in the real
    store instead of staying at zero rows forever."""
    store = Store(tmp_path)
    store.upsert_turn(_turn("t1", started_at=now_ms() - 60_000, kpi_score=None))
    store.close()

    exit_code = cli.main(["--dir", str(tmp_path), "--json", "rollup"])
    assert exit_code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["governor_decisions"] == len(GOVERNOR_DOMAINS)

    store2 = Store(tmp_path)
    domains = {d["domain"] for d in store2.decisions_for("t1")}
    store2.close()
    assert domains == set(GOVERNOR_DOMAINS)


def test_rollup_applies_governor_toml_weights(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """cmd_rollup must score with governor.toml's [kpi.weights] table, not the hardcoded
    kpi.DEFAULT_WEIGHTS — that table is the nightly reviewer's self-tuning knob, and it's dead
    config if the rollup path never reads it. The shipped governor.toml weights a failed tool
    call differently than the equal-weighted default, so the two disagree on this turn's score."""
    from flightdeck.kpi import DEFAULT_WEIGHTS, score_turn
    from flightdeck.models import Event

    store = Store(tmp_path)
    turn = _turn("t1", started_at=now_ms() - 60_000, kpi_score=None)
    store.upsert_turn(turn)
    store.add_events([Event(turn_id="t1", ts=now_ms(), kind="tool_call", ok=0)])
    store.close()

    exit_code = cli.main(["--dir", str(tmp_path), "--json", "rollup"])
    assert exit_code == 0

    store2 = Store(tmp_path)
    scored = store2.get_turn("t1")
    events = store2.events_for("t1")
    store2.close()
    assert scored is not None

    from flightdeck.governor import load_kpi_tuning

    governor_weights, _ = load_kpi_tuning()
    governor_score, _ = score_turn(scored, events, weights=governor_weights)
    default_score, _ = score_turn(scored, events, weights=DEFAULT_WEIGHTS)

    assert governor_score != pytest.approx(default_score)
    assert scored.kpi_score == pytest.approx(governor_score)


def test_rollup_expire_removes_expired_texts(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    store = Store(tmp_path)
    store.upsert_turn(_turn("t1", started_at=now_ms() - 60_000))
    store.add_texts(
        [TextBlob(turn_id="t1", kind="prompt", seq=0, body="hi", expires_at=now_ms() - 1000)]
    )
    store.close()

    exit_code = cli.main(["--dir", str(tmp_path), "--json", "rollup", "--expire"])
    assert exit_code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["expired"] == 1

    store2 = Store(tmp_path)
    texts = store2.texts_for("t1")
    store2.close()
    assert texts == []


def test_rollup_without_expire_does_not_resurrect_expired_text(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """A rebuilt/fresh sqlite db (cursor at 0) replays a JSONL text record whose row an earlier
    --expire run already purged — the JSONL mirror is durability, not subject to expiry.
    Re-ingesting it must not resurrect already-expired text just because this rollup omitted
    --expire."""
    store = Store(tmp_path)
    store.upsert_turn(_turn("t1", started_at=now_ms() - 60_000))
    store.add_texts(
        [TextBlob(turn_id="t1", kind="prompt", seq=0, body="hi", expires_at=now_ms() - 1000)]
    )
    assert store.texts_for("t1") != []  # sanity: the row exists before expiry
    store.expire_texts()
    assert store.texts_for("t1") == []  # sanity: purged, same as a prior --expire rollup
    store.close()

    # An ad-hoc rollup with no --expire, e.g. against a freshly rebuilt sqlite db with no
    # ingest_cursor yet, replays the JSONL from byte 0 — including the already-expired text.
    exit_code = cli.main(["--dir", str(tmp_path), "--json", "rollup"])
    assert exit_code == 0
    capsys.readouterr()

    store2 = Store(tmp_path)
    texts = store2.texts_for("t1")
    store2.close()
    assert texts == []


def test_rollup_probe_replay_twice_does_not_duplicate(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """cmd_rollup replays every events-*.jsonl file on every run. Before the watermark+dedupe
    fix, a second run with no new turns/events would still re-ingest the first run's own probe
    writes (mirrored to today's JSONL) and double every probe ever recorded."""
    store = Store(tmp_path)
    store.upsert_turn(_turn("t1", started_at=now_ms() - 60_000))
    store.close()

    exit_code = cli.main(["--dir", str(tmp_path), "--json", "rollup", "--probe"])
    assert exit_code == 0
    capsys.readouterr()

    store2 = Store(tmp_path)
    first_count = store2.conn.execute("SELECT COUNT(*) FROM probes").fetchone()[0]
    store2.close()
    assert first_count == 6  # one row per probe kind

    exit_code = cli.main(["--dir", str(tmp_path), "--json", "rollup", "--probe"])
    assert exit_code == 0
    capsys.readouterr()

    store3 = Store(tmp_path)
    second_count = store3.conn.execute("SELECT COUNT(*) FROM probes").fetchone()[0]
    store3.close()
    # The second run legitimately adds 6 new rows (each probe mints a fresh ts) — that's real
    # new data. What must not happen is the first run's rows being replayed on top of that.
    assert second_count == 12


# ---------- backfill-tier ----------


def test_backfill_tier_updates_derivable_rows(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    store = Store(tmp_path)
    now = now_ms()
    store.upsert_turn(_turn("claude", started_at=now, source="claude-code"))  # tier derivable
    store.upsert_turn(_turn("has-tier", started_at=now, source="claude-code", tier="frontier"))
    store.close()

    exit_code = cli.main(["--dir", str(tmp_path), "--json", "backfill-tier"])
    assert exit_code == 0
    out = json.loads(capsys.readouterr().out)
    assert out == {"scanned": 1, "updated": 1, "unmapped": 0, "dry_run": False}

    store2 = Store(tmp_path)
    updated = store2.get_turn("claude")
    unchanged = store2.get_turn("has-tier")
    store2.close()
    assert updated is not None
    assert updated.tier == "frontier"
    assert unchanged is not None
    assert unchanged.tier == "frontier"  # untouched, not re-derived


def test_backfill_tier_counts_unmapped_models_without_guessing(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    store = Store(tmp_path)
    now = now_ms()
    turn = _turn("t1", started_at=now, source="crush")
    turn.model = "some-model-nobody-has-mapped"
    store.upsert_turn(turn)
    store.close()

    exit_code = cli.main(["--dir", str(tmp_path), "--json", "backfill-tier"])
    assert exit_code == 0
    out = json.loads(capsys.readouterr().out)
    assert out == {"scanned": 1, "updated": 0, "unmapped": 1, "dry_run": False}

    store2 = Store(tmp_path)
    unchanged = store2.get_turn("t1")
    store2.close()
    assert unchanged is not None
    assert unchanged.tier is None


def test_backfill_tier_dry_run_reports_without_writing(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    store = Store(tmp_path)
    store.upsert_turn(_turn("t1", started_at=now_ms(), source="claude-code"))
    store.close()

    exit_code = cli.main(["--dir", str(tmp_path), "--json", "backfill-tier", "--dry-run"])
    assert exit_code == 0
    out = json.loads(capsys.readouterr().out)
    assert out == {"scanned": 1, "updated": 1, "unmapped": 0, "dry_run": True}

    store2 = Store(tmp_path)
    untouched = store2.get_turn("t1")
    store2.close()
    assert untouched is not None
    assert untouched.tier is None


# ---------- sample ----------


def test_sample_write_brief_reports_dropped_cap(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    store = Store(tmp_path)
    now = now_ms()
    for i in range(5):
        store.upsert_turn(_turn(f"t{i}", started_at=now - 1000))
    store.close()

    exit_code = cli.main(["--dir", str(tmp_path), "sample", "--complex", "2", "--write-brief"])
    assert exit_code == 0
    capsys.readouterr()

    brief_path = tmp_path / "review-brief.md"
    assert brief_path.exists()
    text = brief_path.read_text()
    assert "3 turns were dropped by the cap" in text


def test_sample_write_brief_empty_window_warns(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    exit_code = cli.main(["--dir", str(tmp_path), "sample", "--write-brief"])
    assert exit_code == 0
    capsys.readouterr()

    brief_path = tmp_path / "review-brief.md"
    text = brief_path.read_text()
    assert "window is empty" in text.lower()


# ---------- show ----------


def test_show_unknown_turn_returns_1_and_writes_stderr(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    exit_code = cli.main(["--dir", str(tmp_path), "show", "nope"])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "no such turn: nope" in captured.err
    assert captured.out == ""


def test_show_known_turn_returns_full_record(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    store = Store(tmp_path)
    store.upsert_turn(_turn("t1", started_at=now_ms()))
    store.close()

    exit_code = cli.main(["--dir", str(tmp_path), "show", "t1"])
    assert exit_code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["turn"]["turn_id"] == "t1"
    assert out["events"] == []
    assert out["texts"] == []
    assert out["judgment"] is None
    assert out["governor_decisions"] == []


# ---------- guard ----------


def test_guard_all_allowed_exit_0(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    exit_code = cli.main(
        ["--dir", str(tmp_path), "guard", "--repo", str(tmp_path), "--check", "claude/rules/a.md"]
    )
    assert exit_code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["allowed"] is True
    assert out["rejected"] == {}


def test_guard_any_rejected_exit_1(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    exit_code = cli.main(
        [
            "--dir",
            str(tmp_path),
            "guard",
            "--repo",
            str(tmp_path),
            "--check",
            "claude/rules/a.md",
            "flightdeck/foo.py",
        ]
    )
    assert exit_code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["allowed"] is False
    assert "flightdeck/foo.py" in out["rejected"]


# ---------- sync ----------


def test_sync_reports_hosts_missed_and_warns(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    inbox = tmp_path / "inbox"  # deliberately absent: an offline host still must be reportable
    exit_code = cli.main(
        [
            "--dir",
            str(tmp_path / "store"),
            "sync",
            "--inbox",
            str(inbox),
            "--hosts-reached",
            "1",
            "--hosts-missed",
            "1",
        ]
    )
    assert exit_code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["hosts_missed"] == 1
    assert "warning" in out
    assert "partial" in out["warning"]


# ---------- record-change / reverted round trip ----------


def test_record_change_then_reverted_round_trip(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    exit_code = cli.main(
        [
            "--dir",
            str(tmp_path),
            "record-change",
            "--domain",
            "skills",
            "--path",
            "claude/skills/x/SKILL.md",
            "--summary",
            "tweak weights",
            "--commit",
            "abc123",
            "--evidence",
            json.dumps({"n": 5}),
        ]
    )
    assert exit_code == 0
    out = json.loads(capsys.readouterr().out)
    change_id = out["change_id"]

    exit_code = cli.main(["--dir", str(tmp_path), "reverted"])
    assert exit_code == 0
    reverted = json.loads(capsys.readouterr().out)
    assert reverted == []  # not reverted yet: must not appear on the do-not-retry list

    store = Store(tmp_path)
    store.conn.execute(
        "UPDATE tuning_changes SET reverted_at=?, revert_reason=? WHERE change_id=?",
        (now_ms(), "manual test revert", change_id),
    )
    store.close()

    exit_code = cli.main(["--dir", str(tmp_path), "reverted"])
    assert exit_code == 0
    reverted = json.loads(capsys.readouterr().out)
    assert len(reverted) == 1
    assert reverted[0]["change_id"] == change_id


# ---------- verify --auto-revert ----------


def test_verify_auto_revert_marks_regressed_change(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    now = now_ms()
    applied = now - 5 * DAY_MS  # inside the [min_age=2d, max_age=14d] judgeable window

    store = Store(tmp_path)
    store.add_tuning_change(
        TuningChange(
            change_id="c1",
            applied_at=applied,
            domain="skills",
            path="claude/skills/x/SKILL.md",
            summary="tweak",
            commit_sha="deadbeef",
        )
    )
    before_ts = applied - 1000
    after_ts = applied + 1000
    for i in range(30):
        store.upsert_turn(_turn(f"before{i}", started_at=before_ts, kpi_score=0.9))
    for i in range(30):
        store.upsert_turn(_turn(f"after{i}", started_at=after_ts, kpi_score=0.8))
    store.close()

    exit_code = cli.main(["--dir", str(tmp_path), "verify", "--auto-revert"])
    assert exit_code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["marked_reverted"] == ["c1"]
    assert "git revert --no-edit deadbeef" in out["next"]

    store2 = Store(tmp_path)
    row = store2.conn.execute(
        "SELECT reverted_at FROM tuning_changes WHERE change_id=?", ("c1",)
    ).fetchone()
    store2.close()
    assert row["reverted_at"] is not None


# ---------- kpi / governor on an empty store ----------


def test_kpi_empty_store_no_error(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    exit_code = cli.main(["--dir", str(tmp_path), "kpi"])
    assert exit_code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["empty_window"] is True


def test_governor_status_empty_store_no_error(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    exit_code = cli.main(["--dir", str(tmp_path), "governor", "status"])
    assert exit_code == 0
    out = json.loads(capsys.readouterr().out)
    assert "weights_version" in out
    assert set(out["domains"]) == {"model_tier", "skills", "compaction", "mode_delegation"}
