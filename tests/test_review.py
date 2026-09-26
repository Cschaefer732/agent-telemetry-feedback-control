from __future__ import annotations

from pathlib import Path

import pytest

from flightdeck.models import Event, Judgment, Probe, TuningChange, Turn
from flightdeck.review import (
    DAY_MS,
    Guardrails,
    Sampler,
    _matches,
    aggregates,
    complexity,
    regression_check,
)
from flightdeck.store import Store

NOW = 1_700_000_000_000  # fixed epoch ms; never wall-clock, or tests flake at day boundaries


@pytest.fixture
def store(tmp_path: Path) -> Store:
    s = Store(tmp_path / "store")
    yield s
    s.close()


def _turn(
    turn_id: str,
    *,
    started_at: int,
    session_id: str = "sess-1",
    source: str = "claude-code",
    host: str = "testhost",
    tier: str | None = None,
    kpi_score: float | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
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
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        flagged=flagged,
        judged=judged,
    )


def _event(
    turn_id: str,
    ts: int,
    kind: str,
    *,
    name: str | None = None,
    ok: int | None = None,
    payload: dict | None = None,
) -> Event:
    return Event(turn_id=turn_id, ts=ts, kind=kind, name=name, ok=ok, payload=payload or {})


# ---------- Guardrails: allow/deny ----------


def test_allowed_path_is_allowed(tmp_path: Path) -> None:
    guardrails = Guardrails(tmp_path)
    allowed, reason = guardrails.is_writable("claude/rules/foo.md")
    assert allowed is True
    assert reason.startswith("allowed:")


def test_unlisted_path_rejected(tmp_path: Path) -> None:
    guardrails = Guardrails(tmp_path)
    allowed, reason = guardrails.is_writable("random/file.txt")
    assert allowed is False
    assert reason == "not_in_allowlist"


def test_deny_beats_allow(tmp_path: Path) -> None:
    # A path that would satisfy an allow pattern but also matches a deny pattern (e.g. a .go
    # file living under an otherwise-writable directory) must still be rejected.
    guardrails = Guardrails(
        tmp_path,
        writable=("crush-fork/skills/**/*",),
        denied=("**/*.go",),
    )
    allowed, reason = guardrails.is_writable("crush-fork/skills/foo.go")
    assert allowed is False
    assert reason == "denied:**/*.go"


def test_path_traversal_dotdot_rejected(tmp_path: Path) -> None:
    guardrails = Guardrails(tmp_path)
    allowed, reason = guardrails.is_writable("../../etc/passwd")
    assert allowed is False
    assert reason == "outside_repo"


def test_path_traversal_buried_dotdot_rejected(tmp_path: Path) -> None:
    guardrails = Guardrails(tmp_path)
    allowed, reason = guardrails.is_writable("crush-fork/skills/../../../../tmp/x.md")
    assert allowed is False
    assert reason == "outside_repo"


def test_absolute_path_outside_repo_rejected(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    guardrails = Guardrails(repo)
    allowed, reason = guardrails.is_writable("/etc/passwd")
    assert allowed is False
    assert reason == "outside_repo"


def test_symlink_escaping_repo_rejected(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "secret.md"
    target.write_text("x")

    link_dir = repo / "claude" / "rules"
    link_dir.mkdir(parents=True)
    link = link_dir / "escape.md"
    link.symlink_to(target)

    guardrails = Guardrails(repo)
    allowed, reason = guardrails.is_writable(link)
    assert allowed is False
    assert reason == "outside_repo"


def test_nonexistent_path_still_evaluated(tmp_path: Path) -> None:
    # The reviewer creates new files, so a path that isn't on disk yet must still be checked
    # against the allow/deny lists rather than being waved through (or blocked) on that basis.
    guardrails = Guardrails(tmp_path)

    allowed, reason = guardrails.is_writable("claude/rules/new_file.md")
    assert allowed is True
    assert reason.startswith("allowed:")

    allowed2, reason2 = guardrails.is_writable("some/random/new_file.md")
    assert allowed2 is False
    assert reason2 == "not_in_allowlist"


def test_check_returns_only_rejections(tmp_path: Path) -> None:
    guardrails = Guardrails(tmp_path)
    rejected = guardrails.check(["claude/rules/a.md", "random/file.txt", "flightdeck/review.py"])
    assert set(rejected) == {"random/file.txt", "flightdeck/review.py"}
    assert rejected["random/file.txt"] == "not_in_allowlist"
    assert rejected["flightdeck/review.py"].startswith("denied:")


def test_check_empty_dict_means_proceed(tmp_path: Path) -> None:
    guardrails = Guardrails(tmp_path)
    rejected = guardrails.check(["claude/rules/a.md", "claude/skills/foo/SKILL.md"])
    assert rejected == {}


def test_matches_double_star_multi_segment() -> None:
    assert _matches("a/b/c.md", "a/**/c.md") is True


def test_matches_double_star_zero_segment() -> None:
    assert _matches("a/c.md", "a/**/c.md") is True


def test_git_dir_rejected(tmp_path: Path) -> None:
    guardrails = Guardrails(tmp_path)
    allowed, reason = guardrails.is_writable(".git/config")
    assert allowed is False
    assert reason.startswith("denied:")


def test_nested_git_dir_rejected(tmp_path: Path) -> None:
    guardrails = Guardrails(tmp_path)
    allowed, reason = guardrails.is_writable("sub/.git/config")
    assert allowed is False
    assert reason.startswith("denied:")


# ---------- complexity ----------


def test_complexity_ranks_busy_turn_above_trivial() -> None:
    busy = _turn("busy", started_at=NOW, prompt_tokens=4000, completion_tokens=1000)
    busy_events = [
        _event("busy", NOW, "tool_call", name="Bash"),
        _event("busy", NOW, "tool_call", name="Read"),
        _event("busy", NOW + 1, "edit", payload={"path": "a.py"}),
        _event("busy", NOW + 2, "edit", payload={"path": "b.py"}),
    ]
    trivial = _turn("trivial", started_at=NOW, prompt_tokens=10, completion_tokens=5)

    assert complexity(busy, busy_events) > complexity(trivial, [])


# ---------- complex_turns ----------


def test_complex_turns_dropped_accurate(store: Store) -> None:
    for i in range(5):
        store.upsert_turn(_turn(f"t{i}", started_at=NOW - 1000, prompt_tokens=(i + 1) * 1000))
    sampler = Sampler(store, now=NOW)
    sample = sampler.complex_turns(limit=2)

    assert len(sample.turns) == 2
    assert sample.dropped == 3
    assert {t.turn_id for t in sample.turns} == {"t4", "t3"}  # highest token counts


# ---------- low_kpi_turns ----------


def test_low_kpi_turns_skips_judged_and_orders(store: Store) -> None:
    store.upsert_turn(_turn("t1", started_at=NOW - 1000, kpi_score=0.9))
    store.upsert_turn(_turn("t2", started_at=NOW - 900, kpi_score=0.3))
    store.upsert_turn(_turn("t3", started_at=NOW - 800, kpi_score=0.1))
    store.add_judgment(Judgment(turn_id="t3", judge_model="m", verdict="ok", created_at=NOW))

    sampler = Sampler(store, now=NOW)
    sample = sampler.low_kpi_turns(limit=10)

    assert [t.turn_id for t in sample.turns] == ["t2", "t1"]
    assert sample.dropped == 0


def test_low_kpi_turns_excludes_null_kpi(store: Store) -> None:
    store.upsert_turn(_turn("t1", started_at=NOW - 1000, kpi_score=0.5))
    store.upsert_turn(_turn("t2", started_at=NOW - 900, kpi_score=None))

    sampler = Sampler(store, now=NOW)
    sample = sampler.low_kpi_turns(limit=10)

    assert [t.turn_id for t in sample.turns] == ["t1"]


# ---------- sweep: repeated_tool_failure ----------


def test_sweep_repeated_tool_failure_fires(store: Store) -> None:
    store.upsert_turn(_turn("t1", started_at=NOW - 1000))
    store.add_events(
        [
            _event("t1", NOW - 900, "tool_call", name="Bash", ok=0),
            _event("t1", NOW - 800, "tool_call", name="Bash", ok=0),
            _event("t1", NOW - 700, "tool_call", name="Bash", ok=0),
        ]
    )
    signals = Sampler(store, now=NOW).sweep()
    matches = [s for s in signals if s.signal == "repeated_tool_failure"]
    assert len(matches) == 1
    assert matches[0].detail == {"tool": "Bash", "count": 3}


def test_sweep_repeated_tool_failure_below_threshold(store: Store) -> None:
    store.upsert_turn(_turn("t1", started_at=NOW - 1000))
    store.add_events(
        [
            _event("t1", NOW - 900, "tool_call", name="Bash", ok=0),
            _event("t1", NOW - 800, "tool_call", name="Bash", ok=0),
        ]
    )
    signals = Sampler(store, now=NOW).sweep()
    assert not [s for s in signals if s.signal == "repeated_tool_failure"]


# ---------- sweep: permission_denials ----------


def test_sweep_permission_denials_fires(store: Store) -> None:
    store.upsert_turn(_turn("t1", started_at=NOW - 1000))
    store.add_events(
        [
            _event("t1", NOW - 900, "permission", payload={"decision": "deny"}),
            _event("t1", NOW - 800, "permission", payload={"decision": "deny"}),
        ]
    )
    signals = Sampler(store, now=NOW).sweep()
    matches = [s for s in signals if s.signal == "permission_denials"]
    assert len(matches) == 1
    assert matches[0].detail["count"] == 2


def test_sweep_permission_denials_below_threshold(store: Store) -> None:
    store.upsert_turn(_turn("t1", started_at=NOW - 1000))
    store.add_events([_event("t1", NOW - 900, "permission", payload={"decision": "deny"})])
    signals = Sampler(store, now=NOW).sweep()
    assert not [s for s in signals if s.signal == "permission_denials"]


# ---------- sweep: hook_error ----------


def test_sweep_hook_error_fires(store: Store) -> None:
    store.upsert_turn(_turn("t1", started_at=NOW - 1000))
    store.add_events([_event("t1", NOW - 900, "hook", name="PreToolUse", ok=0)])
    signals = Sampler(store, now=NOW).sweep()
    matches = [s for s in signals if s.signal == "hook_error"]
    assert len(matches) == 1
    assert matches[0].detail == {"hook": "PreToolUse", "count": 1}


# ---------- sweep: edit_revert_loop ----------


def test_sweep_edit_revert_loop_fires(store: Store) -> None:
    store.upsert_turn(_turn("t1", started_at=NOW - 1000))
    store.add_events(
        [
            _event("t1", NOW - 900, "revert", payload={"path": "a.md"}),
            _event("t1", NOW - 800, "revert", payload={"path": "a.md"}),
        ]
    )
    signals = Sampler(store, now=NOW).sweep()
    matches = [s for s in signals if s.signal == "edit_revert_loop"]
    assert len(matches) == 1
    assert matches[0].detail == {"path": "a.md", "count": 2}


def test_sweep_edit_revert_loop_below_threshold(store: Store) -> None:
    store.upsert_turn(_turn("t1", started_at=NOW - 1000))
    store.add_events([_event("t1", NOW - 900, "revert", payload={"path": "a.md"})])
    signals = Sampler(store, now=NOW).sweep()
    assert not [s for s in signals if s.signal == "edit_revert_loop"]


# ---------- sweep: repeated_interrupts ----------


def test_sweep_repeated_interrupts_fires(store: Store) -> None:
    store.upsert_turn(_turn("t1", started_at=NOW - 1000))
    store.add_events(
        [
            _event("t1", NOW - 900, "interrupt"),
            _event("t1", NOW - 800, "interrupt"),
        ]
    )
    signals = Sampler(store, now=NOW).sweep()
    matches = [s for s in signals if s.signal == "repeated_interrupts"]
    assert len(matches) == 1
    assert matches[0].detail["count"] == 2


def test_sweep_repeated_interrupts_below_threshold(store: Store) -> None:
    store.upsert_turn(_turn("t1", started_at=NOW - 1000))
    store.add_events([_event("t1", NOW - 900, "interrupt")])
    signals = Sampler(store, now=NOW).sweep()
    assert not [s for s in signals if s.signal == "repeated_interrupts"]


# ---------- _probe_signals ----------


def test_probe_signals_unhealthy_and_healthy(store: Store) -> None:
    store.add_probe(Probe(ts=NOW - 500, host="h1", kind="hook_liveness", ok=1, total=2, detail={}))
    store.add_probe(Probe(ts=NOW - 500, host="h1", kind="judge_queue", ok=3, total=3, detail={}))

    signals = Sampler(store, now=NOW)._probe_signals()

    kinds = {s.signal for s in signals}
    assert "probe_hook_liveness" in kinds
    assert "probe_judge_queue" not in kinds


def test_probe_never_ran_fires_on_empty_window(store: Store) -> None:
    # The single most important sweep behaviour: a collector that produced nothing is a finding,
    # not silence. No probes at all must not read as a clean night.
    signals = Sampler(store, now=NOW)._probe_signals()
    assert any(s.signal == "probe_never_ran" for s in signals)


# ---------- regression_check ----------


def test_regression_check_reverts_on_kpi_drop(store: Store) -> None:
    applied = NOW - 5 * DAY_MS
    store.add_tuning_change(
        TuningChange(
            change_id="c1",
            applied_at=applied,
            domain="skills",
            path="claude/skills/x/SKILL.md",
            summary="tweak",
        )
    )
    before_ts = applied - 1000
    after_ts = applied + 1000
    for i in range(30):
        store.upsert_turn(_turn(f"before{i}", started_at=before_ts, kpi_score=0.9))
    for i in range(30):
        store.upsert_turn(_turn(f"after{i}", started_at=after_ts, kpi_score=0.8))

    verdicts = regression_check(store, now=NOW)
    v = next(v for v in verdicts if v.change_id == "c1")

    assert v.should_revert is True
    assert v.reason == "kpi_regression"
    assert v.n_before == 30
    assert v.n_after == 30


def test_regression_check_insufficient_samples(store: Store) -> None:
    applied = NOW - 5 * DAY_MS
    store.add_tuning_change(
        TuningChange(change_id="c2", applied_at=applied, domain="skills", path="p", summary="s")
    )
    for i in range(5):
        store.upsert_turn(_turn(f"b{i}", started_at=applied - 1000, kpi_score=0.9))
    for i in range(5):
        store.upsert_turn(_turn(f"a{i}", started_at=applied + 1000, kpi_score=0.8))

    verdicts = regression_check(store, now=NOW)
    v = next(v for v in verdicts if v.change_id == "c2")

    assert v.should_revert is False
    assert v.reason == "insufficient_samples"


def test_regression_check_improvement_within_tolerance(store: Store) -> None:
    applied = NOW - 5 * DAY_MS
    store.add_tuning_change(
        TuningChange(change_id="c3", applied_at=applied, domain="skills", path="p", summary="s")
    )
    for i in range(30):
        store.upsert_turn(_turn(f"b{i}", started_at=applied - 1000, kpi_score=0.6))
    for i in range(30):
        store.upsert_turn(_turn(f"a{i}", started_at=applied + 1000, kpi_score=0.8))

    verdicts = regression_check(store, now=NOW)
    v = next(v for v in verdicts if v.change_id == "c3")

    assert v.should_revert is False
    assert v.reason == "within_tolerance"


def test_regression_check_excludes_out_of_age_range(store: Store) -> None:
    too_young = NOW - 1 * DAY_MS  # default min_age_days=2
    too_old = NOW - 20 * DAY_MS  # default max_age_days=14
    store.add_tuning_change(
        TuningChange(
            change_id="young", applied_at=too_young, domain="skills", path="p", summary="s"
        )
    )
    store.add_tuning_change(
        TuningChange(change_id="old", applied_at=too_old, domain="skills", path="p", summary="s")
    )

    verdicts = regression_check(store, now=NOW)
    change_ids = {v.change_id for v in verdicts}

    assert "young" not in change_ids
    assert "old" not in change_ids


def test_regression_check_skips_reverted(store: Store) -> None:
    applied = NOW - 5 * DAY_MS
    store.add_tuning_change(
        TuningChange(
            change_id="c5",
            applied_at=applied,
            domain="skills",
            path="p",
            summary="s",
            reverted_at=applied + 500,
        )
    )
    verdicts = regression_check(store, now=NOW)
    assert all(v.change_id != "c5" for v in verdicts)


# ---------- aggregates ----------


def test_aggregates_empty_window(store: Store) -> None:
    result = aggregates(store, since_ms=NOW - DAY_MS, until_ms=NOW)
    assert result["empty_window"] is True
    assert result["turns"] == 0


def test_aggregates_means_by_tier_source_host(store: Store) -> None:
    store.upsert_turn(
        _turn("t1", started_at=NOW - 900, tier="fast", source="crush", host="h1", kpi_score=0.5)
    )
    store.upsert_turn(
        _turn(
            "t2", started_at=NOW - 800, tier="fast", source="claude-code", host="h2", kpi_score=0.7
        )
    )
    store.upsert_turn(
        _turn("t3", started_at=NOW - 700, tier="balanced", source="crush", host="h1", kpi_score=0.9)
    )

    result = aggregates(store, since_ms=NOW - DAY_MS, until_ms=NOW)

    assert result["kpi_by_tier"]["fast"] == pytest.approx(0.6)
    assert result["kpi_by_tier"]["balanced"] == pytest.approx(0.9)
    assert result["kpi_by_source"]["crush"] == pytest.approx(0.7)
    assert result["kpi_by_source"]["claude-code"] == pytest.approx(0.7)
    assert result["kpi_by_host"]["h1"] == pytest.approx(0.7)
    assert result["kpi_by_host"]["h2"] == pytest.approx(0.7)


def test_aggregates_tool_failure_leaderboard(store: Store) -> None:
    store.upsert_turn(_turn("t1", started_at=NOW - 900))
    store.add_events(
        [
            _event("t1", NOW - 890, "tool_call", name="Bash", ok=1),
            _event("t1", NOW - 880, "tool_call", name="Bash", ok=0),
            _event("t1", NOW - 870, "tool_call", name="Read", ok=1),
        ]
    )

    result = aggregates(store, since_ms=NOW - DAY_MS, until_ms=NOW)
    board = {row["tool"]: row for row in result["tool_failure_leaderboard"]}

    assert board["Bash"]["failures"] == 1
    assert board["Bash"]["calls"] == 2
    assert "Read" not in board  # zero failures never enters the leaderboard counter


def test_aggregates_context_snapshot_not_counted_as_compaction(store: Store) -> None:
    store.upsert_turn(_turn("t1", started_at=NOW - 900))
    store.add_events(
        [
            _event("t1", NOW - 890, "compaction", name="context_snapshot"),
            _event("t1", NOW - 880, "compaction", name="full_compact"),
        ]
    )

    result = aggregates(store, since_ms=NOW - DAY_MS, until_ms=NOW)
    assert result["compactions"] == 1
