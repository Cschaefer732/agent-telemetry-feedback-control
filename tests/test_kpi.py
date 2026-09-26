from __future__ import annotations

import json
import math

import pytest

from flightdeck.kpi import (
    DEFAULT_THRESHOLDS,
    DEFAULT_WEIGHTS,
    Components,
    composite,
    score_and_persist,
    score_components,
    score_turn,
    should_flag,
    trailing_baseline,
)
from flightdeck.models import Event, Turn
from flightdeck.store import Store


def make_turn(**overrides) -> Turn:
    defaults = dict(
        turn_id="t1",
        session_id="s1",
        source="claude-code",
        host="h1",
        started_at=1_000_000,
        outcome="ok",
    )
    defaults.update(overrides)
    return Turn(**defaults)


def ev(kind: str, **kw) -> Event:
    return Event(turn_id="t1", ts=1_000_000, kind=kind, **kw)


# ---------- DEFAULT_WEIGHTS / DEFAULT_THRESHOLDS ----------


def test_default_weights_sum_to_one():
    assert set(DEFAULT_WEIGHTS) == {
        "completion",
        "tool_reliability",
        "efficiency",
        "focus",
        "context_health",
        "autonomy",
    }
    assert sum(DEFAULT_WEIGHTS.values()) == 1.0


def test_default_thresholds_present():
    assert DEFAULT_THRESHOLDS["tool_error_rate"] == 0.25
    assert DEFAULT_THRESHOLDS["low_kpi"] == 0.5


# ---------- completion ----------


def test_completion_ratio_from_todo_events():
    turn = make_turn()
    events = [ev("todo", payload={"opened": 4, "closed": 2})]
    c = score_components(turn, events)
    assert c.completion == 0.5


def test_completion_no_todo_events_ok_outcome():
    turn = make_turn(outcome="ok")
    c = score_components(turn, [])
    assert c.completion == 1.0


def test_completion_no_todo_events_outcome_none_is_neutral():
    # outcome None (orphan-backfilled / outcome-blind turn) is unmeasurable, not a failure —
    # it must default optimistically, not be scored as 0.0.
    turn = make_turn(outcome=None)
    c = score_components(turn, [])
    assert c.completion == 1.0


def test_completion_no_todo_events_outcome_error_is_zero():
    turn = make_turn(outcome="error")
    c = score_components(turn, [])
    assert c.completion == 0.0


def test_completion_critic_fail_caps_at_half():
    turn = make_turn(outcome="ok")
    events = [
        ev("todo", payload={"opened": 2, "closed": 2}),
        ev("critic", payload={"verdict": "fail"}),
    ]
    c = score_components(turn, events)
    assert c.completion == 0.5


def test_completion_error_outcome_forces_zero():
    turn = make_turn(outcome="error")
    events = [ev("todo", payload={"opened": 2, "closed": 2})]
    c = score_components(turn, events)
    assert c.completion == 0.0


def test_completion_cancelled_outcome_forces_zero():
    turn = make_turn(outcome="cancelled")
    c = score_components(turn, [])
    assert c.completion == 0.0


# ---------- tool_reliability ----------


def test_tool_reliability_no_tool_calls():
    turn = make_turn()
    c = score_components(turn, [])
    assert c.tool_reliability == 1.0


def test_tool_reliability_partial_failures():
    turn = make_turn()
    events = [
        ev("tool_call", ok=1),
        ev("tool_call", ok=0),
        ev("tool_call", ok=1),
        ev("tool_call", ok=0),
    ]
    c = score_components(turn, events)
    assert c.tool_reliability == 0.5


def test_tool_reliability_all_failed():
    turn = make_turn()
    events = [ev("tool_call", ok=0), ev("tool_call", ok=0)]
    c = score_components(turn, events)
    assert c.tool_reliability == 0.0


# ---------- efficiency ----------


def test_efficiency_no_baseline():
    turn = make_turn(prompt_tokens=500, completion_tokens=500)
    c = score_components(turn, [], baseline_tokens=None)
    assert c.efficiency == 1.0


def test_efficiency_zero_total_tokens():
    turn = make_turn(prompt_tokens=0, completion_tokens=0)
    c = score_components(turn, [], baseline_tokens=100.0)
    assert c.efficiency == 1.0


def test_efficiency_at_baseline_scores_one():
    turn = make_turn(prompt_tokens=50, completion_tokens=50)
    c = score_components(turn, [], baseline_tokens=100.0)
    assert c.efficiency == 1.0


def test_efficiency_above_baseline_scores_less_than_one():
    turn = make_turn(prompt_tokens=100, completion_tokens=100)
    c = score_components(turn, [], baseline_tokens=100.0)
    assert c.efficiency == 0.5


# ---------- focus ----------


def test_focus_no_edits_is_perfect():
    turn = make_turn()
    c = score_components(turn, [])
    assert c.focus == 1.0


def test_focus_repeat_edit_penalized():
    turn = make_turn()
    events = [
        ev("edit", payload={"path": "a.py"}),
        ev("edit", payload={"path": "a.py"}),
    ]
    c = score_components(turn, events)
    # churn = (1 repeat + 0) / 2 edits = 0.5
    assert c.focus == 0.5


def test_focus_revert_weighs_double():
    turn = make_turn()
    events = [
        ev("edit", payload={"path": "a.py"}),
        ev("revert", payload={"path": "a.py"}),
    ]
    c = score_components(turn, events)
    # churn = (0 repeat + 2*1 revert) / 1 edit = 2 -> clamped
    assert c.focus == 0.0


# ---------- context_health ----------


def test_context_health_no_window_is_perfect():
    turn = make_turn(context_peak=None, context_window=None)
    c = score_components(turn, [])
    assert c.context_health == 1.0


def test_context_health_occupancy_penalty():
    turn = make_turn(context_peak=50, context_window=100)
    c = score_components(turn, [])
    assert c.context_health == 0.5


def test_context_health_midturn_compaction_penalty():
    turn = make_turn(context_peak=50, context_window=100)
    events = [ev("compaction")]
    c = score_components(turn, events)
    # 1 - 0.5 - 0.2 = 0.3
    assert c.context_health == pytest.approx(0.3)


# ---------- autonomy ----------


def test_autonomy_clean_turn():
    turn = make_turn()
    c = score_components(turn, [])
    assert c.autonomy == 1.0


def test_autonomy_interrupts_and_denies():
    turn = make_turn()
    events = [
        ev("interrupt"),
        ev("permission", payload={"decision": "deny"}),
        ev("permission", payload={"decision": "allow"}),
    ]
    c = score_components(turn, events)
    # (1 interrupt + 1 deny) / 3 = 0.667 -> autonomy = 0.333
    assert c.autonomy == pytest.approx(1 - 2 / 3)


def test_autonomy_floors_at_zero_with_many_interrupts():
    turn = make_turn()
    events = [ev("interrupt") for _ in range(5)]
    c = score_components(turn, events)
    assert c.autonomy == 0.0


# ---------- all-NULL claude-code turn ----------


def test_all_null_claude_code_turn_never_raises():
    turn = Turn(
        turn_id="null-turn",
        session_id="s1",
        source="claude-code",
        host="h1",
        started_at=1_000_000,
        model_ms=None,
        context_window=None,
        retries=None,
        prompt_tokens=None,
        completion_tokens=None,
        context_peak=None,
        wall_ms=None,
        outcome=None,
    )
    components = score_components(turn, [])
    for value in components.as_dict().values():
        assert 0.0 <= value <= 1.0
        assert not math.isnan(value)
    score, _ = score_turn(turn, [])
    assert 0.0 <= score <= 1.0
    flagged, reasons = should_flag(turn, [], components)
    assert isinstance(flagged, bool)
    assert isinstance(reasons, list)


# ---------- adversarial clamp ----------


def test_adversarial_inputs_never_leave_unit_interval():
    turn = make_turn(
        prompt_tokens=-500,
        completion_tokens=-500,
        context_peak=-10,
        context_window=5,
        outcome="ok",
    )
    events = [
        ev("todo", payload={"opened": 1, "closed": 50}),  # closed > opened
        ev("tool_call", ok=0),
        ev("tool_call", ok=1),
        ev("edit", payload={"path": "a.py"}),
        ev("edit", payload={"path": "a.py"}),
        ev("edit", payload={"path": "a.py"}),
        ev("revert", payload={"path": "a.py"}),
        ev("revert", payload={"path": "a.py"}),
        ev("compaction"),
        ev("compaction"),
        ev("compaction"),
        ev("interrupt"),
        ev("permission", payload={"decision": "deny"}),
    ]
    components = score_components(turn, events, baseline_tokens=999999.0)
    for name, value in components.as_dict().items():
        assert 0.0 <= value <= 1.0, f"{name}={value} out of range"


# ---------- composite ----------


def test_composite_default_weights():
    components = Components(
        completion=1.0,
        tool_reliability=1.0,
        efficiency=1.0,
        focus=1.0,
        context_health=1.0,
        autonomy=1.0,
    )
    assert composite(components) == pytest.approx(1.0)


def test_composite_custom_weights():
    components = Components(
        completion=1.0,
        tool_reliability=0.0,
        efficiency=0.0,
        focus=0.0,
        context_health=0.0,
        autonomy=0.0,
    )
    weights = {
        "completion": 1.0,
        "tool_reliability": 0.0,
        "efficiency": 0.0,
        "focus": 0.0,
        "context_health": 0.0,
        "autonomy": 0.0,
    }
    assert composite(components, weights) == 1.0


# ---------- should_flag reasons ----------


def _perfect_components() -> Components:
    return Components(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)


def test_flag_outcome_not_ok():
    turn = make_turn(outcome="error")
    flagged, reasons = should_flag(turn, [], _perfect_components())
    assert flagged
    assert reasons == ["outcome_not_ok"]


def test_flag_critic_fail():
    turn = make_turn(outcome="ok")
    events = [ev("critic", payload={"verdict": "fail"})]
    flagged, reasons = should_flag(turn, events, _perfect_components())
    assert flagged
    assert reasons == ["critic_fail"]


def test_flag_tool_error_rate():
    turn = make_turn(outcome="ok")
    events = [ev("tool_call", ok=0), ev("tool_call", ok=0), ev("tool_call", ok=1)]
    flagged, reasons = should_flag(turn, events, _perfect_components())
    assert flagged
    assert reasons == ["tool_error_rate"]


def test_flag_slow():
    turn = make_turn(outcome="ok", wall_ms=5000)
    flagged, reasons = should_flag(turn, [], _perfect_components(), p95_wall_ms=1000.0)
    assert flagged
    assert reasons == ["slow"]


def test_flag_midturn_compaction():
    turn = make_turn(outcome="ok")
    events = [ev("compaction")]
    flagged, reasons = should_flag(turn, events, _perfect_components())
    assert flagged
    assert reasons == ["midturn_compaction"]


def test_flag_edit_revert():
    turn = make_turn(outcome="ok")
    events = [ev("revert", payload={"path": "a.py"})]
    flagged, reasons = should_flag(turn, events, _perfect_components())
    assert flagged
    assert reasons == ["edit_revert"]


def test_flag_user_interrupt():
    turn = make_turn(outcome="ok")
    events = [ev("interrupt")]
    flagged, reasons = should_flag(turn, events, _perfect_components())
    assert flagged
    assert reasons == ["user_interrupt"]


def test_flag_low_kpi():
    turn = make_turn(outcome="ok")
    low_components = Components(0.1, 0.1, 0.1, 0.1, 0.1, 0.1)
    flagged, reasons = should_flag(turn, [], low_components)
    assert flagged
    assert reasons == ["low_kpi"]


def test_no_flag_on_clean_turn():
    turn = make_turn(outcome="ok", wall_ms=100)
    flagged, reasons = should_flag(turn, [], _perfect_components(), p95_wall_ms=5000.0)
    assert not flagged
    assert reasons == []


def test_flag_low_kpi_uses_passed_weights_not_default():
    """should_flag's low_kpi check must recompute the composite with the SAME weights that
    produced the stored score, not silently fall back to DEFAULT_WEIGHTS — otherwise the flag
    can disagree with the score that was actually persisted. tool_reliability=0.0, everything
    else 1.0: under DEFAULT_WEIGHTS (equal 1/6 each) the composite is 5/6 (not low); a weight
    table that puts all the weight on tool_reliability drives it to 0.0 (low)."""
    turn = make_turn(outcome="ok")
    components = Components(
        completion=1.0,
        tool_reliability=0.0,
        efficiency=1.0,
        focus=1.0,
        context_health=1.0,
        autonomy=1.0,
    )
    weights = {
        "completion": 0.0,
        "tool_reliability": 1.0,
        "efficiency": 0.0,
        "focus": 0.0,
        "context_health": 0.0,
        "autonomy": 0.0,
    }

    default_flagged, _ = should_flag(turn, [], components)
    weighted_flagged, weighted_reasons = should_flag(turn, [], components, weights=weights)

    assert default_flagged is False
    assert weighted_flagged is True
    assert weighted_reasons == ["low_kpi"]


# ---------- score_and_persist ----------


def test_score_and_persist_empty_store_round_trip(tmp_path):
    store = Store(tmp_path)
    try:
        turn = Turn(
            turn_id="rt-1",
            session_id="s1",
            source="claude-code",
            host="h1",
            started_at=1_000_000,
            tier="fast",
            outcome="ok",
            wall_ms=200,
            prompt_tokens=100,
            completion_tokens=100,
        )
        store.upsert_turn(turn)
        store.add_events(
            [
                Event(turn_id="rt-1", ts=1_000_000, kind="tool_call", ok=1),
                Event(
                    turn_id="rt-1",
                    ts=1_000_001,
                    kind="todo",
                    payload={"opened": 2, "closed": 2},
                ),
            ]
        )

        score, components, reasons = score_and_persist(store, turn)

        assert 0.0 <= score <= 1.0
        assert reasons == []
        assert turn.kpi_score == score
        assert turn.flagged == 0

        persisted = store.get_turn("rt-1")
        assert persisted is not None
        assert persisted.kpi_score == pytest.approx(score)
        assert json.loads(persisted.kpi_components) == components.as_dict()
        assert persisted.flagged == 0
    finally:
        store.close()


def test_score_and_persist_uses_trailing_baseline(tmp_path):
    store = Store(tmp_path)
    try:
        day_ms = 24 * 60 * 60 * 1000
        base_time = 10 * day_ms

        for i in range(3):
            prior = Turn(
                turn_id=f"prior-{i}",
                session_id="s1",
                source="claude-code",
                host="h1",
                started_at=base_time - (i + 1) * day_ms,
                tier="fast",
                outcome="ok",
                prompt_tokens=50,
                completion_tokens=50,
            )
            store.upsert_turn(prior)

        turn = Turn(
            turn_id="new-1",
            session_id="s1",
            source="claude-code",
            host="h1",
            started_at=base_time,
            tier="fast",
            outcome="ok",
            prompt_tokens=50,
            completion_tokens=50,
        )
        store.upsert_turn(turn)

        score, components, _ = score_and_persist(store, turn)
        # baseline (100) / total_tokens (100) == 1.0 -> efficiency perfect
        assert components.efficiency == pytest.approx(1.0)
    finally:
        store.close()


# ---------- output equivalence: old full-row-pull baseline vs new SQL percentile ----------


def _percentile_of(values: list[float], pct: float) -> float:
    """Old algorithm, verbatim: score_and_persist used to inline this. Kept here only as the
    equivalence oracle for test_score_and_persist_matches_old_full_scan_baseline below."""
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(pct * (len(ordered) - 1))))
    return float(ordered[idx])


def _old_baseline(store: Store, turn: Turn) -> tuple[float | None, float | None]:
    """Reference reimplementation of score_and_persist's pre-fix baseline lookup: pull every
    candidate turn's full row via iter_turns and rank it in Python. This is the O(W) per-turn,
    O(W^2) per-rollup path the fix replaces with Store.percentile SQL calls."""
    window_start = turn.started_at - 7 * 24 * 60 * 60 * 1000
    candidates = [
        t
        for t in store.iter_turns(
            since_ms=window_start, until_ms=turn.started_at, source=turn.source
        )
        if t.tier == turn.tier and t.turn_id != turn.turn_id
    ]
    token_values = [float(t.total_tokens) for t in candidates if t.total_tokens]
    baseline_tokens = _percentile_of(token_values, 0.5) if token_values else None
    wall_values = [float(t.wall_ms) for t in candidates if t.wall_ms is not None]
    p95_wall_ms = _percentile_of(wall_values, 0.95) if wall_values else None
    return baseline_tokens, p95_wall_ms


def test_score_and_persist_applies_custom_weights_and_thresholds(tmp_path):
    """governor.toml's [kpi.weights]/[thresholds] tables must actually reach score_and_persist —
    passing them through score_turn/should_flag, not just DEFAULT_WEIGHTS/DEFAULT_THRESHOLDS.
    A single failed tool call makes tool_reliability=0.0 while every other component is 1.0, so a
    weight table that de-emphasizes tool_reliability produces a strictly higher composite than the
    equal-weighted default — and a permissive tool_error_rate threshold un-flags what the default
    threshold flags."""
    store = Store(tmp_path)
    try:
        turn = Turn(
            turn_id="rt-weighted",
            session_id="s1",
            source="claude-code",
            host="h1",
            started_at=1_000_000,
            outcome="ok",
        )
        store.upsert_turn(turn)
        store.add_events([Event(turn_id="rt-weighted", ts=1_000_000, kind="tool_call", ok=0)])

        default_score, _, default_reasons = score_and_persist(store, turn)

        turn2 = Turn(**{**turn.__dict__, "kpi_score": None, "flagged": 0})
        custom_weights = {
            "completion": 0.5,
            "tool_reliability": 0.0,
            "efficiency": 0.1,
            "focus": 0.1,
            "context_health": 0.15,
            "autonomy": 0.15,
        }
        custom_thresholds = {"tool_error_rate": 1.0, "low_kpi": 0.0}
        custom_score, _, custom_reasons = score_and_persist(
            store, turn2, weights=custom_weights, thresholds=custom_thresholds
        )

        assert custom_score != pytest.approx(default_score)
        assert "tool_error_rate" in default_reasons
        assert "tool_error_rate" not in custom_reasons
    finally:
        store.close()


def test_score_and_persist_matches_old_full_scan_baseline(tmp_path):
    """Regression test for the O(W^2) rollup fix: score_and_persist's default (per-turn) baseline
    lookup, now backed by Store.percentile SQL calls, must produce the identical kpi_score the
    old full-row-pull-and-rank-in-Python path produced, on a fixture that stresses every filter
    the baseline honors (source, tier, trailing window, self-exclusion)."""
    store = Store(tmp_path)
    try:
        day_ms = 24 * 60 * 60 * 1000
        base_time = 30 * day_ms

        # In-window, matching source+tier -> counted.
        for i, (tokens, wall) in enumerate([(80, 500), (120, 700), (200, 900), (60, 1100)]):
            store.upsert_turn(
                Turn(
                    turn_id=f"in-{i}",
                    session_id="s1",
                    source="claude-code",
                    host="h1",
                    started_at=base_time - (i + 1) * day_ms,
                    tier="fast",
                    outcome="ok",
                    prompt_tokens=tokens // 2,
                    completion_tokens=tokens - tokens // 2,
                    wall_ms=wall,
                )
            )
        # Wrong tier -> must be excluded from candidates.
        store.upsert_turn(
            Turn(
                turn_id="wrong-tier",
                session_id="s1",
                source="claude-code",
                host="h1",
                started_at=base_time - day_ms,
                tier="balanced",
                outcome="ok",
                prompt_tokens=9000,
                completion_tokens=9000,
                wall_ms=99000,
            )
        )
        # Wrong source -> must be excluded from candidates.
        store.upsert_turn(
            Turn(
                turn_id="wrong-source",
                session_id="s1",
                source="crush",
                host="h1",
                started_at=base_time - day_ms,
                tier="fast",
                outcome="ok",
                prompt_tokens=9000,
                completion_tokens=9000,
                wall_ms=99000,
            )
        )
        # Outside the trailing 7-day window -> must be excluded.
        store.upsert_turn(
            Turn(
                turn_id="too-old",
                session_id="s1",
                source="claude-code",
                host="h1",
                started_at=base_time - 8 * day_ms,
                tier="fast",
                outcome="ok",
                prompt_tokens=1,
                completion_tokens=1,
                wall_ms=1,
            )
        )

        turn = Turn(
            turn_id="scored",
            session_id="s1",
            source="claude-code",
            host="h1",
            started_at=base_time,
            tier="fast",
            outcome="ok",
            prompt_tokens=60,
            completion_tokens=60,
            wall_ms=850,
        )
        store.upsert_turn(turn)
        store.add_events(
            [
                Event(turn_id="scored", ts=base_time, kind="tool_call", ok=1),
                Event(
                    turn_id="scored", ts=base_time, kind="todo", payload={"opened": 2, "closed": 2}
                ),
            ]
        )

        old_baseline_tokens, old_p95_wall = _old_baseline(store, turn)
        new_baseline_tokens, new_p95_wall = trailing_baseline(
            store,
            source=turn.source,
            tier=turn.tier,
            before_ms=turn.started_at,
            exclude_turn_id=turn.turn_id,
        )
        assert new_baseline_tokens == pytest.approx(old_baseline_tokens)
        assert new_p95_wall == pytest.approx(old_p95_wall)

        old_score, old_components = score_turn(
            turn, store.events_for(turn.turn_id), baseline_tokens=old_baseline_tokens
        )

        new_score, new_components, _ = score_and_persist(store, turn)

        assert new_score == pytest.approx(old_score)
        assert new_components.as_dict() == pytest.approx(old_components.as_dict())
    finally:
        store.close()
