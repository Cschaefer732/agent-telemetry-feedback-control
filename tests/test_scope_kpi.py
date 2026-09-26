from __future__ import annotations

import json
from pathlib import Path

import pytest

from flightdeck.scope_kpi import (
    ZERO_STREAK_LIMIT,
    aggregate,
    health_exit_code,
    health_verdict,
    zero_streak_verdict,
)
from flightdeck.store import DEFAULT_DIR, Store
from flightdeck.synth.provenance import refuse_if_live_store
from flightdeck.synth.scope_rows import ScopeRowParams, generate_rows


def _generate(store, count=60, *, seed=20260905, host="synth"):
    """Test-local helper composing the pure generator with the store, replacing
    sample_data.generate(store, count=...). Not production code -- no shim needed there."""
    refuse_if_live_store(store.directory, DEFAULT_DIR, what="synthetic scope_records")
    rows = generate_rows(ScopeRowParams(count=count, seed=seed, host=host))
    for row in rows:
        store.add_scope_record(row)
    return count


FIXTURE = Path(__file__).parent / "fixtures" / "correction_labels.json"


# ------------------------------------------------------------------ zero-streak detection


def test_no_history_is_silent_not_pass():
    assert zero_streak_verdict([]) == "silent"


def test_sustained_zero_is_a_failure():
    """The dreamer logged 'playbooks': 0 nightly for weeks and exited 0 every time."""
    assert zero_streak_verdict([0] * ZERO_STREAK_LIMIT) == "fail"


def test_short_zero_run_is_not_yet_a_failure():
    assert zero_streak_verdict([0] * (ZERO_STREAK_LIMIT - 1)) == "pass"


def test_recent_activity_clears_the_streak():
    assert zero_streak_verdict([0, 0, 0, 0, 5]) == "pass"


def test_trailing_zeroes_after_activity_still_fail():
    assert zero_streak_verdict([9, 9] + [0] * ZERO_STREAK_LIMIT) == "fail"


# ------------------------------------------------------------------------- aggregation


def test_empty_window_is_silent_not_zero():
    """No rows means the writer may be dead. Reporting 0.0 would read as 'no late
    discovery', i.e. perfect, which is the fail-open shape."""
    summary = aggregate([])
    assert summary["verdict"] == "silent"
    assert summary["rows"] == 0


def _row(**kw):
    base = dict(
        verdict="pass",
        tier="full",
        found_total=10,
        late_discovered=2,
        files_edited=10,
        rework_files=3,
        questions_asked=4,
        questions_valuable=1,
        ceremony_tokens=1000,
        ceremony_ms=30_000,
        turns_to_done=12,
        corrections=2,
        assumptions=4,
        assumptions_overridden=1,
        divergence_flagged=8,
        divergence_kept=2,
    )
    base.update(kw)
    return base


def test_kpi_pairs_are_computed():
    summary = aggregate([_row(), _row()])
    assert summary["late_discovery_rate"] == 0.2
    assert summary["rework_rate"] == 0.3
    assert summary["clarification_value_rate"] == 0.25
    assert summary["divergence_precision"] == 0.25
    assert summary["assumption_override_rate"] == 0.25
    assert summary["verdict"] == "pass"


def test_quality_figure_never_returned_without_its_cost_figure():
    """late_discovery_rate is gameable by scoping forever; the ceremony figures bound it."""
    summary = aggregate([_row()])
    assert (summary["late_discovery_rate"] is None) == (summary["ceremony_tokens_mean"] is None)
    assert summary["unpaired_metrics"] == []


def test_missing_cost_figure_is_named_not_hidden():
    """Caught by running it end to end: late_discovery_rate printed 0.1667 beside a null
    ceremony figure, i.e. exactly the unbounded coverage number the pairing exists to stop."""
    summary = aggregate([_row(ceremony_tokens=None, ceremony_ms=None)])
    assert summary["late_discovery_rate"] is not None
    assert "late_discovery_rate has no ceremony cost recorded" in summary["unpaired_metrics"]


def test_ceremony_ms_alone_satisfies_the_pairing():
    summary = aggregate([_row(ceremony_tokens=None, ceremony_ms=45_000)])
    assert summary["unpaired_metrics"] == []
    assert summary["ceremony_ms_mean"] == 45_000


def test_rows_with_no_healthy_verdict_fail():
    assert aggregate([_row(verdict="fail"), _row(verdict="silent")])["verdict"] == "fail"


# ------------------------------------------------------------------ health_verdict / exit code


def test_health_verdict_passes_through_pass_and_fail():
    assert health_verdict(aggregate([_row()])) == "pass"
    assert health_verdict(aggregate([_row(verdict="fail")])) == "fail"


def test_health_verdict_silent_without_history_stays_silent():
    """Before any history is tracked (or before the caller bothers to pass it), a silent
    window must not fail -- this is the legitimately-quiet-week case."""
    summary = aggregate([])
    assert health_verdict(summary) == "silent"
    assert health_exit_code(health_verdict(summary)) == 0


def test_health_verdict_silent_short_streak_stays_silent():
    summary = aggregate([])
    verdict = health_verdict(summary, rows_history=[0] * (ZERO_STREAK_LIMIT - 2))
    assert verdict == "silent"
    assert health_exit_code(verdict) == 0


def test_health_verdict_permanently_silent_becomes_a_failure():
    """A stream silent for ZERO_STREAK_LIMIT consecutive windows, this one included, is the
    dreamer's zero-playbooks shape and must exit nonzero."""
    summary = aggregate([])
    verdict = health_verdict(summary, rows_history=[0] * (ZERO_STREAK_LIMIT - 1))
    assert verdict == "fail"
    assert health_exit_code(verdict) == 1


def test_health_verdict_recent_activity_in_history_clears_the_streak():
    summary = aggregate([])
    verdict = health_verdict(summary, rows_history=[*([0] * (ZERO_STREAK_LIMIT - 1)), 5])
    assert verdict == "silent"


def test_division_by_zero_yields_none_not_zero():
    summary = aggregate([_row(questions_asked=0, questions_valuable=0, divergence_flagged=0)])
    assert summary["clarification_value_rate"] is None
    assert summary["divergence_precision"] is None


# ------------------------------------------------------------------------- sample data


def test_generator_refuses_the_live_store(tmp_path, monkeypatch):
    """Synthetic rows in the live DB would be read back as measurement forever."""
    with Store(tmp_path) as store:
        monkeypatch.setattr(store, "directory", Path(DEFAULT_DIR).expanduser())
        with pytest.raises(ValueError, match="refusing to write synthetic"):
            _generate(store, count=1)


def test_generated_rows_are_marked_synthetic(tmp_path):
    with Store(tmp_path) as store:
        _generate(store, count=20)
        rows = store.scope_records()
    assert len(rows) == 20
    assert all(json.loads(r["provenance"])["synthetic"] is True for r in rows)


def test_generated_rows_satisfy_the_disposition_invariant(tmp_path):
    with Store(tmp_path) as store:
        _generate(store, count=40)
        rows = store.scope_records()
    for row in rows:
        assert row["committed"] + row["non_goals"] + row["assumptions"] == row["found_total"]


def test_generation_is_deterministic(tmp_path):
    with Store(tmp_path / "a") as a, Store(tmp_path / "b") as b:
        _generate(a, count=10, seed=7)
        _generate(b, count=10, seed=7)
        left = [r["row_hash"] for r in a.scope_records()]
        right = [r["row_hash"] for r in b.scope_records()]
        assert left == right


def test_sample_data_aggregates(tmp_path):
    with Store(tmp_path) as store:
        _generate(store, count=60)
        summary = aggregate(store.scope_records())
    assert summary["rows"] == 60
    assert 0.0 <= summary["late_discovery_rate"] <= 1.0
    assert sum(summary["tiers"].values()) == 60


# ---------------------------------------------------------------------- labeled fixture


def test_labeled_fixture_is_intact():
    """Fixture note: this is a SYNTHETIC gold set (invented example utterances, not sampled
    from any real session) built to exhibit the same statistical shape a real hand-labelled
    corpus has -- see the module docstring on flightdeck.scope_eval. The numbers below are
    only reproducible against this exact set."""
    data = json.loads(FIXTURE.read_text())
    assert len(data) == 139
    assert sum(d["label"] for d in data) == 53
    assert sum(d["regex"] for d in data) == 46


def test_regex_baseline_has_not_silently_regressed():
    from sklearn.metrics import cohen_kappa_score

    data = json.loads(FIXTURE.read_text())
    y = [d["label"] for d in data]
    r = [d["regex"] for d in data]
    # unweighted on the stratified sample; population-weighted kappa is much lower once
    # reweighted by flightdeck.scope_eval -- see check_population_drift and POPULATION.
    assert cohen_kappa_score(y, r) > 0.65
