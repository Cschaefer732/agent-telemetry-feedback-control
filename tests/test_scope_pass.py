from __future__ import annotations

import pytest

from flightdeck.scope_gate import classify
from flightdeck.scope_pass import MAX_AGE_SECONDS, FoundItem, ScopePass
from flightdeck.store import Store


def _pass(**kw) -> ScopePass:
    base = dict(session_id="s1", cwd="/repo", goal="add a save feature")
    base.update(kw)
    decision = classify("add a save feature")
    return ScopePass.open(decision=decision, **base)


def _complete(p: ScopePass) -> ScopePass:
    p.add("save writes to disk", "committed", "file exists after save with the same bytes")
    p.add("cloud sync", "non_goal", "no backend exists yet")
    p.add("format is JSON", "assumption", "chose JSON; say so to override")
    return p


# ------------------------------------------------------------------ the invariant


def test_committed_item_needs_an_acceptance_criterion():
    p = _pass()
    p.add("save writes to disk", "committed")
    assert any("acceptance criterion" in x for x in p.problems())


def test_non_goal_needs_a_reason():
    p = _pass()
    p.add("cloud sync", "non_goal")
    assert any("without a reason" in x for x in p.problems())


def test_assumption_needs_a_chosen_default():
    p = _pass()
    p.add("format is JSON", "assumption")
    assert any("chosen default" in x for x in p.problems())


def test_unknown_disposition_is_refused():
    with pytest.raises(ValueError, match="unknown disposition"):
        FoundItem(text="x", disposition="maybe")


def test_a_pass_that_found_nothing_did_not_run():
    assert any("did not run" in x for x in _pass().problems())


def test_a_pass_that_committed_to_nothing_is_invalid():
    p = _pass()
    p.add("cloud sync", "non_goal", "no backend")
    assert any("nothing committed" in x for x in p.problems())


def test_a_complete_pass_is_valid():
    assert _complete(_pass()).valid


def test_problems_are_reported_all_at_once():
    p = _pass()
    p.add("a", "committed")
    p.add("b", "non_goal")
    assert len(p.problems()) >= 2


def test_every_found_item_lands_in_exactly_one_disposition():
    p = _complete(_pass())
    counts = p.counts()
    assert sum(counts.values()) == len(p.found) == 3


# ------------------------------------------------------------------ late discovery


def test_items_found_before_implementation_are_not_late():
    p = _complete(_pass())
    assert all(not i.late for i in p.found)


def test_ceremony_cost_is_captured_when_implementation_begins():
    """The cost side of the headline pair. Without it late_discovery_rate is unbounded."""
    p = _complete(_pass())
    p.begin_implementation(now=p.created_at + 300)
    assert p.to_record("r1").ceremony_ms == 300_000


def test_a_pass_that_never_started_building_reports_no_ceremony_cost():
    assert _complete(_pass()).to_record("r1").ceremony_ms is None


def test_items_found_after_implementation_began_are_late_automatically():
    """The caller does not get to decide whether its own miss counts."""
    p = _complete(_pass())
    p.begin_implementation()
    p.add("permissions check", "committed", "denied user gets 403")
    assert p.to_record("r1").late_discovered == 1


# ------------------------------------------------------------------ freshness


def test_a_fresh_pass_in_the_same_cwd_is_reusable():
    assert _pass().is_fresh_for("/repo")


def test_a_pass_from_another_directory_is_refused():
    """The bleed: a session started in $HOME inherited an unrelated project's spec."""
    reason = _pass().stale_reason("/other/repo")
    assert reason and "was made in /repo" in reason


def test_an_old_pass_is_refused_even_in_the_same_cwd():
    p = _pass()
    reason = p.stale_reason("/repo", now=p.created_at + MAX_AGE_SECONDS + 1)
    assert reason and "old" in reason


def test_staleness_is_checked_not_assumed():
    p = _pass()
    assert p.stale_reason("/repo", now=p.created_at + 60) is None


# ------------------------------------------------------------------ gate coupling


def test_a_none_tier_request_must_not_open_a_pass():
    """Opening one anyway is precisely the ceremony the gate exists to prevent."""
    with pytest.raises(ValueError, match="do not open a pass"):
        ScopePass.open("s1", "/repo", "fix the typo in store.py",
                       classify("fix the typo in flightdeck/store.py"))


# ------------------------------------------------------------------ persistence


def test_round_trip_through_disk(tmp_path):
    p = _complete(_pass())
    p.ask("which format?", changed_commitment=True)
    p.save(tmp_path)
    loaded = ScopePass.load("s1", tmp_path)
    assert loaded is not None
    assert loaded.counts() == p.counts()
    assert loaded.questions[0].changed_commitment is True
    assert loaded.gate_reasons == p.gate_reasons


def test_loading_an_absent_session_returns_none(tmp_path):
    assert ScopePass.load("nope", tmp_path) is None


def test_state_is_keyed_by_session_not_cwd(tmp_path):
    """Two sessions in the same directory must not share one scope."""
    a = _complete(_pass(session_id="a"))
    b = _complete(_pass(session_id="b", goal="something else"))
    a.save(tmp_path)
    b.save(tmp_path)
    assert ScopePass.load("a", tmp_path).goal != ScopePass.load("b", tmp_path).goal


# ------------------------------------------------------------------ emission


def test_close_writes_a_scope_record(tmp_path):
    p = _complete(_pass())
    p.ask("which format?", changed_commitment=True)
    p.ask("what colour?", changed_commitment=False)
    with Store(tmp_path) as store:
        record = p.close(store, "r1")
        rows = store.scope_records()
    assert record.found_total == 3
    assert record.committed == 1 and record.non_goals == 1 and record.assumptions == 1
    assert record.questions_asked == 2 and record.questions_valuable == 1
    assert len(rows) == 1 and rows[0]["tier"] == "mini"


def test_an_invalid_pass_emits_a_fail_verdict(tmp_path):
    p = _pass()
    p.add("save writes to disk", "committed")   # no acceptance criterion
    with Store(tmp_path) as store:
        record = p.close(store, "r1")
    assert record.verdict == "fail"


def test_record_carries_the_gate_reasoning(tmp_path):
    record = _complete(_pass()).to_record("r1")
    assert record.provenance["gate_reasons"]
