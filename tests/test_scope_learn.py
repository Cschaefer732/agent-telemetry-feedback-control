"""The closed loop: what it promotes, and -- more importantly -- what it refuses to.

Measured on the real corpus at the time of writing: 65 sessions, 972 turns, ONE usable
false-none event. So these tests run on fixtures where the signal exists by construction.
That is not a substitute for the real evidence and is not treated as one: the loop's
correct behaviour on the real corpus is to ship nothing, and that case is asserted too.
"""

from __future__ import annotations

import json
from pathlib import Path

from flightdeck.scope_learn import (
    MAX_NONE_SHARE_INCREASE,
    Event,
    Turn,
    apply,
    mine,
    none_share,
    propose,
    replay,
    score,
    session_turns,
    suppression_check,
)
from flightdeck.scope_registry import (
    FREEZE_ENV,
    Entry,
    apply_overlay,
    load,
    save,
)
from flightdeck.store import Store


def _session(root: Path, name: str, prompts: list[str]) -> None:
    target = root / "proj" / f"{name}.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps({
            "type": "user",
            "promptSource": "typed",
            "message": {"role": "user", "content": text},
        })
        for text in prompts
    ]
    target.write_text("\n".join(lines) + "\n")


# ------------------------------------------------------------------ overlay


def test_overlay_is_additive_only(tmp_path: Path) -> None:
    """An overlay that could REMOVE a hand-written phrase would be a learned change that
    deleting the file cannot roll back."""
    path = tmp_path / "reg.json"
    save([Entry(phrase="spruce up", registry="VAGUE_MARKERS", helpful=3)], path)
    result = apply_overlay("VAGUE_MARKERS", ("better", "tidy"), path)
    assert set(result) >= {"better", "tidy", "spruce up"}


def test_missing_overlay_leaves_the_defaults_in_force(tmp_path: Path) -> None:
    base = ("better", "tidy")
    assert apply_overlay("VAGUE_MARKERS", base, tmp_path / "absent.json") == base


def test_corrupt_overlay_leaves_the_defaults_in_force(tmp_path: Path) -> None:
    """A broken registry must not out-decide the tuned baseline -- same fail-open posture
    the hook has."""
    path = tmp_path / "reg.json"
    path.write_text("{ not json")
    assert apply_overlay("VAGUE_MARKERS", ("better",), path) == ("better",)
    assert load(path) == []


def test_retired_entries_do_not_reach_the_gate(tmp_path: Path) -> None:
    path = tmp_path / "reg.json"
    save([Entry(phrase="noisy", registry="VAGUE_MARKERS", status="retired")], path)
    assert "noisy" not in apply_overlay("VAGUE_MARKERS", ("better",), path)


def test_freeze_env_refuses_writes(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(FREEZE_ENV, "1")
    try:
        save([Entry(phrase="x", registry="VAGUE_MARKERS")], tmp_path / "reg.json")
    except RuntimeError as exc:
        assert FREEZE_ENV in str(exc)
    else:
        raise AssertionError("frozen registry accepted a write")


def test_save_backs_up_the_previous_version(tmp_path: Path) -> None:
    path = tmp_path / "reg.json"
    save([Entry(phrase="one", registry="VAGUE_MARKERS")], path)
    save([Entry(phrase="two", registry="VAGUE_MARKERS")], path)
    assert list(tmp_path.glob("reg-*.json")), "no rollback copy written"


# ------------------------------------------------------------------ replay + mining


def test_scope_opens_once_and_stays_open(tmp_path: Path) -> None:
    """Most turns in a session are refinements; re-scoping each is the ceremony tax paid
    over and over."""
    turns = [Turn("s", 0, "improve everything across the whole fleet and then tidy it"),
             Turn("s", 1, "now fix the other one")]
    events = replay(turns)
    assert events[0].tier != "none"
    assert events[1].had_active_scope is True


def test_false_none_needs_all_three_conditions() -> None:
    turn = Turn("s", 0, "fix the typo in store.py", corrected_next=True)
    events = replay([turn])
    assert events[0].tier == "none"
    assert events[0].false_none is True


def test_a_correction_after_ceremony_is_not_a_false_none() -> None:
    """The gate DID fire; a later correction does not indict the decision to scope."""
    turns = [Turn("s", 0, "improve everything across the fleet and then clean it up",
                  corrected_next=True)]
    assert replay(turns)[0].false_none is False


def test_candidates_exclude_phrases_already_known(tmp_path: Path) -> None:
    events = [Event(turn=Turn("s", 0, "please tidy the config", corrected_next=True),
                    tier="none", had_active_scope=False, false_none=True)]
    assert not any("tidy" in c.phrase for c in mine(events))


def test_scoring_counts_harm_outside_the_evidence(tmp_path: Path) -> None:
    """A phrase that fires everywhere looks perfect on its own evidence and ships damage."""
    events = [
        Event(Turn("s", 0, "sort out the config", corrected_next=True), "none", False, True),
        Event(Turn("s", 1, "sort out the imports", corrected_next=False), "none", False),
        Event(Turn("s", 2, "sort out the tests", corrected_next=False), "none", False),
    ]
    from flightdeck.scope_learn import Candidate

    scored = score(Candidate(phrase="sort out", registry="VAGUE_MARKERS"), events)
    assert scored.helpful == 1
    assert scored.harmful == 2
    assert scored.ratio is not None and scored.ratio < 0.5


# ------------------------------------------------------------------ the guard


def test_suppression_check_rejects_a_regime_shift() -> None:
    """Runs before any quality arithmetic: a rate over a window the update emptied is a
    small-sample artifact, not a result."""
    before = [Event(Turn("s", i, "x"), "mini", False) for i in range(10)]
    after = [Event(Turn("s", i, "x"), "none", False) for i in range(10)]
    result = suppression_check(before, after)
    assert result["passed"] is False
    assert result["delta"] > MAX_NONE_SHARE_INCREASE


def test_suppression_check_allows_more_scoping() -> None:
    before = [Event(Turn("s", i, "x"), "none", False) for i in range(10)]
    after = [Event(Turn("s", i, "x"), "mini", False) for i in range(10)]
    assert suppression_check(before, after)["passed"] is True


def test_none_share_of_an_empty_window_is_zero() -> None:
    assert none_share([]) == 0.0


# ------------------------------------------------------------------ end to end


def test_a_repeated_pattern_clears_the_bar(tmp_path: Path) -> None:
    """M3's failable check: a phrase that precedes a correction three times, with
    counter-examples, is promoted -- and writes exactly one TuningChange row."""
    for i in range(3):
        _session(tmp_path, f"hit{i}", ["can you sort out the login flow", "no that's wrong"])
    _session(tmp_path, "miss0", ["can you sort out the login flow", "thanks"])

    result = apply(tmp_path, store=None, dry_run=True)
    assert result["false_none_events"] == 3, result
    # 3 helpful : 1 harmful = 0.75, just over the 0.70 bar. At 3:2 (0.60) the same phrase
    # is correctly refused -- the bar is what stops a phrase that fires everywhere.
    assert any("sort" in c["phrase"] for c in result["clearing_threshold"]), result
    assert result["applied"] is False          # dry run writes nothing


def test_apply_writes_one_tuning_change(tmp_path: Path, monkeypatch) -> None:
    corpus = tmp_path / "corpus"
    for i in range(3):
        _session(corpus, f"hit{i}", ["can you sort out the login flow", "no that's wrong"])
    monkeypatch.setattr("flightdeck.scope_registry.OVERLAY_PATH", tmp_path / "reg.json")
    import flightdeck.scope_learn as learn

    monkeypatch.setattr(learn, "save", lambda entries: None)
    store = Store(tmp_path / "store")
    result = apply(corpus, store=store, dry_run=False)
    assert result["applied"] is True
    rows = store.conn.execute(
        "SELECT * FROM tuning_changes WHERE domain='scope_gate'"
    ).fetchall()
    assert len(rows) == 1
    assert json.loads(rows[0]["evidence"])["promoted"]


def test_nothing_is_promoted_without_evidence(tmp_path: Path) -> None:
    """Shipping nothing is the correct outcome most weeks at this arrival rate."""
    _session(tmp_path, "quiet", ["fix the typo in store.py", "thanks"])
    result = propose(tmp_path)
    assert result["verdict"] == "silent"
    assert result["clearing_threshold"] == []


def test_session_turns_skips_non_human_records(tmp_path: Path) -> None:
    target = tmp_path / "proj" / "s.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join([
        json.dumps({"type": "user", "promptSource": "typed",
                    "message": {"role": "user", "content": "real prompt"}}),
        json.dumps({"type": "user", "isMeta": True,
                    "message": {"role": "user", "content": "injected skill body"}}),
    ]) + "\n")
    sessions = list(session_turns(tmp_path))
    assert [t.text for t in sessions[0]] == ["real prompt"]
