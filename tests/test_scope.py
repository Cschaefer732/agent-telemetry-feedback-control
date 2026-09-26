from __future__ import annotations

import json

import pytest

from flightdeck.models import (
    CORRECTION_FAMILIES,
    DISPOSITIONS,
    HEALTHY_SCOPE_VERDICTS,
    SCOPE_TIERS,
    SCOPE_VERDICTS,
    ScopeRecord,
)
from flightdeck.scope_baseline import (
    CORRECTION_PATTERNS,
    analyze_session,
    classify_correction,
    edit_episodes,
    human_prompt_text,
    is_human_prompt,
)
from flightdeck.store import Store

# --------------------------------------------------------------------------- vocabulary


def test_every_healthy_verdict_is_a_verdict_the_writer_emits():
    """The dreamer guard. It promoted on outcome == 'succeeded' while the writer emitted
    ok|error|cancelled|unknown|truncated: 183 rows in, 0 out, timer green for weeks. Any
    filter constant must be a subset of the registry it filters."""
    assert set(HEALTHY_SCOPE_VERDICTS) <= set(SCOPE_VERDICTS)


def test_correction_pattern_keys_match_the_registry():
    assert set(CORRECTION_PATTERNS) == set(CORRECTION_FAMILIES)


def test_verdicts_are_three_valued():
    """A missing writer must be distinguishable from a passing one."""
    assert "silent" in SCOPE_VERDICTS
    assert {"pass", "fail"} <= set(SCOPE_VERDICTS)


def test_registries_are_non_empty():
    for registry in (SCOPE_TIERS, SCOPE_VERDICTS, DISPOSITIONS, CORRECTION_FAMILIES):
        assert registry, "an empty registry silently disables every check that reads it"


# ------------------------------------------------------------------- structural filtering


def _user(text, **extra):
    return {"type": "user", "message": {"role": "user", "content": text}, **extra}


SKILL_BODY = (
    "Base directory for this skill: /Users/x/.claude/skills/learning # Learning\n"
    "The MOMENT the user corrects you, says remember/always/never/'I told you already'"
)


def test_injected_skill_body_is_not_a_human_prompt():
    """The bug this module exists for: learning/SKILL.md contains the literal string
    'I told you already', so an unfiltered regex scored a correction every time a skill
    loaded. Precision was 24%. The fix is structural -- isMeta -- not more patterns."""
    record = _user(SKILL_BODY, isMeta=True, promptSource="system")
    assert is_human_prompt(record) is False
    assert human_prompt_text(record) is None
    # and the text really would have matched, which is why the filter has to catch it
    assert classify_correction(SKILL_BODY)


@pytest.mark.parametrize(
    "extra",
    [
        {"isMeta": True},
        {"isCompactSummary": True},
        {"isSidechain": True},
        {"promptSource": "system"},
    ],
)
def test_non_human_records_are_excluded(extra):
    assert human_prompt_text(_user("you forgot the tests", **extra)) is None


@pytest.mark.parametrize("source", ["typed", "queued", "suggestion_accepted"])
def test_human_sources_are_kept(source):
    assert human_prompt_text(_user("you forgot the tests", promptSource=source))


def test_tool_results_are_not_human_turns():
    record = _user([{"type": "tool_result", "content": "ok"}], promptSource="typed")
    assert human_prompt_text(record) is None


def test_ansi_is_stripped_before_any_text_test():
    record = _user("\x1b[1mno, that's wrong\x1b[22m", promptSource="typed")
    assert human_prompt_text(record) == "no, that's wrong"


def test_literal_noise_is_dropped():
    assert human_prompt_text(_user("[Request interrupted by user]", promptSource="typed")) is None


# ------------------------------------------------------------------------ classification


@pytest.mark.parametrize(
    "text,family",
    [
        ("no, that is not good", "negate"),
        ("you forgot to add the tests", "missed"),
        ("you never moved the todo calendar", "missed"),
        ("i already told you not to touch the browser", "repeat"),
        ("actually i want the other layout", "redirect"),
    ],
)
def test_families_match(text, family):
    assert family in classify_correction(text)


def test_symptom_phrased_corrections_are_a_known_miss():
    """Documented floor, not a bug to paper over with more regex. Hand-audit found 5-6 of
    these per 25 unflagged turns; a kappa-validated judge is the only real fix."""
    assert classify_correction("the page appears blank, no elements are loading") == []


# ------------------------------------------------------------------------------- rework


def test_returning_to_a_file_is_a_second_episode():
    episodes = edit_episodes([("1", "a.py"), ("2", "a.py"), ("3", "b.py"), ("4", "a.py")])
    assert episodes == {"a.py": 2, "b.py": 1}


def test_consecutive_edits_are_one_episode():
    assert edit_episodes([("1", "a.py"), ("2", "a.py")]) == {"a.py": 1}


def test_episodes_order_by_timestamp_not_arrival():
    """Subagent edits are merged in from other files; unsorted they fake extra episodes."""
    assert edit_episodes([("3", "a.py"), ("1", "a.py"), ("2", "a.py")]) == {"a.py": 1}


def test_analyze_session_counts_subagent_edits(tmp_path):
    path = tmp_path / "s1.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {"type": "user", "timestamp": "1", "promptSource": "typed",
                 "message": {"role": "user", "content": "build it"}},
                {"type": "assistant", "timestamp": "2", "message": {
                    "role": "assistant", "content": [
                        {"type": "tool_use", "name": "Edit", "input": {"file_path": "a.py"}}]}},
                {"type": "user", "timestamp": "3", "promptSource": "typed",
                 "message": {"role": "user", "content": "you forgot the tests"}},
            ]
        )
    )
    row = analyze_session(path, subagent_edits=[("4", "b.py"), ("5", "a.py")])
    assert row["human_turns"] == 2
    assert row["correction_turns"] == 1
    assert row["files_edited"] == 1          # main thread touched a.py only
    assert row["files_edited_incl_sub"] == 2
    assert row["rework_files_incl_sub"] == 1  # a.py revisited after b.py


# -------------------------------------------------------------------------------- store


def _record(**kw):
    base = dict(
        record_id="r1", session_id="s1", created_at=1000, host="mac",
        tier="full", verdict="pass",
    )
    base.update(kw)
    return ScopeRecord(**base)


def test_scope_record_round_trip(tmp_path):
    with Store(tmp_path) as store:
        store.add_scope_record(_record(found_total=7, committed=4, non_goals=2, assumptions=1,
                                       correction_families={"missed": 2},
                                       provenance={"analyzer": "1"}))
        rows = store.scope_records()
    assert len(rows) == 1
    assert rows[0]["found_total"] == 7
    assert json.loads(rows[0]["correction_families"]) == {"missed": 2}
    assert json.loads(rows[0]["provenance"]) == {"analyzer": "1"}


def test_row_hash_chains_to_previous(tmp_path):
    with Store(tmp_path) as store:
        store.add_scope_record(_record(record_id="r1", created_at=1000))
        store.add_scope_record(_record(record_id="r2", created_at=2000))
        rows = store.scope_records()
    assert rows[0]["prev_hash"] is None
    assert rows[1]["prev_hash"] == rows[0]["row_hash"]
    assert rows[0]["row_hash"] != rows[1]["row_hash"]


def test_dispositions_are_exhaustive_for_found_items(tmp_path):
    """Every discovered item lands in exactly one disposition; the sum is the invariant
    that keeps discovery unbounded while commitment stays budgeted."""
    record = _record(found_total=6, committed=3, non_goals=2, assumptions=1)
    assert record.committed + record.non_goals + record.assumptions == record.found_total


def test_migration_created_the_table(tmp_path):
    with Store(tmp_path) as store:
        assert store.version >= 12
        cols = {r["name"] for r in store.conn.execute("PRAGMA table_info(scope_records)")}
    assert {"late_discovered", "ceremony_tokens", "verdict", "row_hash"} <= cols
