from __future__ import annotations

import pytest

from flightdeck.models import SCOPE_TIERS
from flightdeck.scope_gate import FULL, MINI, NONE, classify


@pytest.mark.parametrize(
    "prompt,expected",
    [
        # The measured ledger-tax case: naming the files IS the specification.
        ("create five files a.txt..e.txt, each containing its own letter", NONE),
        ("fix the typo in flightdeck/store.py", NONE),
        # Short and utterly unspecified.
        ("add a save feature", MINI),
        ("setup moonlight with the mac mini", MINI),
        # Unbounded in kind AND plural in extent.
        ("refactor the whole harness across both machines", FULL),
        ("research the options then rewrite all the prompts", FULL),
        # Not requests for work at all.
        ("/compact", NONE),
        ("did you apply the new wordmark to the dev tui", NONE),
        ("why did you add a tab system at the top of the terminal", NONE),
        ("do it", NONE),
        ("is it done", NONE),
    ],
)
def test_known_classifications(prompt, expected):
    assert classify(prompt).tier == expected


def test_polite_request_is_not_filtered_as_a_question():
    """'can you research X' is an imperative with a please on it."""
    assert classify("can you research these github repos and compare them").tier != NONE


def test_full_needs_unbounded_kind_and_plural_extent():
    kind_only = classify("refactor the auth module")
    extent_only = classify("make the icon smaller, also move the button")
    both = classify("refactor the auth module and also rewrite all the tests")
    assert kind_only.tier == MINI
    assert extent_only.tier == MINI
    assert both.tier == FULL


def test_incidental_path_in_a_long_chained_request_is_not_a_specification():
    """A filename mentioned in passing does not bound a design discussion. This exact turn
    scored NONE under the first rule and is the reason bounded_by_target exists."""
    prompt = (
        "keep it, can you research skills and a mechanism to work with ledger so we can get "
        "feedback on catagorical behaviors, then we should have each .md file be seperate so "
        "we can track the changes we make via the files. for example, i would want a scope.md "
        "and scoring to track if the agent correctly added enough features"
    )
    assert classify(prompt).tier != NONE


def test_pasted_output_beats_every_other_signal():
    assert classify("<bash-stderr>sudo: a terminal is required</bash-stderr>").tier == NONE


# ------------------------------------------------------------------ active scope


def test_refinement_of_an_active_scope_costs_nothing():
    """Most turns in a session refine work already scoped. Re-running the pass on each is
    the ceremony tax paid over and over."""
    assert classify("make the icon 2px smaller", has_active_scope=True).tier == NONE


def test_widening_an_active_scope_reopens_it_as_an_amendment():
    decision = classify("also rewrite the config loader", has_active_scope=True)
    assert decision.tier == MINI
    assert "amends the active scope" in decision.reasons[0]


def test_active_scope_never_triggers_a_full_pass():
    """A scope already exists; the expensive tier is for creating one, not amending it."""
    prompt = "refactor everything across both machines and also rewrite all the prompts"
    assert classify(prompt, has_active_scope=True).tier != FULL
    assert classify(prompt, has_active_scope=False).tier == FULL


# --------------------------------------------------------------------- invariants


@pytest.mark.parametrize(
    "prompt",
    ["", "   ", "add a thing", "/compact", "refactor everything across all the repos"],
)
def test_every_decision_is_auditable(prompt):
    """A tier with no reason cannot be disputed. This gate will be wrong sometimes; it has
    to be wrong legibly."""
    decision = classify(prompt)
    assert decision.tier in SCOPE_TIERS
    assert decision.reasons and all(r.strip() for r in decision.reasons)


def test_empty_prompt_is_not_ceremonial():
    assert classify("").ceremonial is False


def test_ceremonial_matches_tier():
    assert classify("add a save feature").ceremonial is True
    assert classify("/compact").ceremonial is False


# --------------------------------------------------------------------- fix regressions


def test_negated_heavy_verb_does_not_reopen_a_bounded_request():
    """'don't refactor' is not a request to refactor -- the target-naming NONE exemption
    must not be blocked by a verb the user explicitly forbade."""
    decision = classify(
        "add a submit button to the form in Login.tsx, do not refactor anything else there"
    )
    assert decision.tier == NONE


def test_should_we_is_recognized_as_interrogative():
    assert classify("should we just rewrite the parser instead").tier == NONE


def test_question_without_terminal_punctuation_is_detected_mid_sentence():
    """81.2% of this user's real prompts carry no terminal punctuation. A leading filler
    word ('so', 'hey') must not defeat question detection the way a bare '?' check would."""
    decision = classify("so why did you add a tab system at the top of the terminal")
    assert decision.tier == NONE


def test_leading_dollar_sign_does_not_grant_a_free_none():
    decision = classify("$ please build a file uploader, this line is not actual shell output")
    assert decision.tier != NONE


def test_pasted_error_not_at_string_start_is_still_recognized():
    """_PASTED used ^ anchors without re.MULTILINE, so a one-line preamble before the
    traceback defeated the whole rule."""
    decision = classify(
        "here's what I'm seeing:\nTraceback (most recent call last):\n  File \"x.py\", line 3"
    )
    assert decision.tier == NONE


def test_path_like_prefix_is_not_a_slash_command():
    """A real slash command is one token of [a-z-]; a prompt merely starting with the
    character a path also starts with must not be mistaken for one."""
    decision = classify("/dev/null is fine, can we add log rotation")
    assert decision.signals["slash_command"] is False


def test_long_fully_specified_request_keeps_its_none_exemption():
    """Length was never the signal -- ambiguity was. A long request that names its file and
    has no chain/heavy/vague/discovery markers must not lose the NONE exemption for being
    long."""
    padding = " ".join(["some", "extra", "unrelated", "commentary"] * 10)
    decision = classify(f"build the login page in src/pages/login.tsx ({padding})")
    assert decision.tier == NONE


def test_build_verb_as_the_last_token_is_still_detected():
    """`v + " "` matched a verb only if something followed it on the same line, so a verb
    as the FINAL token of the prompt fired no signal at all: 'merge' classified as
    tier=none, reasons=['no ambiguity signal fired']. A bare verb still names no target,
    so it earns MINI, not NONE."""
    assert classify("merge").tier == MINI
    assert classify("commit and ship").tier == MINI


def test_first_word_verb_as_the_entire_prompt_is_detected():
    """Same end-of-string gap in `_first_word_verb`'s `startswith(v + " ")`: a prompt that
    IS the verb, with nothing after it, never matched the leading-word form either."""
    assert classify("ship").tier == MINI


def test_precision_guard_state_questions_stay_none():
    """These must never flip to ceremonial just because the boundary fix now matches verbs
    it used to miss -- they are questions about state, filtered before build_verb is ever
    consulted."""
    assert classify("did you fix the spinner").tier == NONE
    assert classify("why is the word sparky purple").tier == NONE
    assert classify("what command is to change the effort").tier == NONE
