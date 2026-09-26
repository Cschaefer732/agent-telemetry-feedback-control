"""Prompt text + expected gate/judge label pairs -- the artifact kind sample_data.py never
produced (it only ever emitted ScopeRecord counters, never natural-language turn text).

Two corpora, never merged (report 08 "Balance resolution"):

- `seed_cases()` -- the BALANCED rule-coverage corpus. 63 seeds drafted in
  08-synth-coverage.md, transcribed here verbatim as real content (not invented
  replacements). Deliberately balanced across gate rules/boundaries, not proportional to
  real-world frequency -- do not use this corpus to report a rate.
- `generate_frequency_realistic()` -- the FREQUENCY-REALISTIC corpus. Preserves the
  measured marginals from report 07 (n=972 real prompts) and report 08's rework/correction
  rates (0.236 / 0.197). Use this corpus for KPI/aggregator dashboards, never for rule
  regression.

# generator

No import of `flightdeck.scope_gate` or `flightdeck.scope_judge` -- labels here come from
the seed's own drafted intent (case-authoring time) or from template intent, never from
running the classifier on the generated text. See provenance.LABEL_SOURCES.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Literal

from flightdeck.models import CORRECTION_FAMILIES
from flightdeck.synth.profiles import (
    REAL_CORRECTION_CORPUS_RATE,
    REAL_PROMPT_ALL_LOWERCASE_RATE,
    REAL_PROMPT_CHAINED_AND_RATE,
    REAL_PROMPT_NO_TERMINAL_PUNCT_RATE,
    REAL_PROMPT_QUESTION_MARK_RATE,
    REAL_PROMPT_VERY_SHORT_RATE,
    REAL_REWORK_CORPUS_RATE,
)
from flightdeck.synth.provenance import stamp_row


@dataclass(frozen=True)
class PromptLabel:
    """One evaluation case for the scope gate or the judge -- NOT a ScopeRecord."""

    case_id: str
    prompt: str
    expected_tier: str | None = None  # one of SCOPE_TIERS, or None if N/A (judge-only case)
    expected_signals: dict[str, bool] = field(default_factory=dict)
    expected_verdict: str | None = None  # judge: pass/fail/silent, or None if N/A
    template_id: str = ""
    difficulty: Literal["easy", "boundary", "adversarial"] = "easy"
    label_source: str = "template_intent"
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "prompt": self.prompt,
            "expected_tier": self.expected_tier,
            "expected_signals": dict(self.expected_signals),
            "expected_verdict": self.expected_verdict,
            "template_id": self.template_id,
            "difficulty": self.difficulty,
            "label_source": self.label_source,
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class PromptLabelParams:
    count: int = 60
    seed: int = 20260905
    rework_rate: float = REAL_REWORK_CORPUS_RATE
    correction_rate: float = REAL_CORRECTION_CORPUS_RATE


# ---------------------------------------------------------------------------------------
# Balanced seed-case corpus (report 08, "60+ seed cases" table). Transcribed verbatim as
# real content; tier is None where the report itself marks the case as judge-only,
# predicted-but-not-executed, or a known boundary/limitation rather than a settled tier.
# (case_id, prompt, expected_tier, rule, difficulty)
_SEED_ROWS: tuple[tuple[str, str, str | None, str, str], ...] = (
    (
        "seed-01",
        "fix the typo in flightdeck/store.py",
        "none",
        ":179-182 bounded_by_target",
        "easy",
    ),
    ("seed-02", "add a login button to src/components/Nav.tsx", "none", ":179-182", "easy"),
    (
        "seed-03",
        "rename the `getUser` function to `fetchUser` in api.ts",
        "none",
        ":179-182",
        "easy",
    ),
    (
        "seed-04",
        "create five files a.txt..e.txt, each containing its own letter",
        "none",
        ":179-182",
        "easy",
    ),
    ("seed-05", "delete the unused import in flightdeck/scope_gate.py", "none", ":179-182", "easy"),
    ("seed-06", "add a save feature", "mini", ":213-214 build_verb w/o target", "easy"),
    ("seed-07", "make the search better", "mini", ":209 vague", "easy"),
    ("seed-08", "clean up the dashboard code", "mini", ":209 vague", "easy"),
    ("seed-09", "setup moonlight with the mac mini", "mini", ":213-214", "adversarial"),
    ("seed-10", "figure out why the build is slow", "mini", ":209 vague", "easy"),
    ("seed-11", "investigate the memory leak", "mini", ":211 discovery_verb", "easy"),
    ("seed-12", "audit the auth module", "mini", ":207-208 heavy_verb, single surface", "boundary"),
    ("seed-13", "polish the onboarding flow", "mini", ":209 vague", "easy"),
    ("seed-14", "make it production ready", "mini", ":209 vague", "easy"),
    ("seed-15", "ensure we are handling overflow of indecators", "mini", ":213-214", "easy"),
    ("seed-16", "refactor the auth module", "mini", ":207-208", "boundary"),
    (
        "seed-17",
        "make the icon smaller, also move the button",
        "mini",
        ":191-203 extent-only near-miss",
        "boundary",
    ),
    ("seed-18", "refactor the whole harness across both machines", "full", ":193-203", "easy"),
    ("seed-19", "research the options then rewrite all the prompts", "full", ":193-203", "easy"),
    (
        "seed-20",
        "refactor the auth module and also rewrite all the tests",
        "full",
        ":193-203",
        "easy",
    ),
    ("seed-21", "audit the whole codebase and harden every endpoint", "full", ":193-203", "easy"),
    (
        "seed-22",
        "modernize the frontend, migrate the backend, and consolidate the configs",
        "full",
        ":193-203",
        "easy",
    ),
    (
        "seed-23",
        "go through everything and clean up whatever needs it across the repo",
        "full",
        ":193-203",
        "easy",
    ),
    (
        "seed-24",
        "review each of the modules and restructure them end to end",
        "full",
        ":193-203",
        "easy",
    ),
    ("seed-25", "/compact", "none", ":155-156 slash_command", "easy"),
    ("seed-26", "/mode/", "none", ":155-156", "easy"),
    ("seed-27", "/clear", "none", ":155-156", "easy"),
    ("seed-28", "do it", "none", ":157-158 approval", "easy"),
    ("seed-29", "contine", "none", ":157-158", "easy"),
    ("seed-30", "yep", "none", ":157-158", "easy"),
    ("seed-31", "fix it", "none", ":157-158", "easy"),
    ("seed-32", "is it done", "none", ":159-160 question", "easy"),
    (
        "seed-33",
        "why did you add a tab system at the top of the terminal",
        "none",
        ":159-160",
        "easy",
    ),
    ("seed-34", "did you apply the new wordmark to the dev tui", "none", ":159-160", "easy"),
    ("seed-35", "have you pushed the last commit yet", "none", ":159-160", "easy"),
    ("seed-36", "am i missing something in the config", "none", ":159-160", "easy"),
    (
        "seed-37",
        "<bash-stderr>sudo: a terminal is required</bash-stderr>",
        "none",
        ":153-154 pasted_output",
        "easy",
    ),
    (
        "seed-38",
        'curl -H "Authorization: Bearer $API_KEY" ... -X POST https://api.sambanova.ai/v1/chat/completions',
        "none",
        ":153-154",
        "easy",
    ),
    (
        "seed-39",
        'Traceback (most recent call last): File "x.py", line 3',
        "none",
        ":153-154",
        "easy",
    ),
    (
        "seed-40",
        "can you research these github repos and compare them",
        None,
        ":130,159-160 polite_request overrides question",
        "boundary",
    ),
    (
        "seed-41",
        "can you tell me what files reference the old auth module",
        None,
        ":159-160 vs :213-214",
        "boundary",
    ),
    (
        "seed-42",
        "is it worth building a caching layer here",
        "none",
        ":159-160 question, no polite_request (known miss)",
        "adversarial",
    ),
    (
        "seed-43",
        "should we just rewrite the parser instead",
        "none",
        ":159-160 (known miss, micro-gap)",
        "adversarial",
    ),
    (
        "seed-44",
        "keep it, can you research skills and a mechanism to work with ledger so we can get "
        "feedback on catagorical behaviors, then we should have each .md file be seperate so "
        "we can track the changes we make via the files. for example, i would want a scope.md "
        "and scoring to track if the agent correctly added enough features",
        None,
        ":148-150 bounded_by_target correctly False (chained)",
        "adversarial",
    ),
    (
        "seed-45",
        "fix the typo in flightdeck/store.py, then also refactor the whole auth flow",
        "full",
        ":193-203, bounded_by_target False due to chain",
        "boundary",
    ),
    (
        "seed-46",
        "build the login page in src/pages/login.tsx (about 45 words of extra unrelated "
        "commentary padding this sentence past the forty word cutoff purely to test the "
        "length guard on bounded_by_target without adding a chain marker or heavy verb "
        "anywhere in the text)",
        "mini",
        ":149 word-count cliff",
        "adversarial",
    ),
    (
        "seed-47",
        "make the icon 2px smaller",
        "none",
        ":174-175 refine (has_active_scope=True)",
        "boundary",
    ),
    (
        "seed-48",
        "also update the icon color",
        "none",
        ":174-175 refine, contains BUILD_VERB (has_active_scope=True)",
        "adversarial",
    ),
    (
        "seed-49",
        "also rewrite the config loader",
        "mini",
        ":166-174 widen: heavy_verb (has_active_scope=True)",
        "boundary",
    ),
    (
        "seed-50",
        "can you also fix the button on the settings page",
        "mini",
        ":171-172 widen: chained (has_active_scope=True)",
        "boundary",
    ),
    (
        "seed-51",
        "this needs to work across all the environments too",
        "mini",
        ":169-170 widen: multi_surface (has_active_scope=True)",
        "boundary",
    ),
    (
        "seed-52",
        "refactor everything across both machines and also rewrite all the prompts",
        None,
        ":165-175 vs :193-203 (mini w/ active scope, full without)",
        "adversarial",
    ),
    (
        "seed-53",
        "add a save feature, but don't bother writing tests for it",
        "mini",
        ":213-214 (negation coverage gap)",
        "adversarial",
    ),
    (
        "seed-54",
        "don't scope this, just fix the button color in Header.tsx",
        "none",
        ":179-182 bounded_by_target",
        "adversarial",
    ),
    ("seed-55", "yes, go ahead", "none", ":157-158 approval", "easy"),
    (
        "seed-56",
        "yes, and also refactor the whole notification system while you're at it",
        None,
        ":193-203 or :213-216 (approval containing new work)",
        "adversarial",
    ),
    (
        "seed-57",
        "$ this needs a caching layer added everywhere",
        "none",
        ":153-154 pasted_output false-positive",
        "adversarial",
    ),
    (
        "seed-58",
        "/dev/null keeps filling up, can we add log rotation",
        "none",
        ":155-156 slash_command false-positive",
        "adversarial",
    ),
    (
        "seed-59",
        "no what abiut the 9 square spinner we made, it might be somewhere else, and the "
        "statusline",
        None,
        "scope_judge.py STEP 1 negate (judge, not gate)",
        "boundary",
    ),
    (
        "seed-60",
        "why cant you mathimatically center it?",
        None,
        "scope_judge.py:59 challenges the assistant (judge, not gate)",
        "boundary",
    ),
    (
        "seed-61",
        "no, because the goal of this sparky-harness is to be able to build stuff with the "
        "agent. How can we not get it to do such a simple task",
        None,
        "scope_judge.py STEP 1 negate (judge, not gate)",
        "boundary",
    ),
    (
        "seed-62",
        "for right now just make a file of all the repos we need to pull into it. add these "
        "to the file too https://github.com/avelino/awesome-go",
        None,
        ":179-182 or :213-216 depending on whether URL counts as concrete_target",
        "adversarial",
    ),
    (
        "seed-63",
        "move the icon 4px to the left in Header.tsx and also bump the font size there",
        None,
        ":148-150 bounded_by_target requires not chained",
        "adversarial",
    ),
)


def seed_cases() -> list[PromptLabel]:
    """The balanced rule-coverage corpus. Deterministic and fixed -- not randomly
    generated. `expected_tier=None` marks a judge-only case, a boundary the report
    predicted but never executed against `classify()`, or a documented gate limitation;
    see 08-synth-coverage.md for the per-case rationale."""
    cases = []
    for case_id, prompt, tier, rule, difficulty in _SEED_ROWS:
        cases.append(
            PromptLabel(
                case_id=case_id,
                prompt=prompt,
                expected_tier=tier,
                expected_signals={"rule": rule} if rule else {},
                template_id="seed-08-synth-coverage",
                difficulty=difficulty,
                label_source="template_intent",
                provenance=stamp_row(
                    {
                        "generator": "flightdeck.synth.prompt_labels.seed_cases",
                        "label_source": "template_intent",
                        "source_report": "08-synth-coverage.md",
                    }
                ),
            )
        )
    return cases


def seed_corpus() -> list[dict[str, Any]]:
    return to_corpus(seed_cases())


def to_corpus(labels: list[PromptLabel]) -> list[dict[str, Any]]:
    return [label.to_row() for label in labels]


# ---------------------------------------------------------------------------------------
# Frequency-realistic corpus: templates whose marginals target report 07's measured
# distribution (median 10 words, 81.2% no terminal punctuation, 55.5% all-lowercase,
# 24.1% <=3 words, 2.7% "?", 32.3% chained "and", 0% code blocks). These templates are
# style-only -- they carry no gate/judge label at all (label_source is deliberately
# omitted from expected_tier/verdict), because this corpus's job is rate realism for KPI
# dashboards, not rule coverage. Sourced from the report's own exemplar phrasing style,
# genericized (no verbatim third-party or sensitive content).
_VERY_SHORT_TEMPLATES = ("donne", "continuem", "2px more", "ok", "yep", "try again", "fix it")
_SHORT_TEMPLATES = (
    "add a save feature",
    "clean up the dashboard code",
    "fix the typo in the config",
    "make the icon smaller",
    "audit the auth module",
    "investigate the memory leak",
)
_CHAINED_TEMPLATES = (
    "create three separate plans for each feature and do them in parallel",
    "review this session and see why the agent keeps doing that and also fix it",
    "refactor the auth module and also rewrite the tests",
    "go through the repo and clean up whatever needs it and update the docs",
)
_QUESTION_TEMPLATES = (
    "is it worth building a caching layer here",
    "should we just rewrite the parser instead",
    "did you apply the new wordmark to the dev tui?",
)


def _apply_lowercase(rng: random.Random, text: str, rate: float) -> str:
    return text.lower() if rng.random() < rate else text


def _apply_terminal_punct(rng: random.Random, text: str, no_punct_rate: float) -> str:
    if rng.random() < no_punct_rate:
        return text.rstrip(".!?")
    return text if text.endswith((".", "!", "?")) else text + "."


def generate_labels(params: PromptLabelParams) -> list[PromptLabel]:
    """Frequency-realistic prompt generator. Style marginals only -- no gate/judge tier is
    asserted, since these prompts are drawn to match REAL rates, not to exercise a
    specific rule (that corpus is `seed_cases()`). Pure -- deterministic in params.seed."""
    rng = random.Random(params.seed)
    labels: list[PromptLabel] = []
    for index in range(params.count):
        roll = rng.random()
        if roll < REAL_PROMPT_VERY_SHORT_RATE:
            base = rng.choice(_VERY_SHORT_TEMPLATES)
            bucket = "very_short"
        elif roll < REAL_PROMPT_VERY_SHORT_RATE + REAL_PROMPT_QUESTION_MARK_RATE:
            base = rng.choice(_QUESTION_TEMPLATES)
            bucket = "question"
        elif (
            roll
            < REAL_PROMPT_VERY_SHORT_RATE
            + REAL_PROMPT_QUESTION_MARK_RATE
            + REAL_PROMPT_CHAINED_AND_RATE
        ):
            base = rng.choice(_CHAINED_TEMPLATES)
            bucket = "chained"
        else:
            base = rng.choice(_SHORT_TEMPLATES)
            bucket = "short"

        text = _apply_lowercase(rng, base, REAL_PROMPT_ALL_LOWERCASE_RATE)
        if bucket != "question":
            text = _apply_terminal_punct(rng, text, REAL_PROMPT_NO_TERMINAL_PUNCT_RATE)

        is_correction = rng.random() < params.correction_rate
        labels.append(
            PromptLabel(
                case_id=f"freq-{index:04d}",
                prompt=text,
                expected_tier=None,
                expected_signals={"bucket": bucket},
                expected_verdict="pass" if not is_correction else None,
                template_id=f"frequency-realistic/{bucket}",
                difficulty="easy",
                label_source="generator_asserted",
                provenance=stamp_row(
                    {
                        "generator": "flightdeck.synth.prompt_labels.generate_labels",
                        "label_source": "generator_asserted",
                        "correction_family": rng.choice(CORRECTION_FAMILIES)
                        if is_correction
                        else None,
                    }
                ),
            )
        )
    return labels
