"""Decide how much scoping ceremony a request earns, before any of it is paid for.

This gate exists because ceremony is not free. Measured on this fleet: injecting a
spec/ledger step into a one-shot run dropped file creation from 3-4/5 to 0-2/5 -- the turn
was spent writing `.sparky/ledger/*.md` and then ended. A request that names its own files
has already been specified; scoping it again buys nothing and costs the deliverable.

So the signal is AMBIGUITY, never length. "create five files a.txt..e.txt, each containing
its own letter" is long and needs no scoping. "add a save feature" is short and needs it.

Rules are ORDERED and stop at the first match. Stating exceptions flat beside the rules
they modify inverts the error profile -- measured while training the correction judge:
adding carve-outs as more bullets cut false positives 5 -> 1 and pushed false negatives
3 -> 7 at identical accuracy. Precedence is the missing information.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from flightdeck.models import SCOPE_TIERS
from flightdeck.scope_registry import apply_overlay as _overlay

NONE, MINI, FULL = SCOPE_TIERS

#: Verbs whose scope is inherently unbounded -- they name a direction, not a deliverable.
HEAVY_VERBS = (
    "refactor",
    "rewrite",
    "migrate",
    "redesign",
    "revamp",
    "overhaul",
    "consolidate",
    "harden",
    "audit",
    "integrate",
    "port",
    "modernise",
    "modernize",
    "restructure",
    "rearchitect",
    "productionise",
    "productionize",
    "unify",
)

#: Verbs that ask for work but bound it to a named thing.
BUILD_VERBS = (
    "add",
    "make",
    "create",
    "build",
    "fix",
    "implement",
    "write",
    "change",
    "update",
    "remove",
    "delete",
    "wire",
    "hook",
    "set up",
    "setup",
    "install",
    "apply",
    "move",
    "rename",
    "split",
    "merge",
    "generate",
    "draft",
    "design",
    "verify",
    "check",
    "ensure",
    "test",
    "document",
    "enable",
    "disable",
    "connect",
    "deploy",
    "ship",
)

#: Discovery verbs. Their object is usually open-ended ("find ones that supersede others"),
#: so they signal ambiguity even though they ask for no artifact directly.
DISCOVERY_VERBS = (
    "research",
    "investigate",
    "explore",
    "look for",
    "look into",
    "look at",
    "find",
    "review",
    "survey",
    "compare",
    "evaluate",
    "assess",
    "go through",
)

#: More than one thing is being asked for. A chained request has as many scopes as clauses,
#: and collapsing it into one bounded lane is a mis-classification, not a simplification.
CHAIN_MARKERS = (
    " then ",
    " then,",
    " also ",
    "also,",
    " and then ",
    " after that",
    " next,",
    " plus ",
    " as well as ",
)

#: Phrases that name a quality rather than a deliverable. Each needs a decision the
#: request did not make, which is exactly what a scoping pass is for.
VAGUE_MARKERS = (
    "better",
    "nicer",
    "improve",
    "improved",
    "clean up",
    "cleanup",
    "tidy",
    "figure out",
    "whatever",
    "properly",
    "robust",
    "production ready",
    "production-ready",
    "polish",
    "make it good",
    "make it nice",
    "as needed",
    "etc",
    "and so on",
    "or something",
    "the rest",
    "everything else",
    "more of",
    "look into",
)

#: Signals that the work crosses more than one surface, where a single bounded lane is a
#: mis-classification rather than a simplification.
MULTI_SURFACE = (
    "across",
    "everywhere",
    "all the",
    "both",
    "each of",
    "every ",
    "end to end",
    "end-to-end",
    "throughout",
    "fleet",
    "everything",
)

#: Clauses opening these forbid whatever verb follows -- "don't refactor X" is not a
#: request for X to be refactored, so a heavy/build verb inside a negated clause must not
#: count as evidence of that verb's scope.
NEGATIONS = ("don't ", "do not ", "doesn't need ", "no need to ", "never ", "not ")

APPROVALS = frozenset(
    {
        "do it",
        "go ahead",
        "yes",
        "yep",
        "ok",
        "okay",
        "sure",
        "apply",
        "apply it",
        "continue",
        "contine",
        "proceed",
        "commit",
        "push",
        "both",
        "fix it",
        "so fix it",
        "try again",
        "run it",
        "do that",
        "next",
        "more",
    }
)

_PATH = re.compile(
    r"[\w./~-]+\.(py|ts|tsx|js|jsx|md|json|toml|yaml|yml|sh|go|rs|sql|html|css|txt|jsonl|db)\b"
)
_DIRPATH = re.compile(r"(?:^|\s)(?:~|\.{1,2})?/[\w./-]+")
_CODE_IDENT = re.compile(r"`[^`]+`|\b\w+\(\)|\b[a-z_]+\.[a-z_]+\(")
#: Asks about STATE -- "did you apply the wordmark", "why did you add a tab system". These
#: are questions however many build verbs they contain, and scoping one wastes the turn.
INTERROGATIVE = (
    "what",
    "why",
    "how",
    "where",
    "when",
    "who",
    "which",
    "is ",
    "are ",
    "was ",
    "were ",
    "do ",
    "does",
    "did",
    "have you",
    "have we",
    "has ",
    "should i",
    "should we",
    "am i",
    "any ",
)

#: Asks for ACTION in question clothing. "can you research X" is an imperative with a
#: please on it, and must not be filtered out as a question.
POLITE_REQUEST = (
    "can you",
    "could you",
    "will you",
    "would you",
    "can we",
    "could we",
    "can i get",
    "lets ",
    "let's ",
    "i want you to",
    "i want to",
    "i need you to",
    "i'd like you to",
    "please ",
    "would it be possible",
)
#: ^-anchored alternatives need MULTILINE -- a one-line preamble before a pasted traceback
#: ("here's what I'm seeing:\nTraceback ...") otherwise puts the marker past offset 0 and
#: the whole rule silently misses.
_PASTED = re.compile(
    r"<bash-(stdout|stderr|input)>|Traceback \(most recent|"
    r"^[A-Za-z]+Error:|\bcurl -|sudo |\[sudo\]",
    re.MULTILINE,
)
#: A leading shell-prompt glyph is only pasted output if nothing after it reads as a
#: request -- "$ this is not shell output, please build X" starts with "$ " but names a
#: BUILD_VERB, so it must not get the free NONE that a real `$ npm install` earns.
_SHELL_PROMPT_LINE = re.compile(r"^\s*(\$|%|>>>)\s")
_SLASH_COMMAND = re.compile(r"^/[a-z][a-z-]*/?$")


# Learned phrases extend the hand-written tuples above; they never replace or remove one.
# A missing or unreadable overlay leaves the tuned defaults in force, so deleting the file
# is a complete rollback. See flightdeck/scope_registry.py.
HEAVY_VERBS = _overlay("HEAVY_VERBS", HEAVY_VERBS)
VAGUE_MARKERS = _overlay("VAGUE_MARKERS", VAGUE_MARKERS)
MULTI_SURFACE = _overlay("MULTI_SURFACE", MULTI_SURFACE)
DISCOVERY_VERBS = _overlay("DISCOVERY_VERBS", DISCOVERY_VERBS)


@dataclass
class GateDecision:
    """Why is part of the output. A tier with no reason cannot be audited or disputed, and
    this gate will be wrong sometimes -- it needs to be wrong legibly."""

    tier: str
    reasons: list[str] = field(default_factory=list)
    signals: dict[str, bool] = field(default_factory=dict)

    @property
    def ceremonial(self) -> bool:
        return self.tier != NONE


def _has(text: str, needles: tuple[str, ...]) -> bool:
    return any(n in text for n in needles)


def _has_word(text: str, needles: tuple[str, ...]) -> bool:
    """Like `_has`, but requires a word boundary on both sides of the needle.

    The construction this replaces (`v + " "`) required a verb to be followed by a
    literal space, so a verb as the LAST token of the prompt -- "merge", "commit and
    ship" -- matched nothing: there is no trailing space to find. `\\b` matches a
    boundary against end-of-string as well as against punctuation, so a verb at the
    end of the prompt is no longer invisible to the gate.
    """
    return any(re.search(r"\b" + re.escape(n) + r"\b", text) for n in needles)


def _first_word_verb(text: str, verbs: tuple[str, ...]) -> bool:
    """A build verb anywhere is weak evidence ('the fix is wrong'); leading it is strong."""
    stripped = text.lstrip("- *#>").strip()
    return any(stripped == v or stripped.startswith(v + " ") for v in verbs)


def _has_unnegated(text: str, needles: tuple[str, ...]) -> bool:
    """A verb inside a clause the user just forbade ("don't refactor X") is not a request
    for that verb -- only count an occurrence a preceding negation doesn't govern."""
    for n in needles:
        start = 0
        while True:
            i = text.find(n, start)
            if i == -1:
                break
            start = i + 1
            window = text[max(0, i - 20) : i]
            if not any(neg in window for neg in NEGATIONS):
                return True
    return False


#: Words that precede a real question without being part of it -- "so why did you..." is a
#: question with a filler word bolted on front. Stripping only these (not scanning the
#: whole lead for an interrogative token) is what keeps this a start-of-sentence signal
#: instead of a catch-all: "figure out why the build is slow" is an imperative that merely
#: contains "why" in its object clause, and must NOT read as a question.
_LEAD_FILLERS = ("so ", "well ", "hey ", "ok ", "okay ", "and ", "but ", "also ", "just ")


def _question_in_lead(low: str) -> bool:
    """Most of this user's real prompts carry no terminal punctuation, so a bare
    endswith("?") check misses the majority of actual questions. Only a leading filler
    word is stripped before checking; the interrogative token must still lead the
    sentence, not merely appear somewhere in it."""
    stripped = low
    changed = True
    while changed:
        changed = False
        for filler in _LEAD_FILLERS:
            if stripped.startswith(filler):
                stripped = stripped[len(filler) :]
                changed = True
    return stripped.startswith(INTERROGATIVE)


def classify(prompt: str, *, has_active_scope: bool = False) -> GateDecision:
    """Classify one turn. `has_active_scope` is the difference between a task start and a
    refinement of one already scoped -- most turns in a session are the latter, and scoping
    them again is pure cost."""
    text = (prompt or "").strip()
    low = text.lower()

    signals = {
        # A real "$ npm install" is pasted output; "$ this isn't shell output, please
        # build X" merely starts with the glyph while naming a verb of its own -- only
        # the former earns the free NONE.
        "pasted_output": bool(_PASTED.search(text))
        or (
            bool(_SHELL_PROMPT_LINE.match(text))
            and not _has(low, HEAVY_VERBS + BUILD_VERBS + DISCOVERY_VERBS)
        ),
        # A real slash command is one token of [a-z-]; "/dev/null is fine" merely starts
        # with the character a path also starts with.
        "slash_command": bool(_SLASH_COMMAND.match(low.strip())),
        "approval": low.strip(".!") in APPROVALS,
        "polite_request": low.startswith(POLITE_REQUEST) or _has(low, POLITE_REQUEST[:6]),
        "question": _question_in_lead(low) or low.endswith("?"),
        "concrete_target": bool(
            _PATH.search(text) or _DIRPATH.search(text) or _CODE_IDENT.search(text)
        ),
        "heavy_verb": _has_unnegated(low, HEAVY_VERBS),
        "build_verb": _has_word(low, BUILD_VERBS) or _first_word_verb(low, BUILD_VERBS),
        "discovery_verb": _has(low, DISCOVERY_VERBS),
        "vague": _has(low, VAGUE_MARKERS),
        "multi_surface": _has(low, MULTI_SURFACE),
        "chained": _has(low, CHAIN_MARKERS),
    }
    # A path mentioned in passing inside a long multi-clause request is not a specification.
    # "fix the typo in store.py" specifies itself; "research a mechanism ... then we should
    # have each .md file be separate" merely contains a filename. The word-count cutoff this
    # replaced was a length proxy for exactly that ramble case, but it fired on length alone
    # -- contradicting the module's own thesis. What actually distinguishes a ramble from a
    # long, fully-specified request is chaining (already checked) and open-ended discovery
    # language: a discovery verb makes the ask inherently unbounded regardless of any path
    # mentioned in passing, the same way chaining does.
    signals["bounded_by_target"] = (
        signals["concrete_target"] and not signals["chained"] and not signals["discovery_verb"]
    )

    # 1. Not a request to build anything. Scoping a question wastes the turn it costs.
    if signals["pasted_output"]:
        return GateDecision(NONE, ["pasted output, not a task"], signals)
    if signals["slash_command"]:
        return GateDecision(NONE, ["slash command"], signals)
    if signals["approval"]:
        return GateDecision(NONE, ["approval or continuation of work already agreed"], signals)
    if signals["question"] and not signals["polite_request"]:
        return GateDecision(NONE, ["question about state, not a request for work"], signals)

    # 2. A scope already exists for this task. Follow-up turns refine it; re-running the
    #    pass on every message is the ceremony tax paid over and over. Only a turn that
    #    widens the work reopens scoping, and then as an amendment, not a fresh pass.
    if has_active_scope:
        widening = []
        if signals["heavy_verb"]:
            widening.append("introduces an unbounded verb")
        if signals["multi_surface"]:
            widening.append("reaches a new surface")
        if signals["chained"]:
            widening.append("chains additional work")
        if widening:
            return GateDecision(MINI, ["amends the active scope: " + "; ".join(widening)], signals)
        return GateDecision(NONE, ["refines work already scoped"], signals)

    # 3. The request names its own specification. This is the case the ledger tax punished:
    #    naming concrete files IS the spec, so re-deriving one is pure cost.
    if signals["bounded_by_target"] and not (
        signals["heavy_verb"] or signals["vague"] or signals["multi_surface"]
    ):
        return GateDecision(NONE, ["names a concrete target and bounds its own scope"], signals)

    # 4. FULL is for work that is unbounded IN KIND and plural IN EXTENT. Either alone is
    #    a mini: two small concrete asks chained together do not need a full pass, and one
    #    open-ended ask on one surface is still one lane. Requiring both keeps the
    #    expensive tier rare enough that its cost stays worth paying.
    #    Known limitation: a request whose unboundedness is implied rather than lexical
    #    ("finish all the incomplete UI work") reads as mini. The reasons are attached to
    #    every decision so that misses are visible in the record rather than silent.
    unbounded_kind = signals["heavy_verb"] or signals["discovery_verb"] or signals["vague"]
    plural_extent = signals["multi_surface"] or signals["chained"]
    if unbounded_kind and plural_extent and not signals["bounded_by_target"]:
        reasons = ["spans multiple surfaces" if signals["multi_surface"] else "chains several asks"]
        if signals["heavy_verb"]:
            reasons.append("unbounded verb")
        if signals["discovery_verb"]:
            reasons.append("open-ended discovery")
        if signals["vague"]:
            reasons.append("names a quality, not a deliverable")
        return GateDecision(FULL, reasons, signals)

    # 5. Anything still ambiguous earns the cheap pass, not the expensive one.
    reasons = []
    if signals["heavy_verb"]:
        reasons.append("unbounded verb, single surface")
    if signals["vague"]:
        reasons.append("names a quality, not a deliverable")
    if signals["discovery_verb"] and not signals["bounded_by_target"]:
        reasons.append("open-ended discovery")
    if signals["build_verb"] and not signals["bounded_by_target"]:
        reasons.append("asks for work without naming what it applies to")
    if reasons:
        return GateDecision(MINI, reasons, signals)

    # 5. Nothing fired. Statements, context, and answers are not tasks.
    return GateDecision(NONE, ["no ambiguity signal fired"], signals)
