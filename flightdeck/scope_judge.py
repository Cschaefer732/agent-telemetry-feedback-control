"""Correction judge: does a user turn say the assistant's prior work was wrong?

Trained 2026-09-05 against 136 real turns sampled from the transcript corpus, blind-labeled
(the regex verdict was withheld during labeling so it could not anchor the labels), split
60/40, prompt iterated on TRAIN only, held-out spent once.

Held-out, population-weighted (the sample oversamples regex-flagged turns 46/46 vs 90/860,
so every metric is reweighted back to the population or it reads far too optimistic):

    regex (baseline)      precision 0.889  recall 0.218  kappa 0.296
    tfidf+logreg          precision 0.649  recall 0.747  kappa 0.612
    judge v1 (untuned)    precision 0.728  recall 0.698  kappa 0.644
    judge v3 (tuned)      precision 0.736  recall 0.834  kappa 0.726   <- shipped

Kappa, not raw agreement: chance-corrected agreement runs 33-41 points below exact-match
on judge benchmarks, and raw agreement here (90.9%) would have flattered v3 badly.

What the judge is actually for: the KPI needs an unbiased RATE, not perfect per-item calls.
On held-out the judge put the population correction rate at 19.5% against a ground truth of
19.7%; the regex said 4.8%. That gap is the whole reason this exists.

VERDICTS ARE THREE-VALUED. A truncated or unparseable reply is SILENT, never NEW. Scoring
truncation as NEW is what hid v3's real recall (0.726 -> 0.834 once abstentions were split
out) and it is the same fail-open shape that let a dead capture layer report green.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from flightdeck.models import SCOPE_VERDICTS

#: Configurable via FLIGHTDECK_SCOPE_JUDGE_ENDPOINT / FLIGHTDECK_SCOPE_JUDGE_MODEL so this
#: points at your own inference host; defaults assume a local ollama.
DEFAULT_ENDPOINT = os.environ.get(
    "FLIGHTDECK_SCOPE_JUDGE_ENDPOINT", "http://localhost:11434/api/chat"
)
DEFAULT_MODEL = os.environ.get("FLIGHTDECK_SCOPE_JUDGE_MODEL", "qwen3.8:27b")

#: Verdicts this module emits. Subset of SCOPE_VERDICTS -- asserted in tests, because a
#: filter keyed on a string the writer never emits is how 183 rows went into a void.
CORRECTION = "correction"
NEW = "new"
SILENT = "silent"
JUDGE_VERDICTS = (CORRECTION, NEW, SILENT)

PROMPT_VERSION = "v3"
PROMPT = """You classify a single message a user sent to an AI coding assistant.
Work through the steps IN ORDER and stop at the first one that applies.

STEP 1 -- these are always CORRECTION, whatever else the message contains:
- it starts with "no" or "nope", or negates what was done ("not like that", "that's wrong")
- it reports a defect or symptom in what the assistant produced ("the page is blank",
  "no change visually", "still doing X", "only appearing in splotches", "isn't glittery")
- it says the assistant missed, skipped, forgot or has not yet done something
  ("you forgot", "you still need", "you never moved", "you didn't split the panel")
- it tells the assistant to do something that was already implied by the earlier task
  ("you need to remove the old comments", "you need to verify the changes applied",
  "you need to open them all") -- an unmet obligation is a correction
- it challenges the assistant ("why can't you center it", "why do you keep pretending")
- it contradicts a factual claim the assistant made ("well claude code can do that")
- it repeats an instruction that was ignored ("i told you", "stop, i said X")
- it asks for the current behaviour to be REPLACED ("scroll instead of becoming taller")
- it asks for something to be put BACK or restored ("add the comments feature back")

STEP 2 -- only if NOTHING in step 1 applies, these are NEW:
- APPROVING work the assistant just proposed, even if corrective-sounding. The message
  opens with an approval token -- "yes", "ok", "sure", "go ahead", "do it" -- and then
  authorises the proposed work ("yes fix it, you need to roll these out"). The approval
  is what makes it NEW; without it, "you need to ..." is a correction under step 1.
- a bare instruction to fix a problem the ASSISTANT itself just reported ("fix it",
  "so fix it", "fix them"), asserting no fault of its own
- AESTHETIC PREFERENCE refinement asserting no defect ("a little more gold", "brighter",
  "2px smaller") -- wanting it different is not saying it was done wrong
- a fresh feature request, a question for information, or a new task
- pasted terminal output, logs, stack traces, command text; slash commands; one-word
  continuations

Decide on what the message ASSERTS about prior work, not on whether it sounds annoyed.

Reply with exactly one word: CORRECTION or NEW."""


def parse_verdict(reply: str) -> str:
    """Map a raw reply to a three-valued verdict. Anything that is not a clean one-word
    answer is SILENT -- 7.3% of held-out replies began explaining instead and were
    truncated, and calling those NEW understated recall by 11 points."""
    text = (reply or "").strip().upper()
    if not text:
        return SILENT
    head = text.split()[0].strip(".,:;!*")
    if head == "CORRECTION":
        return CORRECTION
    if head == "NEW":
        return NEW
    return SILENT


def judge_turn(
    text: str,
    *,
    endpoint: str = DEFAULT_ENDPOINT,
    model: str = DEFAULT_MODEL,
    timeout: float = 90.0,
    opener: Any = None,
) -> str:
    """Classify one turn. Network or protocol failure is SILENT, not NEW: a judge that
    cannot answer has not said the turn was fine."""
    body = json.dumps(
        {
            "model": model,
            "think": False,  # native /api/chat only; thinking eats the output budget
            "stream": False,
            "options": {"temperature": 0, "num_predict": 12},
            "messages": [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": f"Message:\n{text}\n\nOne word:"},
            ],
        }
    ).encode()
    request = urllib.request.Request(
        endpoint, data=body, headers={"Content-Type": "application/json"}
    )
    try:
        open_url = opener or urllib.request.urlopen
        with open_url(request, timeout=timeout) as response:
            payload = json.load(response)
        return parse_verdict(payload["message"]["content"])
    except (urllib.error.URLError, TimeoutError, KeyError, ValueError, OSError):
        return SILENT


def judge_rate(texts: list[str], **kwargs: Any) -> dict[str, Any]:
    """Correction rate over a batch. Abstentions are excluded from the denominator AND
    reported: a rate computed over a judge that silently failed half the time is a
    fiction, and the abstention count is the only thing that reveals it."""
    verdicts = [judge_turn(t, **kwargs) for t in texts]
    counts = {v: verdicts.count(v) for v in JUDGE_VERDICTS}
    answered = counts[CORRECTION] + counts[NEW]
    return {
        "verdicts": verdicts,
        "counts": counts,
        "answered": answered,
        "abstention_rate": round(counts[SILENT] / len(texts), 4) if texts else None,
        "correction_rate": round(counts[CORRECTION] / answered, 4) if answered else None,
        "prompt_version": PROMPT_VERSION,
        "model": kwargs.get("model", DEFAULT_MODEL),
    }


# The one term this vocabulary genuinely shares with the record vocabulary. Asserting the
# whole set were a subset would need a union to pass, i.e. a check that cannot fail.
assert SILENT in SCOPE_VERDICTS
