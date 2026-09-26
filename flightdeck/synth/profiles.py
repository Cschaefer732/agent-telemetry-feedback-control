"""Tunable constants shared by the generators. Values grounded from a real measurement are
labeled with their source; values that are plausible-but-unmeasured say so, matching the
posture `sample_data.py` already took (never blur the two claims together).

# generator
"""

from __future__ import annotations

#: Measured on 90 real sessions, 2026-08-02..2026-09-04 (ported from sample_data.py).
#: Changing these means the sample no longer resembles the system it stands in for --
#: update from a real run, never by taste.
REAL_REWORK_RATE = 0.236
REAL_CORRECTION_RATE = 0.197

#: Ceremony cost rises with tier and late discovery falls -- the tradeoff the KPI pair
#: exists to expose. Values are plausible, not measured (ported verbatim from sample_data.py).
TIER_PROFILE = {
    "none": {"ceremony": (0, 120), "late": (0.25, 0.55), "questions": (0, 1)},
    "mini": {"ceremony": (400, 1200), "late": (0.10, 0.30), "questions": (0, 2)},
    "full": {"ceremony": (1500, 4000), "late": (0.02, 0.15), "questions": (1, 4)},
}

#: Tier weights for the frequency-realistic corpus. Not yet measured from real classify()
#: runs against real transcripts (see 08-synth-coverage.md, "Balance resolution") --
#: plausible only, ported verbatim from sample_data.py's rng.choices weights.
TIER_WEIGHTS = (0.45, 0.35, 0.20)  # none / mini / full

# --- real-prompt marginals (report 07-synth-realism.md, n=972) ----------------------
# Measured on this user's own ~/.claude/projects transcripts, 2026-08-02..2026-09-05.
# Used as the frequency-realistic prompt generator's target rates -- see prompt_labels.py.
REAL_PROMPT_MEDIAN_CHARS = 57
REAL_PROMPT_MEDIAN_WORDS = 10
REAL_PROMPT_NO_TERMINAL_PUNCT_RATE = 0.812
REAL_PROMPT_ALL_LOWERCASE_RATE = 0.555
REAL_PROMPT_VERY_SHORT_RATE = 0.241  # <=3 words
REAL_PROMPT_QUESTION_MARK_RATE = 0.027
REAL_PROMPT_CHAINED_AND_RATE = 0.323
REAL_PROMPT_CODE_BLOCK_RATE = 0.0

# --- frequency-realistic corpus rates (report 08-synth-coverage.md) -----------------
REAL_REWORK_CORPUS_RATE = 0.236
REAL_CORRECTION_CORPUS_RATE = 0.197
