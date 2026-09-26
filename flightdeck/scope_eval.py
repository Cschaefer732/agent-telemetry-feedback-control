"""Score the correction classifiers against the hand-labelled gold set, reproducibly.

This module exists because the judge's headline numbers could not be re-derived. The
docstring on `scope_judge` reports accuracy, precision, recall and kappa; nothing in the
repository regenerated them, the fixture carried no split assignment, and the numbers were
measured in a session and never committed as a check. A number nobody can recompute is a
claim. This is the command that turns them back into results.

Two things it refuses to get wrong:

WEIGHTING. The gold set is STRATIFIED, not random: 46 of its 139 rows are regex-positive
turns and 93 are regex-negative, drawn from a population holding POPULATION["regex1"] and
POPULATION["regex0"] of each (illustrative example figures below -- re-measure your own
with `measure_population()`). So a regex-positive row stands for many more turns than a
regex-negative row does. Reading a rate straight off the fixture overstates the regex
recall severalfold. Every population figure here is reweighted; every in-sample figure is
labelled as in-sample.

GROUPING. The gold rows come from multiple sessions and one session supplies a
disproportionate share of them. Turns within a session share topic, phrasing and the
user's mood, so a row-wise split leaks: the same session lands on both sides and the
held-out score reports memorisation. Splits here are always by session, never by row.

The held-out half is spent, not consulted. Every evaluation against it appends to
holdout-log.jsonl, and that log is the peek-prevention mechanism -- not a convention. If a
prompt version has been scored against held-out more than once, the second number is not
independent evidence and this module says so in its output.

The shipped `tests/fixtures/correction_labels.json` is a SYNTHETIC gold set (invented
example utterances, not sampled from any real session) built to reproduce the same
statistical shape a real hand-labelled corpus has: a regex baseline with high precision but
low population-level recall, because most real corrections are phrased as symptoms
("the page is blank") rather than with an obvious keyword. Point GOLD_PATH at your own
labelled data to get real numbers.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

GOLD_PATH = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "correction_labels.json"

#: ILLUSTRATIVE EXAMPLE figures, sized to match the shipped synthetic gold set's shape.
#: A real deployment measures its own corpus with `python3 -m flightdeck.scope_baseline`
#: and re-pins these (including a corpus_sha via `flightdeck.scope_snapshot`) rather than
#: adjusting them by taste -- the corpus rolls on a retention window, so this pin goes
#: stale continuously, not just once, and that is exactly what `check_population_drift`
#: below exists to catch.
POPULATION = {"regex1": 50, "regex0": 3000}
POPULATION_TURNS = POPULATION["regex1"] + POPULATION["regex0"]
POPULATION_MEASURED_AT = "example"
#: `flightdeck.scope_snapshot.corpus_sha` of the corpus at measurement time, in a real
#: deployment. Not reproducible from the fixture -- it identifies the live corpus, not the
#: gold set -- but it makes clear which corpus produced POPULATION and lets a
#: re-measurement be compared against a name instead of a vibe. Unset here since this
#: repo ships no live corpus.
POPULATION_CORPUS_SHA = None
POPULATION_HUMAN_TURNS = 3050
POPULATION_CORRECTION_TURNS = 50

#: Relative tolerance per stratum before the population pin is considered drifted. Set to
#: 2%: at ~50 rows/session and one session supplying disproportionate weight (see the
#: GROUPING note above), a single session rolling off retention moves a stratum count by
#: roughly this much. Anything inside 2% is normal daily churn; anything past it means the
#: pin needs re-measuring, not that the corpus had a bad day.
POPULATION_DRIFT_TOLERANCE = 0.02

#: Held-out fraction, by SESSION. On a modestly sized gold set this is a small number of
#: sessions and rows -- below the n>=20-per-arm floor for anything but a large effect, and
#: that limit is reported rather than hidden.
HOLDOUT_FRACTION = 0.4
SPLIT_SEED = 20260905

#: Under this many rows in an arm, a rate is noise. Reported as "silent", never as a number
#: that invites a decision.
MIN_ARM_ROWS = 20


@dataclass(frozen=True)
class GoldRow:
    id: str
    text: str
    label: int
    regex: int
    session: str
    split: str = "train"

    @property
    def weight(self) -> float:
        """How many population turns this row stands for."""
        stratum = "regex1" if self.regex else "regex0"
        drawn = SAMPLE_STRATA[stratum]
        return POPULATION[stratum] / drawn if drawn else 0.0


def load_gold(path: Path | str = GOLD_PATH) -> list[GoldRow]:
    raw = json.loads(Path(path).read_text())
    rows = [
        GoldRow(
            id=r["id"],
            text=r["text"],
            label=int(r["label"]),
            regex=int(r["regex"]),
            session=r["session"],
        )
        for r in raw
    ]
    return assign_splits(rows)


def _stratum_counts(rows: Sequence[GoldRow]) -> dict[str, int]:
    return {
        "regex1": sum(1 for r in rows if r.regex),
        "regex0": sum(1 for r in rows if not r.regex),
    }


#: Filled from the fixture at import so the weights track the file rather than a comment.
SAMPLE_STRATA = _stratum_counts(
    [
        GoldRow(
            id=r["id"],
            text=r["text"],
            label=int(r["label"]),
            regex=int(r["regex"]),
            session=r["session"],
        )
        for r in json.loads(GOLD_PATH.read_text())
    ]
)


def assign_splits(
    rows: Sequence[GoldRow], *, holdout: float = HOLDOUT_FRACTION, seed: int = SPLIT_SEED
) -> list[GoldRow]:
    """Deterministic split BY SESSION. The same seed always produces the same partition, so
    a split can be quoted in a report and reproduced from it."""
    sessions = sorted({r.session for r in rows})
    rng = random.Random(seed)
    shuffled = sessions[:]
    rng.shuffle(shuffled)
    cut = max(1, round(len(shuffled) * holdout))
    held = set(shuffled[:cut])
    return [
        GoldRow(
            r.id, r.text, r.label, r.regex, r.session, "holdout" if r.session in held else "train"
        )
        for r in rows
    ]


# ------------------------------------------------------------------ metrics


def confusion(rows: Sequence[GoldRow], predict: Callable[[GoldRow], int | None]) -> dict[str, Any]:
    """Counts both raw and population-weighted, plus abstentions kept separate.

    An abstention is not a negative. Scoring four truncated judge replies as NEW once
    understated recall here by 11 points; three-valued verdicts exist so that an
    unanswered item is visible as unanswered.
    """
    tp = fp = tn = fn = 0
    wtp = wfp = wtn = wfn = 0.0
    abstained = 0
    weighted_abstained = 0.0
    for row in rows:
        guess = predict(row)
        if guess is None:
            abstained += 1
            weighted_abstained += row.weight
            continue
        if row.label and guess:
            tp += 1
            wtp += row.weight
        elif row.label and not guess:
            fn += 1
            wfn += row.weight
        elif not row.label and guess:
            fp += 1
            wfp += row.weight
        else:
            tn += 1
            wtn += row.weight
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "abstained": abstained,
        "weighted": {
            "tp": round(wtp, 1),
            "fp": round(wfp, 1),
            "tn": round(wtn, 1),
            "fn": round(wfn, 1),
            "abstained": round(weighted_abstained, 1),
        },
        "scored": tp + fp + tn + fn,
    }


def _ratio(num: float, den: float) -> float | None:
    return round(num / den, 4) if den else None


def scores(matrix: dict[str, Any], *, weighted: bool) -> dict[str, Any]:
    m = matrix["weighted"] if weighted else matrix
    tp, fp, tn, fn = m["tp"], m["fp"], m["tn"], m["fn"]
    total = tp + fp + tn + fn
    return {
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "accuracy": _ratio(tp + tn, total),
        "kappa": cohen_kappa(tp, fp, tn, fn),
        "n": round(total, 1),
    }


def cohen_kappa(tp: float, fp: float, tn: float, fn: float) -> float | None:
    """Chance-corrected agreement.

    Reported instead of raw agreement because raw agreement on a skewed class is mostly the
    skew: a classifier that answers "not a correction" every time scores 95% agreement on a
    5% base rate and a kappa of 0.
    """
    total = tp + fp + tn + fn
    if not total:
        return None
    observed = (tp + tn) / total
    expected = (((tp + fn) * (tp + fp)) + ((fp + tn) * (fn + tn))) / (total * total)
    if expected == 1:
        return None
    return round((observed - expected) / (1 - expected), 4)


def population_rate(
    rows: Sequence[GoldRow], predict: Callable[[GoldRow], int | None]
) -> dict[str, Any]:
    """The number this classifier actually exists to produce.

    The judge's job is estimating the CORRECTION RATE, not classifying each turn. An
    unbiased rate built on mediocre per-item accuracy is a success here and a per-item score
    that hides a biased rate is not.
    """
    weight_total = answered = 0.0
    positive = 0.0
    for row in rows:
        guess = predict(row)
        weight_total += row.weight
        if guess is None:
            continue
        answered += row.weight
        if guess:
            positive += row.weight
    truth = sum(r.weight for r in rows if r.label) / weight_total if weight_total else None
    estimate = positive / answered if answered else None
    return {
        "estimated_rate": round(estimate, 4) if estimate is not None else None,
        "true_rate": round(truth, 4) if truth is not None else None,
        "rate_error": (
            round(estimate - truth, 4) if estimate is not None and truth is not None else None
        ),
        "answered_weight": round(answered, 1),
        "abstained_weight": round(weight_total - answered, 1),
    }


def memoize(
    rows: Sequence[GoldRow], predict: Callable[[GoldRow], int | None]
) -> Callable[[GoldRow], int | None]:
    """Score every row ONCE and serve the stored verdict thereafter.

    The bootstrap below resamples the corpus 2,000 times. Calling a classifier inside each
    draw turns a 68-item evaluation into 136,000 -- with a judge on a serialised endpoint
    that is a run that never finishes, which is exactly how the first version of this
    harness behaved. A prediction is a property of the row, so it is computed with the row.
    """
    cache = {row.id: predict(row) for row in rows}
    return lambda row: cache[row.id]


def session_bootstrap_ci(
    rows: Sequence[GoldRow],
    predict: Callable[[GoldRow], int | None],
    *,
    draws: int = 2000,
    seed: int = SPLIT_SEED,
) -> dict[str, Any]:
    """Resample SESSIONS, not rows. Rows inside a session are correlated, so a row-wise
    bootstrap reports a confidence interval several times narrower than the data supports."""
    by_session: dict[str, list[GoldRow]] = {}
    for row in rows:
        by_session.setdefault(row.session, []).append(row)
    sessions = list(by_session)
    if len(sessions) < 2:
        return {"verdict": "silent", "reason": "fewer than two sessions"}
    rng = random.Random(seed)
    estimates = []
    for _ in range(draws):
        picked = [by_session[rng.choice(sessions)] for _ in sessions]
        flat = [r for group in picked for r in group]
        rate = population_rate(flat, predict)["estimated_rate"]
        if rate is not None:
            estimates.append(rate)
    if not estimates:
        return {"verdict": "silent", "reason": "no estimate produced"}
    estimates.sort()
    lo = estimates[int(0.025 * len(estimates))]
    hi = estimates[min(len(estimates) - 1, int(0.975 * len(estimates)))]
    return {
        "ci95": [round(lo, 4), round(hi, 4)],
        "width": round(hi - lo, 4),
        "sessions": len(sessions),
        "draws": len(estimates),
    }


def measure_population(root: Path | str | None = None) -> dict[str, Any]:
    """Recompute the population strata from the live corpus, right now.

    This is the only path that is allowed to say what POPULATION should be. The module
    constant is a pin taken from a run of this function; it is never hand-adjusted.
    """
    from flightdeck import scope_baseline, scope_snapshot

    kwargs = {} if root is None else {"root": Path(root).expanduser()}
    _rows, summary = scope_baseline.analyze(**kwargs)
    snapshot = scope_snapshot.take(**kwargs)
    human_turns = summary["human_turns"]
    correction_turns = summary["correction_turns"]
    return {
        "regex1": correction_turns,
        "regex0": human_turns - correction_turns,
        "human_turns": human_turns,
        "correction_turns": correction_turns,
        "corpus_sha": snapshot.corpus_sha,
        "session_files": snapshot.session_files,
    }


def check_population_drift(
    root: Path | str | None = None,
    *,
    pinned: dict[str, int] | None = None,
    tolerance: float = POPULATION_DRIFT_TOLERANCE,
) -> dict[str, Any]:
    """Re-measure the population and compare it against the pinned POPULATION constant.

    Weight = POPULATION[stratum] / SAMPLE_STRATA[stratum], so any drift in the pin scales
    linearly into every population-weighted number this module produces, silently, unless
    something re-measures and checks. This is that something.
    """
    pinned = pinned if pinned is not None else POPULATION
    live = measure_population(root)
    strata: dict[str, Any] = {}
    drifted: list[str] = []
    for stratum in ("regex1", "regex0"):
        before = pinned[stratum]
        now = live[stratum]
        rel = abs(now - before) / before if before else (0.0 if now == before else math.inf)
        strata[stratum] = {"pinned": before, "live": now, "relative_diff": round(rel, 4)}
        if rel > tolerance:
            drifted.append(stratum)
    return {
        "verdict": "drift" if drifted else "ok",
        "drifted_strata": drifted,
        "tolerance": tolerance,
        "strata": strata,
        "live_corpus_sha": live["corpus_sha"],
        "pinned_corpus_sha": POPULATION_CORPUS_SHA,
        "corpus_sha_matches": live["corpus_sha"] == POPULATION_CORPUS_SHA,
    }


# ------------------------------------------------------------------ classifiers


def regex_classifier_version() -> str:
    """Content hash of the actual regex logic, not a literal someone forgets to bump.

    The hardcoded "v3" the CLI used to log under meant editing the regex and re-running
    logged the new numbers as belonging to the old version -- indistinguishable in the
    holdout log from a repeat spend of the same classifier.
    """
    import inspect

    from flightdeck import scope_baseline

    source = inspect.getsource(scope_baseline.classify_correction)
    return "regex-" + hashlib.sha256(source.encode()).hexdigest()[:12]


def _infer_classifier_version(predict: Callable[[GoldRow], int | None]) -> str:
    if predict is judge_predict:
        from flightdeck.scope_judge import PROMPT_VERSION

        return PROMPT_VERSION
    if predict is regex_predict:
        return regex_classifier_version()
    import inspect

    try:
        source = inspect.getsource(predict)
    except (OSError, TypeError):
        source = repr(predict)
    return "fn-" + hashlib.sha256(source.encode()).hexdigest()[:12]


def regex_predict(row: GoldRow) -> int:
    """The committed regex baseline, re-run rather than trusted from the fixture column."""
    from flightdeck.scope_baseline import classify_correction

    return 1 if classify_correction(row.text) else 0


def judge_predict(row: GoldRow) -> int | None:
    """The LLM judge. Returns None on SILENT -- abstention, network failure and truncation
    all land here, and none of them is a negative."""
    from flightdeck.scope_judge import CORRECTION, judge_turn

    verdict = judge_turn(row.text)
    if verdict == CORRECTION:
        return 1
    from flightdeck.scope_judge import NEW

    return 0 if verdict == NEW else None


# ------------------------------------------------------------------ the report


def evaluate(
    rows: Sequence[GoldRow],
    predict: Callable[[GoldRow], int | None],
    *,
    name: str,
    split: str = "train",
    spend_dir: Path | str | None = None,
    version: str | None = None,
) -> dict[str, Any]:
    """Run one classifier over one split and report every statistic derived from it.

    A held-out split is spent the moment this function is called with it -- see
    `record_holdout_spend` below, invoked from here unconditionally so no caller (test,
    notebook, REPL) can evaluate against holdout without it landing in the log.
    """
    arm = [r for r in rows if r.split == split]
    # One pass over the classifier; every statistic below reads the same stored verdicts,
    # so the report is internally consistent as well as affordable.
    scored = memoize(arm, predict)
    matrix = confusion(arm, scored)
    report: dict[str, Any] = {
        "classifier": name,
        "split": split,
        "rows": len(arm),
        "sessions": len({r.session for r in arm}),
        "confusion": matrix,
        "in_sample": scores(matrix, weighted=False),
        "population": scores(matrix, weighted=True),
        "rate": population_rate(arm, scored),
        "rate_ci": session_bootstrap_ci(arm, scored),
        "population_measured_at": POPULATION_MEASURED_AT,
        "sample_strata": SAMPLE_STRATA,
    }
    # Precision rests on the predicted-positive arm, recall on the actual-positive arm --
    # not on the split's total row count. A 68-row train split can still carry a
    # tp+fp of 19: below the floor, silently, while len(arm) sails past it. Check every
    # arm a ratio in `scores` divides by.
    tp, fp, tn, fn = matrix["tp"], matrix["fp"], matrix["tn"], matrix["fn"]
    class_arms = {
        "predicted_positive (tp+fp, precision denominator)": tp + fp,
        "actual_positive (tp+fn, recall denominator)": tp + fn,
        "predicted_negative (tn+fn)": tn + fn,
        "actual_negative (tn+fp)": tn + fp,
    }
    short_arms = {label: n for label, n in class_arms.items() if n < MIN_ARM_ROWS}
    if len(arm) < MIN_ARM_ROWS:
        report["verdict"] = "silent"
        report["reason"] = f"{len(arm)} rows is below the {MIN_ARM_ROWS}-row floor for a rate"
    elif short_arms:
        report["verdict"] = "silent"
        report["reason"] = "; ".join(
            f"{label} has n={n}, below the {MIN_ARM_ROWS}-row floor"
            for label, n in short_arms.items()
        )
        report["short_arms"] = short_arms
    else:
        report["verdict"] = "pass"
    if split == "holdout":
        from flightdeck.store import DEFAULT_DIR

        report["holdout_spend"] = record_holdout_spend(
            spend_dir if spend_dir is not None else DEFAULT_DIR,
            classifier=name,
            version=version or _infer_classifier_version(predict),
            report=report,
        )
    return report


def holdout_log_path(directory: Path | str) -> Path:
    return Path(directory).expanduser() / "scope" / "holdout-log.jsonl"


def record_holdout_spend(
    directory: Path | str, *, classifier: str, version: str, report: dict[str, Any]
) -> dict[str, Any]:
    """Append-only ledger of every held-out evaluation.

    This IS the peek-prevention mechanism. A held-out set consulted repeatedly while a
    prompt is tuned is a training set with extra steps, and the only thing that makes that
    visible afterwards is a record of how many times it was spent.
    """
    path = holdout_log_path(directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    prior = 0
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if entry.get("classifier") == classifier and entry.get("version") == version:
                prior += 1
    entry = {
        "classifier": classifier,
        "version": version,
        "spend_number": prior + 1,
        "independent": prior == 0,
        "rows": report.get("rows"),
        "population": report.get("population"),
        "rate": report.get("rate"),
        "fingerprint": hashlib.sha256(
            json.dumps(report, sort_keys=True, default=str).encode()
        ).hexdigest()[:16],
    }
    with path.open("a") as handle:
        handle.write(json.dumps(entry) + "\n")
    if prior:
        entry["warning"] = (
            f"held-out has been spent {prior + 1} times on {classifier} {version}; "
            "only the first is independent evidence"
        )
    return entry


def summarize(reports: Sequence[dict[str, Any]]) -> str:
    return json.dumps(list(reports), indent=2, sort_keys=True, default=str)


def _mean(values: Sequence[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


assert not math.isnan(HOLDOUT_FRACTION)
