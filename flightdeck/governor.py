"""Selector governor: picks model tier / skills / compaction offset / mode+delegation policy.

Every domain starts in shadow mode — the governor computes and records what it would have done
via `record()`, but `enabled(domain)` returns False so callers must not act on the choice. A
domain graduates to live only by an explicit edit to `governor.toml` (the nightly reviewer's job,
not this module's).

`choose()` must never raise: a turn's model-tier resolution, skill injection, etc. cannot be
allowed to fail because a hand-edited TOML file has a typo. A broken or missing config falls back
to the first available arm with `reason="config_error"`, and the caller is expected to treat that
exactly like a shadow no-op.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flightdeck.models import GOVERNOR_DOMAINS, GovernorDecision, Turn
from flightdeck.store import Store

_LOG = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).with_name("governor.toml")

# Verb vocabulary the feature extractor recognizes in a prompt. Order is the canonical order used
# everywhere a verb needs a stable position (e.g. one-hot keys in Features.as_vector()).
_KNOWN_VERBS = ("fix", "refactor", "explain", "test", "search", "write", "debug", "review")
_VERB_RES = {verb: re.compile(rf"\b{verb}\w*\b", re.IGNORECASE) for verb in _KNOWN_VERBS}

# A "file reference" is a whitespace-delimited token that looks like a path: it contains a slash
# or ends in a short extension. Cheap and good enough for a pre-turn feature, not a real parser.
_PATH_TOKEN_RE = re.compile(r"[^\s,;:()\[\]{}'\"`]+")
_EXT_RE = re.compile(r"\.[A-Za-z]{1,6}$")

# Turn field each domain's "chosen" arm can be compared against for graduation evidence. Domains
# with no single scalar turn field (skills is a set, compaction has no turn column) are omitted;
# graduation_report still reports their sample counts, just no shadow/actual KPI delta.
_ACTUAL_FIELD: dict[str, str] = {
    "model_tier": "tier",
    "mode_delegation": "mode",
}

# Declared min/max for every statically-named numeric field. Dynamic tables (kpi.weights.*,
# arms.model_tier.*.cost, arms.compaction.*.offset, weights.model_tier.*.*) are matched by shape
# in `_bounds_for` instead of being listed here one by one.
# success_table() used to scan every turn ever recorded (unbounded, N+1 into decisions_for) on
# every choose() call. Six months is generous next to the shipped 30-day decay half-life — by
# ~6 half-lives a sample's decay weight is under 2% of a fresh one's, so bounding the scan here
# trades an immaterial long tail for turning that scan into one indexed join.
_SUCCESS_TABLE_LOOKBACK_DAYS = 180.0

_STATIC_BOUNDS: dict[tuple[str, ...], tuple[float, float]] = {
    ("governor", "epsilon"): (0.0, 1.0),
    ("governor", "beta"): (0.0, 5.0),
    ("governor", "gamma"): (0.0, 5.0),
    ("governor", "min_samples"): (1, 500),
    ("governor", "optimistic_prior"): (0.0, 1.0),
    ("governor", "decay_half_life_days"): (0.1, 365.0),
    ("thresholds", "tool_error_rate"): (0.0, 1.0),
    ("thresholds", "low_kpi"): (0.0, 1.0),
}


def _bounds_for(path: tuple[str, ...]) -> tuple[float, float] | None:
    if path in _STATIC_BOUNDS:
        return _STATIC_BOUNDS[path]
    if len(path) == 3 and path[0] == "kpi" and path[1] == "weights":
        return (0.0, 1.0)
    if (
        len(path) == 4
        and path[0] == "arms"
        and path[1] in ("model_tier", "skills")
        and path[3] == "cost"
    ):
        return (0.0, 5.0)
    if len(path) == 4 and path[0] == "arms" and path[1] == "compaction" and path[3] == "offset":
        return (-25.0, 25.0)
    if len(path) == 4 and path[0] == "weights" and path[1] == "model_tier":
        return (-2.0, 2.0)
    return None


def _count_file_refs(text: str) -> int:
    count = 0
    for token in _PATH_TOKEN_RE.findall(text):
        stripped = token.strip(".,;:!?")
        if not stripped:
            continue
        if "/" in stripped or _EXT_RE.search(stripped):
            count += 1
    return count


def _hour_bucket(ts_ms: int) -> int:
    hour = time.gmtime(ts_ms / 1000).tm_hour
    return hour // 6


@dataclass(frozen=True)
class Features:
    prompt_tokens: int
    language: str = "unknown"
    verbs: tuple[str, ...] = ()
    file_refs: int = 0
    prev_turn_failed: bool = False
    is_subagent: bool = False
    context_occupancy: float = 0.0
    hour_bucket: int = 0

    def bucket(self) -> str:
        """Coarse, stable shape-bucket key: fine enough to separate real task shapes, coarse
        enough that a bucket actually accumulates the ~20 samples the success table needs."""
        if self.prompt_tokens < 1000:
            size = "s"
        elif self.prompt_tokens < 4000:
            size = "m"
        else:
            size = "l"
        verb = self.verbs[0] if self.verbs else "none"
        scope = "sub" if self.is_subagent else "top"
        return f"{self.language}:{verb}:{size}:{scope}"

    def as_vector(self) -> dict[str, float]:
        """Normalized numeric features for the score's dot product. Language is deliberately
        excluded — it's open vocabulary, not something a fixed-size weight table can cover; it
        only participates in bucket()."""
        vector = {
            "prompt_tokens": min(self.prompt_tokens / 8000, 1.0),
            "file_refs": min(self.file_refs / 10, 1.0),
            "prev_turn_failed": 1.0 if self.prev_turn_failed else 0.0,
            "is_subagent": 1.0 if self.is_subagent else 0.0,
            "context_occupancy": max(0.0, min(self.context_occupancy, 1.0)),
            "hour_bucket": self.hour_bucket / 3.0,
        }
        for verb in _KNOWN_VERBS:
            vector[f"verb_{verb}"] = 1.0 if verb in self.verbs else 0.0
        return vector


def extract_features(
    prompt: str,
    *,
    turn: Turn | None = None,
    language: str = "unknown",
    prev_turn_failed: bool = False,
) -> Features:
    text = prompt or ""
    if turn is not None and turn.prompt_tokens is not None:
        prompt_tokens = turn.prompt_tokens
    else:
        prompt_tokens = max(0, len(text) // 4)  # chars-per-token heuristic, no tokenizer dep

    verbs = tuple(verb for verb in _KNOWN_VERBS if _VERB_RES[verb].search(text))
    file_refs = _count_file_refs(text)
    is_subagent = bool(turn.is_subagent) if turn is not None else False
    if turn is not None and turn.context_occupancy is not None:
        context_occupancy = turn.context_occupancy
    else:
        context_occupancy = 0.0
    ts_ms = turn.started_at if turn is not None else int(time.time() * 1000)

    return Features(
        prompt_tokens=prompt_tokens,
        language=language,
        verbs=verbs,
        file_refs=file_refs,
        prev_turn_failed=prev_turn_failed,
        is_subagent=is_subagent,
        context_occupancy=context_occupancy,
        hour_bucket=_hour_bucket(ts_ms),
    )


@dataclass
class Choice:
    domain: str
    chosen: str
    scores: dict[str, float]
    shadow: bool
    explored: bool
    reason: str
    # The exploration rate in force when this pick was drawn. Carried on the Choice rather than
    # re-read from governor.toml at record time because the toml is hand-edited between runs:
    # by the time a decision is analysed, the file no longer says what it said at the draw.
    epsilon: float | None = None


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Parse the governor TOML as-is. Callers that will act on the values should pass the result
    through clamp_config first."""
    target = path or DEFAULT_CONFIG_PATH
    with target.open("rb") as handle:
        return tomllib.load(handle)


def load_kpi_tuning(path: Path | None = None) -> tuple[dict[str, float], dict[str, float]]:
    """(weights, thresholds) from governor.toml's [kpi.weights]/[thresholds] tables — the ones the
    nightly reviewer hand-edits to self-tune scoring (see CLAUDE.md's allowed-write list). Values
    are clamped the same as every other config field. Falls back to kpi.DEFAULT_WEIGHTS /
    DEFAULT_THRESHOLDS, per table (not all-or-nothing), when the file is absent, unparseable, or a
    table is missing — score_and_persist must never see an empty weights dict."""
    from flightdeck.kpi import DEFAULT_THRESHOLDS, DEFAULT_WEIGHTS

    try:
        clamped, _violations = clamp_config(load_config(path))
    except Exception:
        return dict(DEFAULT_WEIGHTS), dict(DEFAULT_THRESHOLDS)

    raw_weights = clamped.get("kpi", {}).get("weights")
    raw_thresholds = clamped.get("thresholds")
    weights = dict(raw_weights) if isinstance(raw_weights, dict) and raw_weights else None
    thresholds = (
        dict(raw_thresholds) if isinstance(raw_thresholds, dict) and raw_thresholds else None
    )
    return weights or dict(DEFAULT_WEIGHTS), thresholds or dict(DEFAULT_THRESHOLDS)


def clamp_config(cfg: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Enforce every numeric field's declared min/max. Returns a new config with violations
    corrected in place, plus a human-readable list of what was out of range — the nightly
    reviewer edits this file by hand and must see exactly what it broke."""
    violations: list[str] = []

    def walk(node: Any, path: tuple[str, ...]) -> Any:
        if isinstance(node, dict):
            return {key: walk(value, (*path, key)) for key, value in node.items()}
        if isinstance(node, (bool, str)):
            return node
        if isinstance(node, (int, float)):
            bounds = _bounds_for(path)
            if bounds is None:
                return node
            lo, hi = bounds
            if node < lo or node > hi:
                clamped_value = max(lo, min(hi, node))
                if isinstance(node, int):
                    clamped_value = int(round(clamped_value))
                violations.append(
                    f"{'.'.join(path)}={node} out of range [{lo}, {hi}], clamped to {clamped_value}"
                )
                return clamped_value
            return node
        return node

    clamped = walk(cfg, ())
    return clamped, violations


def _arm_weight(weight_table: dict[str, Any], arm: str, feature: str) -> float:
    entry = weight_table.get(arm)
    if not isinstance(entry, dict):
        return 0.0
    value = entry.get(feature, 0.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)


def _arm_cost(cost_table: dict[str, Any], arm: str) -> float:
    entry = cost_table.get(arm)
    if not isinstance(entry, dict):
        return 0.0
    value = entry.get("cost", 0.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)


class Governor:
    def __init__(
        self,
        config_path: Path | None = None,
        store: Store | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self._config_path = config_path or DEFAULT_CONFIG_PATH
        self._store = store
        self._rng = rng or random.Random()
        self._config_bytes = self._read_bytes()
        self._raw_cfg, self._cfg_error = self._load_safe()

    def _read_bytes(self) -> bytes:
        try:
            return self._config_path.read_bytes()
        except OSError:
            return b""

    def _load_safe(self) -> tuple[dict[str, Any], str | None]:
        try:
            cfg = load_config(self._config_path)
            clamped, violations = clamp_config(cfg)
            for violation in violations:
                _LOG.warning("governor config clamped: %s", violation)
            return clamped, None
        except Exception as exc:  # the config is hand-edited; a bad file must not break turns
            _LOG.warning("governor config unreadable at %s: %s", self._config_path, exc)
            return {}, str(exc)

    @property
    def weights_version(self) -> str:
        """Short hash of the config file's raw bytes. Changes iff the file's contents change."""
        return hashlib.sha256(self._config_bytes).hexdigest()[:12]

    def enabled(self, domain: str) -> bool:
        if self._cfg_error:
            return False
        governor_cfg = self._raw_cfg.get("governor", {})
        if not governor_cfg.get("enabled", False):
            return False
        shadow_cfg = governor_cfg.get("shadow", {})
        return not shadow_cfg.get(domain, True)

    def _default_arms(self, domain: str) -> list[str]:
        arms_cfg = self._raw_cfg.get("arms", {}).get(domain)
        if not arms_cfg:
            return []
        return list(arms_cfg.keys())

    def choose(self, domain: str, features: Features, *, arms: list[str] | None = None) -> Choice:
        domain_arms = arms if arms is not None else self._default_arms(domain)
        if self._cfg_error or not domain_arms:
            if self._cfg_error:
                _LOG.warning(
                    "governor.choose: config error for domain=%s: %s", domain, self._cfg_error
                )
            else:
                _LOG.warning("governor.choose: no arms configured for domain=%s", domain)
            fallback = domain_arms[0] if domain_arms else "default"
            return Choice(
                domain=domain,
                chosen=fallback,
                scores={},
                shadow=True,
                explored=False,
                reason="config_error",
            )
        try:
            return self._choose(domain, features, domain_arms)
        except Exception:
            _LOG.exception("governor.choose: scoring failed for domain=%s", domain)
            return Choice(
                domain=domain,
                chosen=domain_arms[0],
                scores={},
                shadow=True,
                explored=False,
                reason="config_error",
            )

    def _choose(self, domain: str, features: Features, domain_arms: list[str]) -> Choice:
        governor_cfg = self._raw_cfg.get("governor", {})
        epsilon = float(governor_cfg.get("epsilon", 0.10))
        beta = float(governor_cfg.get("beta", 1.0))
        gamma = float(governor_cfg.get("gamma", 0.5))
        min_samples = int(governor_cfg.get("min_samples", 20))
        optimistic_prior = float(governor_cfg.get("optimistic_prior", 0.75))

        weight_table = self._raw_cfg.get("weights", {}).get(domain) or {}
        cost_table = self._raw_cfg.get("arms", {}).get(domain) or {}
        vector = features.as_vector()
        bucket = features.bucket()
        success = self.success_table(domain)

        scores: dict[str, float] = {}
        for arm in domain_arms:
            linear = sum(_arm_weight(weight_table, arm, feat) * val for feat, val in vector.items())
            mean, n = success.get((bucket, arm), (0.0, 0))
            empirical = mean if n >= min_samples else optimistic_prior
            cost = _arm_cost(cost_table, arm)
            scores[arm] = linear + beta * empirical - gamma * cost

        argmax_arm = max(domain_arms, key=lambda a: scores[a])
        chosen = argmax_arm
        explored = False
        if self._rng.random() < epsilon:
            pick = self._rng.choice(domain_arms)
            if pick != argmax_arm:
                chosen = pick
                explored = True

        return Choice(
            domain=domain,
            chosen=chosen,
            scores=scores,
            shadow=not self.enabled(domain),
            explored=explored,
            reason="explore" if explored else "argmax",
            epsilon=epsilon,
        )

    def record(self, turn_id: str, choice: Choice, features: Features) -> None:
        """Write a GovernorDecision for this turn. A missing store is a no-op — recording must
        never be the reason a turn fails."""
        if self._store is None:
            return
        decision = GovernorDecision(
            turn_id=turn_id,
            domain=choice.domain,
            chosen=choice.chosen,
            shadow=1 if choice.shadow else 0,
            alternatives=choice.scores,
            features={**features.as_vector(), "bucket": features.bucket()},
            weights_version=self.weights_version,
            explored=1 if choice.explored else 0,
            reason=choice.reason,
            epsilon=choice.epsilon,
        )
        self._store.add_decision(decision)

    def record_shadow_decisions(
        self, *, since_ms: int, domains: tuple[str, ...] | None = None
    ) -> int:
        """Backfill governor_decisions for turns that already have a kpi_score but no decision
        yet. No production code path calls choose()/record() per turn at decision time — model
        tier, skill injection, etc. are chosen elsewhere without ever consulting the governor,
        so this table only exists retroactively, by replaying history: for each already-scored
        turn, reconstruct Features from that turn's own stored prompt text (via
        Store.texts_for(kind="prompt")) when it hasn't expired yet, or "" once it has, then
        choose()+record() as if the decision were being made now. prev_turn_failed is not
        reconstructed (always False) — that needs an in-order per-session replay this helper
        does not do. Turns are processed oldest-first per domain so each turn's success_table
        reflects decisions already recorded earlier in the same backfill, same as it would in a
        live rolling process. Idempotent: Store.turns_pending_decision only returns turns that
        don't already have a decision for the domain, so a rerun costs nothing on turns already
        covered. Returns the number of decisions recorded."""
        if self._store is None:
            return 0
        recorded = 0
        for domain in domains or GOVERNOR_DOMAINS:
            for turn in self._store.turns_pending_decision(domain, since_ms=since_ms):
                texts = self._store.texts_for(turn.turn_id, kind="prompt")
                prompt = texts[0].body if texts else ""
                features = extract_features(prompt, turn=turn)
                choice = self.choose(domain, features)
                self.record(turn.turn_id, choice, features)
                recorded += 1
        return recorded

    def success_table(self, domain: str) -> dict[tuple[str, str], tuple[float, int]]:
        """(bucket, arm) -> (recency-decayed mean kpi_score, sample count), over the trailing
        `_SUCCESS_TABLE_LOOKBACK_DAYS`. An arm name that never appears in a configured arms table
        is not a crash, just an entry no one scores."""
        if self._store is None:
            return {}
        half_life_days = float(self._raw_cfg.get("governor", {}).get("decay_half_life_days", 30.0))
        half_life_ms = max(half_life_days, 0.01) * 86_400_000
        now = int(time.time() * 1000)
        since = now - int(_SUCCESS_TABLE_LOOKBACK_DAYS * 86_400_000)

        weighted_sum: dict[tuple[str, str], float] = {}
        weight_total: dict[tuple[str, str], float] = {}
        counts: dict[tuple[str, str], int] = {}

        for row in self._store.scored_decisions(domain, since_ms=since):
            arm = row.get("chosen")
            bucket = _bucket_from_row(row)
            if arm is None or bucket is None:
                continue
            started_at = row.get("started_at")
            age_ms = max(0, now - (started_at or now))
            decay = 0.5 ** (age_ms / half_life_ms)
            key = (bucket, arm)
            weighted_sum[key] = weighted_sum.get(key, 0.0) + decay * row["kpi_score"]
            weight_total[key] = weight_total.get(key, 0.0) + decay
            counts[key] = counts.get(key, 0) + 1

        return {
            key: (weighted_sum[key] / weight_total[key] if weight_total[key] else 0.0, counts[key])
            for key in weighted_sum
        }

    def graduation_report(self, domain: str, *, days: int = 7) -> dict[str, Any]:
        """Evidence for the nightly reviewer to decide whether `domain` can flip live: per-arm
        sample counts, whether every configured arm cleared min_samples, and the mean KPI delta
        between turns where the shadow pick matches what actually ran vs turns where it diverged.

        Backed by Store.scored_decisions — one indexed join, in place of the N+1 scan (iter_turns
        then decisions_for per turn) success_table was refactored away from for the identical
        reason. That also means, same as success_table, only turns with a kpi_score count as
        evidence here: an unscored decision proves nothing about the arm yet."""
        min_samples = int(self._raw_cfg.get("governor", {}).get("min_samples", 20))
        since = int(time.time() * 1000) - days * 86_400_000
        actual_attr = _ACTUAL_FIELD.get(domain)

        arm_counts: dict[str, int] = {}
        agree_kpis: list[float] = []
        disagree_kpis: list[float] = []
        considered = 0

        if self._store is not None:
            for row in self._store.scored_decisions(domain, since_ms=since):
                considered += 1
                arm = row.get("chosen")
                if arm is not None:
                    arm_counts[arm] = arm_counts.get(arm, 0) + 1
                if actual_attr is None:
                    continue
                actual = row.get(actual_attr)
                if actual is None:
                    continue
                (agree_kpis if arm == actual else disagree_kpis).append(row["kpi_score"])

        all_arms = self._default_arms(domain) or list(arm_counts)
        all_cleared = bool(all_arms) and all(arm_counts.get(a, 0) >= min_samples for a in all_arms)
        mean_agree = sum(agree_kpis) / len(agree_kpis) if agree_kpis else None
        mean_disagree = sum(disagree_kpis) / len(disagree_kpis) if disagree_kpis else None
        have_both = mean_agree is not None and mean_disagree is not None
        kpi_delta = (mean_agree - mean_disagree) if have_both else None

        return {
            "domain": domain,
            "days": days,
            "decisions": considered,
            "arm_counts": arm_counts,
            "min_samples": min_samples,
            "all_arms_cleared": all_cleared,
            "mean_kpi_delta": kpi_delta,
            "mean_kpi_shadow_matched_actual": mean_agree,
            "mean_kpi_actual_diverged": mean_disagree,
        }


def _bucket_from_row(row: dict[str, Any]) -> str | None:
    raw_features = row.get("features")
    if not raw_features:
        return None
    try:
        parsed = json.loads(raw_features) if isinstance(raw_features, str) else raw_features
    except (TypeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    bucket = parsed.get("bucket")
    return bucket if isinstance(bucket, str) else None
