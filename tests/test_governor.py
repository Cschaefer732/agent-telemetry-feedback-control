from __future__ import annotations

import random
import time
from pathlib import Path

import pytest

from flightdeck.governor import (
    DEFAULT_CONFIG_PATH,
    Features,
    Governor,
    _bucket_from_row,
    clamp_config,
    extract_features,
    load_config,
    load_kpi_tuning,
)
from flightdeck.kpi import DEFAULT_THRESHOLDS, DEFAULT_WEIGHTS
from flightdeck.models import GOVERNOR_DOMAINS, GovernorDecision, Turn
from flightdeck.store import Store

SHIPPED_CONFIG = DEFAULT_CONFIG_PATH


def _write_toml(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def _base_toml(
    *,
    enabled: bool = True,
    epsilon: float = 0.0,
    beta: float = 1.0,
    gamma: float = 0.0,
    min_samples: int = 3,
    optimistic_prior: float = 0.75,
    decay_half_life_days: float = 30.0,
    shadow: bool = True,
) -> str:
    shadow_str = str(shadow).lower()
    return f"""
[governor]
enabled = {str(enabled).lower()}
epsilon = {epsilon}
beta = {beta}
gamma = {gamma}
min_samples = {min_samples}
optimistic_prior = {optimistic_prior}
decay_half_life_days = {decay_half_life_days}

[governor.shadow]
model_tier = {shadow_str}
skills = {shadow_str}
compaction = {shadow_str}
mode_delegation = {shadow_str}

[kpi.weights]
completion = 0.2
tool_reliability = 0.2
efficiency = 0.15
focus = 0.15
context_health = 0.15
autonomy = 0.15

[arms.model_tier]
fast = {{ cost = 0.0 }}
balanced = {{ cost = 0.0 }}
deep = {{ cost = 0.0 }}
frontier = {{ cost = 0.0 }}

[weights.model_tier]
fast = {{ prompt_tokens = 0.0 }}
balanced = {{ prompt_tokens = 0.0 }}
deep = {{ prompt_tokens = 0.0 }}
frontier = {{ prompt_tokens = 0.0 }}

[arms.compaction]
default_recall_off = {{ offset = 0.0, recall = false }}

[arms.mode_delegation]
plan_first = true
critic = true
delegate = true

[thresholds]
tool_error_rate = 0.25
low_kpi = 0.5
"""


def _features(**overrides: object) -> Features:
    base = dict(
        prompt_tokens=100,
        language="python",
        verbs=(),
        file_refs=0,
        prev_turn_failed=False,
        is_subagent=False,
        context_occupancy=0.0,
        hour_bucket=0,
    )
    base.update(overrides)
    return Features(**base)  # type: ignore[arg-type]


# ---------- feature extraction ----------


def test_extract_features_basic() -> None:
    f = extract_features(
        "please fix flightdeck/governor.py and tests/test_governor.py", language="python"
    )
    assert "fix" in f.verbs
    assert f.file_refs == 2
    assert f.language == "python"
    assert f.prompt_tokens > 0


def test_extract_features_uses_turn_metadata() -> None:
    turn = Turn(
        turn_id="t1",
        session_id="s1",
        source="crush",
        host="h1",
        started_at=int(time.time() * 1000),
        prompt_tokens=555,
        is_subagent=1,
        context_peak=800,
        context_window=1000,
    )
    f = extract_features("refactor this", turn=turn, prev_turn_failed=True)
    assert f.prompt_tokens == 555
    assert f.is_subagent is True
    assert f.context_occupancy == pytest.approx(0.8)
    assert f.prev_turn_failed is True


def test_bucket_stable_for_same_shape() -> None:
    f1 = extract_features("fix bug in a.py", language="python")
    f2 = extract_features("fix bug in b.py c.py", language="python")
    assert f1.bucket() == f2.bucket()


def test_bucket_changes_with_language() -> None:
    f1 = extract_features("fix bug", language="python")
    f2 = extract_features("fix bug", language="go")
    assert f1.bucket() != f2.bucket()


# ---------- scoring ----------


def test_dot_product_scoring(tmp_path: Path) -> None:
    text = _base_toml(beta=0.0, gamma=0.0, epsilon=0.0).replace(
        "fast = { prompt_tokens = 0.0 }",
        "fast = { prompt_tokens = 1.0, file_refs = 2.0 }",
    )
    cfg_path = _write_toml(tmp_path / "governor.toml", text)
    gov = Governor(config_path=cfg_path, rng=random.Random(1))

    features = _features(prompt_tokens=8000, file_refs=10)  # both normalize to 1.0
    choice = gov.choose("model_tier", features)

    assert choice.scores["fast"] == pytest.approx(1.0 * 1.0 + 2.0 * 1.0)
    assert choice.scores["balanced"] == pytest.approx(0.0)
    assert choice.chosen == "fast"
    assert choice.explored is False


def test_optimistic_prior_below_min_samples(tmp_path: Path) -> None:
    text = _base_toml(beta=1.0, gamma=0.0, epsilon=0.0, min_samples=5, optimistic_prior=0.9)
    cfg_path = _write_toml(tmp_path / "governor.toml", text)
    store = Store(tmp_path / "store")

    features = _features(prompt_tokens=100, verbs=("fix",))
    bucket = features.bucket()
    now = int(time.time() * 1000)

    # Only 2 samples, well under min_samples=5, with a low actual kpi — the prior must win.
    for i in range(2):
        turn_id = f"t{i}"
        store.upsert_turn(
            Turn(
                turn_id=turn_id,
                session_id="s",
                source="crush",
                host="h",
                started_at=now,
                kpi_score=0.1,
            )
        )
        store.add_decision(
            GovernorDecision(
                turn_id=turn_id,
                domain="model_tier",
                chosen="fast",
                shadow=1,
                alternatives={},
                features={"bucket": bucket},
            )
        )

    gov = Governor(config_path=cfg_path, store=store, rng=random.Random(0))
    choice = gov.choose("model_tier", features)
    assert choice.scores["fast"] == pytest.approx(0.9)  # optimistic_prior, not the 0.1 mean


def test_empirical_mean_used_once_min_samples_reached(tmp_path: Path) -> None:
    text = _base_toml(beta=1.0, gamma=0.0, epsilon=0.0, min_samples=3, optimistic_prior=0.9)
    cfg_path = _write_toml(tmp_path / "governor.toml", text)
    store = Store(tmp_path / "store")

    features = _features(prompt_tokens=100, verbs=("fix",))
    bucket = features.bucket()
    now = int(time.time() * 1000)

    for i in range(3):
        turn_id = f"t{i}"
        store.upsert_turn(
            Turn(
                turn_id=turn_id,
                session_id="s",
                source="crush",
                host="h",
                started_at=now,
                kpi_score=0.1,
            )
        )
        store.add_decision(
            GovernorDecision(
                turn_id=turn_id,
                domain="model_tier",
                chosen="fast",
                shadow=1,
                alternatives={},
                features={"bucket": bucket},
            )
        )

    gov = Governor(config_path=cfg_path, store=store, rng=random.Random(0))
    choice = gov.choose("model_tier", features)
    assert choice.scores["fast"] == pytest.approx(0.1, abs=0.01)


def test_epsilon_exploration_seeded(tmp_path: Path) -> None:
    text = _base_toml(beta=0.0, gamma=0.0, epsilon=1.0, min_samples=1000).replace(
        "fast = { prompt_tokens = 0.0 }",
        "fast = { prompt_tokens = 5.0 }",
    )
    cfg_path = _write_toml(tmp_path / "governor.toml", text)

    features = _features(prompt_tokens=8000)
    gov = Governor(config_path=cfg_path, rng=random.Random(42))
    choice = gov.choose("model_tier", features)

    # Argmax is deterministically "fast" (only nonzero weight). Replay the same rng seed
    # independently to know exactly what epsilon=1.0 must have picked.
    replay = random.Random(42)
    assert replay.random() < 1.0  # always explores
    expected_pick = replay.choice(["fast", "balanced", "deep", "frontier"])

    assert choice.chosen == expected_pick
    assert choice.explored == (expected_pick != "fast")
    assert choice.reason == ("explore" if choice.explored else "argmax")


def test_choose_deterministic_given_seed(tmp_path: Path) -> None:
    text = _base_toml(beta=1.0, gamma=0.0, epsilon=0.3)
    cfg_path = _write_toml(tmp_path / "governor.toml", text)
    features = _features(prompt_tokens=2000, verbs=("test",))

    gov1 = Governor(config_path=cfg_path, rng=random.Random(7))
    gov2 = Governor(config_path=cfg_path, rng=random.Random(7))
    c1 = gov1.choose("model_tier", features)
    c2 = gov2.choose("model_tier", features)

    assert c1.chosen == c2.chosen
    assert c1.explored == c2.explored
    assert c1.scores == c2.scores


# ---------- recency decay ----------


def test_recency_decay_changes_ranking(tmp_path: Path) -> None:
    store = Store(tmp_path / "store")
    now = int(time.time() * 1000)
    old_ts = now - 60 * 86_400_000
    bucket = "python:fix:s:top"

    # Arm A: strong old evidence, weak recent evidence. Arm B: the reverse.
    samples = [
        ("A", old_ts, 1.0),
        ("A", now, 0.4),
        ("B", old_ts, 0.4),
        ("B", now, 0.7),
    ]
    for i, (arm, ts, kpi) in enumerate(samples):
        turn_id = f"t{i}"
        store.upsert_turn(
            Turn(
                turn_id=turn_id,
                session_id="s",
                source="crush",
                host="h",
                started_at=ts,
                kpi_score=kpi,
            )
        )
        store.add_decision(
            GovernorDecision(
                turn_id=turn_id,
                domain="model_tier",
                chosen=arm,
                shadow=1,
                alternatives={},
                features={"bucket": bucket},
            )
        )

    no_decay_cfg = _write_toml(tmp_path / "no_decay.toml", _base_toml(decay_half_life_days=3650.0))
    fast_decay_cfg = _write_toml(tmp_path / "fast_decay.toml", _base_toml(decay_half_life_days=1.0))

    gov_no_decay = Governor(config_path=no_decay_cfg, store=store, rng=random.Random(0))
    gov_fast_decay = Governor(config_path=fast_decay_cfg, store=store, rng=random.Random(0))

    table_no_decay = gov_no_decay.success_table("model_tier")
    table_fast_decay = gov_fast_decay.success_table("model_tier")

    mean_a_slow, _ = table_no_decay[(bucket, "A")]
    mean_b_slow, _ = table_no_decay[(bucket, "B")]
    mean_a_fast, _ = table_fast_decay[(bucket, "A")]
    mean_b_fast, _ = table_fast_decay[(bucket, "B")]

    assert mean_a_slow > mean_b_slow  # old-heavy evidence still wins without decay
    assert mean_b_fast > mean_a_fast  # recency decay flips the ranking toward B


def _old_success_table(
    store: Store, domain: str, *, half_life_days: float
) -> dict[tuple[str, str], tuple[float, int]]:
    """Reference reimplementation of Governor.success_table's pre-fix body: an unbounded scan of
    every turn ever recorded, decisions_for() called once per turn, filtering `domain` in Python.
    Equivalence oracle for test_success_table_matches_old_full_scan below."""
    half_life_ms = max(half_life_days, 0.01) * 86_400_000
    now = int(time.time() * 1000)
    weighted_sum: dict[tuple[str, str], float] = {}
    weight_total: dict[tuple[str, str], float] = {}
    counts: dict[tuple[str, str], int] = {}
    for turn in store.iter_turns():
        if turn.kpi_score is None:
            continue
        for row in store.decisions_for(turn.turn_id):
            if row.get("domain") != domain:
                continue
            arm = row.get("chosen")
            bucket = _bucket_from_row(row)
            if arm is None or bucket is None:
                continue
            age_ms = max(0, now - (turn.started_at or now))
            decay = 0.5 ** (age_ms / half_life_ms)
            key = (bucket, arm)
            weighted_sum[key] = weighted_sum.get(key, 0.0) + decay * turn.kpi_score
            weight_total[key] = weight_total.get(key, 0.0) + decay
            counts[key] = counts.get(key, 0) + 1
    return {
        key: (weighted_sum[key] / weight_total[key] if weight_total[key] else 0.0, counts[key])
        for key in weighted_sum
    }


def test_success_table_matches_old_full_scan(tmp_path: Path) -> None:
    """Regression test for the unbounded-scan fix: success_table's new bounded SQL join must
    return the identical (bucket, arm) -> (decayed mean, count) table the old per-turn
    decisions_for() scan produced, for a fixture spanning multiple buckets, arms, and a decision
    in a domain that must be excluded."""
    cfg_path = _write_toml(tmp_path / "governor.toml", _base_toml(decay_half_life_days=14.0))
    store = Store(tmp_path / "store")
    now = int(time.time() * 1000)
    day_ms = 86_400_000

    samples = [
        # (turn_id, domain, bucket, arm, kpi, age_days)
        ("t0", "model_tier", "python:fix:s:top", "fast", 0.8, 0),
        ("t1", "model_tier", "python:fix:s:top", "fast", 0.6, 2),
        ("t2", "model_tier", "python:fix:s:top", "balanced", 0.4, 1),
        ("t3", "model_tier", "go:test:m:sub", "deep", 0.9, 5),
        ("t4", "skills", "python:fix:s:top", "fast", 0.95, 0),  # different domain, must be excluded
    ]
    for turn_id, domain, bucket, arm, kpi, age_days in samples:
        store.upsert_turn(
            Turn(
                turn_id=turn_id,
                session_id="s",
                source="crush",
                host="h",
                started_at=now - age_days * day_ms,
                kpi_score=kpi,
            )
        )
        store.add_decision(
            GovernorDecision(
                turn_id=turn_id,
                domain=domain,
                chosen=arm,
                shadow=1,
                alternatives={},
                features={"bucket": bucket},
            )
        )

    gov = Governor(config_path=cfg_path, store=store, rng=random.Random(0))

    new_table = gov.success_table("model_tier")
    old_table = _old_success_table(store, "model_tier", half_life_days=14.0)

    assert set(new_table) == set(old_table)
    for key in old_table:
        old_mean, old_count = old_table[key]
        new_mean, new_count = new_table[key]
        assert new_count == old_count
        assert new_mean == pytest.approx(old_mean)


def test_success_table_ignores_unknown_arms(tmp_path: Path) -> None:
    cfg_path = _write_toml(tmp_path / "governor.toml", _base_toml())
    store = Store(tmp_path / "store")
    now = int(time.time() * 1000)
    store.upsert_turn(
        Turn(turn_id="t1", session_id="s", source="crush", host="h", started_at=now, kpi_score=0.5)
    )
    store.add_decision(
        GovernorDecision(
            turn_id="t1",
            domain="model_tier",
            chosen="ghost_arm",
            shadow=1,
            alternatives={},
            features={"bucket": "python:fix:s:top"},
        )
    )

    gov = Governor(config_path=cfg_path, store=store, rng=random.Random(0))
    table = gov.success_table("model_tier")  # must not raise
    assert table[("python:fix:s:top", "ghost_arm")] == (pytest.approx(0.5), 1)


# ---------- clamp_config ----------


def test_clamp_config_violations() -> None:
    cfg = load_config(SHIPPED_CONFIG)
    cfg["governor"]["epsilon"] = 5.0
    clamped, violations = clamp_config(cfg)
    assert clamped["governor"]["epsilon"] == 1.0
    assert any("governor.epsilon" in v for v in violations)


def test_clamp_config_clamps_dynamic_tables() -> None:
    cfg = load_config(SHIPPED_CONFIG)
    cfg["arms"]["model_tier"]["fast"]["cost"] = 999.0
    cfg["weights"]["model_tier"]["frontier"]["prompt_tokens"] = 10.0
    clamped, violations = clamp_config(cfg)
    assert clamped["arms"]["model_tier"]["fast"]["cost"] == 5.0
    assert clamped["weights"]["model_tier"]["frontier"]["prompt_tokens"] == 2.0
    assert len(violations) == 2


def test_clamp_config_leaves_valid_values_untouched() -> None:
    cfg = load_config(SHIPPED_CONFIG)
    clamped, violations = clamp_config(cfg)
    assert violations == []
    assert clamped == cfg


def test_clamp_config_clamps_skills_cost() -> None:
    """[arms.skills] costs get the same (0.0, 5.0) bound [arms.model_tier] costs do — added
    alongside the shipped [arms.skills] table so a hand-edit here is caught like every other
    arm cost, not silently passed through."""
    cfg = load_config(SHIPPED_CONFIG)
    cfg["arms"]["skills"]["suppress"]["cost"] = -1.0
    clamped, violations = clamp_config(cfg)
    assert clamped["arms"]["skills"]["suppress"]["cost"] == 0.0
    assert any("arms.skills.suppress.cost" in v for v in violations)


def test_shipped_config_has_real_skills_arms() -> None:
    """[arms.skills] used to be entirely absent, which sent every skills decision through
    Governor.choose's config_error fallback (a single fake "default" arm) whenever a caller
    didn't supply its own per-turn arms list — record_shadow_decisions is exactly such a caller.
    A real table here means the skills domain can accumulate genuine shadow evidence."""
    cfg = load_config(SHIPPED_CONFIG)
    arms = cfg["arms"]["skills"]
    assert set(arms) == {"suppress", "match_only", "match_plus_related", "always_on"}
    for name, entry in arms.items():
        assert isinstance(entry["cost"], (int, float)), name
        assert 0.0 <= entry["cost"] <= 5.0, name


# ---------- load_kpi_tuning ----------


def test_load_kpi_tuning_reads_shipped_config() -> None:
    weights, thresholds = load_kpi_tuning(SHIPPED_CONFIG)
    assert weights == {
        "completion": 0.20,
        "tool_reliability": 0.20,
        "efficiency": 0.15,
        "focus": 0.15,
        "context_health": 0.15,
        "autonomy": 0.15,
    }
    assert thresholds == {"tool_error_rate": 0.25, "low_kpi": 0.50}


def test_load_kpi_tuning_reflects_hand_edits(tmp_path: Path) -> None:
    """This is the nightly reviewer's whole mechanism: it edits [kpi.weights]/[thresholds] in
    governor.toml, and the next rollup must score with the edited values, not the hardcoded
    kpi.DEFAULT_WEIGHTS/DEFAULT_THRESHOLDS."""
    text = (
        _base_toml()
        .replace("completion = 0.2", "completion = 0.9")
        .replace("tool_error_rate = 0.25", "tool_error_rate = 0.75")
    )
    cfg_path = _write_toml(tmp_path / "governor.toml", text)

    weights, thresholds = load_kpi_tuning(cfg_path)

    assert weights["completion"] == 0.9
    assert thresholds["tool_error_rate"] == 0.75


def test_load_kpi_tuning_clamps_out_of_range_edits(tmp_path: Path) -> None:
    text = _base_toml().replace("completion = 0.2", "completion = 5.0")
    cfg_path = _write_toml(tmp_path / "governor.toml", text)

    weights, _thresholds = load_kpi_tuning(cfg_path)

    assert weights["completion"] == 1.0  # clamped to kpi.weights.* bound [0.0, 1.0]


def test_load_kpi_tuning_falls_back_to_defaults_on_missing_file(tmp_path: Path) -> None:
    weights, thresholds = load_kpi_tuning(tmp_path / "does-not-exist.toml")
    assert weights == DEFAULT_WEIGHTS
    assert thresholds == DEFAULT_THRESHOLDS


def test_load_kpi_tuning_falls_back_to_defaults_on_malformed_toml(tmp_path: Path) -> None:
    cfg_path = _write_toml(tmp_path / "governor.toml", "not [ valid toml")
    weights, thresholds = load_kpi_tuning(cfg_path)
    assert weights == DEFAULT_WEIGHTS
    assert thresholds == DEFAULT_THRESHOLDS


def test_load_kpi_tuning_falls_back_per_table_when_one_is_missing(tmp_path: Path) -> None:
    """A governor.toml with [kpi.weights] but no [thresholds] (or vice versa) must not zero out
    the table that IS present — each table falls back independently."""
    text = "\n".join(
        line
        for line in _base_toml().splitlines()
        if "tool_error_rate" not in line and "low_kpi" not in line and line != "[thresholds]"
    )
    cfg_path = _write_toml(tmp_path / "governor.toml", text)

    weights, thresholds = load_kpi_tuning(cfg_path)

    assert weights["completion"] == 0.2  # present in the file, not the default
    assert thresholds == DEFAULT_THRESHOLDS  # table absent -> fallback


# ---------- shadow mode ----------


def test_shadow_default_shipped_config() -> None:
    gov = Governor(config_path=SHIPPED_CONFIG, rng=random.Random(0))
    for domain in GOVERNOR_DOMAINS:
        assert gov.enabled(domain) is False


def test_choose_from_shipped_config_is_shadow() -> None:
    gov = Governor(config_path=SHIPPED_CONFIG, rng=random.Random(0))
    choice = gov.choose("model_tier", _features())
    assert choice.shadow is True


def test_choose_skills_from_shipped_config_no_longer_config_errors() -> None:
    """Before [arms.skills] existed, every no-arms-supplied skills choice (e.g.
    record_shadow_decisions' batch replay) fell back to reason="config_error" and a single fake
    "default" arm — real evidence could never accumulate. With a real table, an ordinary
    argmax/explore choice comes back instead."""
    gov = Governor(config_path=SHIPPED_CONFIG, rng=random.Random(0))
    choice = gov.choose("skills", _features())
    assert choice.reason != "config_error"
    assert choice.chosen in {"suppress", "match_only", "match_plus_related", "always_on"}


# ---------- weights_version ----------


def test_weights_version_changes_with_file(tmp_path: Path) -> None:
    cfg_path = _write_toml(tmp_path / "governor.toml", _base_toml())
    v1 = Governor(config_path=cfg_path, rng=random.Random(0)).weights_version

    _write_toml(cfg_path, _base_toml(epsilon=0.5))
    v2 = Governor(config_path=cfg_path, rng=random.Random(0)).weights_version

    assert v1 != v2


def test_weights_version_stable_for_same_content(tmp_path: Path) -> None:
    cfg_path = _write_toml(tmp_path / "governor.toml", _base_toml())
    v1 = Governor(config_path=cfg_path, rng=random.Random(0)).weights_version
    v2 = Governor(config_path=cfg_path, rng=random.Random(1)).weights_version
    assert v1 == v2


# ---------- graduation_report ----------


def test_graduation_report(tmp_path: Path) -> None:
    cfg_path = _write_toml(tmp_path / "governor.toml", _base_toml(min_samples=3))
    store = Store(tmp_path / "store")
    now = int(time.time() * 1000)
    arms = ["fast", "balanced", "deep", "frontier"]

    tid = 0
    for arm in arms:
        count = 2 if arm == "frontier" else 3
        for i in range(count):
            tid += 1
            turn_id = f"t{tid}"
            agree = i % 2 == 0
            actual_tier = arm if agree else ("fast" if arm != "fast" else "balanced")
            kpi = 0.9 if agree else 0.3
            store.upsert_turn(
                Turn(
                    turn_id=turn_id,
                    session_id="s1",
                    source="crush",
                    host="h1",
                    started_at=now,
                    tier=actual_tier,
                    kpi_score=kpi,
                )
            )
            store.add_decision(
                GovernorDecision(
                    turn_id=turn_id,
                    domain="model_tier",
                    chosen=arm,
                    shadow=1,
                    alternatives={},
                    features={"bucket": "python:fix:s:top"},
                )
            )

    gov = Governor(config_path=cfg_path, store=store, rng=random.Random(0))
    report = gov.graduation_report("model_tier", days=7)

    assert report["arm_counts"]["fast"] == 3
    assert report["arm_counts"]["frontier"] == 2
    assert report["all_arms_cleared"] is False
    assert report["mean_kpi_delta"] is not None
    assert report["mean_kpi_delta"] > 0


def test_graduation_report_no_store() -> None:
    gov = Governor(config_path=SHIPPED_CONFIG, rng=random.Random(0))
    report = gov.graduation_report("model_tier")
    assert report["decisions"] == 0
    assert report["all_arms_cleared"] is False


def test_graduation_report_does_not_scan_per_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression test for the N+1 fix: graduation_report used to call iter_turns then
    decisions_for(turn_id) per turn — the exact pattern success_table was refactored away from
    via Store.scored_decisions (one indexed join). If graduation_report still touched either of
    those per-turn paths this would raise instead of returning the correct report."""
    cfg_path = _write_toml(tmp_path / "governor.toml", _base_toml(min_samples=3))
    store = Store(tmp_path / "store")
    now = int(time.time() * 1000)
    store.upsert_turn(
        Turn(
            turn_id="t1",
            session_id="s1",
            source="crush",
            host="h1",
            started_at=now,
            tier="fast",
            kpi_score=0.9,
        )
    )
    store.add_decision(
        GovernorDecision(
            turn_id="t1",
            domain="model_tier",
            chosen="fast",
            shadow=1,
            alternatives={},
            features={"bucket": "python:fix:s:top"},
        )
    )

    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("graduation_report must not scan per-turn via iter_turns")

    monkeypatch.setattr(store, "iter_turns", _boom)
    monkeypatch.setattr(store, "decisions_for", _boom)

    gov = Governor(config_path=cfg_path, store=store, rng=random.Random(0))
    report = gov.graduation_report("model_tier", days=7)  # must not raise

    assert report["decisions"] == 1
    assert report["arm_counts"] == {"fast": 1}


def test_graduation_report_ignores_unscored_decisions(tmp_path: Path) -> None:
    """A decision on a turn that hasn't been scored yet proves nothing about the arm's quality —
    scored_decisions filters on kpi_score IS NOT NULL, so it must not inflate arm_counts."""
    cfg_path = _write_toml(tmp_path / "governor.toml", _base_toml(min_samples=3))
    store = Store(tmp_path / "store")
    now = int(time.time() * 1000)
    store.upsert_turn(
        Turn(
            turn_id="unscored",
            session_id="s1",
            source="crush",
            host="h1",
            started_at=now,
            tier="fast",
            kpi_score=None,
        )
    )
    store.add_decision(
        GovernorDecision(
            turn_id="unscored",
            domain="model_tier",
            chosen="fast",
            shadow=1,
            alternatives={},
            features={"bucket": "python:fix:s:top"},
        )
    )

    gov = Governor(config_path=cfg_path, store=store, rng=random.Random(0))
    report = gov.graduation_report("model_tier", days=7)

    assert report["decisions"] == 0
    assert report["arm_counts"] == {}


# ---------- record_shadow_decisions ----------


def _scored_turn(turn_id: str, *, started_at: int, kpi_score: float | None = 0.8) -> Turn:
    return Turn(
        turn_id=turn_id,
        session_id="s1",
        source="crush",
        host="h1",
        started_at=started_at,
        kpi_score=kpi_score,
    )


def test_record_shadow_decisions_backfills_every_domain(tmp_path: Path) -> None:
    """No production code calls choose()/record() per turn — this is the batch replay that
    stands in for that missing live call. It must cover every governor domain, including one
    (skills) that has no [arms.skills] table in this test's local config — the shipped
    governor.toml does have one as of 2026-08-21, but choose() falling back to a single
    "default" arm when a domain has none configured is still real behavior worth covering."""
    cfg_path = _write_toml(tmp_path / "governor.toml", _base_toml(min_samples=3))
    store = Store(tmp_path / "store")
    now = int(time.time() * 1000)
    store.upsert_turn(_scored_turn("t1", started_at=now))
    store.upsert_turn(_scored_turn("t2", started_at=now))

    gov = Governor(config_path=cfg_path, store=store, rng=random.Random(0))
    recorded = gov.record_shadow_decisions(since_ms=now - 1000)

    assert recorded == 2 * len(GOVERNOR_DOMAINS)
    for turn_id in ("t1", "t2"):
        domains = {d["domain"] for d in store.decisions_for(turn_id)}
        assert domains == set(GOVERNOR_DOMAINS)
        for decision in store.decisions_for(turn_id):
            assert decision["shadow"] == 1  # every domain ships in shadow in this config


def test_record_shadow_decisions_is_idempotent(tmp_path: Path) -> None:
    """A turn that already has a decision for a domain is left alone on a rerun (Store's PK is
    (turn_id, domain), but the point is the nightly rollup shouldn't waste work re-deciding
    every turn ever scored on every run — only newly-scored ones)."""
    cfg_path = _write_toml(tmp_path / "governor.toml", _base_toml(min_samples=3))
    store = Store(tmp_path / "store")
    now = int(time.time() * 1000)
    store.upsert_turn(_scored_turn("t1", started_at=now))

    gov = Governor(config_path=cfg_path, store=store, rng=random.Random(0))
    first = gov.record_shadow_decisions(since_ms=now - 1000)
    second = gov.record_shadow_decisions(since_ms=now - 1000)

    assert first == len(GOVERNOR_DOMAINS)
    assert second == 0


def test_record_shadow_decisions_ignores_unscored_turns(tmp_path: Path) -> None:
    """A decision on an unscored turn can never contribute to success_table/graduation_report
    (both filter on kpi_score IS NOT NULL) — deciding for it is pure waste, so it's skipped."""
    cfg_path = _write_toml(tmp_path / "governor.toml", _base_toml(min_samples=3))
    store = Store(tmp_path / "store")
    now = int(time.time() * 1000)
    store.upsert_turn(_scored_turn("unscored", started_at=now, kpi_score=None))

    gov = Governor(config_path=cfg_path, store=store, rng=random.Random(0))
    recorded = gov.record_shadow_decisions(since_ms=now - 1000)

    assert recorded == 0
    assert store.decisions_for("unscored") == []


def test_record_shadow_decisions_no_store() -> None:
    gov = Governor(config_path=SHIPPED_CONFIG, rng=random.Random(0))
    assert gov.record_shadow_decisions(since_ms=0) == 0


# ---------- never raise ----------


def test_choose_never_raises_on_missing_config(tmp_path: Path) -> None:
    gov = Governor(config_path=tmp_path / "missing.toml", rng=random.Random(0))
    choice = gov.choose("model_tier", _features())
    assert choice.reason == "config_error"
    assert choice.shadow is True
    assert isinstance(choice.chosen, str)


def test_choose_never_raises_on_malformed_toml(tmp_path: Path) -> None:
    bad_path = tmp_path / "bad.toml"
    bad_path.write_text("this is not [valid toml")
    gov = Governor(config_path=bad_path, rng=random.Random(0))
    choice = gov.choose("skills", _features(), arms=["skillA", "skillB"])
    assert choice.reason == "config_error"
    assert choice.chosen == "skillA"


def test_choose_never_raises_with_no_arms_configured(tmp_path: Path) -> None:
    cfg_path = _write_toml(tmp_path / "governor.toml", _base_toml())
    gov = Governor(config_path=cfg_path, rng=random.Random(0))
    choice = gov.choose("skills", _features())  # no [arms.skills] in config, none passed
    assert choice.reason == "config_error"
    assert choice.chosen == "default"


def test_record_persists_propensity(tmp_path: Path) -> None:
    """Without explored/epsilon on the row, P(a|x) is unrecoverable and every off-policy
    estimator is undefined on these decisions — not merely imprecise."""
    cfg_path = _write_toml(tmp_path / "governor.toml", _base_toml(epsilon=1.0, min_samples=1000))
    store = Store(tmp_path / "store")

    gov = Governor(config_path=cfg_path, store=store, rng=random.Random(42))
    features = _features(prompt_tokens=8000)
    choice = gov.choose("model_tier", features)
    gov.record("t-prop", choice, features)

    row = store.conn.execute(
        "SELECT chosen, explored, reason, epsilon FROM governor_decisions WHERE turn_id = ?",
        ("t-prop",),
    ).fetchone()

    assert row["chosen"] == choice.chosen
    assert row["explored"] == (1 if choice.explored else 0)
    assert row["reason"] == choice.reason
    assert row["epsilon"] == 1.0


def test_epsilon_recorded_is_the_value_in_force(tmp_path: Path) -> None:
    """Read back from the row, not from governor.toml: the toml is hand-edited between runs, so
    the file cannot answer what epsilon was at the moment of the draw."""
    cfg_path = _write_toml(tmp_path / "governor.toml", _base_toml(epsilon=0.0))
    store = Store(tmp_path / "store")
    gov = Governor(config_path=cfg_path, store=store, rng=random.Random(1))
    features = _features(prompt_tokens=100)
    choice = gov.choose("model_tier", features)
    gov.record("t-eps", choice, features)

    assert choice.explored is False
    row = store.conn.execute(
        "SELECT explored, reason, epsilon FROM governor_decisions WHERE turn_id = ?", ("t-eps",)
    ).fetchone()
    assert (row["explored"], row["reason"], row["epsilon"]) == (0, "argmax", 0.0)
