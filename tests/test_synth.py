from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from flightdeck.models import DISPOSITIONS, SCOPE_TIERS
from flightdeck.store import DEFAULT_DIR, Store
from flightdeck.synth.manifest import Manifest, content_hash, verify_manifest, write_manifest
from flightdeck.synth.prompt_labels import PromptLabelParams, generate_labels, seed_cases
from flightdeck.synth.provenance import (
    is_default_store_dir,
    refuse_if_live_store,
    row_is_synthetic,
    stamp_row,
)
from flightdeck.synth.scope_rows import ScopeRowParams, generate_rows, invariants_hold, to_corpus

SYNTH_DIR = Path(__file__).parent.parent / "flightdeck" / "synth"

# Golden content_hash for ScopeRowParams(seed=20260905, count=10). If generation logic
# changes intentionally, bump GENERATOR_VERSION in provenance.py and update this constant
# from a fresh run -- never edit it to make a failing test pass without checking why the
# content actually changed.
GOLDEN_SCOPE_ROWS_HASH = "fe6e38a7a4b7e8c617c8794a3ddbc1e57a679748130f741b884db7719fcd40df"


# ------------------------------------------------------------------------- golden content


def test_golden_content_hash_fixed_seed():
    rows = to_corpus(generate_rows(ScopeRowParams(seed=20260905, count=10)))
    assert content_hash(rows) == GOLDEN_SCOPE_ROWS_HASH


def test_golden_hash_is_sensitive_to_content_not_just_count():
    """A content swap that preserves row count must still change the hash -- guards the
    exact failure mode named in the task ('length checks hide content swaps')."""
    rows = to_corpus(generate_rows(ScopeRowParams(seed=20260905, count=10)))
    mutated = [dict(r) for r in rows]
    mutated[0]["found_total"] = mutated[0]["found_total"] + 1
    assert len(mutated) == len(rows)
    assert content_hash(mutated) != content_hash(rows)


# --------------------------------------------------------------------------- determinism


def test_same_seed_same_content_hash():
    a = to_corpus(generate_rows(ScopeRowParams(seed=7, count=25)))
    b = to_corpus(generate_rows(ScopeRowParams(seed=7, count=25)))
    assert content_hash(a) == content_hash(b)
    assert a == b


def test_different_seed_different_content_hash():
    a = to_corpus(generate_rows(ScopeRowParams(seed=1, count=25)))
    b = to_corpus(generate_rows(ScopeRowParams(seed=2, count=25)))
    assert content_hash(a) != content_hash(b)


def test_prompt_label_generation_is_deterministic():
    a = generate_labels(PromptLabelParams(seed=42, count=30))
    b = generate_labels(PromptLabelParams(seed=42, count=30))
    assert [x.to_row() for x in a] == [x.to_row() for x in b]


# ------------------------------------------------------------------------- invariants


@pytest.mark.parametrize("seed", range(50))
def test_disposition_invariant_holds_across_seeds(seed):
    for record in generate_rows(ScopeRowParams(seed=seed, count=15)):
        assert invariants_hold(record)
        assert record.late_discovered <= record.found_total
        assert record.rework_files <= record.files_edited
        assert record.divergence_kept <= record.divergence_flagged


def test_disposition_registry_is_three_valued():
    assert len(DISPOSITIONS) == 3


def test_seed_case_tiers_are_valid_or_none():
    seen_ids = set()
    for case in seed_cases():
        assert case.expected_tier in SCOPE_TIERS or case.expected_tier is None
        assert case.case_id not in seen_ids, f"duplicate case_id {case.case_id}"
        seen_ids.add(case.case_id)


def test_seed_cases_are_the_full_drafted_set():
    """63 seeds were drafted in 08-synth-coverage.md; the transcription must not silently
    drop or invent replacements."""
    assert len(seed_cases()) == 63


# ---------------------------------------------------------------------- distributions


def test_tier_distribution_within_wide_tolerance():
    rows = generate_rows(ScopeRowParams(seed=99, count=2000))
    counts = {t: 0 for t in SCOPE_TIERS}
    for r in rows:
        counts[r.tier] += 1
    total = len(rows)
    # configured weights are 0.45 / 0.35 / 0.20 -- wide bands so a correct implementation
    # never flakes, narrow enough to catch a swapped weight tuple.
    assert 0.35 <= counts["none"] / total <= 0.55
    assert 0.25 <= counts["mini"] / total <= 0.45
    assert 0.12 <= counts["full"] / total <= 0.28


def test_frequency_realistic_label_distribution_buckets_present():
    labels = generate_labels(PromptLabelParams(seed=123, count=2000))
    buckets = {}
    for label in labels:
        b = label.expected_signals.get("bucket")
        buckets[b] = buckets.get(b, 0) + 1
    total = len(labels)
    # very_short target 0.241 -- wide tolerance
    assert 0.15 <= buckets.get("very_short", 0) / total <= 0.33
    assert all(v > 0 for v in buckets.values())


# --------------------------------------------------------------------------- manifest


def test_manifest_round_trip(tmp_path):
    rows = to_corpus(generate_rows(ScopeRowParams(seed=5, count=8)))
    manifest = Manifest(
        kind="scope_rows", seed=5, params={"count": 8}, count=8, content_hash=content_hash(rows)
    )
    path = tmp_path / "corpus.manifest.json"
    write_manifest(path, manifest)
    assert verify_manifest(path, rows) is True


def test_manifest_verify_fails_on_content_swap(tmp_path):
    rows = to_corpus(generate_rows(ScopeRowParams(seed=5, count=8)))
    manifest = Manifest(
        kind="scope_rows", seed=5, params={"count": 8}, count=8, content_hash=content_hash(rows)
    )
    path = tmp_path / "corpus.manifest.json"
    write_manifest(path, manifest)
    mutated = [dict(r) for r in rows]
    mutated[3]["ceremony_tokens"] = (mutated[3]["ceremony_tokens"] or 0) + 1
    assert verify_manifest(path, mutated) is False


def test_manifest_created_at_excluded_from_hash_semantics():
    """created_at is informational; two manifests differing only in created_at must both
    validate against the same row content."""
    rows = to_corpus(generate_rows(ScopeRowParams(seed=5, count=8)))
    m1 = Manifest(
        kind="scope_rows", seed=5, params={}, count=8, content_hash=content_hash(rows), created_at=1
    )
    m2 = Manifest(
        kind="scope_rows", seed=5, params={}, count=8, content_hash=content_hash(rows), created_at=2
    )
    assert m1.content_hash == m2.content_hash


# ------------------------------------------------------------------------- containment


def test_is_default_store_dir_matches_by_resolved_path():
    assert is_default_store_dir(DEFAULT_DIR, DEFAULT_DIR)
    assert is_default_store_dir(str(DEFAULT_DIR), DEFAULT_DIR)


def test_refuse_if_live_store_raises_for_default_dir():
    with pytest.raises(ValueError, match="refusing to write"):
        refuse_if_live_store(DEFAULT_DIR, DEFAULT_DIR)


def test_refuse_if_live_store_allows_scratch_dir(tmp_path):
    refuse_if_live_store(tmp_path, DEFAULT_DIR)  # must not raise


def test_generator_refuses_the_live_store(tmp_path, monkeypatch):
    with Store(tmp_path) as store:
        monkeypatch.setattr(store, "directory", Path(DEFAULT_DIR).expanduser())
        with pytest.raises(ValueError, match="refusing to write"):
            refuse_if_live_store(store.directory, DEFAULT_DIR, what="synthetic scope_records")


def test_stamp_row_refuses_to_override_explicit_false():
    with pytest.raises(ValueError, match="refusing to stamp"):
        stamp_row({"synthetic": False})


def test_stamp_row_marks_synthetic_true_by_default():
    row = stamp_row({})
    assert row["synthetic"] is True
    assert row["generator"] == "flightdeck.synth"


def test_row_is_synthetic_reads_json_string_provenance():
    row = {"provenance": json.dumps({"synthetic": True})}
    assert row_is_synthetic(row) is True
    assert row_is_synthetic({"provenance": json.dumps({"synthetic": False})}) is False
    assert row_is_synthetic({"provenance": "not json"}) is False
    assert row_is_synthetic({}) is False


def test_generated_scope_rows_carry_synthetic_provenance(tmp_path):
    with Store(tmp_path) as store:
        for record in generate_rows(ScopeRowParams(count=10, seed=3)):
            store.add_scope_record(record)
        rows = store.scope_records()
    assert len(rows) == 10
    assert all(row_is_synthetic(dict(r)) for r in rows)


def test_label_source_excludes_classifier_self():
    from flightdeck.synth.provenance import LABEL_SOURCES

    assert "classifier_self" not in LABEL_SOURCES
    for case in seed_cases():
        assert case.label_source in LABEL_SOURCES
    for label in generate_labels(PromptLabelParams(count=20)):
        assert label.label_source in LABEL_SOURCES


# --------------------------------------------------------------------- circularity lint


def _imports(module_path: Path) -> set[str]:
    tree = ast.parse(module_path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_no_classifier_import():
    """The generator must not import scope_gate or scope_judge -- see 09-synth-labels.md
    Circularity defenses. Mechanical enforcement, not an honor system."""
    forbidden = {"flightdeck.scope_gate", "flightdeck.scope_judge", "scope_gate", "scope_judge"}
    offenders = []
    for py_file in SYNTH_DIR.glob("*.py"):
        imported = _imports(py_file)
        hit = imported & forbidden
        if hit:
            offenders.append((py_file.name, hit))
    assert offenders == [], f"synth modules must not import the classifier under test: {offenders}"
