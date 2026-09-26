"""Snapshot, ingest and eval: the three pieces that make a scope number reproducible.

Each of these exists because a number was being reported that nobody could recompute.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from flightdeck import scope_snapshot
from flightdeck.models import ScopeRecord
from flightdeck.scope_eval import (
    MIN_ARM_ROWS,
    POPULATION,
    GoldRow,
    assign_splits,
    check_population_drift,
    cohen_kappa,
    confusion,
    evaluate,
    holdout_log_path,
    load_gold,
    population_rate,
    record_holdout_spend,
    regex_predict,
)
from flightdeck.scope_ingest import ingest, record_id_for, verify_chain
from flightdeck.scope_kpi import aggregate
from flightdeck.store import JsonlLog, Store

# ------------------------------------------------------------------ snapshot


def _corpus(tmp_path: Path, files: dict[str, str]) -> Path:
    root = tmp_path / "projects"
    for name, body in files.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body)
    return root


def test_snapshot_is_content_addressed_not_mtime_addressed(tmp_path: Path) -> None:
    """An earlier system here signed artifacts by mtime; the signature went stale across
    checkouts while content was identical, and matched while content differed."""
    root = _corpus(tmp_path, {"a/s1.jsonl": '{"x":1}\n'})
    first = scope_snapshot.take(root)
    (root / "a" / "s1.jsonl").touch()  # mtime moves, content does not
    assert scope_snapshot.take(root).corpus_sha == first.corpus_sha


def test_snapshot_sha_moves_when_content_moves(tmp_path: Path) -> None:
    root = _corpus(tmp_path, {"a/s1.jsonl": '{"x":1}\n'})
    before = scope_snapshot.take(root).corpus_sha
    (root / "a" / "s1.jsonl").write_text('{"x":2}\n')
    assert scope_snapshot.take(root).corpus_sha != before


def test_verify_separates_aged_out_from_rewritten(tmp_path: Path) -> None:
    """Retention dropping an old session is expected drift. A transcript whose CONTENT
    changed under a measurement is not, and conflating them hides the second."""
    root = _corpus(tmp_path, {"a/s1.jsonl": "one\n", "a/s2.jsonl": "two\n"})
    snap = scope_snapshot.take(root)
    (root / "a" / "s1.jsonl").unlink()
    (root / "a" / "s2.jsonl").write_text("changed\n")
    (root / "a" / "s3.jsonl").write_text("new\n")
    result = scope_snapshot.verify(snap, root)
    assert result["counts"] == {"missing": 1, "added": 1, "changed": 1}
    assert result["identical"] is False


def test_snapshot_round_trips(tmp_path: Path) -> None:
    root = _corpus(tmp_path, {"a/s1.jsonl": "one\n"})
    snap = scope_snapshot.take(root)
    path = scope_snapshot.save(snap, tmp_path)
    assert scope_snapshot.load(path).corpus_sha == snap.corpus_sha


# ------------------------------------------------------------------ ingest


def _gate_row(**kw: object) -> dict:
    row = {
        "ts": "2026-09-05T12:00:00+00:00",
        "session_id": "s1",
        "cwd": "/tmp",
        "tier": "mini",
        "reasons": ["unbounded verb"],
        "has_active_scope": False,
        "prompt_sha1": "abc123",
        "prompt_len": 40,
    }
    row.update(kw)
    return row


def _write_log(directory: Path, rows: list[dict]) -> None:
    path = directory / "scope" / "gate-log.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_ingested_rows_are_unobserved_not_passing(tmp_path: Path) -> None:
    """The gate decided; nothing watched what the scoping was worth. Calling that "pass"
    would manufacture the measurement the subsystem exists to earn."""
    _write_log(tmp_path, [_gate_row()])
    store = Store(tmp_path)
    ingest(store)
    assert [r["verdict"] for r in store.scope_records()] == ["silent"]


def test_ingest_is_idempotent(tmp_path: Path) -> None:
    _write_log(tmp_path, [_gate_row(), _gate_row(prompt_sha1="def456")])
    store = Store(tmp_path)
    assert ingest(store)["written"] == 2
    second = ingest(store)
    assert second["written"] == 0
    assert second["already_present"] == 2


def test_suppressed_decisions_are_counted_but_not_ingested(tmp_path: Path) -> None:
    """Traffic the input filter rejected is not the gate's work, and folding it into the
    tier distribution would put the filter's decisions inside the gate's numbers."""
    _write_log(
        tmp_path,
        [
            _gate_row(),
            {
                "ts": "2026-09-05T12:00:01+00:00",
                "tier": None,
                "suppressed": "harness-generated payload",
            },
        ],
    )
    store = Store(tmp_path)
    result = ingest(store)
    assert result["written"] == 1
    assert result["suppressed_not_ingested"] == 1


def test_truncated_tail_does_not_lose_earlier_rows(tmp_path: Path) -> None:
    path = tmp_path / "scope" / "gate-log.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_gate_row()) + "\n{ broken json\n")
    assert ingest(Store(tmp_path))["written"] == 1


def test_record_id_is_stable(tmp_path: Path) -> None:
    assert record_id_for(_gate_row()) == record_id_for(_gate_row())


def test_chain_verifies_in_insertion_order(tmp_path: Path) -> None:
    """Rows written inside one second are ordered by rowid at write time and are
    indistinguishable by timestamp after. Verifying by timestamp reported 7 false breaks
    on a 27-row chain that was intact."""
    rows = [_gate_row(prompt_sha1=f"h{i}") for i in range(8)]  # all the same second
    _write_log(tmp_path, rows)
    store = Store(tmp_path)
    ingest(store)
    result = verify_chain(store)
    assert result["verdict"] == "pass", result["broken"]
    assert result["checked"] == 8


def test_chain_detects_an_edited_row(tmp_path: Path) -> None:
    """The whole point. Written from the first commit and verified nowhere, the chain was
    decoration -- an edited row was undetectable."""
    _write_log(tmp_path, [_gate_row(prompt_sha1=f"h{i}") for i in range(3)])
    store = Store(tmp_path)
    ingest(store)
    store.conn.execute("UPDATE scope_records SET tier='full' WHERE tier='mini'")
    assert verify_chain(store)["verdict"] == "fail"


def test_chain_on_empty_store_is_silent(tmp_path: Path) -> None:
    assert verify_chain(Store(tmp_path))["verdict"] == "silent"


def test_store_refuses_synthetic_rows_into_the_live_store() -> None:
    """The guard belongs at the writer, not only in the generator: a synthetic row that
    arrived by any other path would read back as measurement."""
    from flightdeck.store import DEFAULT_DIR

    record = ScopeRecord(
        record_id="x",
        session_id="s",
        created_at=1,
        host="h",
        tier="mini",
        verdict="pass",
        provenance={"synthetic": True},
    )
    with pytest.raises(ValueError, match="refusing to write a synthetic"):
        Store(DEFAULT_DIR).add_scope_record(record)


# ------------------------------------------------------------------ kpi verdict


def _record(**kw: object) -> dict:
    base = {
        "verdict": "pass",
        "tier": "mini",
        "found_total": 4,
        "late_discovered": 1,
        "files_edited": 2,
        "rework_files": 0,
        "questions_asked": 1,
        "questions_valuable": 1,
        "ceremony_tokens": 100,
        "ceremony_ms": 1000,
        "turns_to_done": 5,
        "corrections": 0,
        "assumptions": 1,
        "assumptions_overridden": 0,
        "divergence_flagged": 1,
        "divergence_kept": 1,
    }
    base.update(kw)
    return base


def test_one_pass_among_many_failures_does_not_read_pass() -> None:
    """The old rule returned pass whenever ANY row was healthy."""
    rows = [_record(verdict="pass")] + [_record(verdict="fail") for _ in range(99)]
    assert aggregate(rows)["verdict"] == "fail"


def test_a_window_of_only_unobserved_rows_is_silent_not_failing() -> None:
    """Thin gate-log rows are not evidence of failure. Reporting fail here would make a
    working ingest look like a broken system."""
    summary = aggregate([_record(verdict="silent") for _ in range(27)])
    assert summary["verdict"] == "silent"
    assert summary["unobserved_rows"] == 27
    assert "none with an observed outcome" in summary["verdict_reason"]


def test_small_windows_name_themselves() -> None:
    assert aggregate([_record()])["sample_warning"] is not None


def test_metrics_are_stratified_by_tier() -> None:
    """Pooling across tiers reports a tier-mix shift as a quality change."""
    summary = aggregate([_record(tier="mini"), _record(tier="full"), _record(tier="full")])
    assert summary["by_tier"]["full"]["rows"] == 2
    assert summary["by_tier"]["mini"]["rows"] == 1


# ------------------------------------------------------------------ eval


def test_splits_never_share_a_session() -> None:
    """One session supplies a disproportionate share of the gold rows. A row-wise split
    puts it on both sides and the held-out score reports memorisation."""
    rows = load_gold()
    train = {r.session for r in rows if r.split == "train"}
    held = {r.session for r in rows if r.split == "holdout"}
    assert train and held
    assert not (train & held)


def test_splits_are_deterministic() -> None:
    assert [r.split for r in load_gold()] == [r.split for r in load_gold()]


def test_stratum_weights_reflect_the_population() -> None:
    """The fixture is a near-census of regex-positive turns but only a small sample of
    the rest. Unweighted metrics off it overstate regex recall several-fold."""
    rows = load_gold()
    positive = next(r for r in rows if r.regex)
    negative = next(r for r in rows if not r.regex)
    assert negative.weight > positive.weight * 5


def test_population_recall_is_far_below_in_sample_recall() -> None:
    """The measured consequence of the stratification, asserted so it cannot silently
    revert to the flattering number."""
    rows = load_gold()
    report = evaluate(rows, regex_predict, name="regex", split="train")
    assert report["in_sample"]["recall"] > report["population"]["recall"] * 2


def test_abstention_is_not_a_negative() -> None:
    """Scoring four truncated judge replies as NEW once understated recall by 11 points."""
    rows = [GoldRow("a", "t", 1, 0, "s1"), GoldRow("b", "t", 1, 0, "s1")]
    matrix = confusion(rows, lambda r: None)
    assert matrix["abstained"] == 2
    assert matrix["scored"] == 0
    assert matrix["fn"] == 0


def test_abstentions_leave_the_rate_denominator() -> None:
    rows = [GoldRow("a", "t", 1, 1, "s1"), GoldRow("b", "t", 0, 1, "s1")]
    rate = population_rate(rows, lambda r: 1 if r.id == "a" else None)
    assert rate["estimated_rate"] == 1.0
    assert rate["abstained_weight"] > 0


def test_kappa_is_zero_for_a_constant_classifier() -> None:
    """Raw agreement on a skewed class is mostly the skew: always answering "no" scores
    95% agreement on a 5% base rate."""
    assert cohen_kappa(tp=0, fp=0, tn=95, fn=5) == 0.0


def test_tiny_arm_is_silent() -> None:
    rows = [GoldRow(str(i), "t", 0, 0, "s1") for i in range(MIN_ARM_ROWS - 1)]
    rows = assign_splits(rows, holdout=0.0)
    assert evaluate(rows, regex_predict, name="r", split="train")["verdict"] == "silent"


def test_classifier_runs_once_per_row(tmp_path: Path) -> None:
    """The bootstrap resamples 2,000 times. Calling the classifier inside each draw turned
    a 68-item judge evaluation into ~136,000 network calls and it never finished."""
    calls: list[str] = []

    def counting(row: GoldRow) -> int:
        calls.append(row.id)
        return 0

    rows = assign_splits([GoldRow(str(i), "t", 0, 0, f"s{i % 4}") for i in range(24)], holdout=0.0)
    train = [r for r in rows if r.split == "train"]
    evaluate(rows, counting, name="counting", split="train")
    assert len(calls) == len(set(calls)) == len(train)


def test_holdout_spend_is_logged_and_repeats_are_flagged(tmp_path: Path) -> None:
    """A held-out set consulted repeatedly while a prompt is tuned is a training set with
    extra steps; the log is what makes that visible afterwards."""
    report = {"rows": 10, "population": {}, "rate": {}}
    first = record_holdout_spend(tmp_path, classifier="judge", version="v3", report=report)
    second = record_holdout_spend(tmp_path, classifier="judge", version="v3", report=report)
    assert first["independent"] is True
    assert second["independent"] is False
    assert "spent 2 times" in second["warning"]


# ------------------------------------------------------------------ defect A: per-class arm floor


def test_arm_floor_checks_class_arms_not_just_split_size() -> None:
    """A split can clear MIN_ARM_ROWS in total while one predicted class is starved."""
    # 30 rows total (clears the split floor), but only 5 are predicted positive by regex.
    positives = [GoldRow(f"p{i}", "please undo that and redo it", 1, 1, f"s{i}") for i in range(5)]
    negatives = [GoldRow(f"n{i}", "just some ordinary text", 0, 0, f"s{i + 5}") for i in range(25)]
    rows = assign_splits(positives + negatives, holdout=0.0)
    report = evaluate(rows, regex_predict, name="regex", split="train")
    assert len(rows) >= MIN_ARM_ROWS  # the split-size floor alone would have passed this
    assert report["verdict"] == "silent"
    assert report["short_arms"]


# ------------------------------------------------------------------ defect B: population drift


def test_population_drift_check_fires_on_a_wrong_pin() -> None:
    """The check must actually be able to fail, not just always agree with the live corpus."""
    wrong = {"regex1": POPULATION["regex1"] * 5, "regex0": POPULATION["regex0"]}
    result = check_population_drift(pinned=wrong, tolerance=0.02)
    assert result["verdict"] == "drift"
    assert "regex1" in result["drifted_strata"]


def test_population_drift_check_passes_a_matching_pin() -> None:
    result = check_population_drift(pinned=POPULATION, tolerance=1.0)
    assert result["verdict"] == "ok"
    assert result["drifted_strata"] == []


# ------------------------------------------------------------------ defect C: mandatory holdout log


def test_evaluate_on_holdout_logs_spend_with_no_cli_involved(tmp_path: Path, monkeypatch) -> None:
    """Calling evaluate(split="holdout") directly -- no CLI, no explicit spend call -- must
    still append to the holdout log. The log is the peek-prevention mechanism, not a
    convention any caller can skip."""
    monkeypatch.setattr("flightdeck.store.DEFAULT_DIR", tmp_path)
    rows = load_gold()
    report = evaluate(rows, regex_predict, name="regex", split="holdout")
    assert "holdout_spend" in report
    log_path = holdout_log_path(tmp_path)
    assert log_path.exists()
    assert json.loads(log_path.read_text().splitlines()[0])["classifier"] == "regex"


def test_regex_holdout_version_is_a_content_hash_not_a_literal(tmp_path: Path) -> None:
    """The regex arm used to log every run under the hardcoded literal "v3", the same
    string the judge's real PROMPT_VERSION happened to be -- indistinguishable in the log
    from a repeat spend of the same classifier after the regex logic changed."""
    rows = load_gold()
    report = evaluate(rows, regex_predict, name="regex", split="holdout", spend_dir=tmp_path)
    version = report["holdout_spend"]["version"]
    assert version != "v3"
    assert version.startswith("regex-")


# ------------------------------------------------------- mirror segment units


def test_scope_record_mirrors_into_its_own_day(tmp_path: Path) -> None:
    """A scope_record's jsonl mirror is partitioned by the day it happened.

    ScopeRecord.created_at is whole seconds; JsonlLog.path_for takes milliseconds. Handing
    it over unconverted filed all 27 live scope_records under events-1970-01-21.jsonl. The
    mirror is what replay reads, so this was not a cosmetic filename bug -- it is the same
    seconds-vs-ms slip that silently empties every windowed query in this fleet.
    """
    created_at = 1788628719  # 2026-09-05, in seconds
    store = Store(tmp_path)
    store.add_scope_record(
        ScopeRecord(
            record_id="unit-check",
            session_id="s",
            created_at=created_at,
            host="h",
            tier="mini",
            verdict="pass",
        )
    )

    segments = sorted(p.name for p in tmp_path.glob("events-*.jsonl"))
    assert segments == ["events-2026-09-05.jsonl"], segments
    assert not list(tmp_path.glob("events-1970-*.jsonl"))


def test_jsonl_path_for_is_milliseconds() -> None:
    """JsonlLog.path_for's argument is epoch MILLISECONDS -- pinned, because the one caller
    that forgot produced a 56-year-old segment and nothing failed.

    Asserting the seconds case lands in 1970 is the point: it is the exact wrong answer the
    bug produced, so this test fails if someone "helpfully" makes path_for accept either.
    """
    seconds = 1788628719
    log = JsonlLog(Path("/tmp/does-not-need-to-exist-for-path-only"))
    assert log.path_for(seconds).name == "events-1970-01-21.jsonl"
    assert log.path_for(seconds * 1000).name == "events-2026-09-05.jsonl"
