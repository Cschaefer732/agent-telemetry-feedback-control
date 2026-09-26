from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from flightdeck import ingest_transcript
from flightdeck.ingest_transcript import (
    Window,
    _iso_to_ms,
    attribute_transcript,
    backfill,
    find_sessions_needing_backfill,
    find_transcript,
    windows_for_turns,
)
from flightdeck.models import Turn
from flightdeck.store import Store

BASE = "2026-01-01T00:00:00.000Z"


def _row_ts(offset_seconds: float) -> str:
    dt = datetime.fromisoformat(BASE) + timedelta(seconds=offset_seconds)
    return dt.isoformat().replace("+00:00", "Z")


def _ms(offset_seconds: float) -> int:
    """A real epoch-ms instant, in the SAME coordinate system `_iso_to_ms` produces -- keeps
    every test's turn boundaries expressed in the actual units the code uses instead of small
    ad-hoc integers that could pass even if seconds/ms were silently swapped somewhere."""
    ts = _iso_to_ms(_row_ts(offset_seconds))
    assert ts is not None
    return ts


def _assistant_block(
    *,
    mid: str,
    ts_offset: float,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read: int = 0,
    model: str = "claude-opus-5",
    block_type: str = "text",
) -> dict:
    return {
        "type": "assistant",
        "timestamp": _row_ts(ts_offset),
        "message": {
            "id": mid,
            "model": model,
            "role": "assistant",
            "content": [{"type": block_type}],
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_read_input_tokens": cache_read,
                "cache_creation_input_tokens": 0,
            },
        },
    }


def _tool_result_row(*, ts_offset: float) -> dict:
    """A generic non-assistant row -- stands in for a tool_result. Its only job in these tests
    is to occupy time between two usage groups without itself opening one."""
    return {
        "type": "user",
        "timestamp": _row_ts(ts_offset),
        "message": {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "tu-1", "content": "ok"}],
        },
    }


def _interrupt_row(*, ts_offset: float) -> dict:
    return {
        "type": "user",
        "timestamp": _row_ts(ts_offset),
        "message": {
            "role": "user",
            "content": [{"type": "text", "text": "[Request interrupted by user]"}],
        },
    }


def _error_row(*, ts_offset: float, error: str = "rate_limit") -> dict:
    return {
        "type": "assistant",
        "timestamp": _row_ts(ts_offset),
        "message": {
            "id": "synthetic-1",
            "model": "<synthetic>",
            "role": "assistant",
            "content": [{"type": "text", "text": "You've hit your session limit"}],
            "usage": {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0},
        },
        "isApiErrorMessage": True,
        "error": error,
    }


def _write_transcript(path: Path, *records: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")


def _turn(
    turn_id: str,
    *,
    started_at: int,
    ended_at: int | None = None,
    session_id: str = "sess-1",
) -> Turn:
    return Turn(
        turn_id=turn_id,
        session_id=session_id,
        source="claude-code",
        host="testhost",
        started_at=started_at,
        ended_at=ended_at,
    )


# ---------- units: the seconds/ms footgun ----------


def test_iso_to_ms_is_milliseconds_not_seconds() -> None:
    ms = _iso_to_ms("2026-01-01T00:00:01.000Z")
    assert ms == 1767225601000

    # Feeding the SECONDS value where the rest of the store (turns.started_at, events.ts) expects
    # MILLISECONDS must not silently look like a plausible timestamp in the same neighborhood --
    # it is 1000x smaller and belongs to a different epoch (1970 vs 2026).
    seconds_value = ms // 1000
    assert seconds_value != ms
    assert ms > 10**12  # real epoch-ms values are 13 digits
    assert seconds_value < 10**11  # the seconds-scale value is not even close


def test_iso_to_ms_rejects_malformed_timestamps() -> None:
    assert _iso_to_ms("") is None
    assert _iso_to_ms("not-a-timestamp") is None


# ---------- windows_for_turns ----------


def test_windows_for_turns_uses_ended_at_when_present() -> None:
    turns = [_turn("t1", started_at=1000, ended_at=1500), _turn("t2", started_at=2000)]
    windows = {w.turn_id: w for w in windows_for_turns(turns)}
    assert windows["t1"].lo == 1000
    assert windows["t1"].hi == 1500


def test_windows_for_turns_null_ended_at_borrows_next_started_at() -> None:
    turns = [_turn("t1", started_at=1000), _turn("t2", started_at=2000, ended_at=2500)]
    windows = {w.turn_id: w for w in windows_for_turns(turns)}
    assert windows["t1"].hi == 2000  # t1 never closed; t2's start is its bound


def test_windows_for_turns_last_turn_with_no_ended_at_is_unbounded() -> None:
    turns = [_turn("t1", started_at=1000, ended_at=1500), _turn("t2", started_at=2000)]
    windows = {w.turn_id: w for w in windows_for_turns(turns)}
    assert windows["t2"].hi is None


# ---------- attribute_transcript: dedupe and model_ms ----------


def test_attribute_transcript_dedupes_usage_by_message_id_not_by_row(tmp_path: Path) -> None:
    """Claude Code repeats the SAME usage on every content-block row of one API response.
    Summing per row rather than per unique message.id would 3x every count here."""
    path = tmp_path / "s.jsonl"
    _write_transcript(
        path,
        _assistant_block(
            mid="m1", ts_offset=1, input_tokens=10, output_tokens=20, block_type="thinking"
        ),
        _assistant_block(
            mid="m1", ts_offset=2, input_tokens=10, output_tokens=20, block_type="tool_use"
        ),
        _assistant_block(
            mid="m1", ts_offset=3, input_tokens=10, output_tokens=20, block_type="text"
        ),
    )
    windows = [Window(turn_id="t1", lo=_ms(0), hi=None)]
    attrs = attribute_transcript(path, windows)
    assert attrs["t1"].requests == 1
    assert attrs["t1"].prompt_tokens == 10
    assert attrs["t1"].completion_tokens == 20


def test_attribute_transcript_model_ms_excludes_tool_execution_time(tmp_path: Path) -> None:
    """Request A: generation takes 1s (t=1 -> t=2). Then a tool runs for 5s (t=2 -> t=7,
    represented by the tool_result row at t=7). Request B's generation then takes 0.2s
    (t=7 -> t=7.2). Total model_ms must be ~1200ms, NOT ~6200ms -- the 5s tool gap must not be
    counted as model time."""
    path = tmp_path / "s.jsonl"
    _write_transcript(
        path,
        _assistant_block(mid="m1", ts_offset=1, input_tokens=1, output_tokens=1),
        _assistant_block(mid="m1", ts_offset=2, input_tokens=1, output_tokens=1),
        _tool_result_row(ts_offset=7),
        _assistant_block(mid="m2", ts_offset=7.2, input_tokens=1, output_tokens=1),
    )
    windows = [Window(turn_id="t1", lo=_ms(0), hi=None)]
    attrs = attribute_transcript(path, windows)
    # group m1: prev_ts is None (first row in file) -> its model_ms is NOT counted (undercount,
    # documented). group m2: prev row is the tool_result at t=7 -> model_ms = 7.2-7 = 200ms.
    assert attrs["t1"].requests == 2
    assert attrs["t1"].model_ms == 200
    assert attrs["t1"].has_model_ms is True


def test_attribute_transcript_first_group_in_file_has_no_model_ms(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    _write_transcript(
        path, _assistant_block(mid="m1", ts_offset=1, input_tokens=5, output_tokens=5)
    )
    windows = [Window(turn_id="t1", lo=_ms(0), hi=None)]
    attrs = attribute_transcript(path, windows)
    assert attrs["t1"].requests == 1
    assert attrs["t1"].prompt_tokens == 5  # tokens still captured
    assert attrs["t1"].has_model_ms is False  # but no preceding row to measure latency from


def test_attribute_transcript_caps_an_implausible_model_ms_sample(tmp_path: Path) -> None:
    """A single request whose measured gap is absurd (a human stepped away for hours between
    two rows inside a supposedly-bounded window) must not be folded into model_ms as if it were
    real generation time -- it is dropped, not clamped."""
    path = tmp_path / "s.jsonl"
    _write_transcript(
        path,
        _assistant_block(mid="m1", ts_offset=1, input_tokens=1, output_tokens=1),
        _tool_result_row(ts_offset=2),
        # 20 minutes later, well past the 15-minute plausibility cap.
        _assistant_block(mid="m2", ts_offset=2 + 20 * 60, input_tokens=1, output_tokens=1),
    )
    # A real (non-None) hi keeps the window from being frozen early by the gap-detection logic
    # below -- this test is specifically about the per-sample cap inside an otherwise-normal,
    # already-bounded window.
    windows = [Window(turn_id="t1", lo=_ms(0), hi=_ms(3600))]
    attrs = attribute_transcript(path, windows)
    assert attrs["t1"].requests == 2  # both requests still counted
    assert attrs["t1"].model_ms == 0  # neither sample was a plausible model-latency measurement
    assert attrs["t1"].has_model_ms is False


def test_attribute_transcript_freezes_an_open_ended_window_on_an_implausible_gap(
    tmp_path: Path,
) -> None:
    """The real bug this guards: a session resumed hours later into the SAME transcript file,
    whose last claude-code turn never saw a Stop (hi=None, genuinely open-ended). Without this,
    the later, unrelated conversation's tokens and a many-hour "model latency" silently attribute
    to the stale turn. Measured on the live corpus: one turn's model_ms reached ~63M ms (17.5h)
    and its (model_ms+tool_ms)/wall_ms ratio hit 110x before this fix."""
    path = tmp_path / "s.jsonl"
    _write_transcript(
        path,
        _assistant_block(mid="m1", ts_offset=1, input_tokens=1, output_tokens=1),
        # 20 minutes later: past the plausibility cap, and this window is open-ended (hi=None),
        # so this gap must close the window right after m1's row rather than admitting m2.
        _assistant_block(mid="m2", ts_offset=1 + 20 * 60, input_tokens=999, output_tokens=999),
    )
    windows = [Window(turn_id="t1", lo=_ms(0), hi=None)]
    attrs = attribute_transcript(path, windows)
    assert attrs["t1"].requests == 1  # only m1
    assert attrs["t1"].prompt_tokens == 1  # m2's tokens are NOT folded in
    assert attrs["t1"].model_ms == 0


def test_attribute_transcript_does_not_freeze_a_window_with_a_real_boundary(
    tmp_path: Path,
) -> None:
    """A window with a real `hi` (Stop was seen, or a later turn exists) already stops admitting
    rows at its own boundary -- the gap-freeze logic must never touch it, and a legitimately
    later turn's window must still pick up rows normally."""
    path = tmp_path / "s.jsonl"
    _write_transcript(
        path,
        _assistant_block(mid="m1", ts_offset=1, input_tokens=1, output_tokens=1),
        # Falls inside t2's window, 20 minutes after t1's only row -- a real second turn, not an
        # anomaly, and must be attributed normally.
        _assistant_block(mid="m2", ts_offset=1 + 20 * 60, input_tokens=1, output_tokens=1),
    )
    windows = [
        Window(turn_id="t1", lo=_ms(0), hi=_ms(30)),
        Window(turn_id="t2", lo=_ms(30), hi=None),
    ]
    attrs = attribute_transcript(path, windows)
    assert attrs["t1"].requests == 1
    assert attrs["t2"].requests == 1
    assert attrs["t2"].prompt_tokens == 1


def test_attribute_transcript_model_is_last_group_wins(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    _write_transcript(
        path,
        _assistant_block(mid="m1", ts_offset=1, model="claude-opus-5"),
        _tool_result_row(ts_offset=2),
        _assistant_block(mid="m2", ts_offset=3, model="claude-sonnet-5"),
    )
    windows = [Window(turn_id="t1", lo=_ms(0), hi=None)]
    attrs = attribute_transcript(path, windows)
    assert attrs["t1"].model == "claude-sonnet-5"


def test_attribute_transcript_drops_a_reappearing_message_id(tmp_path: Path) -> None:
    """A message.id should never reappear after its group closes in a real (append-only)
    transcript. If it does anyway, the second occurrence must be dropped, not double-counted."""
    path = tmp_path / "s.jsonl"
    _write_transcript(
        path,
        _assistant_block(mid="m1", ts_offset=1, input_tokens=10, output_tokens=10),
        _tool_result_row(ts_offset=2),
        _assistant_block(mid="m2", ts_offset=3, input_tokens=1, output_tokens=1),
        _tool_result_row(ts_offset=4),
        _assistant_block(mid="m1", ts_offset=5, input_tokens=999, output_tokens=999),  # anomaly
    )
    windows = [Window(turn_id="t1", lo=_ms(0), hi=None)]
    attrs = attribute_transcript(path, windows)
    assert attrs["t1"].requests == 2  # not 3
    assert attrs["t1"].prompt_tokens == 11  # 10 + 1, never 999 again


def test_attribute_transcript_excludes_synthetic_error_rows_from_usage(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    _write_transcript(path, _error_row(ts_offset=1))
    windows = [Window(turn_id="t1", lo=_ms(0), hi=None)]
    attrs = attribute_transcript(path, windows)
    assert attrs["t1"].requests == 0
    assert attrs["t1"].model is None


def test_attribute_transcript_sets_outcome_error_from_api_error_marker(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    _write_transcript(path, _error_row(ts_offset=1, error="rate_limit"))
    windows = [Window(turn_id="t1", lo=_ms(0), hi=None)]
    attrs = attribute_transcript(path, windows)
    assert attrs["t1"].outcome == "error"
    assert attrs["t1"].error_class == "rate_limit"


def test_attribute_transcript_sets_outcome_interrupted_from_literal_marker(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    _write_transcript(
        path,
        _assistant_block(mid="m1", ts_offset=1, input_tokens=1, output_tokens=1),
        _interrupt_row(ts_offset=2),
    )
    windows = [Window(turn_id="t1", lo=_ms(0), hi=None)]
    attrs = attribute_transcript(path, windows)
    assert attrs["t1"].outcome == "interrupted"


def test_attribute_transcript_never_infers_ok(tmp_path: Path) -> None:
    """A turn whose transcript ends quietly (no error, no interrupt, Stop never seen) must not
    get an outcome at all -- collect_claude.py's rule that a missing Stop means UNKNOWN, not ok,
    holds for transcript evidence too."""
    path = tmp_path / "s.jsonl"
    _write_transcript(
        path, _assistant_block(mid="m1", ts_offset=1, input_tokens=1, output_tokens=1)
    )
    windows = [Window(turn_id="t1", lo=_ms(0), hi=None)]
    attrs = attribute_transcript(path, windows)
    assert attrs["t1"].outcome is None


def test_attribute_transcript_skips_malformed_lines(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    path.write_text(
        json.dumps(_assistant_block(mid="m1", ts_offset=1, input_tokens=1, output_tokens=1))
        + "\n"
        + "{not valid json\n"
        + "\n"
        + json.dumps(_assistant_block(mid="m2", ts_offset=2, input_tokens=2, output_tokens=2))
        + "\n",
        encoding="utf-8",
    )
    windows = [Window(turn_id="t1", lo=_ms(0), hi=None)]
    attrs = attribute_transcript(path, windows)
    assert attrs["t1"].requests == 2
    assert attrs["t1"].prompt_tokens == 3


def test_attribute_transcript_rows_outside_every_window_are_unattributed(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    _write_transcript(
        path, _assistant_block(mid="m1", ts_offset=-100, input_tokens=1, output_tokens=1)
    )
    windows = [Window(turn_id="t1", lo=_ms(0), hi=None)]
    attrs = attribute_transcript(path, windows)
    assert attrs == {}


def test_attribute_transcript_respects_a_closed_window_boundary(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    _write_transcript(
        path,
        _assistant_block(mid="m1", ts_offset=1, input_tokens=1, output_tokens=1),  # inside t1
        _assistant_block(mid="m2", ts_offset=10, input_tokens=2, output_tokens=2),  # inside t2
    )
    windows = [
        Window(turn_id="t1", lo=_ms(0), hi=_ms(5)),
        Window(turn_id="t2", lo=_ms(5), hi=None),
    ]
    attrs = attribute_transcript(path, windows)
    assert attrs["t1"].prompt_tokens == 1
    assert attrs["t2"].prompt_tokens == 2


# ---------- find_transcript / find_sessions_needing_backfill ----------


def test_find_transcript_matches_by_filename_one_level_deep(tmp_path: Path) -> None:
    root = tmp_path / "projects"
    session_file = root / "-Users-me-project" / "sess-1.jsonl"
    _write_transcript(session_file, _assistant_block(mid="m1", ts_offset=1))
    found = find_transcript("sess-1", root)
    assert found == session_file


def test_find_transcript_falls_back_to_nested_search(tmp_path: Path) -> None:
    root = tmp_path / "projects"
    nested = root / "-Users-me-project" / "sess-1" / "subagents" / "sess-1.jsonl"
    _write_transcript(nested, _assistant_block(mid="m1", ts_offset=1))
    found = find_transcript("sess-1", root)
    assert found == nested


def test_find_transcript_returns_none_when_missing(tmp_path: Path) -> None:
    assert find_transcript("no-such-session", tmp_path / "projects") is None


def test_find_sessions_needing_backfill_only_lists_incomplete_claude_code_turns(
    tmp_path: Path,
) -> None:
    store = Store(tmp_path)
    try:
        complete = _turn("t1", started_at=1000, session_id="done")
        complete.prompt_tokens, complete.completion_tokens = 1, 1
        complete.model, complete.model_ms, complete.outcome = "m", 1, "ok"
        store.upsert_turn(complete)
        store.upsert_turn(_turn("t2", started_at=1000, session_id="needs-work"))
        crush = Turn(
            turn_id="t3", session_id="needs-work-2", source="crush", host="h", started_at=1000
        )
        store.upsert_turn(crush)

        sessions = find_sessions_needing_backfill(store)
        assert sessions == ["needs-work"]
    finally:
        store.close()


# ---------- backfill: end-to-end against a Store + fake transcript root ----------


def _seed_session(root: Path, session_id: str, *records: dict) -> None:
    _write_transcript(root / "-Users-me-project" / f"{session_id}.jsonl", *records)


def test_backfill_fills_tokens_model_and_marks_model_ms_estimated(tmp_path: Path) -> None:
    store = Store(tmp_path / "store")
    root = tmp_path / "projects"
    try:
        t0 = _ms(0)
        store.upsert_turn(_turn("t1", started_at=t0, session_id="sess-1"))
        _seed_session(
            root,
            "sess-1",
            _assistant_block(mid="m1", ts_offset=1, input_tokens=5, output_tokens=7, cache_read=2),
        )

        report = backfill(store, root=root)
        assert report["turns_updated_tokens"] == 1
        assert report["turns_updated_model"] == 1
        assert report["sessions_matched"] == 1
        assert report["sessions_unmatched"] == 0

        turn = store.get_turn("t1")
        assert turn is not None
        assert turn.prompt_tokens == 5
        assert turn.completion_tokens == 7
        assert turn.cached_tokens == 2
        assert turn.requests == 1
        assert turn.model == "claude-opus-5"
        # first (only) group in the file has no preceding row -> no model_ms sample.
        assert turn.model_ms is None
        assert turn.estimated == 0
    finally:
        store.close()


def test_backfill_marks_estimated_only_when_model_ms_is_actually_written(tmp_path: Path) -> None:
    store = Store(tmp_path / "store")
    root = tmp_path / "projects"
    try:
        t0 = _ms(0)
        store.upsert_turn(_turn("t1", started_at=t0, session_id="sess-1"))
        _seed_session(
            root,
            "sess-1",
            _assistant_block(mid="m1", ts_offset=1, input_tokens=1, output_tokens=1),
            _tool_result_row(ts_offset=2),
            _assistant_block(mid="m2", ts_offset=2.5, input_tokens=1, output_tokens=1),
        )

        report = backfill(store, root=root)
        assert report["turns_updated_model_ms"] == 1

        turn = store.get_turn("t1")
        assert turn is not None
        assert turn.model_ms == 500
        assert turn.estimated == 1
    finally:
        store.close()


def test_backfill_never_overwrites_existing_prompt_tokens(tmp_path: Path) -> None:
    store = Store(tmp_path / "store")
    root = tmp_path / "projects"
    try:
        t0 = _ms(0)
        turn = _turn("t1", started_at=t0, session_id="sess-1")
        turn.prompt_tokens, turn.completion_tokens = 999, 999
        store.upsert_turn(turn)
        _seed_session(
            root, "sess-1", _assistant_block(mid="m1", ts_offset=1, input_tokens=5, output_tokens=5)
        )

        report = backfill(store, root=root)
        assert report["turns_updated_tokens"] == 0

        unchanged = store.get_turn("t1")
        assert unchanged is not None
        assert unchanged.prompt_tokens == 999
        assert unchanged.completion_tokens == 999
    finally:
        store.close()


def test_backfill_never_fabricates_ok_outcome(tmp_path: Path) -> None:
    store = Store(tmp_path / "store")
    root = tmp_path / "projects"
    try:
        t0 = _ms(0)
        store.upsert_turn(_turn("t1", started_at=t0, session_id="sess-1"))  # ended_at=None
        _seed_session(
            root, "sess-1", _assistant_block(mid="m1", ts_offset=1, input_tokens=1, output_tokens=1)
        )

        backfill(store, root=root)
        turn = store.get_turn("t1")
        assert turn is not None
        assert turn.outcome is None
    finally:
        store.close()


def test_backfill_sets_unambiguous_error_outcome(tmp_path: Path) -> None:
    store = Store(tmp_path / "store")
    root = tmp_path / "projects"
    try:
        t0 = _ms(0)
        store.upsert_turn(_turn("t1", started_at=t0, session_id="sess-1"))
        _seed_session(root, "sess-1", _error_row(ts_offset=1, error="overloaded"))

        report = backfill(store, root=root)
        assert report["turns_updated_outcome"] == 1

        turn = store.get_turn("t1")
        assert turn is not None
        assert turn.outcome == "error"
        assert turn.error_class == "overloaded"
    finally:
        store.close()


def test_backfill_is_idempotent_on_rerun(tmp_path: Path) -> None:
    store = Store(tmp_path / "store")
    root = tmp_path / "projects"
    try:
        t0 = _ms(0)
        store.upsert_turn(_turn("t1", started_at=t0, session_id="sess-1"))
        _seed_session(
            root,
            "sess-1",
            _assistant_block(mid="m1", ts_offset=1, input_tokens=5, output_tokens=7),
            _tool_result_row(ts_offset=2),
            _assistant_block(mid="m2", ts_offset=2.5, input_tokens=1, output_tokens=1),
        )

        first = backfill(store, root=root)
        after_first = store.get_turn("t1")
        assert after_first is not None
        assert first["turns_updated_tokens"] == 1

        second = backfill(store, root=root)
        after_second = store.get_turn("t1")

        assert second["turns_updated_tokens"] == 0
        assert second["turns_updated_model"] == 0
        assert second["turns_updated_model_ms"] == 0
        assert after_second == after_first  # byte-for-byte identical, not just "same value"
    finally:
        store.close()


def test_backfill_dry_run_reports_without_writing(tmp_path: Path) -> None:
    store = Store(tmp_path / "store")
    root = tmp_path / "projects"
    try:
        t0 = _ms(0)
        store.upsert_turn(_turn("t1", started_at=t0, session_id="sess-1"))
        _seed_session(
            root, "sess-1", _assistant_block(mid="m1", ts_offset=1, input_tokens=5, output_tokens=5)
        )

        report = backfill(store, root=root, dry_run=True)
        assert report["turns_updated_tokens"] == 1
        assert report["dry_run"] is True

        untouched = store.get_turn("t1")
        assert untouched is not None
        assert untouched.prompt_tokens is None
    finally:
        store.close()


def test_backfill_reports_unmatched_sessions_without_crashing(tmp_path: Path) -> None:
    store = Store(tmp_path / "store")
    root = tmp_path / "projects"
    root.mkdir(parents=True)
    try:
        store.upsert_turn(_turn("t1", started_at=_ms(0), session_id="no-transcript-for-this-one"))
        report = backfill(store, root=root)
        assert report["sessions_unmatched"] == 1
        assert report["unmatched_session_ids"] == ["no-transcript-for-this-one"]
    finally:
        store.close()


def test_backfill_scopes_writes_to_only_the_columns_it_touches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The store is genuinely live: the hook collector can be closing this exact turn
    concurrently. A write here must never be a bare, unscoped upsert_turn(turn) that could
    round-trip a stale ended_at/wall_ms/cwd back over whatever the hook just wrote."""
    store = Store(tmp_path / "store")
    root = tmp_path / "projects"
    try:
        t0 = _ms(0)
        turn = _turn("t1", started_at=t0, session_id="sess-1")
        turn.cwd = "/original/cwd"
        store.upsert_turn(turn)
        _seed_session(
            root, "sess-1", _assistant_block(mid="m1", ts_offset=1, input_tokens=1, output_tokens=1)
        )

        captured: list[set[str] | None] = []
        original_upsert = Store.upsert_turn

        def spy(self: Store, turn: Turn, *, mirror: bool = True, present=None) -> None:
            captured.append(present)
            original_upsert(self, turn, mirror=mirror, present=present)

        monkeypatch.setattr(Store, "upsert_turn", spy)

        backfill(store, root=root)

        assert captured  # at least one scoped write happened
        for present in captured:
            assert present is not None
            assert "ended_at" not in present
            assert "wall_ms" not in present
            assert "cwd" not in present
    finally:
        store.close()


def test_backfill_since_ms_filters_which_sessions_are_scanned(tmp_path: Path) -> None:
    store = Store(tmp_path / "store")
    root = tmp_path / "projects"
    try:
        store.upsert_turn(_turn("old", started_at=1000, session_id="old-sess"))
        store.upsert_turn(_turn("new", started_at=_ms(0), session_id="new-sess"))
        _seed_session(root, "old-sess", _assistant_block(mid="m1", ts_offset=1))
        _seed_session(root, "new-sess", _assistant_block(mid="m2", ts_offset=1))

        report = backfill(store, root=root, since_ms=_ms(0) - 1)
        assert report["sessions_scanned"] == 1
    finally:
        store.close()


def test_ingest_transcript_module_default_root_is_claude_projects() -> None:
    assert Path("~/.claude/projects").expanduser() == ingest_transcript.DEFAULT_ROOT
