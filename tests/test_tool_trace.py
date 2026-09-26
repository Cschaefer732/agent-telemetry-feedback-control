from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from flightdeck.tool_trace import (
    analyze_session,
    batch_stats,
    failure_stats,
    missed_batch_opportunities,
    parse_transcript,
    prefetchability_stats,
)

BASE = "2026-01-01T00:00:00.000Z"


def _ts(offset_seconds: float) -> str:
    dt = datetime.fromisoformat(BASE) + timedelta(seconds=offset_seconds)
    return dt.isoformat().replace("+00:00", "Z")


def _assistant_tool_uses(
    *, mid: str, ts_offset: float, uses: list[tuple[str, dict, str]], sidechain: bool = False
) -> dict:
    """`uses` is a list of (tool_use_id, input, name) tuples -- N of them makes one batch of N."""
    content = [
        {"type": "tool_use", "id": tuid, "name": name, "input": inp} for tuid, inp, name in uses
    ]
    record = {
        "type": "assistant",
        "timestamp": _ts(ts_offset),
        "message": {"id": mid, "role": "assistant", "content": content},
    }
    if sidechain:
        record["isSidechain"] = True
    return record


def _tool_result(*, ts_offset: float, tool_use_id: str, text: str, is_error: bool = False) -> dict:
    return {
        "type": "user",
        "timestamp": _ts(ts_offset),
        "message": {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": text,
                    "is_error": is_error,
                }
            ],
        },
    }


def _write(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def test_three_tool_batch_counts_as_one_batch(tmp_path: Path) -> None:
    path = tmp_path / "sess1.jsonl"
    _write(
        path,
        [
            _assistant_tool_uses(
                mid="m1",
                ts_offset=0,
                uses=[
                    ("tu1", {"file_path": "/a.py"}, "Read"),
                    ("tu2", {"file_path": "/b.py"}, "Read"),
                    ("tu3", {"pattern": "foo"}, "Grep"),
                ],
            ),
            _tool_result(ts_offset=1, tool_use_id="tu1", text="a content"),
            _tool_result(ts_offset=1, tool_use_id="tu2", text="b content"),
            _tool_result(ts_offset=1, tool_use_id="tu3", text="match"),
        ],
    )
    trace = parse_transcript(path)
    stats = batch_stats(trace)
    assert stats["total_tool_calls"] == 3
    assert stats["messages_multi_tool"] == 1
    assert stats["messages_single_tool"] == 0
    assert stats["calls_traveling_in_a_batch"] == 3
    assert stats["achieved_batch_rate"] == 1.0
    assert stats["batch_size_distribution"] == {3: 1}


def test_dependent_sequential_calls_not_counted_as_missed_batch(tmp_path: Path) -> None:
    """Second call's input references a path that appeared verbatim in the first call's result
    -- a real dependency, must not be flagged as an independent/missable batch opportunity."""
    path = tmp_path / "sess2.jsonl"
    _write(
        path,
        [
            _assistant_tool_uses(
                mid="m1", ts_offset=0, uses=[("tu1", {"pattern": "TODO"}, "Grep")]
            ),
            _tool_result(ts_offset=1, tool_use_id="tu1", text="/repo/src/widget.py:12: TODO fix"),
            _assistant_tool_uses(
                mid="m2", ts_offset=2, uses=[("tu2", {"file_path": "/repo/src/widget.py"}, "Read")]
            ),
            _tool_result(ts_offset=3, tool_use_id="tu2", text="widget contents"),
        ],
    )
    trace = parse_transcript(path)
    missed = missed_batch_opportunities(trace)
    assert missed["consecutive_single_pairs_considered"] == 1
    assert missed["missed_batches_strict"] == 0
    assert missed["missed_batches_loose"] == 0


def test_independent_sequential_calls_counted_as_missed_batch(tmp_path: Path) -> None:
    """Two single-tool messages with no textual relationship between the first result and the
    second call's input -- these could have been issued together."""
    path = tmp_path / "sess3.jsonl"
    _write(
        path,
        [
            _assistant_tool_uses(
                mid="m1", ts_offset=0, uses=[("tu1", {"file_path": "/repo/alpha.py"}, "Read")]
            ),
            _tool_result(ts_offset=1, tool_use_id="tu1", text="alpha contents, nothing special"),
            _assistant_tool_uses(
                mid="m2", ts_offset=2, uses=[("tu2", {"file_path": "/repo/zeta.py"}, "Read")]
            ),
            _tool_result(ts_offset=3, tool_use_id="tu2", text="zeta contents, unrelated"),
        ],
    )
    trace = parse_transcript(path)
    missed = missed_batch_opportunities(trace)
    assert missed["consecutive_single_pairs_considered"] == 1
    assert missed["missed_batches_strict"] == 1
    assert missed["missed_batches_loose"] == 1


def test_tool_use_with_no_matching_result_is_unmatched_not_dropped(tmp_path: Path) -> None:
    path = tmp_path / "sess4.jsonl"
    _write(
        path,
        [
            _assistant_tool_uses(
                mid="m1", ts_offset=0, uses=[("tu1", {"file_path": "/a.py"}, "Read")]
            ),
            # No tool_result row at all -- e.g. turn interrupted mid-call.
        ],
    )
    trace = parse_transcript(path)
    assert len(trace.tool_calls) == 1
    assert trace.tool_calls[0].has_result is False
    stats = failure_stats(trace)
    assert stats["total_unmatched"] == 1
    assert stats["by_tool"]["Read"]["unmatched"] == 1
    assert stats["by_tool"]["Read"]["errored"] == 0


def test_sidechain_records_excluded_by_default(tmp_path: Path) -> None:
    path = tmp_path / "sess5.jsonl"
    _write(
        path,
        [
            _assistant_tool_uses(
                mid="m1",
                ts_offset=0,
                uses=[("tu1", {"file_path": "/sub.py"}, "Read")],
                sidechain=True,
            ),
            _tool_result(ts_offset=1, tool_use_id="tu1", text="sub contents"),
            _assistant_tool_uses(
                mid="m2", ts_offset=2, uses=[("tu2", {"file_path": "/top.py"}, "Read")]
            ),
            _tool_result(ts_offset=3, tool_use_id="tu2", text="top contents"),
        ],
    )
    excluded_trace = parse_transcript(path, include_sidechains=False)
    assert len(excluded_trace.tool_calls) == 1
    assert excluded_trace.tool_calls[0].tool_use_id == "tu2"
    assert excluded_trace.sidechain_calls_excluded == 1

    included_trace = parse_transcript(path, include_sidechains=True)
    assert len(included_trace.tool_calls) == 2
    assert included_trace.sidechain_calls_excluded == 0


def test_error_result_counted_in_failure_stats(tmp_path: Path) -> None:
    path = tmp_path / "sess6.jsonl"
    _write(
        path,
        [
            _assistant_tool_uses(
                mid="m1", ts_offset=0, uses=[("tu1", {"command": "false"}, "Bash")]
            ),
            _tool_result(ts_offset=1, tool_use_id="tu1", text="exit 1", is_error=True),
        ],
    )
    trace = parse_transcript(path)
    stats = failure_stats(trace)
    assert stats["by_tool"]["Bash"]["errored"] == 1
    assert stats["overall_error_rate"] == 1.0


def test_prefetchability_flags_read_only_tools_and_bash_prefixes(tmp_path: Path) -> None:
    path = tmp_path / "sess7.jsonl"
    _write(
        path,
        [
            _assistant_tool_uses(
                mid="m1",
                ts_offset=0,
                uses=[
                    ("tu1", {"file_path": "/a.py"}, "Read"),
                    ("tu2", {"command": "git status"}, "Bash"),
                    ("tu3", {"command": "rm -rf /tmp/x"}, "Bash"),
                    ("tu4", {"file_path": "/a.py", "old_string": "x", "new_string": "y"}, "Edit"),
                ],
            ),
            _tool_result(ts_offset=1, tool_use_id="tu1", text="ok"),
            _tool_result(ts_offset=1, tool_use_id="tu2", text="ok"),
            _tool_result(ts_offset=1, tool_use_id="tu3", text="ok"),
            _tool_result(ts_offset=1, tool_use_id="tu4", text="ok"),
        ],
    )
    trace = parse_transcript(path)
    stats = prefetchability_stats(trace)
    assert stats["total_calls"] == 4
    assert stats["prefetchable_calls"] == 2  # Read + `git status` Bash
    assert stats["prefetchable_share"] == 0.5


def test_repeated_read_of_same_file_is_top_repeated_target(tmp_path: Path) -> None:
    path = tmp_path / "sess8.jsonl"
    records = []
    for i in range(5):
        records.append(
            _assistant_tool_uses(
                mid=f"m{i}",
                ts_offset=i * 2,
                uses=[(f"tu{i}", {"file_path": "/hot.py"}, "Read")],
            )
        )
        records.append(_tool_result(ts_offset=i * 2 + 1, tool_use_id=f"tu{i}", text="hot contents"))
    _write(path, records)
    trace = parse_transcript(path)
    stats = prefetchability_stats(trace)
    top = stats["top_repeated_targets"][0]
    assert top == {"tool": "Read", "target": "/hot.py", "count": 5}


def test_malformed_line_counted_as_parse_issue_not_silently_dropped(tmp_path: Path) -> None:
    path = tmp_path / "sess9.jsonl"
    path.write_text(
        "not json at all\n"
        + json.dumps(
            _assistant_tool_uses(mid="m1", ts_offset=0, uses=[("tu1", {"file_path": "/a"}, "Read")])
        )
        + "\n"
        + json.dumps(_tool_result(ts_offset=1, tool_use_id="tu1", text="ok"))
        + "\n"
    )
    trace = parse_transcript(path)
    assert len(trace.parse_issues) == 1
    assert "json error" in trace.parse_issues[0].reason
    assert len(trace.tool_calls) == 1


def test_analyze_session_returns_all_four_measurement_sections(tmp_path: Path) -> None:
    path = tmp_path / "sess10.jsonl"
    _write(
        path,
        [
            _assistant_tool_uses(
                mid="m1", ts_offset=0, uses=[("tu1", {"file_path": "/a.py"}, "Read")]
            ),
            _tool_result(ts_offset=1, tool_use_id="tu1", text="a contents"),
        ],
    )
    report = analyze_session(path)
    assert set(["batching", "missed_batches", "prefetchability", "failures"]) <= report.keys()
    assert report["parse_issues"] == []
