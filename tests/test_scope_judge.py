from __future__ import annotations

import io
import json

import pytest

from flightdeck.models import SCOPE_VERDICTS
from flightdeck.scope_judge import (
    CORRECTION,
    JUDGE_VERDICTS,
    NEW,
    SILENT,
    judge_rate,
    judge_turn,
    parse_verdict,
)


@pytest.mark.parametrize(
    "reply,expected",
    [
        ("CORRECTION", CORRECTION),
        ("correction", CORRECTION),
        (" NEW ", NEW),
        ("NEW.", NEW),
        ("", SILENT),
        (None, SILENT),
        ("THE MESSAGE CONTAINS", SILENT),   # truncated explanation, seen on 7.3% of held-out
        ("THE USER IS ASKING F", SILENT),
    ],
)
def test_parse_verdict(reply, expected):
    assert parse_verdict(reply) == expected


def test_truncated_explanation_is_silent_not_new():
    """Scoring truncation as NEW understated held-out recall by 11 points (0.726 -> 0.834)
    and is the same fail-open shape as a dead hook reading PASS."""
    assert parse_verdict("THE MESSAGE STARTS W") != NEW


def test_silent_is_shared_with_the_record_vocabulary():
    assert SILENT in SCOPE_VERDICTS


def test_judge_verdicts_are_three_valued():
    assert set(JUDGE_VERDICTS) == {CORRECTION, NEW, SILENT}


def _opener(payload):
    def _open(request, timeout=None):
        return io.BytesIO(json.dumps(payload).encode())
    return _open


def test_judge_turn_parses_a_reply():
    opener = _opener({"message": {"content": "CORRECTION"}})
    assert judge_turn("no that's wrong", opener=opener) == CORRECTION


def test_network_failure_is_silent_not_new():
    def _boom(request, timeout=None):
        raise OSError("connection refused")

    assert judge_turn("anything", opener=_boom) == SILENT


def test_malformed_payload_is_silent():
    assert judge_turn("anything", opener=_opener({"unexpected": True})) == SILENT


def test_judge_rate_excludes_abstentions_from_the_denominator():
    replies = iter(["CORRECTION", "NEW", "THE MESSAGE CONT", "CORRECTION"])

    def _open(request, timeout=None):
        return io.BytesIO(json.dumps({"message": {"content": next(replies)}}).encode())

    result = judge_rate(["a", "b", "c", "d"], opener=_open)
    assert result["counts"][SILENT] == 1
    assert result["answered"] == 3
    assert result["correction_rate"] == round(2 / 3, 4)
    assert result["abstention_rate"] == 0.25


def test_all_silent_reports_no_rate_rather_than_zero():
    def _boom(request, timeout=None):
        raise OSError("down")

    result = judge_rate(["a", "b"], opener=_boom)
    assert result["correction_rate"] is None
    assert result["abstention_rate"] == 1.0
