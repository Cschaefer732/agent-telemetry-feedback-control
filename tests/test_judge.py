from __future__ import annotations

import json
from pathlib import Path

import pytest

from flightdeck import judge
from flightdeck.judge import (
    DEFAULT_ENDPOINT,
    DEFAULT_MODEL,
    JudgeConfig,
    JudgeUnavailable,
    build_prompt,
    ollama_client,
    parse_verdict,
    run_queue,
)
from flightdeck.models import Event, TextBlob, Turn
from flightdeck.store import Store


def make_turn(**overrides) -> Turn:
    defaults = dict(
        turn_id="t1",
        session_id="s1",
        source="claude-code",
        host="h1",
        started_at=1_000_000,
        outcome="error",
        flagged=1,
    )
    defaults.update(overrides)
    return Turn(**defaults)


def ev(turn_id: str, kind: str, **kw) -> Event:
    return Event(turn_id=turn_id, ts=1_000_000, kind=kind, **kw)


def config(**overrides) -> JudgeConfig:
    defaults = dict(endpoint="http://judge.example:11434", model="qwen3.8:27b")
    defaults.update(overrides)
    return JudgeConfig(**defaults)


VALID_RESPONSE = (
    "VERDICT: partial\n"
    "TIER: right\n"
    "SKILLS: missing:context7\n"
    "BLAME: harness\n"
    "LESSON: fetch docs before guessing API shape\n"
    "NOTES: Tool failed twice on a missing arg. Otherwise on track."
)


@pytest.fixture
def store(tmp_path: Path) -> Store:
    s = Store(tmp_path / "store")
    yield s
    s.close()


# ---------- JudgeConfig.from_env ----------


def test_from_env_defaults(monkeypatch):
    monkeypatch.delenv("SPARKY_JUDGE", raising=False)
    monkeypatch.delenv("SPARKY_JUDGE_ENDPOINT", raising=False)
    monkeypatch.delenv("SPARKY_JUDGE_MODEL", raising=False)
    cfg = JudgeConfig.from_env()
    assert cfg.endpoint == DEFAULT_ENDPOINT
    assert cfg.model == DEFAULT_MODEL
    assert cfg.enabled is True


def test_from_env_disabled(monkeypatch):
    monkeypatch.setenv("SPARKY_JUDGE", "0")
    cfg = JudgeConfig.from_env()
    assert cfg.enabled is False


def test_from_env_overrides(monkeypatch):
    monkeypatch.setenv("SPARKY_JUDGE_ENDPOINT", "http://other:1234")
    monkeypatch.setenv("SPARKY_JUDGE_MODEL", "some-model")
    cfg = JudgeConfig.from_env()
    assert cfg.endpoint == "http://other:1234"
    assert cfg.model == "some-model"


# ---------- build_prompt ----------


def test_build_prompt_keeps_tail_over_head():
    turn = make_turn()
    events = [ev("t1", "tool_call", ok=0, name="edit", payload={"error": "bad path"})]
    head_text = "HEAD_MARKER " + ("x" * 500)
    tail_text = "y" * 500 + " TAIL_MARKER"
    texts = [
        TextBlob(turn_id="t1", kind="prompt", seq=0, body=head_text, expires_at=0),
        TextBlob(turn_id="t1", kind="response", seq=0, body=tail_text, expires_at=0),
    ]
    prompt = build_prompt(turn, events, texts, max_context_chars=300)

    assert "TAIL_MARKER" in prompt
    assert "HEAD_MARKER" not in prompt


def test_build_prompt_always_includes_summary_and_reasons():
    turn = make_turn(kpi_score=0.1)
    events = [ev("t1", "revert")]
    prompt = build_prompt(turn, events, [], max_context_chars=12000)

    assert "turn_id=t1" in prompt
    assert "outcome_not_ok" in prompt
    assert "low_kpi" in prompt
    assert "edit_revert" in prompt


def test_build_prompt_lists_tool_failures():
    turn = make_turn()
    events = [
        ev("t1", "tool_call", ok=0, name="bash", payload={"error": "permission denied"}),
        ev("t1", "tool_call", ok=1, name="read"),
    ]
    prompt = build_prompt(turn, events, [], max_context_chars=12000)

    assert "bash" in prompt
    assert "permission denied" in prompt


def test_build_prompt_no_flag_reasons_says_none():
    turn = make_turn(outcome="ok", kpi_score=0.9, flagged=0)
    prompt = build_prompt(turn, [], [], max_context_chars=12000)
    assert "FLAG_REASONS: none" in prompt


# ---------- parse_verdict ----------


def test_parse_verdict_reads_every_field():
    parsed = parse_verdict(VALID_RESPONSE)
    assert parsed["verdict"] == "partial"
    assert parsed["tier"] == "right"
    assert parsed["skills"] == "missing:context7"
    assert parsed["blame"] == "harness"
    assert parsed["lesson"] == "fetch docs before guessing API shape"
    assert "Tool failed twice" in parsed["notes"]


def test_parse_verdict_lesson_none_becomes_none():
    raw = VALID_RESPONSE.replace("LESSON: fetch docs before guessing API shape", "LESSON: NONE")
    parsed = parse_verdict(raw)
    assert parsed["lesson"] is None


@pytest.mark.parametrize("verdict", ["pass", "partial", "fail"])
def test_parse_verdict_accepts_all_valid_verdicts(verdict):
    raw = f"VERDICT: {verdict}\nTIER: right\nSKILLS: used\nBLAME: none\nLESSON: NONE\nNOTES: ok"
    assert parse_verdict(raw)["verdict"] == verdict


def test_parse_verdict_case_insensitive():
    raw = "verdict: Pass\ntier: Right\nskills: used\nblame: none\nlesson: none\nnotes: fine"
    assert parse_verdict(raw)["verdict"] == "pass"


def test_parse_verdict_missing_verdict_raises():
    with pytest.raises(ValueError):
        parse_verdict("this is not the rubric format at all")


def test_parse_verdict_invalid_verdict_value_raises():
    with pytest.raises(ValueError):
        parse_verdict("VERDICT: maybe\nNOTES: unsure")


# ---------- judge_turn / run_queue: parse failure still judges ----------


def test_run_queue_malformed_response_marks_failed(store):
    turn = make_turn()
    store.upsert_turn(turn)

    def garbage_client(prompt: str, cfg: JudgeConfig) -> str:
        return "the model rambled about something unrelated"

    result = run_queue(store, config(), client=garbage_client)

    assert result == {"judged": 0, "skipped": 0, "failed": 1, "backlog": 1}
    stored = store.judgment_for("t1")
    assert stored is not None
    assert stored["verdict"] == "fail"
    assert "parse failure" in stored["notes"]
    reloaded = store.get_turn("t1")
    assert reloaded.judged == 1


# ---------- run_queue: JudgeUnavailable halts the batch ----------


def test_run_queue_halts_on_first_unavailable(store):
    for i in range(3):
        store.upsert_turn(make_turn(turn_id=f"t{i}", started_at=1_000_000 + i))

    calls = []

    def flaky_client(prompt: str, cfg: JudgeConfig) -> str:
        calls.append(prompt)
        raise JudgeUnavailable("connection refused")

    result = run_queue(store, config(), client=flaky_client)

    assert len(calls) == 1
    assert result["failed"] == 0
    assert result["judged"] == 0
    assert result["skipped"] == 3
    assert result["backlog"] == 3
    for i in range(3):
        assert store.get_turn(f"t{i}").judged == 0


# ---------- run_queue: backlog over cap ----------


def test_run_queue_backlog_over_cap_counts_skipped_and_logs(store):
    for i in range(5):
        store.upsert_turn(make_turn(turn_id=f"t{i}", started_at=1_000_000 + i))

    def ok_client(prompt: str, cfg: JudgeConfig) -> str:
        return VALID_RESPONSE

    result = run_queue(store, config(max_backlog=2), client=ok_client)

    assert result["backlog"] == 5
    assert result["skipped"] == 3
    assert result["judged"] == 2

    # the 3 oldest (t0, t1, t2) were dropped, not the 2 newest (t3, t4)
    for tid in ("t3", "t4"):
        assert store.judgment_for(tid) is not None
    for tid in ("t0", "t1", "t2"):
        assert store.judgment_for(tid) is None

    dropped_events = store.conn.execute(
        "SELECT COUNT(*) FROM events WHERE kind='queue' AND ok=0"
    ).fetchone()[0]
    assert dropped_events == 3


# ---------- run_queue: disabled is a no-op ----------


def test_run_queue_disabled_is_noop(store):
    store.upsert_turn(make_turn())
    calls = []

    def should_not_be_called(prompt: str, cfg: JudgeConfig) -> str:
        calls.append(prompt)
        return VALID_RESPONSE

    result = run_queue(store, config(enabled=False), client=should_not_be_called)

    assert result == {"judged": 0, "skipped": 0, "failed": 0, "backlog": 0}
    assert calls == []
    assert store.judgment_for("t1") is None


def test_judge_turn_disabled_returns_none(store):
    turn = make_turn()
    store.upsert_turn(turn)
    result = judge.judge_turn(store, turn, config(enabled=False))
    assert result is None


# ---------- run_queue: full round trip against a real temporary Store ----------


def test_run_queue_full_round_trip(store):
    turn = make_turn(turn_id="t1")
    store.upsert_turn(turn)
    store.add_events([ev("t1", "tool_call", ok=0, name="bash", payload={"error": "boom"})])
    store.add_texts(
        [TextBlob(turn_id="t1", kind="prompt", seq=0, body="fix the bug", expires_at=0)]
    )

    seen_prompts = []

    def recording_client(prompt: str, cfg: JudgeConfig) -> str:
        seen_prompts.append(prompt)
        return VALID_RESPONSE

    result = run_queue(store, config(), client=recording_client)

    assert result == {"judged": 1, "skipped": 0, "failed": 0, "backlog": 1}
    assert len(seen_prompts) == 1
    assert "fix the bug" in seen_prompts[0]
    assert "bash" in seen_prompts[0]

    stored = store.judgment_for("t1")
    assert stored["verdict"] == "partial"
    assert stored["lesson"] == "fetch docs before guessing API shape"
    rubric = json.loads(stored["rubric"])
    assert rubric["blame"] == "harness"

    reloaded = store.get_turn("t1")
    assert reloaded.judged == 1


def test_run_queue_respects_limit(store):
    for i in range(3):
        store.upsert_turn(make_turn(turn_id=f"t{i}", started_at=1_000_000 + i))

    def ok_client(prompt: str, cfg: JudgeConfig) -> str:
        return VALID_RESPONSE

    result = run_queue(store, config(), limit=1, client=ok_client)
    assert result["judged"] == 1
    assert result["backlog"] == 3


# ---------- ollama_client ----------


class _FakeResponse:
    def __init__(self, body: bytes, status: int = 200) -> None:
        self._body = body
        self.status = status

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def test_ollama_client_builds_correct_request_body(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        payload = {"message": {"content": VALID_RESPONSE}}
        return _FakeResponse(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    cfg = config(timeout_s=42.0, max_output_tokens=256)
    result = ollama_client("judge this turn", cfg)

    assert result == VALID_RESPONSE
    # native /api/chat, NOT the OpenAI path: "think" is silently ignored over there and
    # qwen3's reasoning ate the whole output budget (measured live 2026-08-23)
    assert captured["url"] == "http://judge.example:11434/api/chat"
    assert captured["timeout"] == 42.0
    body = captured["body"]
    assert body["model"] == "qwen3.8:27b"
    assert body["messages"] == [{"role": "user", "content": "judge this turn"}]
    assert body["stream"] is False
    assert body["think"] is False
    assert body["options"] == {"temperature": 0, "num_predict": 256}


def test_ollama_client_think_flag_only_for_qwen3(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        payload = {"message": {"content": VALID_RESPONSE}}
        return _FakeResponse(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    ollama_client("judge this turn", config(model="nemotron-3.5-lightning:latest"))
    # non-thinking models must not receive the flag — ollama errors the call
    assert "think" not in captured["body"]


def test_ollama_client_connection_error_raises_judge_unavailable(monkeypatch):
    def fake_urlopen(request, timeout=None):
        raise ConnectionRefusedError("nope")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(JudgeUnavailable):
        ollama_client("prompt", config())


def test_ollama_client_malformed_json_raises_judge_unavailable(monkeypatch):
    def fake_urlopen(request, timeout=None):
        return _FakeResponse(b"not json at all")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(JudgeUnavailable):
        ollama_client("prompt", config())


def test_ollama_client_missing_choices_raises_judge_unavailable(monkeypatch):
    def fake_urlopen(request, timeout=None):
        return _FakeResponse(json.dumps({"unexpected": "shape"}).encode("utf-8"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(JudgeUnavailable):
        ollama_client("prompt", config())


def test_prompt_does_not_leak_kpi_score() -> None:
    """The judge verdict is consumed as an INDEPENDENT signal (training/outcome_score.py weights
    it 2.0 precisely to escape kpi_score's bias; training/reward_model.py calls it an external
    calibration set). Showing the judge the score makes that independence false."""
    turn = make_turn(kpi_score=1.0, outcome="ok", flagged=0)
    prompt = build_prompt(turn, [], [])

    assert "kpi_score" not in prompt
    assert "1.0" not in prompt.splitlines()[0]
