from __future__ import annotations

import copy
import time

import pytest

from flightdeck import redact as redact_module
from flightdeck.redact import RedactionError, redact, redact_or_drop, redact_payload

# ---------- per-pattern kind: positive + near-miss negative ----------


def test_anthropic_key() -> None:
    text = "key is sk-ant-" + "A1b2C3d4E5f6G7h8I9j0" + "_extra-chars"
    scrubbed, counts = redact(text)
    assert "[REDACTED:anthropic_key]" in scrubbed
    assert counts == {"anthropic_key": 1}
    assert "A1b2C3d4E5f6G7h8I9j0" not in scrubbed


def test_anthropic_key_near_miss_too_short() -> None:
    text = "key is sk-ant-short"
    scrubbed, counts = redact(text)
    assert scrubbed == text
    assert counts == {}


def test_openai_key() -> None:
    text = "OPENAI_API_KEY=sk-proj-" + "abcdefghijklmnopqrstuvwxyz012345"
    scrubbed, counts = redact(text)
    assert "[REDACTED:openai_key]" in scrubbed
    assert counts.get("openai_key") == 1
    assert "abcdefghijklmnopqrstuvwxyz012345" not in scrubbed


def test_openai_key_near_miss_too_short() -> None:
    text = "token was sk-short123"
    scrubbed, counts = redact(text)
    assert "openai_key" not in counts


def test_github_token() -> None:
    text = "auth with ghp_" + "A" * 36
    scrubbed, counts = redact(text)
    assert "[REDACTED:github_token]" in scrubbed
    assert counts.get("github_token") == 1


def test_github_token_near_miss_too_short() -> None:
    text = "auth with ghp_tooshort"
    scrubbed, counts = redact(text)
    assert "github_token" not in counts


def test_aws_key() -> None:
    text = "AWS_ACCESS_KEY_ID=AKIA" + "Q" * 16
    scrubbed, counts = redact(text)
    assert "[REDACTED:aws_key]" in scrubbed
    assert counts.get("aws_key") == 1


def test_aws_key_near_miss_lowercase() -> None:
    text = "akia" + "q" * 16 + " is not an aws key"
    scrubbed, counts = redact(text)
    assert "aws_key" not in counts


def test_slack_token() -> None:
    text = "slack token " + "xoxb-1234567890-" + "abcdefghijklmnop"
    scrubbed, counts = redact(text)
    assert "[REDACTED:slack_token]" in scrubbed
    assert counts.get("slack_token") == 1


def test_slack_token_near_miss_bad_letter() -> None:
    text = "not real " + "xoxz-1234567890-" + "abcdef"
    scrubbed, counts = redact(text)
    assert "slack_token" not in counts


def test_jwt() -> None:
    text = (
        "token=eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0."
        "dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
    )
    scrubbed, counts = redact(text)
    assert "eyJhbGciOiJIUzI1NiJ9" not in scrubbed
    # The kv_secret "token=" prefix means the whole thing gets caught either as jwt or
    # kv_secret; either way it must be redacted exactly once with no leftover fragments.
    assert sum(counts.values()) == 1


def test_jwt_near_miss_no_dots() -> None:
    text = "value is eyJhbGciOiJIUzI1NiJ9 alone"
    scrubbed, counts = redact(text)
    assert "jwt" not in counts


def test_pem() -> None:
    text = (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "MIIBogIBAAJBAKj34GkxFhD90vcNLYLInFEr8NF3vGwzTB4qEBrbG"
        "\n-----END RSA PRIVATE KEY-----"
    )
    scrubbed, counts = redact(text)
    assert scrubbed == "[REDACTED:pem]"
    assert counts == {"pem": 1}


def test_pem_near_miss_certificate() -> None:
    text = "-----BEGIN CERTIFICATE-----\nMIIB...\n-----END CERTIFICATE-----"
    scrubbed, counts = redact(text)
    assert "pem" not in counts
    assert scrubbed == text


def test_bearer() -> None:
    text = "Authorization: Bearer abc123.def456-ghi_789"
    scrubbed, counts = redact(text)
    assert scrubbed == "Authorization: Bearer [REDACTED:bearer]"
    assert counts == {"bearer": 1}


def test_bearer_basic_scheme_kept() -> None:
    text = "Authorization: Basic dXNlcjpwYXNz"
    scrubbed, counts = redact(text)
    assert scrubbed == "Authorization: Basic [REDACTED:bearer]"


def test_bearer_near_miss_prose() -> None:
    text = "she was the bearer of good news"
    scrubbed, counts = redact(text)
    assert scrubbed == text
    assert "bearer" not in counts


def test_kv_secret() -> None:
    text = 'password: "hunter2"'
    scrubbed, counts = redact(text)
    assert scrubbed == "password: [REDACTED:kv_secret]"
    assert "hunter2" not in scrubbed
    assert counts == {"kv_secret": 1}


def test_kv_secret_unquoted() -> None:
    text = "api_key=abcdef123456"
    scrubbed, counts = redact(text)
    assert scrubbed == "api_key=[REDACTED:kv_secret]"


def test_kv_secret_near_miss_prose() -> None:
    text = "the password is required before you continue"
    scrubbed, counts = redact(text)
    assert scrubbed == text
    assert "kv_secret" not in counts


def test_url_userinfo() -> None:
    text = "postgres://dbuser:s3cret@db.internal:5432/app"
    scrubbed, counts = redact(text)
    assert scrubbed == "postgres://[REDACTED:url_userinfo]@db.internal:5432/app"
    assert counts == {"url_userinfo": 1}


def test_url_userinfo_near_miss_no_credentials() -> None:
    text = "https://example.com/path?user=carter"
    scrubbed, counts = redact(text)
    assert scrubbed == text
    assert "url_userinfo" not in counts


def test_ips_and_paths_not_redacted() -> None:
    text = "connecting to 198.51.100.20 at /Users/carter/dev/agent-telemetry-feedback-control/flightdeck/redact.py"
    scrubbed, counts = redact(text)
    assert scrubbed == text
    assert counts == {}


# ---------- ordering ----------


def test_jwt_inside_bearer_labeled_bearer_not_jwt() -> None:
    jwt = (
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0."
        "dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
    )
    text = f"Authorization: Bearer {jwt}"
    scrubbed, counts = redact(text)
    assert counts == {"bearer": 1}
    assert jwt not in scrubbed
    assert scrubbed == "Authorization: Bearer [REDACTED:bearer]"


# ---------- performance bound ----------


def test_adversarial_input_stays_fast() -> None:
    # Matches the spec's example shape ("a" * N + "sk-ant-..."): a long run of a single
    # character is exactly what forces catastrophic backtracking in an unbounded-quantifier
    # regex. A leading space keeps the \b boundary so the key pattern still gets a fair shot.
    text = "a" * 200_000 + " sk-ant-" + "B2c3D4e5F6g7H8i9J0k1L2m3N4o5"
    start = time.monotonic()
    scrubbed, counts = redact(text)
    elapsed = time.monotonic() - start
    assert elapsed < 1.0, f"redact() took {elapsed:.3f}s on adversarial input"
    assert counts.get("anthropic_key") == 1
    assert "[REDACTED:anthropic_key]" in scrubbed


def test_adversarial_input_no_boundary_still_fast() -> None:
    # The literal example from the spec: no separator before "sk-ant-", so the \b boundary
    # blocks a match entirely. Still must not blow up — pure backtracking-safety check.
    text = "a" * 100_000 + "sk-ant-"
    start = time.monotonic()
    redact(text)
    elapsed = time.monotonic() - start
    assert elapsed < 1.0, f"redact() took {elapsed:.3f}s on adversarial input"


# ---------- fail closed ----------


def test_redact_or_drop_returns_none_when_apply_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(kind: str, pattern, text: str, counts: dict) -> str:  # noqa: ANN001
        raise RuntimeError("simulated regex engine failure")

    monkeypatch.setattr(redact_module, "_apply", boom)
    assert redact_or_drop("api_key=super-secret-value") is None


def test_redact_raises_when_apply_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(kind: str, pattern, text: str, counts: dict) -> str:  # noqa: ANN001
        raise RuntimeError("simulated regex engine failure")

    monkeypatch.setattr(redact_module, "_apply", boom)
    with pytest.raises(RuntimeError):
        redact("api_key=super-secret-value")


def test_redact_payload_never_leaks_raw_text_when_redaction_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(kind: str, pattern, text: str, counts: dict) -> str:  # noqa: ANN001
        raise RuntimeError("simulated regex engine failure")

    monkeypatch.setattr(redact_module, "_apply", boom)
    secret = "sk-ant-" + "A1b2C3d4E5f6G7h8I9j0"
    result = redact_payload({"prompt": secret})
    assert secret not in result["prompt"]


# ---------- payload nesting + no mutation ----------


def test_redact_payload_nested_and_no_mutation() -> None:
    payload = {
        "prompt": "my password: hunter2",
        "meta": {
            "nested": ["fine", "api_key=abcdef123456", {"deep": "token: xyz789secret"}],
            "count": 3,
            "flag": None,
        },
    }
    original = copy.deepcopy(payload)

    result = redact_payload(payload)

    # original untouched
    assert payload == original

    assert "hunter2" not in result["prompt"]
    assert "abcdef123456" not in result["meta"]["nested"][1]
    assert "xyz789secret" not in result["meta"]["nested"][2]["deep"]
    # non-str leaves pass through untouched
    assert result["meta"]["nested"][0] == "fine"
    assert result["meta"]["count"] == 3
    assert result["meta"]["flag"] is None


def test_redact_payload_within_depth_limit_ok() -> None:
    # redact_payload scrubs recognizable secret shapes inside string values; the dict key
    # itself carries no meaning to it, so the leaf value has to look like a real secret.
    payload: dict = {"note": "api_key=abcdef123456"}
    node = payload
    for _ in range(5):
        node["child"] = {"note": "api_key=abcdef123456"}
        node = node["child"]
    result = redact_payload(payload)
    assert "abcdef123456" not in str(result)


def test_redact_payload_too_deep_raises() -> None:
    payload: dict = {"v": "leaf"}
    node = payload
    for _ in range(20):
        node["child"] = {"v": "leaf"}
        node = node["child"]
    with pytest.raises(RedactionError):
        redact_payload(payload)


# ---------- idempotency ----------


@pytest.mark.parametrize(
    "text",
    [
        "sk-ant-" + "A1b2C3d4E5f6G7h8I9j0" + "extra",
        "Authorization: Bearer abc.def-ghi_123",
        "password: hunter2",
        "postgres://dbuser:s3cret@db.internal/app",
        ("-----BEGIN RSA PRIVATE KEY-----\nMIIBogIBAAJB\n-----END RSA PRIVATE KEY-----"),
        "ghp_" + "A" * 40,
        "AKIA" + "Q" * 16,
        "xoxb-1234567890-" + "abcdefghijklmnop",
        (
            "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0."
            "dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
        ),
    ],
)
def test_idempotent(text: str) -> None:
    once, _ = redact(text)
    twice, _ = redact(once)
    assert once == twice


def test_kv_secret_does_not_eat_pass_fail_or_token_counts() -> None:
    """This store's own subject matter is token counts and pass/fail verdicts. A key pattern
    ending in \\w* made every one of them a "secret" -- 54 live rows, sampled 100% false
    positive -- which redacted the signal the nightly reviewer exists to read."""
    for text in (
        "CheckResult.passed: true",
        "pass_at_1: 0.42",
        "model supports max tokens: 128000",
        "authority: local",
        "passing: 21/21",
    ):
        scrubbed, counts = redact(text)
        assert scrubbed == text, f"over-redacted: {text!r} -> {scrubbed!r}"
        assert "kv_secret" not in counts


def test_kv_secret_still_catches_real_keys() -> None:
    for text, key in (
        ('password: "hunter2"', "password"),
        ("api_key=abcdef123456", "api_key"),
        ("secret_value: swordfish", "secret_value"),
        ("token_id = deadbeef", "token_id"),
    ):
        scrubbed, counts = redact(text)
        assert counts.get("kv_secret") == 1, f"missed a real key: {text!r}"
        assert scrubbed.startswith(key)
