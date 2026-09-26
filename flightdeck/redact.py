"""Scrub secrets out of captured text before it ever touches disk.

Telemetry comes from a machine where the user pastes API keys into prompts, tool args, and tool
results. This module is the only backstop between that text and a file that gets rsynced to
another box and read by an unattended nightly agent. A miss here is a real credential leak, so
every path fails closed: if redaction itself errors, the caller must drop the text rather than
store it raw.
"""

from __future__ import annotations

import re
from typing import Any

REDACTED = "[REDACTED:{kind}]"

_MAX_PAYLOAD_DEPTH = 12


class RedactionError(Exception):
    """Raised when a pattern application fails. Callers see this only via redact_or_drop."""


# Order matters: more specific patterns run first so a token that could match two shapes is
# labeled by its most specific context (e.g. a JWT following "Bearer " is tagged `bearer`, not
# separately caught and double-substituted by the bare `jwt` pattern).
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "pem",
        re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
            re.DOTALL,
        ),
    ),
    (
        "bearer",
        re.compile(
            # "Bearer" is case-sensitive (RFC 6750 casing, keeps false positives on the common
            # English word down); the "Authorization" header name is case-insensitive per HTTP.
            r"(?P<prefix>\bBearer\b\s+|(?i:Authorization):\s*\w+\s+)"
            r"(?P<token>[A-Za-z0-9\-._~+/]+=*)"
        ),
    ),
    (
        "url_userinfo",
        # Bounded quantifiers: an unbounded scheme/userinfo run over a long literal-free string
        # (no "://" or "@" anywhere) forces the engine to retry every start position, which is
        # the catastrophic-backtracking shape requirement 3 tests for.
        re.compile(r"(?P<scheme>[a-zA-Z][a-zA-Z0-9+.\-]{0,15}://)(?P<userinfo>[^\s/@]{1,255})@"),
    ),
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}\b")),
    ("openai_key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_\-]{20,}\b")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b")),
    ("aws_key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]+\b")),
    (
        "jwt",
        re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"),
    ),
    (
        "kv_secret",
        # Generic catch-all, so it runs last. The value alternatives all reject a leading
        # "[REDACTED:" so a value already scrubbed by a more specific pattern above (or by the
        # bearer pattern, which leaves the scheme word in place) is never wrapped a second time.
        #
        # The key alternation is deliberately narrow. It used to end in `\w*`, which made every
        # English word starting with a key name a secret: "passed", "passing", "tokens",
        # "authority" all matched. On a store whose subject matter IS token counts and pass/fail
        # results that is not a rare edge -- 54 live `texts` rows carry [REDACTED:kv_secret] and
        # a sample of them was 100% false positives ("passed": ..., pass_at_1: ..., tokens: ...),
        # destroying exactly the signal the nightly reviewer reads. `(?:[_-]\w+)*` still catches
        # secret_value / token_id / api_key_v2, but a continuation now needs a separator, and
        # bare `pass` is gone so pass_rate and pass_at_1 no longer look like credentials.
        re.compile(
            r"""(?i)(?P<key>\b(?:pass(?:word|wd)|secret|token|api[_-]?key|auth(?!orization)
                |credential)(?:[_-]\w+)*)
                (?P<sep>["']?\s*[:=]\s*)
                (?P<value>"(?!\[REDACTED:)[^"\n]+"|'(?!\[REDACTED:)[^'\n]+'|(?!\[REDACTED:)\S+)""",
            re.VERBOSE,
        ),
    ),
]


def _apply(kind: str, pattern: re.Pattern[str], text: str, counts: dict[str, int]) -> str:
    placeholder = REDACTED.format(kind=kind)

    def _sub(match: re.Match[str]) -> str:
        counts[kind] = counts.get(kind, 0) + 1
        groups = match.groupdict()
        if "prefix" in groups:
            return f"{match.group('prefix')}{placeholder}"
        if "scheme" in groups:
            return f"{match.group('scheme')}{placeholder}@"
        if "key" in groups:
            return f"{match.group('key')}{match.group('sep')}{placeholder}"
        return placeholder

    return pattern.sub(_sub, text)


def redact(text: str) -> tuple[str, dict[str, int]]:
    """Return scrubbed text and a count of substitutions per pattern kind."""
    counts: dict[str, int] = {}
    for kind, pattern in _PATTERNS:
        text = _apply(kind, pattern, text, counts)
    return text, counts


def redact_or_drop(text: str) -> str | None:
    """Redact; return None if redaction itself raised. Callers must drop on None."""
    try:
        scrubbed, _ = redact(text)
        return scrubbed
    except Exception:
        return None


def _redact_value(value: Any, depth: int) -> Any:
    if depth > _MAX_PAYLOAD_DEPTH:
        raise RedactionError(f"payload exceeds max depth {_MAX_PAYLOAD_DEPTH}")
    if isinstance(value, str):
        scrubbed = redact_or_drop(value)
        return scrubbed if scrubbed is not None else REDACTED.format(kind="unredactable")
    if isinstance(value, dict):
        return {k: _redact_value(v, depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_value(v, depth + 1) for v in value]
    return value


def redact_payload(payload: dict) -> dict:
    """Recursively redact string values in an event payload dict (lists/dicts nested)."""
    return _redact_value(payload, 0)
