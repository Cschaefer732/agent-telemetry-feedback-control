"""Model -> tier derivation.

`tier` is flightdeck's own vocabulary (models.TIERS: fast, balanced, deep, frontier) — the same
vocabulary governor.toml's [arms.model_tier] and [weights.model_tier] use, and the only vocabulary
graduation_report's arm-vs-actual comparison can match against (it compares a governor_decisions
row's `chosen` arm, which is always one of those four names, straight against turns.tier). Writing
anything else into this column — including whatever internal model-tier naming your own fleet
uses — would make every turn permanently "disagree" with its own shadow decision instead of
leaving the comparison at null, which is worse than the current gap. Keep this vocabulary and
your fleet's naming separate; map between them in one place if you need both.

Coverage here is deliberately partial and never guesses: an unrecognized model, or a source this
module has no rule for, resolves to None, exactly as if nothing ran at all. `claude-code` is the
one source handled without looking at `model`, because collect_claude.py's hook payloads never
carry a model id (documented on that module) but a claude-code turn is, by construction, always a
frontier (Anthropic) turn — collect_claude.py's own module docstring calls it that.
"""

from __future__ import annotations

# Provider-prefix noise observed in the store (e.g. "ollama/qwen3.8:27b") that isn't part of
# the model's identity for tiering purposes.
_PREFIXES = ("ollama/", "vllm/", "cerebras/")

# Normalized (lowercased, provider-prefix stripped) model id -> flightdeck tier. This mapping is
# an EXAMPLE for one particular fleet of locally-hosted models; edit it for your own model roster.
MODEL_TIER: dict[str, str] = {
    # fast: general/default tier -- the cheapest resident model.
    "qwen3.8:27b": "fast",
    "qwen3:4b-instruct-2507-q4_k_m": "fast",  # smallest model observed; no cheaper slot exists
    # balanced: mid coding tier, between the general (fast) and specialist (deep) tiers, for
    # mid-size non-"-next" coder models.
    "qwen3-coder:30b": "balanced",
    "qwen2.5-coder:32b": "balanced",
    # deep: coding-specialist / heavy tier.
    "qwen3-coder-next:latest": "deep",
    "qwen3-coder-next": "deep",
    "gpt-oss-120b": "deep",
    "gpt-oss:120b": "deep",
}


def _normalize(model: str) -> str:
    normalized = model.strip().lower()
    for prefix in _PREFIXES:
        if normalized.startswith(prefix):
            return normalized[len(prefix) :]
    return normalized


def derive_tier(source: str | None, model: str | None) -> str | None:
    """Best-effort tier for a turn that doesn't have one yet, or None if it can't legitimately be
    derived. Never fabricates: no mapping means no tier, not a guess."""
    if source == "claude-code":
        return "frontier"
    if not model:
        return None
    normalized = _normalize(model)
    if normalized in MODEL_TIER:
        return MODEL_TIER[normalized]
    if "claude" in normalized or normalized.startswith("frontier/"):
        return "frontier"
    return None
